# -*- coding: utf-8 -*-
"""
43_Treino_RViaria.py
Train Deep Learning Model - rede viária.
Experiências (ativo=True para correr):
  - U-Net binário: rede viária, chips 256 (rede_viaria_chip256_unet_binario),
    com LR do LR Finder limitado a LR_LIMITES
  - U-Net multiclasse: pavimentada / não pavimentada / ferrovia (desativado:
    3 tentativas sem aprender)
  - Multi-Task Road Extractor binários (rede viária, ferrovia): já treinados

MODO_TESTE = True  -> 1 época por modelo, sem paragem automática, para validar
                      o fluxo, o LR encontrado, a memória da GPU e o tempo por época.
MODO_TESTE = False -> treino real (até EPOCAS, com paragem automática).

Registo guardado para cada experiência (pasta <modelo>_registo, ao lado da
pasta do modelo, para não interferir com a escrita do modelo):
  - parametros.json : configuração completa, versões, GPU, nº de chips
  - mensagens.txt   : todas as mensagens do geoprocessamento (perdas por época)
  - resumo.json     : início, fim, duração, estado, batch usado, LR encontrado, .emd
  - TensorBoard     : curvas de treino (dentro da pasta do modelo), se ativo
E um resumo de todas as experiências em <PASTA_MODELOS>\\resumo_treinos.csv

Executar na janela Python, num Notebook do ArcGIS Pro 3.6 ou na linha de comandos
(arcgispro-py3).
"""
import os
# Antes de importar arcpy/PyTorch:
#  - matplotlib sem interface gráfica (Tk): evita 'Tcl_AsyncDelete: async handler
#    deleted by the wrong thread', que termina o processo na linha de comandos
#  - reduz a fragmentação da memória da GPU
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import matplotlib
matplotlib.use("Agg", force=True)
import re
import sys
import csv
import subprocess
import json
import glob
import time
import platform
import datetime
import arcpy

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.env.overwriteOutput = True
arcpy.env.processorType = "GPU"
arcpy.env.gpuId = 0

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
MODO_TESTE = False        # True = 1 época por modelo; False = treino real

TRAINING = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\Training"
PASTA_MULTI = os.path.join(TRAINING, "vias_chip256_unet")              # 3 classes (U-Net)
#PASTA_REDE = os.path.join(TRAINING, "rede_viaria_chip512_binario")        # binário
#PASTA_FERRO = os.path.join(TRAINING, "ferrovia_chip512_binario")          # binário
PASTA_MODELOS = r"D:\DL\Modelos\126_RViaria_Resnet50"

EPOCAS = 50
VALIDACAO = 10          # % de validação, como no baseline dos edifícios
CHIP = 256
PARAGEM = "STOP_TRAINING"   # paragem automática quando a perda de validação estagna
MONITOR = "VALID_LOSS"

MTRE_ARGS = "gaussian_thresh 0.76;orient_bin_size 20;mtl_model linknet"

# LR automático limitado: se learning_rate=None e a experiência tiver lr_limites,
# o LR Finder corre antes (processo separado, ~3 min) e o valor encontrado é
# limitado a [mín, máx]; o treino usa esse valor como LR fixo.
LR_LIMITES = (1e-4, 1e-3)
SONDA_TIMEOUT_MIN = 15   # se a sondagem não terminar a tempo, segue com o LR automático
PASTA_REDE_256 = os.path.join(TRAINING, "rede_viaria_chip256_unet_binario")  # binário, chips 256

# Cada experiência: nome -> parâmetros. ativo=False para não correr.
# epocas (opcional) sobrepõe-se a EPOCAS para essa experiência.
# learning_rate=None -> taxa ótima estimada automaticamente (LR Finder).
# batches: tentados por ordem; se faltar memória na GPU passa ao seguinte.
# tensorboard (opcional, por defeito True).
EXPERIENCIAS = {
    # U-Net multiclasse: pavimentada / não pavimentada / ferrovia
    # U-Net binário: rede viária (mesmas etiquetas do Multi-Task Road Extractor)
    "unet_resnet50_rede_binario": dict(
        ativo=True,
        pasta=PASTA_REDE_256,
        model_type="UNET",
        backbone_model="RESNET50",
        arguments="class_balancing True",
        learning_rate=None,
        lr_limites=LR_LIMITES,
        batches=[8, 4],
        tensorboard=False,
        tag="binario_lrlim",
    ),
    # U-Net multiclasse: 3 tentativas sem aprender (LR auto ~5e-5; LR 2e-4 + focal loss)
    "unet_resnet50": dict(
        ativo=False,
        pasta=PASTA_MULTI,
        model_type="UNET",
        backbone_model="RESNET50",
        arguments="class_balancing True",
        learning_rate=0.0002,
        batches=[8, 4],
        tensorboard=False,
        tag="lr2e-4",
    ),
    # # Multi-Task Road Extractor binário: rede viária (pavimentada + não pavimentada)
    # "mtre_rede_viaria_resnet50": dict(
        # ativo=False,
        # pasta=PASTA_REDE,
        # model_type="MULTITASK_ROADEXTRACTOR",
        # backbone_model="RESNET50",
        # arguments=MTRE_ARGS,
        # learning_rate=None,
        # batches=[8, 4],
    # ),
    # # Multi-Task Road Extractor binário: ferrovia
    # "mtre_ferrovia_resnet50": dict(
        # ativo=False,
        # epocas=100,   # poucos chips (~34 batches/época): mais épocas para atualizações suficientes
        # pasta=PASTA_FERRO,
        # model_type="MULTITASK_ROADEXTRACTOR",
        # backbone_model="RESNET50",
        # arguments=MTRE_ARGS,
        # learning_rate=None,
        # batches=[8, 4],
    # ),
}

ERROS_MEMORIA = ("out of memory", "cuda error", "cudnn_status_alloc_failed")


# ---------------------------------------------------------------------------
def agora():
    return datetime.datetime.now().isoformat(timespec="seconds")


def carimbo():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def info_ambiente():
    info = {
        "arcgis_pro": arcpy.GetInstallInfo().get("Version"),
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "processador": arcpy.env.processorType,
        "gpu_id": arcpy.env.gpuId,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import arcgis
        info["arcgis_api"] = arcgis.__version__
    except Exception:
        pass
    return info


def contar_chips(pasta):
    """Conta só os chips .tif (ignora .tfw, .aux.xml e outros auxiliares)."""
    imgs = os.path.join(pasta, "images")
    if not os.path.isdir(imgs):
        return None
    return sum(1 for f in os.listdir(imgs) if f.lower().endswith((".tif", ".tiff")))


def validar_pasta(pasta):
    """Confirma que todos os pares do map.txt existem. Devolve (n_pares, em_falta)."""
    mapa = os.path.join(pasta, "map.txt")
    if not os.path.exists(mapa):
        return 0, ["map.txt inexistente"]
    pares, falta = 0, []
    with open(mapa, encoding="utf-8") as fh:
        for linha in fh:
            partes = linha.split()
            if not partes:
                continue
            pares += 1
            for rel in partes[:2]:
                rel = rel.replace("\\", os.sep).replace("/", os.sep)
                if not os.path.exists(os.path.join(pasta, rel)):
                    falta.append(rel)
    return pares, falta


def ler_emd(pasta):
    for emd in glob.glob(os.path.join(pasta, "*.emd")):
        try:
            with open(emd, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            return {"erro_leitura_emd": str(e)}
    return None


def lr_encontrado(mensagens):
    """Extrai o LR das mensagens: 'slice(a, b, None)' ou um valor único."""
    m = re.search(r"Learning Rate\s*-\s*slice\(([\d.eE+-]+),\s*([\d.eE+-]+)", mensagens)
    if m:
        return {"tipo": "slice", "min": float(m.group(1)), "max": float(m.group(2))}
    m = re.search(r"Learning Rate\s*-\s*([\d.eE+-]+)", mensagens)
    if m:
        return {"tipo": "valor", "valor": float(m.group(1))}
    return None


SONDA = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg", force=True)
from arcgis.learn import prepare_data, UnetClassifier, MultiTaskRoadExtractor
cfg = json.loads(sys.argv[1])
data = prepare_data(cfg["pasta"], batch_size=cfg["batch"], chip_size=cfg["chip"],
                    val_split_pct=cfg["val"] / 100)
if cfg["model_type"] == "UNET":
    m = UnetClassifier(data, backbone=cfg["backbone"].lower(), **cfg["kwargs"])
else:
    m = MultiTaskRoadExtractor(data, backbone=cfg["backbone"].lower(), **cfg["kwargs"])
try:
    lr = m.lr_find(allow_plot=False)
except TypeError:
    lr = m.lr_find()
print("LR_SONDA=" + repr(float(lr)))
"""


def kwargs_de_argumentos(texto):
    """'class_balancing True;mtl_model linknet' -> {'class_balancing': True, ...}"""
    out = {}
    for par in (texto or "").split(";"):
        if not par.strip():
            continue
        k, v = par.strip().split(None, 1)
        v = v.strip()
        if v in ("True", "False"):
            v = v == "True"
        else:
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
        out[k] = v
    return out


def sondar_lr(p, batch):
    """Corre o LR Finder num processo separado e devolve (lr_encontrado, lr_usado)."""
    cfg = dict(pasta=p["pasta"], batch=batch, chip=CHIP, val=VALIDACAO,
               model_type=p["model_type"], backbone=p["backbone_model"],
               kwargs=kwargs_de_argumentos(p["arguments"]))
    t0 = time.time()
    print(f"  sondagem do LR (até {SONDA_TIMEOUT_MIN} min)...", flush=True)
    try:
        r = subprocess.run([sys.executable, "-c", SONDA, json.dumps(cfg)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=SONDA_TIMEOUT_MIN * 60)
    except subprocess.TimeoutExpired:
        print(f"  AVISO: sondagem do LR sem resposta ao fim de {SONDA_TIMEOUT_MIN} min "
              "(arranque do python.exe bloqueado?)")
        return None, None
    m = re.search(r"LR_SONDA=([\d.eE+-]+)", r.stdout)
    if not m:
        print(f"  AVISO: sondagem do LR falhou ({time.time() - t0:.0f} s):\n"
              f"{(r.stderr or r.stdout)[-1500:]}")
        return None, None
    lr = float(m.group(1))
    lo, hi = p["lr_limites"]
    usado = min(max(lr, lo), hi)
    nota = "" if usado == lr else f" -> limitado a {usado:g}"
    print(f"  sondagem do LR: {lr:.3g}{nota} (intervalo {lo:g}–{hi:g}, {time.time() - t0:.0f} s)")
    return lr, usado


def falta_memoria(mensagens):
    texto = mensagens.lower()
    return any(e in texto for e in ERROS_MEMORIA)


def treinar(nome, p, ambiente):
    n_chips = contar_chips(p["pasta"])
    epocas = 1 if MODO_TESTE else p.get("epocas", EPOCAS)
    paragem = "CONTINUE_TRAINING" if MODO_TESTE else PARAGEM
    sufixo = "teste" if MODO_TESTE else "treino"
    tensorboard = "ENABLE_TENSORBOARD" if p.get("tensorboard", True) else "DISABLE_TENSORBOARD"

    for batch in p["batches"]:
        lr_sonda, lr_usado = None, p["learning_rate"]
        if p["learning_rate"] is None and p.get("lr_limites"):
            lr_sonda, lr_usado = sondar_lr(p, batch)
            if lr_usado is None:
                print("  a seguir com o LR automático do Train Deep Learning Model")
        tag = p.get("tag") or ("lrauto" if p["learning_rate"] is None else f"lr{p['learning_rate']:g}")
        saida = os.path.join(PASTA_MODELOS, f"{nome}_b{batch}_{tag}_{sufixo}_{carimbo()}")
        registo = saida + "_registo"
        os.makedirs(registo, exist_ok=True)

        parametros = {
            "experiencia": nome,
            "modo_teste": MODO_TESTE,
            "pasta_treino": p["pasta"],
            "n_chips": n_chips,
            "max_epocas": epocas,
            "paragem": paragem,
            "monitor": MONITOR,
            "validacao_pct": VALIDACAO,
            "batch": batch,
            "chip": CHIP,
            "model_type": p["model_type"],
            "backbone_model": p["backbone_model"],
            "arguments": p["arguments"],
            "learning_rate": lr_usado or "automatico",
            "lr_sonda": lr_sonda,
            "lr_limites": p.get("lr_limites"),
            "tensorboard": tensorboard,
            "ambiente": ambiente,
        }
        with open(os.path.join(registo, "parametros.json"), "w", encoding="utf-8") as fh:
            json.dump(parametros, fh, indent=2, ensure_ascii=False)

        lr_txt = f"{lr_usado:g}" if lr_usado else "automático"
        print(f"\n=== {nome} | {n_chips} chips | batch={batch} | épocas<={epocas} | lr={lr_txt} "
              f"| paragem={paragem} | {tensorboard} === ({agora()})")
        inicio, t0, estado = agora(), time.time(), "concluido"
        try:
            arcpy.ia.TrainDeepLearningModel(
                in_folder=p["pasta"],
                out_folder=saida,
                max_epochs=epocas,
                model_type=p["model_type"],
                batch_size=batch,
                arguments=p["arguments"],
                learning_rate=lr_usado,
                backbone_model=p["backbone_model"],
                validation_percentage=VALIDACAO,
                stop_training=paragem,
                freeze="FREEZE_MODEL",
                augmentation="DEFAULT",
                chip_size=CHIP,
                monitor=MONITOR,
                tensorboard=tensorboard,
            )
        except arcpy.ExecuteError:
            estado = "erro"
        duracao = (time.time() - t0) / 60

        mensagens = arcpy.GetMessages()
        with open(os.path.join(registo, "mensagens.txt"), "w", encoding="utf-8") as fh:
            fh.write(mensagens)
        print(mensagens)

        lr = lr_encontrado(mensagens)
        resumo = {
            "experiencia": nome,
            "modo_teste": MODO_TESTE,
            "estado": estado,
            "batch": batch,
            "inicio": inicio,
            "fim": agora(),
            "duracao_min": round(duracao, 1),
            "lr_encontrado": lr,
            "lr_sonda": lr_sonda,
            "lr_usado": lr_usado,
            "pasta_modelo": saida,
            "emd": ler_emd(saida),
        }
        with open(os.path.join(registo, "resumo.json"), "w", encoding="utf-8") as fh:
            json.dump(resumo, fh, indent=2, ensure_ascii=False)

        print(f"{estado.upper()} em {duracao:.1f} min | LR encontrado: {lr} -> {saida}")
        if MODO_TESTE and estado == "concluido":
            print(f"  (teste) tempo de 1 época com batch {batch}: {duracao:.0f} min "
                  f"-> {EPOCAS} épocas no máximo: ~{duracao * EPOCAS / 60:.1f} h")

        if estado == "erro" and falta_memoria(mensagens):
            print(f"  memória insuficiente com batch {batch}; a tentar o seguinte")
            continue
        return resumo

    return resumo   # último resultado, se todos os batches falharam


# ---------------------------------------------------------------------------
os.makedirs(PASTA_MODELOS, exist_ok=True)
ambiente = info_ambiente()
print(f"Modo teste: {MODO_TESTE} | matplotlib: {matplotlib.get_backend()}")
ATIVAS = {n: p for n, p in EXPERIENCIAS.items() if p["ativo"]}
if not ATIVAS:
    raise SystemExit("Nenhuma experiência ativa: pôr ativo=True nas que se quer treinar.")

PRONTAS = {}
for n, p in ATIVAS.items():
    chips = contar_chips(p["pasta"])
    pares, falta = validar_pasta(p["pasta"])
    if not chips:
        print(f"  SALTADA {n}: sem chips em {p['pasta']} (exportação em falta?)")
    elif falta:
        print(f"  SALTADA {n}: {len(falta)} ficheiros do map.txt em falta (ex.: {falta[:3]})")
    else:
        print(f"  {n}: {chips} chips, {pares} pares em {p['pasta']}")
        PRONTAS[n] = p
if not PRONTAS:
    raise SystemExit("Nenhuma experiência com dados válidos.")
print(f"Experiências a treinar: {', '.join(PRONTAS)}")

resumos = [treinar(n, p, ambiente) for n, p in PRONTAS.items()]

csv_path = os.path.join(PASTA_MODELOS, "resumo_treinos.csv")
novo = not os.path.exists(csv_path)
with open(csv_path, "a", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh, delimiter=";")
    if novo:
        w.writerow(["experiencia", "modo_teste", "estado", "batch", "inicio", "fim",
                    "duracao_min", "lr_min", "lr_max", "lr_sonda", "lr_usado",
                    "chip", "pasta_modelo"])
    for r in resumos:
        lr = r["lr_encontrado"] or {}
        lr_min = lr.get("min", lr.get("valor", ""))
        lr_max = lr.get("max", lr.get("valor", ""))
        w.writerow([r["experiencia"], r["modo_teste"], r["estado"], r["batch"],
                    r["inicio"], r["fim"], r["duracao_min"], lr_min, lr_max,
                    r.get("lr_sonda") or "", r.get("lr_usado") or "",
                    CHIP, r["pasta_modelo"]])
print(f"\nResumo -> {csv_path}")