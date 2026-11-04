# -*- coding: utf-8 -*-
r"""
treinar_fase1_ocupacao.py  (ArcGIS Pro 3.6, extensão Image Analyst)

FASE 1 do modelo de OCUPAÇÃO DO SOLO: treina, em sequência e com os mesmos chips,
  - U-Net      ResNet-50
  - DeepLabV3  ResNet-50
  - PSPNet     ResNet-50
  - SAMLoRA

Parâmetros comuns (comparação justa): mesmos chips (256/128), mesma validação,
mesma paragem antecipada, ignore_classes=[8] (Ignorar) e class_balancing=True.

Se um treino falhar por falta de memória da GPU, repete com o batch size seguinte
da lista do modelo (ex.: 16 -> 8 -> 4).

Cada modelo fica numa pasta própria. Se um modelo falhar, o erro fica no relatório
e o script passa ao seguinte. Modelos já treinados (pasta com .emd) são saltados,
para poder retomar depois de uma interrupção.

Saídas em PASTA_MODELOS:
  <modelo>\                       modelo treinado (.emd, .pth, .dlpk, model_metrics.html)
  treino_fase1_relatorio.txt      parâmetros, tempos e estado de cada modelo
  treino_fase1_resumo.csv         uma linha por modelo

Correr no Python Command Prompt (de preferência com o ArcGIS Pro fechado, para
libertar a memória da GPU).
"""
import os
import csv
import shutil
import glob
import json
import time
import datetime
import traceback
import arcpy

# =============================== PARÂMETROS ===============================
CHIPS = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\23_Training\chips_ocupacao_256_128"
PASTA_MODELOS = r"D:\DL\Modelos\modelos_ocupacao_fase1"

IGNORE = 8
CHIP_SIZE = 256
EPOCHS = 50                 # máximo; com STOP_TRAINING para quando a validação deixa de melhorar
VALIDACAO = 10              # %
LEARNING_RATE = 0.0002      # o mesmo dos edifícios; None = o ArcGIS escolhe (lr_find)
SALTAR_SE_EXISTIR = True    # não volta a treinar modelos que já tenham .emd

# Backbone congelado (o valor por defeito, como nos edifícios e nas vias).
# "UNFREEZE_MODEL" afinaria a rede toda, incluindo os pesos da banda NIR.
FREEZE = "FREEZE_MODEL"
# Inicialização dos pesos das bandas que não são RGB (só imagens multiespectrais).
# None = valor por defeito da ferramenta (RANDOM), como nos treinos anteriores;
# "RED_BAND" copiaria para o NIR os pesos pré-treinados do vermelho.
# Só é aplicado aos modelos com backbone ResNet (não ao SAMLoRA).
WEIGHT_INIT = None

# Argumentos comuns (formato "nome valor;nome valor" do Train Deep Learning Model)
ARGS_COMUNS = f"class_balancing True;ignore_classes [{IGNORE}]"

# nome da pasta: (model_type, backbone, batch sizes a tentar, argumentos, learning_rate)
# Os batch sizes são tentados por ordem: se o treino falhar por falta de memória
# da GPU, apaga a tentativa e repete com o seguinte.
# backbone None = o valor por defeito da ferramenta
MODELOS = {
    "unet_r50":      ("UNET",    "RESNET50", [16, 8, 4], ARGS_COMUNS, LEARNING_RATE),
    "deeplabv3_r50": ("DEEPLAB", "RESNET50", [16, 8, 4], ARGS_COMUNS, LEARNING_RATE),
    "pspnet_r50":    ("PSPNET",  "RESNET50", [16, 8, 4], ARGS_COMUNS, LEARNING_RATE),
    "samlora":       ("SAMLORA", "VIT_B",    [8, 4, 2],  ARGS_COMUNS, LEARNING_RATE),
}
# Texto que identifica um erro de memória da GPU (só estes erros levam a nova tentativa)
ERROS_MEMORIA = ("out of memory", "cuda error", "cudnn", "memoryerror",
                 "not enough memory", "alloc")
# Ordem de treino (o SAMLoRA primeiro por ser o mais rápido: confirma logo que tudo funciona)
ORDEM = ["samlora", "unet_r50", "deeplabv3_r50", "pspnet_r50"]
# ==========================================================================

arcpy.env.overwriteOutput = True
arcpy.env.processorType = "GPU"
arcpy.env.gpuId = 0


class Relatorio:
    def __init__(self, caminho):
        self.caminho = caminho
        self.linhas = []

    def __call__(self, txt=""):
        print(txt)
        self.linhas.append(str(txt))
        self.gravar()   # grava sempre: o relatório sobrevive a uma interrupção

    def gravar(self):
        os.makedirs(os.path.dirname(self.caminho), exist_ok=True)
        with open(self.caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas) + "\n")


def ja_treinado(pasta):
    return os.path.isdir(pasta) and bool(glob.glob(os.path.join(pasta, "*.emd")))


def ler_emd(pasta):
    """Informação útil do .emd (classes e, se existirem, métricas guardadas)."""
    emds = glob.glob(os.path.join(pasta, "*.emd"))
    if not emds:
        return {}
    try:
        with open(emds[0], encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}
    info = {"emd": emds[0]}
    cls = d.get("Classes") or []
    info["classes"] = [c.get("Value", c.get("ClassValue")) for c in cls]
    info["bandas"] = d.get("ExtractBands")
    for k in ("accuracy", "Accuracy", "metrics", "Metrics", "ModelMetrics"):
        if k in d:
            info["metricas"] = d[k]
    return info


def validar(rel):
    if not os.path.isdir(CHIPS):
        raise FileNotFoundError(f"Pasta de chips não existe: {CHIPS}")
    n_img = len(glob.glob(os.path.join(CHIPS, "images", "*.tif")))
    n_lab = len(glob.glob(os.path.join(CHIPS, "labels", "*.tif")))
    rel(f"Chips: {n_img} imagens | {n_lab} rótulos")
    if n_img == 0 or n_img != n_lab:
        raise RuntimeError("Chips em falta ou número de imagens diferente do de rótulos.")
    if not os.path.exists(os.path.join(CHIPS, "esri_model_definition.emd")):
        rel("  AVISO: esri_model_definition.emd não encontrado na pasta dos chips")
    if arcpy.CheckExtension("ImageAnalyst") != "Available":
        raise RuntimeError("Extensão Image Analyst não disponível.")
    arcpy.CheckOutExtension("ImageAnalyst")
    rel()


def libertar_gpu():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def erro_de_memoria(texto):
    t = (texto or "").lower()
    return any(k in t for k in ERROS_MEMORIA)


def treinar(nome, cfg, rel):
    model_type, backbone, batches, args, lr = cfg
    pasta = os.path.join(PASTA_MODELOS, nome)
    linha = {"modelo": nome, "model_type": model_type, "backbone": backbone or "defeito",
             "batch": "", "batches_tentados": "", "argumentos": args,
             "learning_rate": lr if lr else "lr_find",
             "estado": "", "minutos": "", "pasta": pasta}

    rel("-" * 78)
    rel(f"MODELO: {nome}  ({model_type}, backbone {backbone or 'defeito'}, "
        f"batch sizes a tentar {batches})")
    rel(f"  argumentos: {args}")
    rel(f"  learning rate: {lr if lr else 'automático (lr_find)'}")

    if SALTAR_SE_EXISTIR and ja_treinado(pasta):
        rel("  já treinado -> saltado")
        linha["estado"] = "saltado (já existia)"
        return linha

    tentados = []
    t_total = time.time()
    for batch in batches:
        tentados.append(str(batch))
        linha["batches_tentados"] = ",".join(tentados)
        if os.path.isdir(pasta):            # restos de uma tentativa anterior
            shutil.rmtree(pasta, ignore_errors=True)
        libertar_gpu()

        kwargs = dict(
            in_folder=CHIPS,
            out_folder=pasta,
            max_epochs=EPOCHS,
            model_type=model_type,
            batch_size=batch,
            arguments=args,
            validation_percentage=VALIDACAO,
            stop_training="STOP_TRAINING",
            freeze=FREEZE,
            chip_size=CHIP_SIZE,
            monitor="VALID_LOSS",
        )
        if backbone:
            kwargs["backbone_model"] = backbone
        if lr:
            kwargs["learning_rate"] = lr
        if WEIGHT_INIT and model_type != "SAMLORA":
            kwargs["weight_init_scheme"] = WEIGHT_INIT

        t = time.time()
        rel(f"  batch {batch} - início: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        try:
            arcpy.ia.TrainDeepLearningModel(**kwargs)
        except Exception:
            mins = (time.time() - t) / 60
            msgs = arcpy.GetMessages(2) or ""
            texto = msgs + "\n" + traceback.format_exc()
            if erro_de_memoria(texto) and batch != batches[-1]:
                rel(f"  batch {batch}: sem memória na GPU ao fim de {mins:.1f} min "
                    "-> nova tentativa com batch menor")
                continue
            rel(f"  ERRO com batch {batch} ao fim de {mins:.1f} min")
            rel("  " + (msgs or traceback.format_exc()).replace("\n", "\n  "))
            linha.update(estado="ERRO", batch=batch,
                         minutos=round((time.time() - t_total) / 60, 1))
            libertar_gpu()
            return linha

        mins = (time.time() - t) / 60
        rel(f"  OK com batch {batch} - {mins:.1f} min")
        linha.update(estado="OK", batch=batch, minutos=round(mins, 1))
        if batch != batches[0]:
            rel(f"  NOTA: treinado com batch {batch} (e não {batches[0]}) - registar na comparação")
        info = ler_emd(pasta)
        if info.get("classes"):
            rel(f"  classes no modelo: {info['classes']}")
        if info.get("bandas") is not None:
            rel(f"  bandas usadas (ExtractBands): {info['bandas']}")
        if info.get("metricas"):
            rel(f"  métricas no .emd: {info['metricas']}")
        html = os.path.join(pasta, "model_metrics.html")
        if os.path.exists(html):
            rel(f"  métricas de validação: {html}")
        libertar_gpu()
        return linha

    return linha


def main():
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    rel = Relatorio(os.path.join(PASTA_MODELOS, "treino_fase1_relatorio.txt"))
    t0 = time.time()
    rel("=" * 78)
    rel("RELATÓRIO - TREINO FASE 1 - OCUPAÇÃO DO SOLO")
    rel("=" * 78)
    rel(f"Data/hora:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rel(f"ArcGIS Pro:  {arcpy.GetInstallInfo().get('Version', '?')}")
    rel()
    rel("PARÂMETROS COMUNS")
    for n in ("CHIPS", "PASTA_MODELOS", "IGNORE", "CHIP_SIZE", "EPOCHS", "VALIDACAO",
              "LEARNING_RATE", "ARGS_COMUNS", "FREEZE", "WEIGHT_INIT"):
        rel(f"  {n:<14} = {globals()[n]}")
    rel(f"  ordem          = {ORDEM}")
    rel()

    validar(rel)

    resumo = []
    for nome in ORDEM:
        resumo.append(treinar(nome, MODELOS[nome], rel))
        with open(os.path.join(PASTA_MODELOS, "treino_fase1_resumo.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(resumo[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(resumo)

    rel("=" * 78)
    rel("RESUMO")
    for r in resumo:
        rel(f"  {r['modelo']:<15} {r['estado']:<22} batch {r['batch'] or '-':<4} {r['minutos']} min")
    rel(f"Tempo total: {(time.time() - t0) / 3600:.2f} h")
    rel()
    rel("PRÓXIMO PASSO: Classify Pixels Using Deep Learning nas 3 ortofotos de teste")
    rel("(126-I-3, 126-II-4, 126-IV-3) e avaliação com rotulos_teste.shp")
    arcpy.CheckInExtension("ImageAnalyst")


if __name__ == "__main__":
    main()