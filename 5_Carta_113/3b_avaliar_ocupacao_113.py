r"""
3b_avaliar_ocupacao_113.py  (ArcGIS Pro 3.6)

Avaliacao dos modelos de OCUPACAO DO SOLO treinados na 126 na FOLHA 113, contra a
referencia criada pelo 3a_rotulos_ocupacao_113.py (mesmas regras dos rotulos da 126).

E o 30_avaliar_ocupacao.py aplicado a 113: as funcoes de classificacao, matriz de
confusao, metricas e agrupamentos sao COPIADAS SEM ALTERACOES (mesmas classes, mesmos
pixeis excluidos [0, 8], mesmos agrupamentos "completa" e "generalizada", mesmo mapa
das previsoes do SAMLoRA v1 {0: 9}, mesmos argumentos de inferencia com TTA), para os
resultados serem comparaveis com os da 126 (D:\DL\Avaliacao\126_ocupacao_TTA). Mudam:

  1. CLASSIFICACAO: uma vez por modelo, no ortomosaico 113 inteiro ORIGINAL (sem
     calibracao; ver 0_verificar_dominio_126_113.py). As classificacoes ja feitas sao
     reutilizadas se o modelo, os argumentos e o ortomosaico forem os mesmos (.json ao lado).
  2. UNIDADES DE AVALIACAO: em vez das 3 ortos de teste, os 16 BLOCOS da grelha 4x4 do
     0_verificar_dominio (mesmos nomes, ex.: 113_bloco_L4C4), na coluna 'orto' dos CSV,
     e o TOTAL da folha (soma das matrizes dos blocos).
  3. No fim, resumo_blocos.txt cruza o desempenho de cada bloco com a radiometria do
     0_verificar_dominio (correlacao de Spearman).

Com MODO_TESTE a True (teste RAPIDO) classifica so com o primeiro modelo, numa janela de
TESTE_LADO_M x TESTE_LADO_M (com CONTEXTO_M de margem), e corre o resto do pipeline
(referencia, matrizes, metricas, tabelas), so para confirmar que funciona. Da tambem
uma estimativa grosseira do tempo da folha inteira (so vale com a GPU livre).

Saidas em PASTA_AVALIACAO: metricas_classe.csv, metricas_globais.csv,
tabela_comparativa.txt, resumo_blocos.txt, matrizes\..., classificacoes\<modelo>\...,
referencia_113.tif, grelha_blocos.shp, log_avaliacao.txt

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro fechado), na pasta do
script (precisa do comum_113.py na mesma pasta):
    python 3b_avaliar_ocupacao_113.py
    python 3b_avaliar_ocupacao_113.py --cpu     (sempre no CPU)
"""
print("script iniciado, a carregar o arcpy...", flush=True)

import arcpy

print("arcpy carregado", flush=True)
import os
import sys
import csv
import glob
import time
import subprocess
import json
import traceback
import numpy as np
from datetime import datetime

import comum_113 as comum

# ============================================================
# CONFIGURACAO - folha 113
# ============================================================
ORTOMOSAICO = r"D:\Users\Cap_Almeida\Projeto\113\Ortomosaico\113.tif"     # original, sem calibracao
REFERENCIA = r"D:\Users\Cap_Almeida\Projeto\113\113_OcupacaoSolo\rotulos_ocupacao\rotulos_ocupacao.shp"
CAMPO_CLASSE = "classe"
PASTA_VERIFICACAO = r"D:\Users\Cap_Almeida\Projeto\113\Verificacoes_dominio"   # None = sem radiometria

# Argumentos de inferencia (iguais ao 30)
ARGS_V1 = "padding 64;batch_size 4;predict_background True;test_time_augmentation False"
ARGS_V2 = "padding 64;batch_size 4;predict_background False;test_time_augmentation False"
ARGS_V1_TTA = "padding 64;batch_size 4;predict_background True;test_time_augmentation True"
ARGS_V2_TTA = "padding 64;batch_size 4;predict_background False;test_time_augmentation True"

# Modelos: (rotulo, pasta do modelo, mapa previsao -> codigo da referencia, argumentos)
# Com TTA, como na ultima avaliacao da 126 (D:\DL\Avaliacao\126_ocupacao_TTA).
MODELOS = [
    # sem TTA -> PASTA_AVALIACAO = r"D:\DL\Avaliacao\113_ocupacao"
    # ("samlora_v1",       r"D:\DL\Modelos\126_ocupacao_samlora\samlora", {0: 9}, ARGS_V1),
    # ("unet_r50_v2",      r"D:\DL\Modelos\126_ocupacao_2v\unet_r50", {}, ARGS_V2),
    # ("deeplabv3_r50_v2", r"D:\DL\Modelos\126_ocupacao_2v\deeplabv3_r50", {}, ARGS_V2),
    # ("pspnet_r50_v2",    r"D:\DL\Modelos\126_ocupacao_2v\pspnet_r50", {}, ARGS_V2),

    # com TTA -> PASTA_AVALIACAO = r"D:\DL\Avaliacao\113_ocupacao_TTA"
    ("samlora_v1_tta",       r"D:\DL\Modelos\126_ocupacao_samlora\samlora", {0: 9}, ARGS_V1_TTA),
    ("unet_r50_v2_tta",      r"D:\DL\Modelos\126_ocupacao_2v\unet_r50", {}, ARGS_V2_TTA),
    ("deeplabv3_r50_v2_tta", r"D:\DL\Modelos\126_ocupacao_2v\deeplabv3_r50", {}, ARGS_V2_TTA),
    ("pspnet_r50_v2_tta",    r"D:\DL\Modelos\126_ocupacao_2v\pspnet_r50", {}, ARGS_V2_TTA),
]

PASTA_AVALIACAO = r"D:\DL\Avaliacao\113_ocupacao_TTA"

# Classes, exclusoes e agrupamentos (iguais ao 30)
CLASSES = {1: "Vinha", 2: "Pomar_vinha", 3: "Pomar", 4: "Arvoredo_denso",
           5: "Arvoredo_esparso", 6: "Mato_denso", 7: "Agua", 9: "Sem_representacao"}
IGNORAR_REF = {0, 8}
SEM_REF = 255
AGRUPAMENTOS = {
    "completa": {},
    "generalizada": {2: 1},          # Pomar e vinha passa a Vinha
}

CELL_SIZE = 0.5
MEM_GPU_MIN_GB = 6
GPU_ID = 0
FORCAR_CPU = "--cpu" in sys.argv

MODO_TESTE = False
TESTE_LADO_M = 1500
TESTE_CENTRO = None       # None = centro do ortomosaico; ou (x, y) no sistema do ortomosaico
CONTEXTO_M = 256          # margem de contexto a volta do bloco de teste (como no 30)
# ============================================================

FICHEIRO_LOG = os.path.join(PASTA_AVALIACAO, "log_avaliacao.txt")
TOLERANCIA_FORA = 0.1
SINAIS_BLOQUEIO = ("winerror 32", "being used by another process")


# ------------------------------------------------------------------ funcoes copiadas do 30
def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(FICHEIRO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def nome_seguro(txt):
    return txt.replace("-", "_").replace(" ", "_").replace(".", "_")


def apagar(*itens):
    for it in itens:
        if it and arcpy.Exists(it):
            arcpy.management.Delete(it)


def contar(fc):
    return int(arcpy.management.GetCount(fc)[0])


def ler_grelha(raster, grelha, nodata):
    """Le o raster na grelha exata do raster 'grelha'; o NoData passa a 'nodata'."""
    g = arcpy.Raster(grelha)
    canto = arcpy.Point(g.extent.XMin, g.extent.YMin)
    return arcpy.RasterToNumPyArray(raster, canto, g.width, g.height, nodata_to_value=nodata)


def rasterizar(fc, campo, saida, grelha):
    """Rasteriza os poligonos com o valor do 'campo' na grelha exata do raster 'grelha'."""
    with arcpy.EnvManager(extent=grelha, snapRaster=grelha, cellSize=CELL_SIZE,
                          outputCoordinateSystem=grelha):
        arcpy.conversion.PolygonToRaster(fc, campo, saida, "CELL_CENTER", "NONE", CELL_SIZE)
    return saida


def encontrar_dlpk(caminho):
    """Ficheiro do modelo a usar: o .emd (preferido) ou o .dlpk."""
    if caminho.lower().endswith((".emd", ".dlpk")):
        return caminho if os.path.exists(caminho) else None
    for padrao in ("*.emd", "*.dlpk"):
        f = sorted(glob.glob(os.path.join(caminho, padrao)))
        if f:
            return f[0]
    # procura tambem nas subpastas (ex.: ArcGIS guarda por vezes em <pasta>\<nome>\)
    for padrao in ("*.emd", "*.dlpk"):
        f = sorted(glob.glob(os.path.join(caminho, "**", padrao), recursive=True))
        f = [x for x in f if "checkpoint" not in x.lower()]
        if f:
            return f[0]
    return None


def listar_pasta(caminho):
    """Para diagnostico: o que existe realmente na pasta do modelo."""
    if not os.path.exists(caminho):
        pai = os.path.dirname(caminho)
        irmas = sorted(os.listdir(pai)) if os.path.isdir(pai) else "pasta-mae tambem nao existe"
        return f"a pasta NAO existe. Conteudo de {pai}: {irmas}"
    itens = []
    for raiz, pastas, ficheiros in os.walk(caminho):
        for fx in ficheiros:
            itens.append(os.path.relpath(os.path.join(raiz, fx), caminho))
        if len(itens) > 40:
            break
    return f"{len(itens)} ficheiros: {itens[:40]}"


def bandas_do_modelo(modelo):
    """N.o de bandas com que o modelo foi treinado (ExtractBands do .emd), ou None."""
    emd = modelo if modelo.lower().endswith(".emd") else \
        next(iter(sorted(glob.glob(os.path.join(os.path.dirname(modelo), "*.emd")))), None)
    if not emd:
        return None
    try:
        with open(emd, encoding="utf-8") as f:
            eb = json.load(f).get("ExtractBands")
        return len(eb) if eb else None
    except Exception:
        return None


def assinatura(modelo, args):
    """O que define uma classificacao: modelo (pasta, ficheiro e data), argumentos e ortomosaico."""
    # o modelo e identificado pela sua pasta, nome do ficheiro e data (e nao pelo caminho
    # completo), para que mudar o nome das pastas de cima nao obrigue a classificar de novo
    return {"modelo": os.path.join(os.path.basename(os.path.dirname(os.path.abspath(modelo))),
                                   os.path.basename(modelo)),
            "modelo_data": round(os.path.getmtime(modelo), 0),
            "argumentos": args,
            "ortomosaico": os.path.abspath(ORTOMOSAICO)}


def ler_assinatura(cl_tif):
    js = os.path.splitext(cl_tif)[0] + ".json"
    if not os.path.exists(js):
        return None
    try:
        with open(js, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("legado", None)
        d.pop("data_classificacao", None)
        return d
    except Exception:
        return {}


def gravar_assinatura(cl_tif, sig, legado=False):
    d = dict(sig, data_classificacao=f"{datetime.now():%Y-%m-%d %H:%M}")
    if legado:
        d["legado"] = True
    with open(os.path.splitext(cl_tif)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def memoria_gpu_livre_gb():
    """Memoria livre da GPU em GB (nvidia-smi), ou None se nao for possivel saber."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        valores = [float(v) for v in out.strip().splitlines() if v.strip()]
        return valores[GPU_ID] / 1024 if len(valores) > GPU_ID else None
    except Exception:
        return None


def escolher_processador():
    livre = memoria_gpu_livre_gb()
    if FORCAR_CPU:
        return "CPU", livre
    if livre is not None and livre >= MEM_GPU_MIN_GB:
        return "GPU", livre
    return "CPU", livre


def classificar(orto_tif, dlpk, out_tif, args):
    proc, livre = escolher_processador()
    txt_livre = f"{livre:.1f} GB livres" if livre is not None else "memoria livre desconhecida"
    motivo = "forcado pela opcao --cpu" if FORCAR_CPU else f"minimo para usar a GPU: {MEM_GPU_MIN_GB} GB"
    log(f"      processador: {proc} (GPU com {txt_livre}; {motivo})")
    for tentativa in range(3):
        try:
            env = dict(cellSize=CELL_SIZE, snapRaster=orto_tif, processorType=proc)
            if proc == "GPU":
                env["gpuId"] = GPU_ID
            with arcpy.EnvManager(**env):
                res = arcpy.ia.ClassifyPixelsUsingDeepLearning(orto_tif, dlpk, args)
                res.save(out_tif)
            return proc
        except Exception as e:
            if any(s in str(e).lower() for s in SINAIS_BLOQUEIO) and tentativa < 2:
                log("      ficheiro bloqueado; nova tentativa dentro de 2 min")
                time.sleep(120)
                continue
            raise


def matriz_confusao(ref, pred, valido):
    """Matriz 256 x 256 (linhas = referencia, colunas = previsao)."""
    r = ref[valido].astype(np.int64)
    p = np.clip(pred[valido].astype(np.int64), 0, 255)
    return np.bincount(r * 256 + p, minlength=256 * 256).reshape(256, 256).astype(np.int64)


def agrupar(M, mapa):
    if not mapa:
        return M, sorted(CLASSES)
    M2 = np.zeros_like(M)
    f = lambda c: mapa.get(c, c)
    for i, j in zip(*np.nonzero(M)):
        M2[f(i), f(j)] += M[i, j]
    return M2, sorted({f(c) for c in CLASSES})


def nome_classe(c, mapa):
    juntas = [CLASSES[k] for k in CLASSES if mapa.get(k, k) == c]
    return "+".join(juntas) if juntas else str(c)


def calcular_metricas(M, labels):
    n = int(M[labels, :].sum())
    por_classe = []
    for c in labels:
        tp = int(M[c, c])
        sup = int(M[c, :].sum())
        prev = int(M[labels, c].sum())
        fp, fn = prev - tp, sup - tp
        prec = tp / prev if prev else 0.0
        rec = tp / sup if sup else 0.0
        por_classe.append(dict(classe=c, suporte=sup, previstos=prev, TP=tp, FP=fp, FN=fn,
                               precision=prec, recall=rec,
                               F1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                               IoU=tp / (tp + fp + fn) if (tp + fp + fn) else 0.0))
    com_sup = [m for m in por_classe if m["suporte"] > 0]
    oa = sum(m["TP"] for m in por_classe) / n if n else 0.0
    pe = sum(m["suporte"] * m["previstos"] for m in por_classe) / (n * n) if n else 0.0
    fora = n - sum(m["previstos"] for m in por_classe)
    glob_ = dict(
        pixeis=n,
        exatidao_global=oa,
        exatidao_media=float(np.mean([m["recall"] for m in com_sup])) if com_sup else 0.0,
        kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
        mIoU=float(np.mean([m["IoU"] for m in com_sup])) if com_sup else 0.0,
        fwIoU=sum(m["suporte"] / n * m["IoU"] for m in com_sup) if n else 0.0,
        F1_macro=float(np.mean([m["F1"] for m in com_sup])) if com_sup else 0.0,
        F1_ponderado=sum(m["suporte"] / n * m["F1"] for m in com_sup) if n else 0.0,
        pct_previsoes_fora=100 * fora / n if n else 0.0,
    )
    return por_classe, glob_


def gravar_matriz(M, labels, mapa, caminho):
    outros = [c for c in range(256) if c not in labels and M[labels, c].sum() > 0]
    cab = ["referencia \\ previsao"] + [nome_classe(c, mapa) for c in labels] + \
          [f"outro_{c}" for c in outros] + ["total", "recall_%"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["CONTAGENS (pixeis)"])
        w.writerow(cab)
        for r in labels:
            linha = [int(M[r, c]) for c in labels] + [int(M[r, c]) for c in outros]
            tot = int(M[r, :].sum())
            w.writerow([nome_classe(r, mapa)] + linha + [tot, round(100 * M[r, r] / tot, 2) if tot else ""])
        w.writerow([])
        w.writerow(["PERCENTAGEM DE CADA LINHA (para onde vai cada classe da referencia)"])
        w.writerow(cab[:-2])
        for r in labels:
            tot = M[r, :].sum()
            w.writerow([nome_classe(r, mapa)] +
                       [round(100 * M[r, c] / tot, 2) if tot else "" for c in labels + outros])


# ------------------------------------------------------------------ NOVO (folha 113)
def recortar(ras, ext, saida):
    with arcpy.EnvManager(snapRaster=ORTOMOSAICO, cellSize=ORTOMOSAICO):
        arcpy.management.Clip(ras, f"{ext[0]} {ext[1]} {ext[2]} {ext[3]}", saida, "#", "#", "NONE",
                              "MAINTAIN_EXTENT")
    return saida


def preparar_regiao(blocos):
    """Raster a classificar, grelha de leitura e blocos (em pixeis dessa grelha)."""
    if not MODO_TESTE:
        return ORTOMOSAICO, ORTOMOSAICO, blocos, ""
    b = dict(nome="teste_janela", ext=comum.janela_teste(ORTOMOSAICO, TESTE_LADO_M, TESTE_CENTRO))
    log(f"MODO TESTE: janela de {TESTE_LADO_M:g} m {tuple(round(v) for v in b['ext'])}")
    pasta = os.path.join(PASTA_AVALIACAO, "modo_teste")
    os.makedirs(pasta, exist_ok=True)
    m = arcpy.Describe(ORTOMOSAICO).extent
    x0, y0, x1, y1 = b["ext"]
    ctx = (max(x0 - CONTEXTO_M, m.XMin), max(y0 - CONTEXTO_M, m.YMin),
           min(x1 + CONTEXTO_M, m.XMax), min(y1 + CONTEXTO_M, m.YMax))
    orto = os.path.join(pasta, "orto_teste_janela_contexto.tif")
    grelha = os.path.join(pasta, "grelha_teste_janela.tif")
    apagar(orto, grelha)                 # a janela pode ter mudado
    recortar(ORTOMOSAICO, ctx, orto)
    recortar(ORTOMOSAICO, b["ext"], grelha)
    g = arcpy.Raster(grelha)
    return orto, grelha, [dict(b, r0=0, r1=g.height, c0=0, c1=g.width)], "_modo_teste"


def resumo_blocos(l_glb, rad, caminho):
    """Metricas globais por bloco (agrupamento completo) vs radiometria (EMD vs 126)."""
    L = ["OCUPACAO DO SOLO 113 - DESEMPENHO POR BLOCO vs RADIOMETRIA (agrupamento 'completa')",
         "rho = Spearman entre o EMD medio do bloco (vs 126) e a metrica; negativo = blocos",
         "radiometricamente mais diferentes da 126 tem pior desempenho. Com 16 blocos e indicativo.",
         "Atencao: a composicao da paisagem tambem muda de bloco para bloco.", ""]
    f = lambda v: f"{v:.3f}" if v is not None else "-"
    for rotulo in dict.fromkeys(l[0] for l in l_glb):
        ls = {l[2]: l for l in l_glb if l[0] == rotulo and l[1] == "completa" and l[2] != "TOTAL"}
        if not ls:
            continue
        # colunas de l_glb: modelo, agr, orto, pixeis, OA, exat_media, kappa, mIoU, fwIoU, F1mac, F1pond, fora
        L.append(f"MODELO {rotulo}")
        L.append(f"  {'bloco':<20}{'EMD':>7}{'fora126':>9}{'pixeis':>12}{'OA':>7}{'mIoU':>7}{'F1mac':>7}")
        for b in sorted(ls):
            r, l = rad.get(b, {}), ls[b]
            emd = r.get("emd_medio")
            L.append(f"  {b:<20}{(format(emd, '.1f') if emd is not None else '-'):>7}"
                     f"{r.get('fora_126', '-'):>9}{l[3]:>12,}{l[4]:>7.3f}{l[7]:>7.3f}{l[9]:>7.3f}")
        if rad:
            bl = sorted(ls)
            emd = [rad.get(b, {}).get("emd_medio") for b in bl]
            L.append(f"  rho(EMD, OA) = {f(comum.spearman(emd, [ls[b][4] for b in bl]))} | "
                     f"rho(EMD, mIoU) = {f(comum.spearman(emd, [ls[b][7] for b in bl]))} | "
                     f"rho(EMD, F1mac) = {f(comum.spearman(emd, [ls[b][9] for b in bl]))}")
        L.append("")
    with open(caminho, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("\n".join(L))


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(PASTA_AVALIACAO, exist_ok=True)
    os.makedirs(os.path.join(PASTA_AVALIACAO, "matrizes"), exist_ok=True)
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.CheckOutExtension("Spatial")

    log("=" * 60)
    log(f"Avaliacao 113 iniciada | modo_teste={MODO_TESTE} | forcar_cpu={FORCAR_CPU} | "
        f"modelos={[m[0] for m in MODELOS]}")
    for nome, c in (("ORTOMOSAICO", ORTOMOSAICO), ("REFERENCIA", REFERENCIA)):
        if not arcpy.Exists(c):
            raise FileNotFoundError(f"{nome} nao existe: {c}")
    r_mos = arcpy.Raster(ORTOMOSAICO)
    log(f"ORTOMOSAICO: {ORTOMOSAICO} | {r_mos.bandCount} bandas | pixel {r_mos.meanCellWidth} m")
    if abs(r_mos.meanCellWidth - CELL_SIZE) > 1e-6:
        raise ValueError(f"O pixel do ORTOMOSAICO ({r_mos.meanCellWidth} m) e diferente de "
                         f"CELL_SIZE ({CELL_SIZE} m)")
    livre = memoria_gpu_livre_gb()
    log(f"GPU: {f'{livre:.1f} GB livres' if livre is not None else 'nao foi possivel consultar'} "
        f"-> {escolher_processador()[0]}")

    # blocos (mesma grelha do 0_verificar_dominio) e regiao a classificar
    blocos = comum.blocos_raster(ORTOMOSAICO)
    comum.criar_fc_blocos(blocos, r_mos.spatialReference, os.path.join(PASTA_AVALIACAO, "grelha_blocos.shp"))
    orto_cl, grelha, blocos, sufixo = preparar_regiao(blocos)
    rad = comum.ler_radiometria(PASTA_VERIFICACAO)
    log(f"{len(blocos)} bloco(s) | radiometria do 0_verificar: {'sim' if rad else 'nao'}")

    # referencia ao pixel na grelha (refeita se a shapefile for mais recente)
    ref_tif = os.path.join(PASTA_AVALIACAO, f"referencia_113{sufixo}.tif")
    if MODO_TESTE:
        apagar(ref_tif)                  # a janela pode ter mudado
    shp_mtime = max(os.path.getmtime(f) for f in glob.glob(os.path.splitext(REFERENCIA)[0] + ".*"))
    if os.path.exists(ref_tif) and os.path.getmtime(ref_tif) < shp_mtime:
        log("   referencia rasterizada mais antiga do que a shapefile -> refeita")
        apagar(ref_tif)
    if not os.path.exists(ref_tif):
        t0 = time.time()
        rasterizar(REFERENCIA, CAMPO_CLASSE, ref_tif, grelha)
        log(f"   referencia rasterizada em {(time.time() - t0) / 60:.1f} min")
    ref = ler_grelha(ref_tif, grelha, SEM_REF).astype(np.uint8)
    valido = ~np.isin(ref, list(IGNORAR_REF | {SEM_REF}))
    vals, cont = np.unique(ref, return_counts=True)
    log("   referencia (pixeis): " + ", ".join(
        f"{CLASSES.get(int(v), {0: 'residual', 8: 'Ignorar', SEM_REF: 'sem_ref'}.get(int(v), v))}={c:,}"
        for v, c in zip(vals, cont)))
    log(f"   avaliados: {int(valido.sum()):,} pixeis")

    l_cls, l_glb, comparativa = [], [], []
    pasta_cl_base = os.path.join(PASTA_AVALIACAO, "classificacoes_modo_teste" if MODO_TESTE else "classificacoes")

    for rotulo, caminho, mapa_pred, args in (MODELOS[:1] if MODO_TESTE else MODELOS):
        dlpk = encontrar_dlpk(caminho)
        if not dlpk:
            log(f"ERRO {rotulo}: nenhum .emd ou .dlpk em {caminho}")
            log(f"   {listar_pasta(caminho)}")
            continue
        log(f"MODELO {rotulo} | {dlpk} | mapa das previsoes: {mapa_pred or 'nenhum'} | {args}")
        nb_modelo = bandas_do_modelo(dlpk)
        log(f"   bandas: modelo {nb_modelo if nb_modelo else '?'} | ortomosaico {r_mos.bandCount}")
        if nb_modelo and nb_modelo != r_mos.bandCount:
            log(f"ERRO {rotulo}: bandas diferentes - modelo ignorado")
            continue
        pasta_cl = os.path.join(pasta_cl_base, rotulo)
        os.makedirs(pasta_cl, exist_ok=True)
        try:
            cl_tif = os.path.join(pasta_cl, "class_113.tif")
            sig = assinatura(dlpk, args)
            if MODO_TESTE:
                apagar(cl_tif)           # a janela pode ter mudado: refazer sempre
            if os.path.exists(cl_tif):
                antiga = ler_assinatura(cl_tif)
                if antiga is not None and antiga != sig:
                    dif = [k for k in sig if antiga.get(k) != sig[k]]
                    log(f"   classificacao existente feita com outro(s) {dif} -> refeita")
                    apagar(cl_tif)
            if not os.path.exists(cl_tif):
                t0 = time.time()
                proc = classificar(orto_cl, dlpk, cl_tif, args)
                gravar_assinatura(cl_tif, sig)
                mins = (time.time() - t0) / 60
                r = arcpy.Raster(orto_cl)
                km2 = r.width * r.height * r.meanCellWidth * r.meanCellHeight / 1e6
                log(f"   classificacao em {mins:.1f} min ({proc}) para {km2:.1f} km2")
                if MODO_TESTE:
                    km2_folha = r_mos.width * r_mos.height * CELL_SIZE ** 2 / 1e6
                    log(f"   MODO TESTE: folha inteira ({km2_folha:.0f} km2) ~{mins / km2 * km2_folha / 60:.1f} h "
                        f"neste processador, so para este modelo")
            else:
                log("   classificacao ja existente, reutilizada")

            pred = ler_grelha(cl_tif, grelha, 0)
            if mapa_pred:
                pred_m = pred.copy()
                for de, para in mapa_pred.items():
                    pred_m[pred == de] = para
                pred = pred_m
            n_val = int(valido.sum())
            vals, cont = np.unique(pred[valido], return_counts=True)
            fora = {int(v): 100 * c / n_val for v, c in zip(vals, cont) if int(v) not in CLASSES}
            if sum(fora.values()) >= TOLERANCIA_FORA:
                log("   AVISO previsoes fora das classes avaliadas (% dos pixeis avaliados): " +
                    ", ".join(f"{v}: {p:.2f}%" for v, p in sorted(fora.items())))
        except Exception as e:
            log(f"ERRO {rotulo}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")
            continue

        M_total = np.zeros((256, 256), np.int64)
        for b in blocos:
            sl = (slice(b["r0"], b["r1"]), slice(b["c0"], b["c1"]))
            M = matriz_confusao(ref[sl], pred[sl], valido[sl])
            M_total += M
            if M.sum() == 0:
                log(f"   {b['nome']}: sem pixeis avaliados")
                continue
            for agr, mapa in AGRUPAMENTOS.items():
                Ma, labels = agrupar(M, mapa)
                pc, gl = calcular_metricas(Ma, labels)
                for m in pc:
                    l_cls.append([rotulo, agr, b["nome"], m["classe"], nome_classe(m["classe"], mapa),
                                  m["suporte"], m["previstos"], m["TP"], m["FP"], m["FN"]] +
                                 [round(m[k], 4) for k in ("precision", "recall", "F1", "IoU")])
                l_glb.append([rotulo, agr, b["nome"]] + [round(v, 4) if isinstance(v, float) else v
                                                         for v in gl.values()])
                gravar_matriz(Ma, labels, mapa, os.path.join(
                    PASTA_AVALIACAO, "matrizes", f"{rotulo}_{agr}_{b['nome']}{sufixo}.csv"))
                if agr == "completa":
                    log(f"   {b['nome']}: OA={gl['exatidao_global']:.3f} kappa={gl['kappa']:.3f} "
                        f"mIoU={gl['mIoU']:.3f} F1macro={gl['F1_macro']:.3f}")

        if M_total.sum() == 0:
            continue
        for agr, mapa in AGRUPAMENTOS.items():
            Ma, labels = agrupar(M_total, mapa)
            pc, gl = calcular_metricas(Ma, labels)
            for m in pc:
                l_cls.append([rotulo, agr, "TOTAL", m["classe"], nome_classe(m["classe"], mapa),
                              m["suporte"], m["previstos"], m["TP"], m["FP"], m["FN"]] +
                             [round(m[k], 4) for k in ("precision", "recall", "F1", "IoU")])
            l_glb.append([rotulo, agr, "TOTAL"] + [round(v, 4) if isinstance(v, float) else v
                                                   for v in gl.values()])
            gravar_matriz(Ma, labels, mapa, os.path.join(PASTA_AVALIACAO, "matrizes",
                                                         f"{rotulo}_{agr}_TOTAL{sufixo}.csv"))
            comparativa.append((rotulo, agr, labels, mapa, pc, gl))
            log(f"   TOTAL {rotulo} | {agr}: OA={gl['exatidao_global']:.3f} kappa={gl['kappa']:.3f} "
                f"mIoU={gl['mIoU']:.3f} fwIoU={gl['fwIoU']:.3f} F1macro={gl['F1_macro']:.3f}")
        del pred

    # --- CSV (mesmas colunas do 30; 'orto' = bloco ou TOTAL)
    def gravar(nome, cab, linhas):
        with open(os.path.join(PASTA_AVALIACAO, nome.replace(".csv", f"{sufixo}.csv")), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas_classe.csv",
           ["modelo", "agrupamento", "orto", "classe", "nome", "suporte_px", "previstos_px", "TP", "FP",
            "FN", "precision", "recall", "F1", "IoU"], l_cls)
    gravar("metricas_globais.csv",
           ["modelo", "agrupamento", "orto", "pixeis", "exatidao_global", "exatidao_media", "kappa",
            "mIoU", "fwIoU", "F1_macro", "F1_ponderado", "pct_previsoes_fora"], l_glb)

    # --- tabela comparativa (total da folha), como no 30
    linhas = ["AVALIACAO DOS MODELOS DE OCUPACAO DO SOLO NA FOLHA 113 - total da folha",
              f"Data: {datetime.now():%Y-%m-%d %H:%M} | referencia: {REFERENCIA}",
              "Ortomosaico original, sem calibracao radiometrica."]
    for agr, mapa in AGRUPAMENTOS.items():
        bloco = sorted((c for c in comparativa if c[1] == agr), key=lambda x: x[5]["mIoU"], reverse=True)
        if not bloco:
            continue
        labels = bloco[0][2]
        cab = (f"{'modelo':<22}{'OA':>7}{'kappa':>7}{'mIoU':>7}{'fwIoU':>7}{'F1mac':>7} | " +
               "".join(f"{nome_classe(c, mapa)[:9]:>10}" for c in labels))
        linhas += ["", f"AGRUPAMENTO: {agr}  (F1 por classe a direita)", cab, "-" * len(cab)]
        for rotulo, _, lb, _, pc, gl in bloco:
            f1 = {m["classe"]: m["F1"] for m in pc}
            linhas.append(f"{rotulo[:21]:<22}{gl['exatidao_global']:>7.3f}{gl['kappa']:>7.3f}"
                          f"{gl['mIoU']:>7.3f}{gl['fwIoU']:>7.3f}{gl['F1_macro']:>7.3f} | " +
                          "".join(f"{f1.get(c, 0):>10.3f}" for c in labels))
    linhas += ["", "OA: exatidao global. kappa: coeficiente kappa. mIoU: IoU medio das classes. "
               "fwIoU: IoU ponderado pela area de cada classe. F1mac: F1 medio das classes. "
               f"Excluidos da avaliacao: pixeis da referencia com {sorted(IGNORAR_REF)}."]
    with open(os.path.join(PASTA_AVALIACAO, f"tabela_comparativa{sufixo}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    log("\n".join(linhas))
    if not MODO_TESTE:
        resumo_blocos(l_glb, rad, os.path.join(PASTA_AVALIACAO, "resumo_blocos.txt"))
    log("Avaliacao terminada")


if __name__ == "__main__":
    main()
