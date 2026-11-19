# -*- coding: utf-8 -*-
r"""
4_teste_sem_nir.py  (ArcGIS Pro 3.6)

Quanto dependem do NIR os modelos treinados na 126? Para aplicar os modelos (4 bandas:
R, G, B, NIR) a folhas so com RGB (235, 222, 289), a 4.a banda e substituida por uma
CONSTANTE igual a media do NIR no treino de cada modelo (band_mean_values[3] do .emd):
depois da normalizacao vale ~0, ou seja, o modelo nao recebe informacao do NIR.

Para cada modelo, numa orto de teste da 126 (que nao entrou no treino desse tema):
  1. recorta a orto do mosaico 126 com CONTEXTO_M de margem (contexto real a volta);
  2. classifica DUAS versoes: com_nir (original) e nir_constante (banda 4 = constante;
     os pixeis NoData continuam a 0);
  3. compara as duas com a referencia, na extensao da orto, com as MESMAS metricas nas
     duas versoes:
       construcoes: precision, recall, F1 e IoU ao pixel (classe 1)
       rede viaria: as mesmas ao pixel (referencia = buffer dos eixos com as larguras do
                    44, menos o desvio do eixo) e F1 relaxado com tolerancias 2/3/5 m
       ocupacao:    OA, kappa, mIoU, F1 macro e F1/IoU por classe (sem os pixeis 0 e 8
                    da referencia; SAMLoRA v1: previsao 0 -> 9), agrupamentos completo e
                    generalizado
O que interessa e a DIFERENCA entre as duas versoes (perda sem NIR). Os valores absolutos
nao sao iguais aos das avaliacoes 09/44/30 (sem regularizacao nem vetorizacao, orto inteira
em vez da zona de teste), mas sao comparaveis entre versoes e entre modelos.

Saidas em PASTA_SAIDA:
  resultados_nir.csv           todas as metricas (tema, modelo, versao, metrica, valor)
  ocupacao_por_classe_nir.csv  F1 e IoU por classe na ocupacao do solo
  tabela_nir.txt               resumo: com NIR, sem NIR e diferenca
  classificacoes\<versao>_<modelo>.tif, rasters de entrada e referencia, log_teste_nir.txt

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro FECHADO):
    python 4_teste_sem_nir.py
"""
print("a carregar o arcpy...", flush=True)
import os
import csv
import glob
import json
import time
import traceback

import numpy as np
import arcpy
from datetime import datetime   # DEPOIS do arcpy: o arcpy volta a ligar o nome 'datetime' ao modulo

# ============================================================
# CONFIGURACAO
# ============================================================
MOSAICO_126 = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"
PASTA_ORTOS = r"D:\Users\Cap_Almeida\Projeto\M888_Ortos\126_Ortofotos"   # so para a extensao de cada orto
PASTA_SAIDA = r"D:\DL\Avaliacao\126_teste_sem_nir"

TEMAS = ["construcoes", "rviaria", "ocupacao"]     # tirar os que nao se quer correr

# orto de teste de cada tema (tem de ser de TESTE nesse tema)
ORTO = {
    "construcoes": "126-I-3",    # teste das construcoes: 126-I-3, 126-II-3, 126-IV-1
    "rviaria": "126-I-2",        # teste das vias: 126-I-2 (tem ferrovia), 126-III-4, 126-IV-1
    "ocupacao": "126-I-3",       # teste da ocupacao: 126-I-3, 126-II-4, 126-IV-3
}
CONTEXTO_M = 128                 # margem de imagem real a volta da orto (recortada depois)

# --- construcoes
REF_CONSTR = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf\a_constr_Limpa.shp"
PASTA_M_CONSTR = r"D:\DL\Modelos\126_construcoes"
MODELOS_CONSTR = ["unet_resnet50", "unet_resnet152", "deeplab_resnet101", "samlora_vit_l"]  # [] = todos
ARGS_CONSTR = "padding 64;batch_size 4;predict_background True;test_time_augmentation False"  # = 09

# --- rede viaria (= 44_avaliar_RViaria.py)
EIXOS = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP\l_vias_IGeoE.shp"
CAMPO_SOURCE = "source"
VIAS = ["Auto_estrada", "Estrada_larga", "Estrada_estreita", "Arruamento",
        "Acesso_auto", "Caminho_carreteiro", "Caminho_a_pe_posto"]
FERROVIA = ["Caminho_de_ferro_via_larga_unica"]
MODELOS_RV = {
    "mtre_rede_viaria": (r"D:\DL\Modelos\126_RViaria\mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_1137", VIAS),
    "unet_rede_viaria": (r"D:\DL\Modelos\126_RViaria_Resnet50\unet_resnet50_rede_binario_b8_binario_lrlim_treino_20261102_1429", VIAS),
    "mtre_ferrovia": (r"D:\DL\Modelos\126_RViaria\mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209", FERROVIA),
}
LARGURA = {"Auto_estrada": 11.0, "Estrada_larga": 7.0, "Estrada_estreita": 4.0, "Arruamento": 5.0,
           "Acesso_auto": 3.0, "Caminho_carreteiro": 2.5, "Caminho_a_pe_posto": 1.5,
           "Caminho_de_ferro_via_larga_unica": 3.5}
DESVIO_EIXO = 0.75
LARGURA_MIN = 1.0
TOLERANCIAS = [2.0, 3.0, 5.0]
BATCH_RV = 2
PADDING_MAX = 128

# --- ocupacao do solo (= 30_avaliar_ocupacao.py)
REF_OCUP = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\32_separacao_unet\rotulos_teste.shp"
CAMPO_CLASSE = "classe"
TTA = False      # True = como a avaliacao final da 126 (mais lento); a comparacao com/sem NIR vale nos dois
_t = "True" if TTA else "False"
MODELOS_OCUP = [
    ("samlora_v1", r"D:\DL\Modelos\126_ocupacao_samlora\samlora", {0: 9},
     f"padding 64;batch_size 4;predict_background True;test_time_augmentation {_t}"),
    ("unet_r50_v2", r"D:\DL\Modelos\126_ocupacao_2v\unet_r50", {},
     f"padding 64;batch_size 4;predict_background False;test_time_augmentation {_t}"),
    ("deeplabv3_r50_v2", r"D:\DL\Modelos\126_ocupacao_2v\deeplabv3_r50", {},
     f"padding 64;batch_size 4;predict_background False;test_time_augmentation {_t}"),
    ("pspnet_r50_v2", r"D:\DL\Modelos\126_ocupacao_2v\pspnet_r50", {},
     f"padding 64;batch_size 4;predict_background False;test_time_augmentation {_t}"),
]
CLASSES = {1: "Vinha", 2: "Pomar_vinha", 3: "Pomar", 4: "Arvoredo_denso",
           5: "Arvoredo_esparso", 6: "Mato_denso", 7: "Agua", 9: "Sem_representacao"}
IGNORAR_REF = {0, 8}
SEM_REF = 255
AGRUPAMENTOS = {"completa": {}, "generalizada": {2: 1}}

CELL = 0.5
VERSOES = ["com_nir", "nir_constante"]
# ============================================================

LOG = os.path.join(PASTA_SAIDA, "log_teste_nir.txt")


# ------------------------------------------------------------------ utilitarios
def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def apagar(*itens):
    for it in itens:
        if it and arcpy.Exists(it):
            arcpy.management.Delete(it)


def ler_emd(pasta_ou_emd):
    if pasta_ou_emd.lower().endswith(".emd"):
        return pasta_ou_emd
    f = sorted(glob.glob(os.path.join(pasta_ou_emd, "*.emd")))
    if not f:
        raise FileNotFoundError(f"sem .emd em {pasta_ou_emd}")
    return f[0]


def media_nir(emd):
    """Media do NIR no treino (band_mean_values[3] do .emd)."""
    with open(emd, encoding="utf-8") as f:
        d = json.load(f)
    if len(d.get("ExtractBands") or []) != 4:
        raise RuntimeError(f"{emd}: o modelo nao usa 4 bandas")
    return float(d["NormalizationStats"]["band_mean_values"][3]), int(d.get("ImageHeight") or 256)


def ler_grelha(raster, grelha, nodata):
    g = arcpy.Raster(grelha)
    return arcpy.RasterToNumPyArray(raster, arcpy.Point(g.extent.XMin, g.extent.YMin),
                                    g.width, g.height, nodata_to_value=nodata)


def rasterizar(fc, campo, saida, grelha):
    with arcpy.EnvManager(extent=grelha, snapRaster=grelha, cellSize=CELL, outputCoordinateSystem=grelha):
        arcpy.conversion.PolygonToRaster(fc, campo, saida, "CELL_CENTER", "NONE", CELL)
    return saida


def recortar(ras, ext, saida):
    with arcpy.EnvManager(snapRaster=MOSAICO_126, cellSize=MOSAICO_126):
        arcpy.management.Clip(ras, f"{ext[0]} {ext[1]} {ext[2]} {ext[3]}", saida, "#", "#", "NONE",
                              "NO_MAINTAIN_EXTENT")
    return saida


def poligono_extensao(ext, sr, saida):
    x0, y0, x1, y1 = ext
    pol = arcpy.Polygon(arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x0, y1), arcpy.Point(x1, y1),
                                     arcpy.Point(x1, y0), arcpy.Point(x0, y0)]), sr)
    apagar(saida)
    arcpy.management.CopyFeatures([pol], saida)
    return saida


# ------------------------------------------------------------------ preparar a orto
PREP = {}


def preparar_orto(orto):
    """Grelha da orto (extensao exata), entrada com contexto, e mascara de dados."""
    if orto in PREP:
        return PREP[orto]
    pasta = os.path.join(PASTA_SAIDA, "entradas")
    os.makedirs(pasta, exist_ok=True)
    so = orto.replace("-", "_")
    tif = os.path.join(PASTA_ORTOS, orto + ".tif")
    e = arcpy.Describe(tif).extent
    m = arcpy.Describe(MOSAICO_126).extent
    ext = (max(e.XMin, m.XMin), max(e.YMin, m.YMin), min(e.XMax, m.XMax), min(e.YMax, m.YMax))
    ctx = (max(ext[0] - CONTEXTO_M, m.XMin), max(ext[1] - CONTEXTO_M, m.YMin),
           min(ext[2] + CONTEXTO_M, m.XMax), min(ext[3] + CONTEXTO_M, m.YMax))
    grelha = os.path.join(pasta, f"grelha_{so}.tif")
    entrada = os.path.join(pasta, f"com_nir_{so}.tif")
    if not os.path.exists(grelha):
        recortar(MOSAICO_126, ext, grelha)
    if not os.path.exists(entrada):
        recortar(MOSAICO_126, ctx, entrada)
    g = ler_grelha(grelha, grelha, 0)
    dados = ~np.all(g[:3] == 0, axis=0)          # NoData = RGB a 0
    del g
    r = arcpy.Raster(grelha)
    log(f"Orto {orto}: {r.width} x {r.height} px ({r.width * r.height * CELL ** 2 / 1e6:.1f} km2), "
        f"contexto {CONTEXTO_M} m, {int(dados.sum()):,} pixeis com dados")
    PREP[orto] = dict(grelha=grelha, entrada=entrada, ext=ext, dados=dados, so=so,
                      sr=arcpy.Describe(MOSAICO_126).spatialReference)
    return PREP[orto]


def entrada_nir_constante(P, valor):
    """Copia da entrada com a banda 4 = valor (inteiro); NoData (RGB a 0) fica a 0."""
    v = int(round(valor))
    saida = os.path.join(PASTA_SAIDA, "entradas", f"nir_{v}_{P['so']}.tif")
    if os.path.exists(saida):
        return saida
    r = arcpy.Raster(P["entrada"])
    a = arcpy.RasterToNumPyArray(P["entrada"])
    if a.ndim != 3 or a.shape[0] != 4:
        raise RuntimeError(f"{P['entrada']}: esperadas 4 bandas, encontrado {a.shape}")
    sem = np.all(a[:3] == 0, axis=0)
    a[3] = np.where(sem, 0, v).astype(a.dtype)
    out = arcpy.NumPyArrayToRaster(a, arcpy.Point(r.extent.XMin, r.extent.YMin), r.meanCellWidth,
                                   r.meanCellHeight)
    out.save(saida)
    arcpy.management.DefineProjection(saida, r.spatialReference)
    log(f"   entrada com NIR constante = {v} -> {os.path.basename(saida)}")
    return saida


def classificar(entrada, emd, args, saida):
    if os.path.exists(saida):
        log(f"   {os.path.basename(saida)}: ja existe, reutilizada")
        return saida
    t0 = time.time()
    with arcpy.EnvManager(cellSize=CELL, snapRaster=MOSAICO_126, processorType="GPU", gpuId=0):
        arcpy.ia.ClassifyPixelsUsingDeepLearning(entrada, emd, args).save(saida)
    log(f"   {os.path.basename(saida)}: {(time.time() - t0) / 60:.1f} min")
    return saida


def previsoes(P, nome, emd, args):
    """{versao: array previsto na grelha da orto} para as duas versoes."""
    nir, _ = media_nir(emd)
    pasta = os.path.join(PASTA_SAIDA, "classificacoes")
    os.makedirs(pasta, exist_ok=True)
    out = {}
    for versao in VERSOES:
        entrada = P["entrada"] if versao == "com_nir" else entrada_nir_constante(P, nir)
        cl = classificar(entrada, emd, args, os.path.join(pasta, f"{versao}_{nome}_{P['so']}.tif"))
        out[versao] = ler_grelha(cl, P["grelha"], 0)
    return out, nir


# ------------------------------------------------------------------ metricas
def binarias(pred, ref, valido):
    tp = int(np.sum(pred & ref & valido)); fp = int(np.sum(pred & ~ref & valido))
    fn = int(np.sum(~pred & ref & valido)); tn = int(np.sum(~pred & ~ref & valido))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=p, recall=r, F1=2 * p * r / (p + r) if p + r else 0.0,
                IoU=tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                pixeis_previstos=tp + fp, pixeis_referencia=tp + fn)


def relaxadas(pred, ref, valido):
    """F1 relaxado (como o 44): previsao correta se houver referencia a <= tol, e vice-versa."""
    from scipy.ndimage import distance_transform_edt
    pred, ref = pred & valido, ref & valido
    d_ref = distance_transform_edt(~ref) * CELL if ref.any() else None
    d_pred = distance_transform_edt(~pred) * CELL if pred.any() else None
    out = {}
    for tol in TOLERANCIAS:
        vp_p = int(np.sum(pred & (d_ref <= tol))) if d_ref is not None else 0
        vp_r = int(np.sum(ref & (d_pred <= tol))) if d_pred is not None else 0
        p = vp_p / pred.sum() if pred.sum() else 0.0
        r = vp_r / ref.sum() if ref.sum() else 0.0
        out[f"F1_tol{tol:g}m"] = 2 * p * r / (p + r) if p + r else 0.0
        out[f"precision_tol{tol:g}m"] = p
        out[f"recall_tol{tol:g}m"] = r
    return out


def matriz(ref, pred, valido):
    r = ref[valido].astype(np.int64)
    p = np.clip(pred[valido].astype(np.int64), 0, 255)
    return np.bincount(r * 256 + p, minlength=256 * 256).reshape(256, 256)


def metricas_ocup(M, mapa):
    f = lambda c: mapa.get(c, c)
    if mapa:
        M2 = np.zeros_like(M)
        for i, j in zip(*np.nonzero(M)):
            M2[f(i), f(j)] += M[i, j]
        M = M2
    labels = sorted({f(c) for c in CLASSES})
    n = int(M[labels, :].sum())
    pc = {}
    for c in labels:
        tp, sup, prev = int(M[c, c]), int(M[c, :].sum()), int(M[labels, c].sum())
        p = tp / prev if prev else 0.0
        r = tp / sup if sup else 0.0
        pc[c] = dict(suporte=sup, F1=2 * p * r / (p + r) if p + r else 0.0,
                     IoU=tp / (prev + sup - tp) if prev + sup - tp else 0.0, recall=r, previstos=prev)
    com = [c for c in labels if pc[c]["suporte"] > 0]
    oa = sum(M[c, c] for c in labels) / n if n else 0.0
    pe = sum(pc[c]["suporte"] * pc[c]["previstos"] for c in labels) / n ** 2 if n else 0.0
    glob_ = dict(OA=oa, kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
                 mIoU=float(np.mean([pc[c]["IoU"] for c in com])) if com else 0.0,
                 F1_macro=float(np.mean([pc[c]["F1"] for c in com])) if com else 0.0)
    return glob_, pc


# ------------------------------------------------------------------ temas
def tema_construcoes(R):
    P = preparar_orto(ORTO["construcoes"])
    pasta = os.path.join(PASTA_SAIDA, "entradas")
    ref_tif = os.path.join(pasta, f"ref_construcoes_{P['so']}.tif")
    if not os.path.exists(ref_tif):
        tmp, area = "memory/ref_c", "memory/area_c"
        poligono_extensao(P["ext"], P["sr"], area)
        apagar(tmp)
        arcpy.analysis.PairwiseClip(REF_CONSTR, area, tmp)
        arcpy.management.AddField(tmp, "um", "SHORT")
        arcpy.management.CalculateField(tmp, "um", "1", "PYTHON3")
        rasterizar(tmp, "um", ref_tif, P["grelha"])
        apagar(tmp, area)
    ref = ler_grelha(ref_tif, P["grelha"], 0) == 1
    dlpks = sorted(glob.glob(os.path.join(PASTA_M_CONSTR, "*", "*.emd")))
    if MODELOS_CONSTR:
        dlpks = [f for f in dlpks if any(os.path.basename(os.path.dirname(f)).startswith(p) for p in MODELOS_CONSTR)]
    for emd in dlpks:
        nome = os.path.basename(os.path.dirname(emd))
        log(f"CONSTRUCOES {nome} | orto {ORTO['construcoes']}")
        try:
            prev, nir = previsoes(P, nome, emd, ARGS_CONSTR)
            for versao, a in prev.items():
                for k, v in binarias(a == 1, ref, P["dados"]).items():
                    R.append(["construcoes", nome, ORTO["construcoes"], versao, round(nir, 1), k, v])
        except Exception as e:
            log(f"ERRO {nome}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")


def classe_origem(valor, chaves):
    n = os.path.splitext(os.path.basename(str(valor or "").strip()))[0].lower().replace(" ", "_").replace("-", "_")
    norm = {k.lower(): k for k in chaves}
    if n in norm:
        return norm[n]
    for k in sorted(norm, key=len, reverse=True):
        if k in n:
            return norm[k]
    return None


def tema_rviaria(R):
    P = preparar_orto(ORTO["rviaria"])
    pasta = os.path.join(PASTA_SAIDA, "entradas")
    refs = {}
    for nome, (pasta_m, chaves) in MODELOS_RV.items():
        chave_ref = "ferrovia" if chaves is FERROVIA else "vias"
        if chave_ref not in refs:
            ref_tif = os.path.join(pasta, f"ref_{chave_ref}_{P['so']}.tif")
            if not os.path.exists(ref_tif):
                area, sel, buf, rec = "memory/area_v", "memory/sel_v", "memory/buf_v", "memory/rec_v"
                poligono_extensao(P["ext"], P["sr"], area)
                apagar(sel, buf, rec)
                arcpy.analysis.PairwiseClip(EIXOS, area, sel)
                arcpy.management.AddField(sel, "BUF_M", "DOUBLE")
                with arcpy.da.UpdateCursor(sel, [CAMPO_SOURCE, "BUF_M"]) as cur:
                    for row in cur:
                        c = classe_origem(row[0], chaves)
                        if c is None:
                            cur.deleteRow()
                            continue
                        row[1] = max(LARGURA_MIN, LARGURA[c] - 2 * DESVIO_EIXO) / 2.0
                        cur.updateRow(row)
                if int(arcpy.management.GetCount(sel)[0]) == 0:
                    log(f"   AVISO: sem {chave_ref} de referencia na orto {ORTO['rviaria']}")
                    refs[chave_ref] = None
                    apagar(area, sel)
                    continue
                arcpy.analysis.Buffer(sel, buf, "BUF_M", "FULL", "ROUND", "ALL")
                arcpy.analysis.PairwiseClip(buf, area, rec)
                arcpy.management.AddField(rec, "VAL", "SHORT")
                arcpy.management.CalculateField(rec, "VAL", "1", "PYTHON3")
                rasterizar(rec, "VAL", ref_tif, P["grelha"])
                apagar(area, sel, buf, rec)
            refs[chave_ref] = ler_grelha(ref_tif, P["grelha"], 0) == 1
        ref = refs[chave_ref]
        if ref is None:
            continue
        emd = ler_emd(pasta_m)
        _, chip = media_nir(emd)
        args = f"padding {min(PADDING_MAX, chip // 4)};batch_size {BATCH_RV}"   # = 44
        log(f"REDE VIARIA {nome} | orto {ORTO['rviaria']} | {args}")
        try:
            prev, nir = previsoes(P, nome, emd, args)
            for versao, a in prev.items():
                m = binarias(a == 1, ref, P["dados"])
                m.update(relaxadas(a == 1, ref, P["dados"]))
                for k, v in m.items():
                    R.append(["rviaria", nome, ORTO["rviaria"], versao, round(nir, 1), k, v])
        except Exception as e:
            log(f"ERRO {nome}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")


def tema_ocupacao(R, RC):
    P = preparar_orto(ORTO["ocupacao"])
    ref_tif = os.path.join(PASTA_SAIDA, "entradas", f"ref_ocupacao_{P['so']}.tif")
    if not os.path.exists(ref_tif):
        rasterizar(REF_OCUP, CAMPO_CLASSE, ref_tif, P["grelha"])
    ref = ler_grelha(ref_tif, P["grelha"], SEM_REF).astype(np.uint8)
    valido = P["dados"] & ~np.isin(ref, list(IGNORAR_REF | {SEM_REF}))
    log(f"   ocupacao: {int(valido.sum()):,} pixeis avaliados")
    for nome, pasta_m, mapa_pred, args in MODELOS_OCUP:
        log(f"OCUPACAO {nome} | orto {ORTO['ocupacao']} | {args}")
        try:
            emd = ler_emd(pasta_m)
            prev, nir = previsoes(P, nome, emd, args)
            for versao, a in prev.items():
                if mapa_pred:
                    b = a.copy()
                    for de, para in mapa_pred.items():
                        b[a == de] = para
                    a = b
                M = matriz(ref, a, valido)
                for agr, mapa in AGRUPAMENTOS.items():
                    gl, pc = metricas_ocup(M, mapa)
                    for k, v in gl.items():
                        R.append(["ocupacao", nome, ORTO["ocupacao"], versao, round(nir, 1), f"{k}_{agr}", v])
                    for c, d in pc.items():
                        nomes = "+".join(CLASSES[k] for k in CLASSES if mapa.get(k, k) == c)
                        RC.append([nome, versao, agr, c, nomes, d["suporte"], round(d["F1"], 4), round(d["IoU"], 4)])
        except Exception as e:
            log(f"ERRO {nome}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")


# ------------------------------------------------------------------ resumo
PRINCIPAIS = {
    "construcoes": ["F1", "IoU", "precision", "recall"],
    "rviaria": ["F1", "IoU", "F1_tol3m", "recall_tol3m", "precision_tol3m"],
    "ocupacao": ["OA_completa", "kappa_completa", "mIoU_completa", "F1_macro_completa", "mIoU_generalizada"],
}


def tabela(R, RC):
    val = {(t, m, v, k): x for t, m, o, v, n, k, x in R}
    L = ["TESTE SEM NIR - modelos da 126 com a banda 4 substituida pela media do NIR do treino",
         f"Data: {datetime.now():%Y-%m-%d %H:%M} | ortos: {ORTO}",
         "com = original (RGB+NIR) | sem = NIR constante | dif = sem - com (negativo = perda sem NIR)", ""]
    for tema in TEMAS:
        modelos = list(dict.fromkeys(m for t, m, *_ in R if t == tema))
        if not modelos:
            continue
        L.append(f"=== {tema.upper()} (orto {ORTO[tema]})")
        L.append(f"  {'modelo':<38}{'metrica':<20}{'com':>8}{'sem':>8}{'dif':>8}")
        for m in modelos:
            for k in PRINCIPAIS[tema]:
                a, b = val.get((tema, m, "com_nir", k)), val.get((tema, m, "nir_constante", k))
                if a is None or b is None:
                    continue
                L.append(f"  {m[:37]:<38}{k:<20}{a:>8.3f}{b:>8.3f}{b - a:>+8.3f}")
            L.append("")
    if RC:
        L.append("=== OCUPACAO - F1 POR CLASSE (agrupamento completo): com -> sem (dif)")
        f1 = {(m, v, a, c): (n, f) for m, v, a, c, n, s, f, i in RC}
        sup = {c: s for m, v, a, c, n, s, f, i in RC if a == "completa"}
        for m in dict.fromkeys(r[0] for r in RC):
            L.append(f"  {m}")
            for c in sorted(CLASSES):
                a = f1.get((m, "com_nir", "completa", c))
                b = f1.get((m, "nir_constante", "completa", c))
                if a and b and sup.get(c, 0) > 0:
                    L.append(f"    {a[0]:<20}{a[1]:>7.3f} -> {b[1]:>6.3f} ({b[1] - a[1]:+.3f})")
        L.append("")
    L += ["Leitura: perda pequena (ex.: F1 a cair poucas centesimas) -> o modelo pode ser aplicado a",
          "folhas sem NIR com a banda constante (indicar na tese); perda grande -> e preciso um",
          "modelo treinado so com RGB para essas folhas."]
    with open(os.path.join(PASTA_SAIDA, "tabela_nir.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.CheckOutExtension("Spatial")
    log("=" * 60)
    log(f"Teste sem NIR | temas {TEMAS} | ortos {ORTO} | TTA ocupacao {TTA}")
    if arcpy.Raster(MOSAICO_126).bandCount != 4:
        raise RuntimeError("O mosaico 126 tem de ter 4 bandas")
    R, RC = [], []
    if "construcoes" in TEMAS:
        tema_construcoes(R)
    if "rviaria" in TEMAS:
        tema_rviaria(R)
    if "ocupacao" in TEMAS:
        tema_ocupacao(R, RC)

    with open(os.path.join(PASTA_SAIDA, "resultados_nir.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["tema", "modelo", "orto", "versao", "nir_constante_usado", "metrica", "valor"])
        w.writerows([r[:6] + [round(r[6], 4) if isinstance(r[6], float) else r[6]] for r in R])
    with open(os.path.join(PASTA_SAIDA, "ocupacao_por_classe_nir.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["modelo", "versao", "agrupamento", "classe", "nome", "suporte_px", "F1", "IoU"])
        w.writerows(RC)
    tabela(R, RC)
    log("Teste terminado")


if __name__ == "__main__":
    main()
