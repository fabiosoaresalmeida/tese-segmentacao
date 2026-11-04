# -*- coding: utf-8 -*-
r"""
09_avaliar_modelos.py  (ArcGIS Pro 3.6)

Avaliacao completa de todos os modelos de edificios nas ortos de teste, contra
uma ou mais REFERENCIAS na POSICAO ORIGINAL (por exemplo, o produto da edicao
cartografica, com que os modelos foram treinados, e o produto da fotogrametria).
A classificacao e os poligonos de cada modelo sao feitos uma so vez; as metricas
sao calculadas para cada referencia e guardadas com a coluna "referencia".

Os modelos sao organizados em GRUPOS, cada um com a sua pasta de modelos e o seu
ortomosaico (por exemplo, fase A com 4 bandas e fase B com 5 bandas, com o nDSM).
Para cada grupo:
  1. prepara as ortos de teste a partir do ortomosaico do grupo, com uma margem de
     contexto e as bordas exteriores espelhadas (feito uma vez e reutilizado);
  2. rasteriza a referencia e a mascara da zona de teste na mesma grelha (0,5 m).
Para cada modelo x orto:
  3. classifica a orto (Classify Pixels Using Deep Learning), reutilizando
     classificacoes ja feitas;
  4. converte em poligonos, recorta pela orto, preenche buracos pequenos, remove
     manchas pequenas e regulariza (Regularize Building Footprint);
  5. calcula as metricas:
     AO PIXEL, para a classificacao e para os poligonos regularizados (o produto final):
        matriz de confusao (TP, FP, FN, TN), exatidao do utilizador (UA) e do
        produtor (PA) das classes edificio e fundo, exatidao global, kappa, F1, IoU;
     POR OBJETO, para os poligonos limpos e regularizados:
        correspondencia 1 para 1 com IoU >= LIMIAR_IOU (detetados, omitidos, falsos),
        precision, recall, F1, cobertura (>= 50 % da area), fusoes, IoU medio,
        erro de area (com sinal e absoluto), deslocamento do centroide, distancia
        media e mediana entre contornos, e detecao e cobertura por classe de area.

Para cada referencia (ficheiro com TODOS os edificios):
  - ao pixel: todos os edificios, incluindo a parte dos edificios das margens dentro
    da zona de teste; so contam os pixeis dentro da zona de teste;
  - por objeto: os edificios inteiramente dentro da zona de teste; os que atravessam
    o limite (margens) sao identificados pelo script, e as previsoes sobre eles nao
    contam como falsos positivos.

Saidas em PASTA_AVALIACAO:
  metricas_pixel.csv, metricas_objeto.csv, metricas_por_area.csv,
  tabela_comparativa.txt (todos os modelos e referencias, total das ortos), log_avaliacao.txt,
  classificacoes\<grupo>\<modelo>\class_<orto>.tif e poligonos.gdb (poligonos finais).

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro FECHADO e sem
treinos a correr na GPU):
    python "D:\Users\Cap_Almeida\Projeto\Python\20_avaliar_modelos.py"
"""
print("script iniciado, a carregar o arcpy...", flush=True)

import arcpy

print("arcpy carregado", flush=True)
import os
import csv
import json
import glob
import time
import traceback
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURACAO
# ============================================================
SHP = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf"

# Grupos de modelos: (rotulo, pasta dos modelos, ortomosaico usado no treino)
# O ortomosaico tem de ter as mesmas bandas com que os modelos foram treinados.
GRUPOS = [
    # ("faseA_4bandas", r"D:\DL\Modelos\126_original",
     # r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"),
    ("faseB_ndsm", r"D:\DL\Modelos\126_original_ndsm",
     r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126_ndsm\126_nDSM_5bandas_v2.tif"),
]
# Modelos a avaliar (nomes das pastas). Lista vazia = todos os que tiverem .dlpk.
MODELOS = []

# Referencias: (nome, ficheiro com TODOS os edificios, na posicao original e no
# sistema de coordenadas do ortomosaico). Os modelos foram treinados com a edicao.
REFERENCIAS = [
    ("edicao",        os.path.join(SHP, "a_constr_Limpa.shp")),
    ("fotogrametria", os.path.join(SHP, "construcoes_fotogrametria_Limpa.shp")),
]
ZONA_TESTE = os.path.join(SHP, "zona_teste.shp")                  # um poligono por orto
CAMPO_NOME  = "NOME"
ORTOS_TESTE = ["126-I-3", "126-II-3", "126-IV-1"]

PASTA_AVALIACAO = r"D:\DL\Avaliacao\126_original_ndsm"

# Classificacao
CELL_SIZE = 0.5
VALOR_EDIFICIO = 1
ARGS_INFERENCIA = "padding 64;batch_size 4;predict_background True;test_time_augmentation False"
CONTEXTO_M = 256          # margem de contexto a volta de cada orto de teste
ESPELHAR_BORDAS = True    # preenche a margem fora do ortomosaico com o reflexo da imagem

# Limpeza e regularizacao (m2; 1 m2 = 4 pixeis)
AREA_MIN = 10
FURO_MAX = 20
METODO_REG = "RIGHT_ANGLES_AND_DIAGONALS"
TOLERANCIA, DENSIFICACAO, PRECISAO, PENALIZACAO_DIAGONAL = 1.0, 1.0, 0.25, 1.5

# Avaliacao por objeto
LIMIAR_IOU = 0.5
CLASSES_AREA = [0, 25, 50, 100, 200, float("inf")]   # m2
PASSO_CONTORNO_M = 1.0                               # espacamento para a distancia entre contornos

# Modo teste: True = so o primeiro modelo do primeiro grupo e a primeira orto
MODO_TESTE = False
# ============================================================

FICHEIRO_LOG = os.path.join(PASTA_AVALIACAO, "log_avaliacao.txt")
GDB = os.path.join(PASTA_AVALIACAO, "poligonos.gdb")
SINAIS_BLOQUEIO = ("winerror 32", "being used by another process")


# ------------------------------------------------------------------ utilitarios
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


def ler_grelha(raster, grelha):
    """Le o raster na grelha exata do raster 'grelha'. Usa arcpy.Raster, e nao
    arcpy.Describe, porque o Describe de um raster com varias bandas (a orto)
    nao tem largura nem altura."""
    g = arcpy.Raster(grelha)
    canto = arcpy.Point(g.extent.XMin, g.extent.YMin)
    return arcpy.RasterToNumPyArray(raster, canto, g.width, g.height, nodata_to_value=0)


def rasterizar(fc, saida, grelha):
    """Rasteriza os poligonos com valor 1 na grelha exata do raster 'grelha'."""
    tmp = os.path.join("memory", "rast_tmp")
    apagar(tmp)
    arcpy.management.CopyFeatures(fc, tmp)
    arcpy.management.AddField(tmp, "um", "SHORT")
    arcpy.management.CalculateField(tmp, "um", "1", "PYTHON3")
    with arcpy.EnvManager(extent=grelha, snapRaster=grelha, cellSize=CELL_SIZE):
        arcpy.conversion.PolygonToRaster(tmp, "um", saida, "CELL_CENTER", "NONE", CELL_SIZE)
    apagar(tmp)
    return saida


def bandas_do_modelo(dlpk):
    """Numero de bandas com que o modelo foi treinado (do .emd), ou None."""
    emds = glob.glob(os.path.join(os.path.dirname(dlpk), "*.emd"))
    if not emds:
        return None
    try:
        with open(emds[0], encoding="utf-8") as f:
            emd = json.load(f)
        eb = emd.get("ExtractBands")
        return len(eb) if eb else None
    except Exception:
        return None


# ------------------------------------------------------------------ referencias
REFS = {}


def preparar_referencias():
    """Para cada referencia: edificios de teste (inteiramente dentro da zona de teste)
    e das margens (atravessam o limite), a partir do ficheiro com todos os edificios."""
    uniao = os.path.join("memory", "zona_uniao")
    apagar(uniao)
    arcpy.management.Dissolve(ZONA_TESTE, uniao)
    sr_z = arcpy.Describe(ZONA_TESTE).spatialReference
    for nome, todos in REFERENCIAS:
        if not arcpy.Exists(todos):
            raise FileNotFoundError(f"Referencia {nome} nao existe: {todos}")
        sr = arcpy.Describe(todos).spatialReference
        if sr.name != sr_z.name:
            raise RuntimeError(f"A referencia {nome} esta em {sr.name}, e a zona de teste em "
                               f"{sr_z.name}. Projeta-a para o sistema do ortomosaico.")
        teste = os.path.join("memory", f"teste_{nome}")
        marg = os.path.join("memory", f"marg_{nome}")
        apagar(teste, marg)
        lyr = arcpy.management.MakeFeatureLayer(todos, "lyr_ref")[0]
        arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", uniao)
        arcpy.management.CopyFeatures(lyr, teste)
        arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", uniao)
        arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", uniao,
                                               selection_type="REMOVE_FROM_SELECTION")
        arcpy.management.CopyFeatures(lyr, marg)
        arcpy.management.Delete(lyr)
        REFS[nome] = dict(todos=todos, teste=teste, margens=marg)
        log(f"Referencia {nome}: {contar(teste)} edificios na zona de teste, {contar(marg)} nas margens "
            f"({os.path.basename(todos)})")
    apagar(uniao)


# ------------------------------------------------------------------ preparacao das ortos
def espelhar(entrada, saida, falta):
    d = arcpy.Raster(entrada)
    sr, cs = d.spatialReference, d.meanCellWidth
    arr = arcpy.RasterToNumPyArray(entrada)
    pad = ((falta["cima"], falta["baixo"]), (falta["esquerda"], falta["direita"]))
    if arr.ndim == 3:
        pad = ((0, 0),) + pad
    arr = np.pad(arr, pad, mode="reflect")
    canto = arcpy.Point(d.extent.XMin - falta["esquerda"] * cs, d.extent.YMin - falta["baixo"] * cs)
    ras = arcpy.NumPyArrayToRaster(arr, canto, cs, cs)
    ras.save(saida)
    arcpy.management.DefineProjection(saida, sr)


def preparar_grupo(rotulo, ortomosaico):
    """Prepara, para cada orto de teste, a orto com contexto, a referencia ao pixel e a
    mascara da zona, e seleciona a referencia por objeto e as margens."""
    pasta = os.path.join(PASTA_AVALIACAO, "ortos_teste", rotulo)
    os.makedirs(pasta, exist_ok=True)
    mos = arcpy.Describe(ortomosaico).extent
    preparadas = {}
    for o in ORTOS_TESTE if not MODO_TESTE else ORTOS_TESTE[:1]:
        so = nome_seguro(o)
        orto_tif = os.path.join(pasta, f"orto_{so}.tif")
        msk_tif = os.path.join(pasta, f"mascara_{so}.tif")

        zona = os.path.join("memory", f"z_{rotulo}_{so}")
        apagar(zona)
        arcpy.analysis.Select(ZONA_TESTE, zona, f"{CAMPO_NOME} = '{o}'")
        if contar(zona) != 1:
            raise RuntimeError(f"Nao encontrei exatamente 1 poligono para {o} em {ZONA_TESTE}")

        if not os.path.exists(orto_tif):
            log(f"   [{rotulo}] a recortar {o} com {CONTEXTO_M} m de contexto")
            ctx = os.path.join("memory", "ctx")
            apagar(ctx)
            arcpy.analysis.Buffer(zona, ctx, f"{CONTEXTO_M} Meters")
            ext = arcpy.Describe(ctx).extent
            xmin, ymin = max(ext.XMin, mos.XMin), max(ext.YMin, mos.YMin)
            xmax, ymax = min(ext.XMax, mos.XMax), min(ext.YMax, mos.YMax)
            ret = f"{xmin} {ymin} {xmax} {ymax}"
            falta = {
                "esquerda": int(round((mos.XMin - ext.XMin) / CELL_SIZE)) if ext.XMin < mos.XMin else 0,
                "direita":  int(round((ext.XMax - mos.XMax) / CELL_SIZE)) if ext.XMax > mos.XMax else 0,
                "baixo":    int(round((mos.YMin - ext.YMin) / CELL_SIZE)) if ext.YMin < mos.YMin else 0,
                "cima":     int(round((ext.YMax - mos.YMax) / CELL_SIZE)) if ext.YMax > mos.YMax else 0,
            }
            if ESPELHAR_BORDAS and any(falta.values()):
                base = os.path.join(pasta, f"base_{so}.tif")
                arcpy.management.Clip(ortomosaico, ret, base, "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
                espelhar(base, orto_tif, falta)
                arcpy.management.Delete(base)
                log(f"      bordas espelhadas: " + ", ".join(f"{k} {v} px" for k, v in falta.items() if v))
            else:
                arcpy.management.Clip(ortomosaico, ret, orto_tif, "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
            apagar(ctx)

        if not os.path.exists(msk_tif):
            rasterizar(zona, msk_tif, orto_tif)
        msk = ler_grelha(msk_tif, orto_tif) == 1

        refs = {}
        for nome_ref, R in REFS.items():
            ref_tif = os.path.join(pasta, f"ref_{nome_ref}_{so}.tif")
            if not os.path.exists(ref_tif):
                rasterizar(R["todos"], ref_tif, orto_tif)
            ref_o = os.path.join("memory", f"r_{rotulo}_{nome_ref}_{so}")
            marg_o = os.path.join("memory", f"g_{rotulo}_{nome_ref}_{so}")
            apagar(ref_o, marg_o)
            for origem, destino, rel in ((R["teste"], ref_o, "HAVE_THEIR_CENTER_IN"),
                                         (R["margens"], marg_o, "INTERSECT")):
                lyr = arcpy.management.MakeFeatureLayer(origem, "lyr_sel")[0]
                arcpy.management.SelectLayerByLocation(lyr, rel, zona)
                arcpy.management.CopyFeatures(lyr, destino)
                arcpy.management.Delete(lyr)
            refs[nome_ref] = dict(ref_o=ref_o, marg_o=marg_o,
                                  ref=(ler_grelha(ref_tif, orto_tif) == 1) & msk)
            log(f"   [{rotulo}] {o} | {nome_ref}: {contar(ref_o)} edificios de referencia, "
                f"{contar(marg_o)} nas margens")
        preparadas[o] = dict(orto=orto_tif, zona=zona, msk=msk, refs=refs)
    return preparadas


# ------------------------------------------------------------------ classificacao e poligonos
def classificar(orto_tif, dlpk, out_tif):
    for tentativa in range(3):
        try:
            with arcpy.EnvManager(cellSize=CELL_SIZE, snapRaster=orto_tif):
                res = arcpy.ia.ClassifyPixelsUsingDeepLearning(orto_tif, dlpk, ARGS_INFERENCIA)
                res.save(out_tif)
            return
        except Exception as e:
            if any(s in str(e).lower() for s in SINAIS_BLOQUEIO) and tentativa < 2:
                log("      ficheiro bloqueado; nova tentativa dentro de 2 min")
                time.sleep(120)
                continue
            raise


def criar_poligonos(class_tif, zona, nome_base):
    m = "memory"
    bruto, edif, recort, sem_fu = (os.path.join(m, x) for x in ("p_bruto", "p_edif", "p_rec", "p_sf"))
    limpo, reg = os.path.join(GDB, f"{nome_base}_limpo"), os.path.join(GDB, f"{nome_base}_reg")
    apagar(bruto, edif, recort, sem_fu, limpo, reg)
    arcpy.conversion.RasterToPolygon(class_tif, bruto, "NO_SIMPLIFY", "Value")
    arcpy.analysis.Select(bruto, edif, f"gridcode = {VALOR_EDIFICIO}")
    arcpy.analysis.Clip(edif, zona, recort)
    arcpy.management.EliminatePolygonPart(recort, sem_fu, "AREA", f"{FURO_MAX} SquareMeters",
                                          0, "CONTAINED_ONLY")
    arcpy.management.CopyFeatures(sem_fu, limpo)
    with arcpy.da.UpdateCursor(limpo, ["SHAPE@AREA"]) as uc:
        for (a,) in uc:
            if a < AREA_MIN:
                uc.deleteRow()
    arcpy.ddd.RegularizeBuildingFootprint(limpo, reg, METODO_REG, TOLERANCIA, DENSIFICACAO,
                                          PRECISAO, PENALIZACAO_DIAGONAL)
    apagar(bruto, edif, recort, sem_fu)
    return limpo, reg


# ------------------------------------------------------------------ metricas
def contagens(pred, ref, msk):
    tp = int(np.sum(pred & ref)); fp = int(np.sum(pred & ~ref & msk))
    fn = int(np.sum(~pred & ref)); tn = int(np.sum(~pred & ~ref & msk))
    return tp, fp, fn, tn


def metricas_pixel(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    ua = tp / (tp + fp) if (tp + fp) else 0.0
    pa = tp / (tp + fn) if (tp + fn) else 0.0
    ua_f = tn / (tn + fn) if (tn + fn) else 0.0
    pa_f = tn / (tn + fp) if (tn + fp) else 0.0
    oa = (tp + tn) / n if n else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0
    return dict(UA_edificio=ua, PA_edificio=pa, UA_fundo=ua_f, PA_fundo=pa_f,
                exatidao_global=oa, kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
                F1=2 * ua * pa / (ua + pa) if (ua + pa) else 0.0,
                IoU=tp / (tp + fp + fn) if (tp + fp + fn) else 0.0)


def dist_contornos(gp, gr):
    def media(a, b):
        la, lb = a.boundary(), b.boundary()
        n = max(int(la.length / PASSO_CONTORNO_M), 4)
        ds = [lb.distanceTo(la.positionAlongLine(i / n, True)) for i in range(n)]
        return sum(ds) / len(ds)
    return (media(gp, gr) + media(gr, gp)) / 2


def avaliar_objetos(pred_fc, ref_fc, marg_fc):
    m = "memory"
    p, r, inter = (os.path.join(m, x) for x in ("e_pred", "e_ref", "e_inter"))
    apagar(p, r, inter)
    arcpy.management.CopyFeatures(pred_fc, p)
    arcpy.management.CopyFeatures(ref_fc, r)
    for fc, campo in ((p, "PID"), (r, "RID")):
        arcpy.management.AddField(fc, campo, "LONG")
        arcpy.management.CalculateField(fc, campo, f"!{arcpy.Describe(fc).OIDFieldName}!", "PYTHON3")
    preds = {i: g for i, g in arcpy.da.SearchCursor(p, ["PID", "SHAPE@"]) if g is not None}
    refs = {i: g for i, g in arcpy.da.SearchCursor(r, ["RID", "SHAPE@"]) if g is not None}

    pares = {}
    if preds and refs:
        arcpy.analysis.Intersect([p, r], inter, "NO_FID")
        for pid, rid, a in arcpy.da.SearchCursor(inter, ["PID", "RID", "SHAPE@AREA"]):
            pares[(pid, rid)] = pares.get((pid, rid), 0.0) + a
    cand = sorted(((ai / (preds[pid].area + refs[rid].area - ai), pid, rid)
                   for (pid, rid), ai in pares.items()), reverse=True)
    up, ur, ok = set(), set(), []
    for iou, pid, rid in cand:
        if iou >= LIMIAR_IOU and pid not in up and rid not in ur:
            up.add(pid); ur.add(rid); ok.append((iou, pid, rid))

    excluidas = set()
    if marg_fc and arcpy.Exists(marg_fc) and contar(marg_fc):
        lyr = arcpy.management.MakeFeatureLayer(p, "lyr_pm")[0]
        arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", marg_fc)
        excluidas = {pid for (pid,) in arcpy.da.SearchCursor(lyr, ["PID"])} - up
        arcpy.management.Delete(lyr)

    cob = {}
    for (pid, rid), ai in pares.items():
        cob[rid] = cob.get(rid, 0.0) + ai
    coberto = {rid for rid, g in refs.items() if cob.get(rid, 0.0) >= 0.5 * g.area}
    por_pred = {}
    for (pid, rid), ai in pares.items():
        if ai >= 0.5 * refs[rid].area:
            por_pred[pid] = por_pred.get(pid, 0) + 1

    ious, erros, desl, dcont = [], [], [], []
    for iou, pid, rid in ok:
        gp, gr = preds[pid], refs[rid]
        cp, cr = gp.trueCentroid, gr.trueCentroid
        ious.append(iou)
        erros.append((gp.area - gr.area) / gr.area * 100)
        desl.append(((cp.X - cr.X) ** 2 + (cp.Y - cr.Y) ** 2) ** 0.5)
        dcont.append(dist_contornos(gp, gr))

    detetados = {rid for _, _, rid in ok}
    por_classe = []
    for a0, a1 in zip(CLASSES_AREA[:-1], CLASSES_AREA[1:]):
        ids = [rid for rid, g in refs.items() if a0 <= g.area < a1]
        por_classe.append([a0, a1, len(ids), sum(1 for i in ids if i in detetados),
                           sum(1 for i in ids if i in coberto)])
    apagar(p, r, inter)
    return dict(n_ref=len(refs), n_pred=len(preds), tp=len(ok),
                fp=len(preds) - len(ok) - len(excluidas), fn=len(refs) - len(ok),
                excluidas=len(excluidas), cobertos=len(coberto),
                fusoes=sum(1 for v in por_pred.values() if v >= 2),
                ious=ious, erros=erros, desl=desl, dcont=dcont, por_classe=por_classe)


def juntar(lista):
    t = dict(n_ref=0, n_pred=0, tp=0, fp=0, fn=0, excluidas=0, cobertos=0, fusoes=0,
             ious=[], erros=[], desl=[], dcont=[], por_classe=None)
    for res in lista:
        for k in ("n_ref", "n_pred", "tp", "fp", "fn", "excluidas", "cobertos", "fusoes"):
            t[k] += res[k]
        for k in ("ious", "erros", "desl", "dcont"):
            t[k] += res[k]
        if t["por_classe"] is None:
            t["por_classe"] = [list(c) for c in res["por_classe"]]
        else:
            for c, n in zip(t["por_classe"], res["por_classe"]):
                c[2] += n[2]; c[3] += n[3]; c[4] += n[4]
    return t


def resumo_objetos(res):
    tp, fp, fn = res["tp"], res["fp"], res["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    med = lambda v: float(np.mean(v)) if v else 0.0
    return dict(precision=p, recall=r, F1=2 * p * r / (p + r) if (p + r) else 0.0,
                cobertura=res["cobertos"] / res["n_ref"] if res["n_ref"] else 0.0,
                IoU_medio=med(res["ious"]), erro_area_pct=med(res["erros"]),
                erro_area_abs_pct=med([abs(x) for x in res["erros"]]),
                desloc_centroide_m=med(res["desl"]),
                dist_contorno_media_m=med(res["dcont"]),
                dist_contorno_mediana_m=float(np.median(res["dcont"])) if res["dcont"] else 0.0)


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(PASTA_AVALIACAO, exist_ok=True)
    if not arcpy.Exists(GDB):
        arcpy.management.CreateFileGDB(PASTA_AVALIACAO, "poligonos.gdb")
    arcpy.env.overwriteOutput = True
    arcpy.env.processorType = "GPU"
    arcpy.env.gpuId = 0
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("3D")

    log("=" * 60)
    log(f"Avaliacao iniciada | modo_teste={MODO_TESTE} | grupos={[g[0] for g in GRUPOS]} | "
        f"referencias={[r[0] for r in REFERENCIAS]}")
    if not arcpy.Exists(ZONA_TESTE):
        raise FileNotFoundError(f"Nao existe: {ZONA_TESTE}")
    preparar_referencias()

    l_pix, l_obj, l_area, comparativa = [], [], [], []
    grupos = GRUPOS[:1] if MODO_TESTE else GRUPOS

    for rotulo, pasta_modelos, ortomosaico in grupos:
        log(f"GRUPO {rotulo} | modelos em {pasta_modelos} | ortomosaico {ortomosaico}")
        if not os.path.isdir(pasta_modelos):
            log(f"ERRO: a pasta dos modelos nao existe: {pasta_modelos}")
            continue
        if not arcpy.Exists(ortomosaico):
            log(f"ERRO: o ortomosaico nao existe: {ortomosaico}")
            continue
        n_bandas_orto = arcpy.Describe(ortomosaico).bandCount
        # procura os .dlpk em qualquer nivel de subpastas
        dlpks = sorted(f for f in glob.glob(os.path.join(pasta_modelos, "**", "*.dlpk"), recursive=True)
                       if not os.path.basename(os.path.dirname(f)).startswith("_"))
        if not dlpks:
            log(f"   AVISO: nenhum .dlpk em {pasta_modelos}. Conteudo da pasta: "
                f"{sorted(os.listdir(pasta_modelos))[:20]}")
        if MODELOS:
            dlpks = [f for f in dlpks if os.path.basename(os.path.dirname(f)) in MODELOS]
        if MODO_TESTE:
            dlpks = dlpks[:1]
        log(f"   {len(dlpks)} modelos: {[os.path.basename(os.path.dirname(f)) for f in dlpks]}")
        if not dlpks:
            continue
        preparadas = preparar_grupo(rotulo, ortomosaico)

        for dlpk in dlpks:
            modelo = os.path.basename(os.path.dirname(dlpk))
            nb = bandas_do_modelo(dlpk)
            if nb and nb != n_bandas_orto:
                log(f"ERRO {modelo}: treinado com {nb} bandas, mas o ortomosaico tem {n_bandas_orto}; ignorado")
                continue
            log(f"MODELO {rotulo} | {modelo}" + (f" | {nb} bandas" if nb else ""))
            # acumuladores por referencia
            acc = {r: dict(cl=np.zeros(4, np.int64), pg=np.zeros(4, np.int64), lim=[], reg=[])
                   for r in REFS}

            for o, P in preparadas.items():
                so = nome_seguro(o)
                try:
                    pasta_cl = os.path.join(PASTA_AVALIACAO, "classificacoes", rotulo, modelo)
                    os.makedirs(pasta_cl, exist_ok=True)
                    cl_tif = os.path.join(pasta_cl, f"class_{so}.tif")
                    t0 = time.time()
                    if not os.path.exists(cl_tif):
                        classificar(P["orto"], dlpk, cl_tif)
                        log(f"   {o}: classificacao em {(time.time() - t0) / 60:.1f} min")

                    # classificacao e poligonos: feitos uma so vez
                    pred_cl = (ler_grelha(cl_tif, P["orto"]) == VALOR_EDIFICIO) & P["msk"]
                    base = nome_seguro(f"{rotulo}_{modelo}_{o}")
                    limpo, reg = criar_poligonos(cl_tif, P["zona"], base)
                    pg_tif = os.path.join(pasta_cl, f"poligonos_{so}.tif")
                    rasterizar(reg, pg_tif, P["orto"])
                    pred_pg = (ler_grelha(pg_tif, P["orto"]) == 1) & P["msk"]

                    # metricas para cada referencia
                    for nome_ref, R in P["refs"].items():
                        A = acc[nome_ref]
                        for produto, pred, chave in (("classificacao", pred_cl, "cl"),
                                                     ("poligonos_reg", pred_pg, "pg")):
                            c = contagens(pred, R["ref"], P["msk"])
                            A[chave] += c
                            mp = metricas_pixel(*c)
                            l_pix.append([nome_ref, rotulo, modelo, o, produto, *c] +
                                         [round(v, 4) for v in mp.values()])
                            if chave == "cl":
                                f1_cl = mp["F1"]
                        for etapa, fc, chave in (("limpo", limpo, "lim"), ("regularizado", reg, "reg")):
                            res = avaliar_objetos(fc, R["ref_o"], R["marg_o"])
                            A[chave].append(res)
                            so_ = resumo_objetos(res)
                            l_obj.append([nome_ref, rotulo, modelo, o, etapa, res["n_ref"], res["n_pred"],
                                          res["tp"], res["fp"], res["fn"], res["excluidas"], res["fusoes"]] +
                                         [round(v, 4) for v in so_.values()])
                            if etapa == "regularizado":
                                for a0, a1, n, det, cb in res["por_classe"]:
                                    l_area.append([nome_ref, rotulo, modelo, o, f"{a0:g}-{a1:g}", n, det,
                                                   round(det / n, 4) if n else "", cb,
                                                   round(cb / n, 4) if n else ""])
                        log(f"   {o} | {nome_ref}: pixel F1={f1_cl:.3f} | objeto F1={so_['F1']:.3f} "
                            f"R={so_['recall']:.3f} fusoes={res['fusoes']} IoU={so_['IoU_medio']:.3f} "
                            f"dist.contorno={so_['dist_contorno_media_m']:.2f} m")
                    log(f"   {o}: concluida em {(time.time() - t0) / 60:.1f} min")
                except Exception as e:
                    log(f"ERRO {modelo} | {o}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")

            # totais do modelo, por referencia
            for nome_ref, A in acc.items():
                if not A["reg"]:
                    continue
                for produto, chave in (("classificacao", "cl"), ("poligonos_reg", "pg")):
                    t = [int(x) for x in A[chave]]
                    l_pix.append([nome_ref, rotulo, modelo, "TOTAL", produto, *t] +
                                 [round(v, 4) for v in metricas_pixel(*t).values()])
                for etapa, chave in (("limpo", "lim"), ("regularizado", "reg")):
                    tot = juntar(A[chave])
                    so_ = resumo_objetos(tot)
                    l_obj.append([nome_ref, rotulo, modelo, "TOTAL", etapa, tot["n_ref"], tot["n_pred"],
                                  tot["tp"], tot["fp"], tot["fn"], tot["excluidas"], tot["fusoes"]] +
                                 [round(v, 4) for v in so_.values()])
                    if etapa == "regularizado":
                        for a0, a1, n, det, cb in tot["por_classe"]:
                            l_area.append([nome_ref, rotulo, modelo, "TOTAL", f"{a0:g}-{a1:g}", n, det,
                                           round(det / n, 4) if n else "", cb, round(cb / n, 4) if n else ""])
                        mcl = metricas_pixel(*[int(x) for x in A["cl"]])
                        mpg = metricas_pixel(*[int(x) for x in A["pg"]])
                        comparativa.append((nome_ref, rotulo, modelo, mcl, mpg, so_, tot))
                        log(f"   TOTAL {modelo} | {nome_ref}: pixel F1={mcl['F1']:.3f} | objeto F1={so_['F1']:.3f} "
                            f"fusoes={tot['fusoes']} dist.contorno={so_['dist_contorno_media_m']:.2f} m")

    # --- CSV
    def gravar(nome, cab, linhas):
        with open(os.path.join(PASTA_AVALIACAO, nome), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas_pixel.csv",
           ["referencia", "grupo", "modelo", "orto", "produto", "TP", "FP", "FN", "TN", "UA_edificio",
            "PA_edificio", "UA_fundo", "PA_fundo", "exatidao_global", "kappa", "F1", "IoU"], l_pix)
    gravar("metricas_objeto.csv",
           ["referencia", "grupo", "modelo", "orto", "etapa", "n_ref", "n_pred", "detetados", "falsos",
            "omitidos", "previsoes_margens_excluidas", "fusoes", "precision", "recall", "F1", "cobertura50",
            "IoU_medio", "erro_area_pct", "erro_area_abs_pct", "desloc_centroide_m",
            "dist_contorno_media_m", "dist_contorno_mediana_m"], l_obj)
    gravar("metricas_por_area.csv",
           ["referencia", "grupo", "modelo", "orto", "classe_area_m2", "n_ref", "detetados", "taxa_detecao",
            "cobertos50", "taxa_cobertura"], l_area)

    # --- tabela comparativa (total das ortos): uma secao por referencia, ordenada pelo F1 por objeto
    cab = (f"{'grupo':<16}{'modelo':<38}{'UA':>7}{'PA':>7}{'OA':>8}{'kappa':>7}{'F1px':>7}{'F1pg':>7} | "
           f"{'P':>6}{'R':>6}{'F1obj':>7}{'cob':>6}{'fus':>5}{'IoU':>6}{'erroA%':>8}{'desl':>6}{'dcont':>7}")
    linhas = ["AVALIACAO DOS MODELOS - total das ortos de teste, referencias na posicao original",
              f"Data: {datetime.now():%Y-%m-%d %H:%M}"]
    for nome_ref, _ in REFERENCIAS:
        bloco = sorted((c for c in comparativa if c[0] == nome_ref), key=lambda x: x[5]["F1"], reverse=True)
        if not bloco:
            continue
        linhas += ["", f"REFERENCIA: {nome_ref}  ({REFS[nome_ref]['todos']})", cab, "-" * len(cab)]
        for _, rotulo, modelo, mcl, mpg, so_, tot in bloco:
            linhas.append(f"{rotulo[:15]:<16}{modelo[:37]:<38}{mcl['UA_edificio']:>7.3f}{mcl['PA_edificio']:>7.3f}"
                          f"{mcl['exatidao_global']:>8.4f}{mcl['kappa']:>7.3f}{mcl['F1']:>7.3f}{mpg['F1']:>7.3f} | "
                          f"{so_['precision']:>6.3f}{so_['recall']:>6.3f}{so_['F1']:>7.3f}{so_['cobertura']:>6.3f}"
                          f"{tot['fusoes']:>5d}{so_['IoU_medio']:>6.3f}{so_['erro_area_pct']:>8.1f}"
                          f"{so_['desloc_centroide_m']:>6.2f}{so_['dist_contorno_media_m']:>7.2f}")
    linhas += ["", "UA, PA, OA, kappa, F1px: ao pixel, classificacao, classe edificio. F1pg: F1 ao pixel dos "
               "poligonos regularizados (produto final). P, R, F1obj: por objeto, poligonos regularizados "
               f"(IoU >= {LIMIAR_IOU}). cob: cobertura >= 50 %. fus: fusoes. erroA%: erro de area medio. "
               "desl: deslocamento do centroide (m). dcont: distancia media entre contornos (m)."]
    with open(os.path.join(PASTA_AVALIACAO, "tabela_comparativa.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    log("\n".join(linhas))
    log("Avaliacao terminada")


if __name__ == "__main__":
    main()