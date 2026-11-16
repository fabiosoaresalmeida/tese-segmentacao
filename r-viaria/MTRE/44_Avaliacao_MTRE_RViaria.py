# -*- coding: utf-8 -*-
"""
44 - Avaliação dos Multi-Task Road Extractor (rede viária e ferrovia)
nas 3 ortofotos de teste independente.

Para cada modelo e cada ortofoto de teste:
  1. Inferência (Classify Pixels Using Deep Learning) -> raster binário, sobre a
     orto espelhada para fora (MARGEM_ESPELHO_M metros) e recortada de volta à
     extensão original, para eliminar o ruído das margens
  2. Vetorização: Thin -> Raster to Polyline -> remoção de troços curtos
  3. Referência: eixos restituídos (já no sistema da orto), recortados à orto
  4. Métricas por píxel (deteção remota): matriz de confusão contra a referência
     rasterizada com as mesmas larguras das etiquetas de treino -> OA, exatidão
     do utilizador/produtor, F1, IoU, mIoU, Kappa
  5. Métricas sobre os eixos (Wiedemann), para várias tolerâncias:
       completeness = comprimento da referência a <= tol da extração / referência
       correctness  = comprimento da extração a <= tol da referência / extração
       quality      = extração correta / (extração + referência não detetada)
     e completeness por classe de origem (que tipos de via ficam por detetar).

Saídas em PASTA_AVAL:
  <modelo>\\<folha>_classificado.tif, <folha>_eixos.shp, <folha>_referencia.shp
  matriz_confusao.csv      : por modelo e ortofoto (+ TOTAL), métricas por píxel
  metricas_eixos.csv       : por modelo, ortofoto (+ TOTAL) e tolerância
  completeness_classes.csv : por modelo, ortofoto (+ TOTAL), tolerância e classe
  
  Ficheiro
  
    126-I-2_classificado.tif	        - Resultado da inferência, recortado à orto
    126-I-2_via.tif	                    - Só os píxeis de via
    126-I-2_thin.tif	                - Esqueleto após o Thin, com o fundo a 0
    126-I-2_esqueleto.tif	            - Esqueleto limpo, só com o valor 1
    avaliacao.gdb\eixos_brutos_126_I_2	- Linhas antes do filtro dos 10 m
    126-I-2_eixos.shp                   - Rede viária, Ferrovia
    126-I-2_referencia.shp	            - Eixos de referência recortados à orto
    126-I-2_referencia_pol.shp	        - Referência com buffer, usada na matriz de confusão

Executar na linha de comandos (arcgispro-py3) ou na janela Python do ArcGIS Pro 3.6.
Requer Image Analyst e Spatial Analyst.
"""
import os
import csv
import numpy as np
import glob
import time
import datetime
import arcpy
from arcpy.sa import Con, Raster, SetNull, Thin

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.processorType = "GPU"
arcpy.env.gpuId = 0

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
MODO_TESTE = True            # True = só ORTO_TESTE; False = as 3 ortos de teste
ORTO_TESTE = "126-I-2"     # tem todas as classes, incluindo a ferrovia

MIN_GPU_LIVRE_GB = 6.0       # não arranca a inferência com menos memória livre na GPU

PASTA_ORTOS = r"D:\Users\Cap_Almeida\Projeto\M888_Ortos\126_Ortofotos"
MAX_FRACAO_VIA = 0.30   # acima disto o raster classificado é suspeito: não vetorizar
TESTE = ["126-I-2", "126-III-4", "126-IV-1"]

EIXOS_REF = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP\l_vias_IGeoE.shp"
CAMPO_SOURCE = "source"
PASTA_AVAL = r"D:\DL\Avaliacao\126_RViaria"

VIAS = ["Auto_estrada", "Estrada_larga", "Estrada_estreita", "Arruamento",
        "Acesso_auto", "Caminho_carreteiro", "Caminho_a_pe_posto"]
FERROVIA = ["Caminho_de_ferro_via_larga_unica"]

MODELOS = {
    "mtre_rede_viaria": dict(
        pasta=r"D:\DL\Modelos\126_RViaria\mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_1137",
        referencia=VIAS,
    ),
    "mtre_ferrovia": dict(
        pasta=r"D:\DL\Modelos\126_RViaria\mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209",
        referencia=FERROVIA,
    ),
}

MARGEM_ESPELHO_M = 32.0  # metros de imagem espelhada à volta da orto (usar o mesmo dos edifícios)
APAGAR_ESPELHO = True   # apagar a orto espelhada temporária depois da inferência
ARGS_INFERENCIA = "padding 128;batch_size 2"   # batch pequeno: a GPU está partilhada com o treino
TOLERANCIAS = [2.0, 3.0, 5.0]   # metros; 3 m é o valor principal, 2 e 5 para sensibilidade
ESPESSURA_MAX = 20.0            # metros: espessura máxima das manchas para o Thin
TROCO_MIN = 10.0                # metros: remove esporões e troços isolados mais curtos
REFAZER_INFERENCIA = True       # True: repetir a inferência (agora necessário: passou a usar espelho)

# Larguras da referência por píxel: as MESMAS do 41 (etiquetas de treino)
LARGURA = {
    "Auto_estrada": 11.0,
    "Estrada_larga": 7.0,
    "Estrada_estreita": 4.0,
    "Arruamento": 5.0,
    "Acesso_auto": 3.0,
    "Caminho_carreteiro": 2.5,
    "Caminho_a_pe_posto": 1.5,
    "Caminho_de_ferro_via_larga_unica": 3.5,
}
DESVIO_EIXO = 0.75
LARGURA_MIN = 1.0


# ---------------------------------------------------------------------------
def agora():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{agora()}] {msg}", flush=True)


def normalizar(valor):
    base = os.path.splitext(os.path.basename(str(valor or "").strip()))[0]
    return base.lower().replace(" ", "_").replace("-", "_")


def classe_origem(valor, chaves):
    """Devolve a classe de 'chaves' correspondente ao valor (tolera o prefixo L_)."""
    n = normalizar(valor)
    norm = {normalizar(k): k for k in chaves}
    if n in norm:
        return norm[n]
    for k in sorted(norm, key=len, reverse=True):
        if k in n:
            return norm[k]
    return None


def comprimento(fc, campo=None, chaves=None):
    """Comprimento total (m) ou, com campo, por classe."""
    if not arcpy.Exists(fc) or int(arcpy.management.GetCount(fc)[0]) == 0:
        return {} if campo else 0.0
    if not campo:
        with arcpy.da.SearchCursor(fc, ["SHAPE@LENGTH"]) as cur:
            return sum(r[0] for r in cur)
    tot = {}
    with arcpy.da.SearchCursor(fc, [campo, "SHAPE@LENGTH"]) as cur:
        for v, l in cur:
            c = classe_origem(v, chaves)
            if c:
                tot[c] = tot.get(c, 0.0) + l
    return tot


def razao(a, b):
    return a / b if b > 0 else None


def modelo_emd(pasta):
    emds = glob.glob(os.path.join(pasta, "*.emd"))
    if not emds:
        raise SystemExit(f"Sem .emd em {pasta}")
    return emds[0]


def poligono_orto(tif, saida):
    desc = arcpy.Describe(tif)
    poli = arcpy.Polygon(desc.extent.polygon.getPart(0), desc.spatialReference)
    arcpy.management.CopyFeatures([poli], saida)
    return saida


def espelhar(tif, saida, margem):
    """Cria uma cópia da orto com 'margem' píxeis espelhados em cada lado."""
    r = arcpy.Raster(tif)
    ext, cw, ch = r.extent, r.meanCellWidth, r.meanCellHeight
    a = arcpy.RasterToNumPyArray(tif)
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    a = np.pad(a, ((0, 0), (margem, margem), (margem, margem)), mode="reflect")
    canto = arcpy.Point(ext.XMin - margem * cw, ext.YMin - margem * ch)
    out = arcpy.NumPyArrayToRaster(a if a.shape[0] > 1 else a[0], canto, cw, ch)
    out.save(saida)
    arcpy.management.DefineProjection(saida, r.spatialReference)
    return saida


def inferir(emd, tif, saida):
    if arcpy.Exists(saida) and not REFAZER_INFERENCIA:
        log(f"    inferência já existe: {os.path.basename(saida)}")
        return
    t0 = time.time()
    base = os.path.splitext(saida)[0]
    margem_px = int(round(MARGEM_ESPELHO_M / arcpy.Raster(tif).meanCellWidth))
    espelho = espelhar(tif, base + "_espelho.tif", margem_px)
    log(f"    orto espelhada (+{MARGEM_ESPELHO_M:.0f} m = {margem_px} px por lado): "
        f"{time.time() - t0:.0f} s")

    t1 = time.time()
    pred = arcpy.ia.ClassifyPixelsUsingDeepLearning(
        in_raster=espelho,
        in_model_definition=emd,
        arguments=ARGS_INFERENCIA,
        processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
    )
    pred_pad = base + "_espelho_classificado.tif"
    pred.save(pred_pad)
    log(f"    inferência: {(time.time() - t1) / 60:.1f} min")

    # recortar à extensão exata da orto, na grelha da orto
    ext = arcpy.Describe(tif).extent
    with arcpy.EnvManager(snapRaster=tif, cellSize=tif):
        arcpy.management.Clip(pred_pad, f"{ext.XMin} {ext.YMin} {ext.XMax} {ext.YMax}",
                              saida, maintain_clipping_extent="MAINTAIN_EXTENT")
    r_o, r_c = arcpy.Raster(tif), arcpy.Raster(saida)
    if (r_o.width, r_o.height) != (r_c.width, r_c.height):
        log(f"    ATENÇÃO: recorte {r_c.width}x{r_c.height} != orto {r_o.width}x{r_o.height}")

    if APAGAR_ESPELHO:
        for f in (espelho, pred_pad):
            arcpy.management.Delete(f)


def fracao_via(ras):
    """Fração de píxeis classificados como via (1) e valores presentes."""
    r = arcpy.Raster(ras)
    a = arcpy.RasterToNumPyArray(ras, r.extent.lowerLeft, r.width, r.height, nodata_to_value=0)
    vals, cont = np.unique(a, return_counts=True)
    return float(np.count_nonzero(a == 1)) / a.size, dict(zip(vals.tolist(), cont.tolist()))


def contar_pixeis(ras):
    r = arcpy.Raster(ras)
    a = arcpy.RasterToNumPyArray(ras, r.extent.lowerLeft, r.width, r.height, nodata_to_value=0)
    return int(np.count_nonzero(a == 1))


def vetorizar(raster_class, saida, prefixo):
    """Linha central de cada faixa de via: isolar via -> Thin -> Raster to Polyline.
    Grava os intermédios (<prefixo>_thin.tif, <prefixo>_esqueleto.tif e as linhas brutas
    em avaliacao.gdb) para diagnóstico."""
    t0 = time.time()
    via = Con(Raster(raster_class) == 1, 1)                # só a classe via; resto NoData
    via_tif = prefixo + "_via.tif"
    via.save(via_tif)
    log(f"      1. via isolada: {contar_pixeis(via_tif):,} píxeis ({time.time() - t0:.0f} s)")

    t0 = time.time()
    fino = Thin(via_tif, "NODATA", "FILTER", "ROUND", ESPESSURA_MAX)
    fino_tif = prefixo + "_thin.tif"
    fino.save(fino_tif)
    r = arcpy.Raster(fino_tif)
    a = arcpy.RasterToNumPyArray(fino_tif, r.extent.lowerLeft, r.width, r.height,
                                 nodata_to_value=255)
    vals, cont = np.unique(a, return_counts=True)
    log(f"      2. thin: {contar_pixeis(fino_tif):,} píxeis de esqueleto | valores "
        f"(255 = sem dados): {dict(zip(vals.tolist(), cont.tolist()))} ({time.time() - t0:.0f} s)")

    # Só o esqueleto (valor 1) é vetorizado; o resto, incluindo um eventual fundo a 0,
    # passa a NoData. Sem isto o Raster to Polyline vetoriza também o fundo.
    esq_tif = prefixo + "_esqueleto.tif"
    SetNull(Raster(fino_tif) != 1, 1).save(esq_tif)

    t0 = time.time()
    gdb = os.path.join(os.path.dirname(prefixo), "avaliacao.gdb")
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(os.path.dirname(gdb), "avaliacao.gdb")
    nome_fc = "eixos_brutos_" + os.path.basename(prefixo).replace("-", "_")
    bruto = os.path.join(gdb, nome_fc)          # geodatabase: sem o limite de 2 GB do shapefile
    arcpy.conversion.RasterToPolyline(esq_tif, bruto, "NODATA", 0, "SIMPLIFY")
    n_bruto = int(arcpy.management.GetCount(bruto)[0])
    log(f"      3. raster to polyline: {n_bruto:,} linhas, "
        f"{comprimento(bruto) / 1000:.1f} km ({time.time() - t0:.0f} s)")

    arcpy.management.CopyFeatures(bruto, saida)
    with arcpy.da.UpdateCursor(saida, ["SHAPE@LENGTH"]) as cur:   # esporões e troços curtos
        for (l,) in cur:
            if l < TROCO_MIN:
                cur.deleteRow()
    log(f"      4. sem troços < {TROCO_MIN:.0f} m: "
        f"{int(arcpy.management.GetCount(saida)[0]):,} linhas, {comprimento(saida) / 1000:.1f} km")


def referencia(chaves, area, saida):
    sel = "memory/ref_sel"
    arcpy.management.CopyFeatures(EIXOS_REF, sel)
    with arcpy.da.UpdateCursor(sel, [CAMPO_SOURCE]) as cur:
        for (v,) in cur:
            if classe_origem(v, chaves) is None:
                cur.deleteRow()
    arcpy.analysis.PairwiseClip(sel, area, saida)
    arcpy.management.Delete(sel)


def gpu_livre_gb():
    try:
        import torch
        if torch.cuda.is_available():
            livre, total = torch.cuda.mem_get_info()
            return livre / 2**30, total / 2**30
    except Exception:
        pass
    return None, None


def referencia_poligonos(ref_linhas, chaves, area, saida):
    """Buffer dos eixos de referência com as larguras efetivas das etiquetas de treino."""
    tmp, buf = "memory/refp_l", "memory/refp_b"
    arcpy.management.CopyFeatures(ref_linhas, tmp)
    arcpy.management.AddField(tmp, "BUF_M", "DOUBLE")
    with arcpy.da.UpdateCursor(tmp, [CAMPO_SOURCE, "BUF_M"]) as cur:
        for row in cur:
            c = classe_origem(row[0], chaves)
            row[1] = max(LARGURA_MIN, LARGURA[c] - 2 * DESVIO_EIXO) / 2.0
            cur.updateRow(row)
    arcpy.analysis.Buffer(tmp, buf, "BUF_M", "FULL", "ROUND", "ALL")
    arcpy.analysis.PairwiseClip(buf, area, saida)
    arcpy.management.AddField(saida, "VAL", "SHORT")
    arcpy.management.CalculateField(saida, "VAL", "1")
    for fc in (tmp, buf):
        arcpy.management.Delete(fc)


def matriz_confusao(ras_class, ref_pol):
    """Conta VP, FP, FN, VN na grelha do raster classificado (via = 1)."""
    r = arcpy.Raster(ras_class)
    ext, ncol, nlin, cel = r.extent, r.width, r.height, r.meanCellWidth
    pred = arcpy.RasterToNumPyArray(ras_class, ext.lowerLeft, ncol, nlin,
                                    nodata_to_value=0) == 1
    ref = np.zeros_like(pred)
    if int(arcpy.management.GetCount(ref_pol)[0]) > 0:
        with arcpy.EnvManager(snapRaster=ras_class, extent=ext, cellSize=cel):
            ref_ras = "memory/ref_ras"
            arcpy.conversion.PolygonToRaster(ref_pol, "VAL", ref_ras, "CELL_CENTER", "", cel)
            ref = arcpy.RasterToNumPyArray(ref_ras, ext.lowerLeft, ncol, nlin,
                                           nodata_to_value=0) == 1
            arcpy.management.Delete(ref_ras)
    return dict(
        vp=int(np.count_nonzero(pred & ref)),
        fp=int(np.count_nonzero(pred & ~ref)),
        fn=int(np.count_nonzero(~pred & ref)),
        vn=int(np.count_nonzero(~pred & ~ref)),
    )


def metricas_pixel(c):
    vp, fp, fn, vn = c["vp"], c["fp"], c["fn"], c["vn"]
    n = vp + fp + fn + vn
    oa = razao(vp + vn, n)
    pe = ((vp + fp) * (vp + fn) + (fn + vn) * (fp + vn)) / n**2 if n else None
    kappa = (oa - pe) / (1 - pe) if (oa is not None and pe is not None and pe < 1) else None
    ua_via, pa_via = razao(vp, vp + fp), razao(vp, vp + fn)
    f1 = razao(2 * vp, 2 * vp + fp + fn)
    iou_via, iou_fundo = razao(vp, vp + fp + fn), razao(vn, vn + fp + fn)
    miou = (iou_via + iou_fundo) / 2 if None not in (iou_via, iou_fundo) else None
    return dict(oa=oa, kappa=kappa,
                ua_via=ua_via, pa_via=pa_via,
                ua_fundo=razao(vn, vn + fn), pa_fundo=razao(vn, vn + fp),
                f1_via=f1, iou_via=iou_via, iou_fundo=iou_fundo, miou=miou)


def medir(ref, ext, tol, chaves):
    """Comprimentos (m) para as métricas de Wiedemann com tolerância tol."""
    l_ref, l_ext = comprimento(ref), comprimento(ext)
    l_ref_ok, l_ext_ok, por_classe_ok = 0.0, 0.0, {}
    if l_ref > 0 and l_ext > 0:
        b_ext, b_ref = "memory/b_ext", "memory/b_ref"
        r_ok, e_ok = "memory/r_ok", "memory/e_ok"
        arcpy.analysis.PairwiseBuffer(ext, b_ext, f"{tol} Meters", "ALL")
        arcpy.analysis.PairwiseBuffer(ref, b_ref, f"{tol} Meters", "ALL")
        arcpy.analysis.PairwiseClip(ref, b_ext, r_ok)
        arcpy.analysis.PairwiseClip(ext, b_ref, e_ok)
        l_ref_ok, l_ext_ok = comprimento(r_ok), comprimento(e_ok)
        por_classe_ok = comprimento(r_ok, CAMPO_SOURCE, chaves)
        for fc in (b_ext, b_ref, r_ok, e_ok):
            arcpy.management.Delete(fc)
    return dict(l_ref=l_ref, l_ext=l_ext, l_ref_ok=l_ref_ok, l_ext_ok=l_ext_ok,
                ref_classe=comprimento(ref, CAMPO_SOURCE, chaves),
                ok_classe=por_classe_ok)


def metricas(m):
    falso_neg = m["l_ref"] - m["l_ref_ok"]
    return dict(
        completeness=razao(m["l_ref_ok"], m["l_ref"]),
        correctness=razao(m["l_ext_ok"], m["l_ext"]),
        quality=razao(m["l_ext_ok"], m["l_ext"] + falso_neg),
    )


CAMPOS_PIXEL = ["oa", "kappa", "ua_via", "pa_via", "ua_fundo", "pa_fundo",
                "f1_via", "iou_via", "iou_fundo", "miou"]


def fmt(x):
    return "" if x is None else f"{x:.4f}"


# ---------------------------------------------------------------------------
os.makedirs(PASTA_AVAL, exist_ok=True)
arcpy.env.outputCoordinateSystem = arcpy.Describe(
    os.path.join(PASTA_ORTOS, TESTE[0] + ".tif")).spatialReference

FOLHAS = [ORTO_TESTE] if MODO_TESTE else TESTE

livre, total = gpu_livre_gb()
if livre is not None:
    log(f"GPU: {livre:.1f} GB livres de {total:.1f} GB")
    if livre < MIN_GPU_LIVRE_GB:
        raise SystemExit(f"Menos de {MIN_GPU_LIVRE_GB} GB livres na GPU: não arrancar "
                         "para não pôr em risco o treino em curso.")

linhas_m, linhas_c, linhas_p = [], [], []
log(f"Avaliação | modo teste: {MODO_TESTE} | ortos: {', '.join(FOLHAS)} "
    f"| tolerâncias: {TOLERANCIAS} m")

for nome, cfg in MODELOS.items():
    emd = modelo_emd(cfg["pasta"])
    pasta_m = os.path.join(PASTA_AVAL, nome)
    os.makedirs(pasta_m, exist_ok=True)
    log(f"=== {nome} | {emd}")
    acum = {tol: [] for tol in TOLERANCIAS}
    acum_px = []

    for folha in FOLHAS:
        tif = os.path.join(PASTA_ORTOS, folha + ".tif")
        log(f"  {folha}")
        area = poligono_orto(tif, "memory/area")
        ras = os.path.join(pasta_m, f"{folha}_classificado.tif")
        ext = os.path.join(pasta_m, f"{folha}_eixos.shp")
        ref = os.path.join(pasta_m, f"{folha}_referencia.shp")

        inferir(emd, tif, ras)
        frac, valores = fracao_via(ras)
        log(f"    píxeis via: {frac:.1%} | valores no raster: {valores}")
        if frac > MAX_FRACAO_VIA:
            log(f"    ATENÇÃO: mais de {MAX_FRACAO_VIA:.0%} classificado como via: "
                "raster suspeito, orto saltada (ver o raster no ArcGIS)")
            continue
        log("    vetorização")
        vetorizar(ras, ext, os.path.join(pasta_m, folha))
        referencia(cfg["referencia"], area, ref)
        ref_pol = os.path.join(pasta_m, f"{folha}_referencia_pol.shp")
        referencia_poligonos(ref, cfg["referencia"], area, ref_pol)
        arcpy.management.Delete(area)

        cm = matriz_confusao(ras, ref_pol)
        acum_px.append(cm)
        mp = metricas_pixel(cm)
        linhas_p.append([nome, folha, cm["vp"], cm["fp"], cm["fn"], cm["vn"]]
                        + [fmt(mp[k]) for k in CAMPOS_PIXEL])
        log(f"    píxel: OA {fmt(mp['oa'])} | F1 via {fmt(mp['f1_via'])} | "
            f"IoU via {fmt(mp['iou_via'])} | Kappa {fmt(mp['kappa'])}")
        log(f"    referência {comprimento(ref) / 1000:.1f} km | "
            f"extração {comprimento(ext) / 1000:.1f} km")

        for tol in TOLERANCIAS:
            m = medir(ref, ext, tol, cfg["referencia"])
            acum[tol].append(m)
            r = metricas(m)
            linhas_m.append([nome, folha, tol, f"{m['l_ref'] / 1000:.3f}",
                             f"{m['l_ext'] / 1000:.3f}", fmt(r["completeness"]),
                             fmt(r["correctness"]), fmt(r["quality"])])
            for c, l in m["ref_classe"].items():
                linhas_c.append([nome, folha, tol, c, f"{l / 1000:.3f}",
                                 f"{m['ok_classe'].get(c, 0) / 1000:.3f}",
                                 fmt(razao(m["ok_classe"].get(c, 0), l))])
            if tol == 3.0:
                log(f"    tol 3 m: completeness {fmt(r['completeness'])} | "
                    f"correctness {fmt(r['correctness'])} | quality {fmt(r['quality'])}")

    # Total por píxel: soma das matrizes de confusão
    tot_cm = {k: sum(c[k] for c in acum_px) for k in ("vp", "fp", "fn", "vn")}
    mp = metricas_pixel(tot_cm)
    linhas_p.append([nome, "TOTAL", tot_cm["vp"], tot_cm["fp"], tot_cm["fn"], tot_cm["vn"]]
                    + [fmt(mp[k]) for k in CAMPOS_PIXEL])
    log(f"  TOTAL píxel: OA {fmt(mp['oa'])} | F1 via {fmt(mp['f1_via'])} | "
        f"IoU via {fmt(mp['iou_via'])} | Kappa {fmt(mp['kappa'])}")

    # Total das ortos: soma dos comprimentos (não a média das razões)
    for tol, lista in acum.items():
        tot = {k: sum(m[k] for m in lista) for k in ("l_ref", "l_ext", "l_ref_ok", "l_ext_ok")}
        r = metricas(tot)
        linhas_m.append([nome, "TOTAL", tol, f"{tot['l_ref'] / 1000:.3f}",
                         f"{tot['l_ext'] / 1000:.3f}", fmt(r["completeness"]),
                         fmt(r["correctness"]), fmt(r["quality"])])
        classes = {c for m in lista for c in m["ref_classe"]}
        for c in sorted(classes):
            l = sum(m["ref_classe"].get(c, 0) for m in lista)
            ok = sum(m["ok_classe"].get(c, 0) for m in lista)
            linhas_c.append([nome, "TOTAL", tol, c, f"{l / 1000:.3f}",
                             f"{ok / 1000:.3f}", fmt(razao(ok, l))])
        if tol == 3.0:
            log(f"  TOTAL tol 3 m: completeness {fmt(r['completeness'])} | "
                f"correctness {fmt(r['correctness'])} | quality {fmt(r['quality'])}")

with open(os.path.join(PASTA_AVAL, "matriz_confusao.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["modelo", "folha", "VP", "FP", "FN", "VN"] + CAMPOS_PIXEL)
    w.writerows(linhas_p)
with open(os.path.join(PASTA_AVAL, "metricas_eixos.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["modelo", "folha", "tolerancia_m", "ref_km", "extracao_km",
                "completeness", "correctness", "quality"])
    w.writerows(linhas_m)
with open(os.path.join(PASTA_AVAL, "completeness_classes.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["modelo", "folha", "tolerancia_m", "classe", "ref_km",
                "detetado_km", "completeness"])
    w.writerows(linhas_c)
log(f"Concluído -> {PASTA_AVAL}")