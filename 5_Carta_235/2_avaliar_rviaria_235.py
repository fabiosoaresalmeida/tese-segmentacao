# -*- coding: utf-8 -*-
"""
2_avaliar_rviaria_235.py - Avaliacao dos modelos da REDE VIARIA treinados na 126
(Multi-Task Road Extractor rede viaria e ferrovia, U-Net binario) na FOLHA 235.

E o 44_avaliar_RViaria.py aplicado a 235: as funcoes de pos-processamento, vetorizacao
e metricas sao COPIADAS SEM ALTERACOES (mesmas versoes bruto / pos / centerline, mesmas
tolerancias, mesma referencia por pixel com as larguras das etiquetas de treino), para
os resultados serem comparaveis com os da 126. Mudam apenas:

  1. INFERENCIA: uma vez por modelo, no ortomosaico 235 inteiro ORIGINAL (sem calibracao;
     ver 0_verificar_dominio_126_235.py), com a mesma margem espelhada a volta do mosaico
     e os mesmos argumentos (padding = 1/4 do chip, batch 2).
  2. UNIDADES DE AVALIACAO: em vez das 3 ortos de teste da 126, os 16 BLOCOS da grelha
     4x4 do 0_verificar_dominio (mesmos nomes, ex.: 235_bloco_L4C4). Cada bloco e
     recortado do raster classificado e passa pelo pipeline do 44 como se fosse uma orto.
     O TOTAL soma as matrizes e os comprimentos (como no 44). As linhas que atravessam
     o limite entre blocos ficam cortadas nesse limite (efeito pequeno, igual na
     referencia e na extracao).
  3. REFERENCIA: l_vias.shp da 235 (Datum Lisboa) projetado para o sistema das ortos com
     a MESMA transformacao da 126 (Lisbon_Lisbon_To_WGS_1984_2).
  4. CLASSES NOVAS na 235: Estrada_com_separador (nao existe na 126) e incluida na rede
     viaria com 7 m de largura (como a estrada larga).
  5. No fim, resumo_blocos_<corrida>.txt cruza o desempenho de cada bloco com a
     radiometria do 0_verificar_dominio (correlacao de Spearman).

Saidas em PASTA_AVAL (iguais as do 44) + resumo_blocos_<corrida>.txt e grelha_blocos.shp.

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro FECHADO), na pasta do
script (precisa do comum_235.py na mesma pasta):
    python 2_avaliar_rviaria_235.py
Requer Image Analyst e Spatial Analyst; a versao 'centerline' requer a extensao do
Polygon to Centerline (Topographic Production). Precisa do scipy (como o 44).
"""
import os
import csv
import json
import math
import numpy as np
import glob
import time
import datetime
import arcpy
from arcpy.sa import (Con, Expand, FocalStatistics, IsNull, Lookup, NbrRectangle,
                      Raster, RegionGroup, SetNull, Shrink, Thin)

import comum_235 as comum

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.CheckOutExtension("Spatial")
ESTADO_EXTENSOES = {}
for _ext in ("Foundation", "Defense", "Production", "Topographic"):
    try:
        ESTADO_EXTENSOES[_ext] = arcpy.CheckOutExtension(_ext)
    except Exception as _e:
        ESTADO_EXTENSOES[_ext] = f"erro: {_e}"
arcpy.env.overwriteOutput = True
arcpy.env.processorType = "GPU"
arcpy.env.gpuId = 0

# ---------------------------------------------------------------------------
# CONFIGURACAO - folha 235
# ---------------------------------------------------------------------------
# Modo teste RAPIDO: True = so o primeiro modelo, numa janela de TESTE_LADO_M x TESTE_LADO_M
# (centro do ortomosaico, ou TESTE_CENTRO = (x, y)). Corre o pipeline todo (inferencia,
# 3 versoes, metricas, resumo) em poucos minutos, so para confirmar que funciona.
MODO_TESTE = False
TESTE_LADO_M = 1500
TESTE_CENTRO = None

ORTOMOSAICO = r"D:\Users\Cap_Almeida\Projeto\235\Ortomosaico\235.tif"     # original, sem calibracao
EIXOS_ORIGINAL = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\235_2020_5\l_vias.shp"
TRANSFORMACAO = "Lisbon_Lisbon_To_WGS_1984_2"    # a mesma da 126 (40a / 41)
CAMPO_SOURCE = "source"
PASTA_AVAL = r"D:\DL\Avaliacao\235_RViaria"
PASTA_VERIFICACAO = r"D:\Users\Cap_Almeida\Projeto\235\Verificacoes_dominio"  # None = sem radiometria

MIN_GPU_LIVRE_GB = 6.0
MAX_FRACAO_VIA = 0.30   # acima disto o bloco classificado e suspeito: nao vetorizar

VIAS = ["Auto_estrada", "Estrada_larga", "Estrada_estreita", "Arruamento",
        "Acesso_auto", "Caminho_carreteiro", "Caminho_a_pe_posto",
        "Estrada_com_separador"]                 # nova na 235 (nao existe na 126)
FERROVIA = ["Caminho_de_ferro_via_larga_unica"]

MODELOS = {
    "mtre_rede_viaria": dict(
        pasta=r"D:\DL\Modelos\126_RViaria\mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_2357",
        referencia=VIAS,
    ),
    "unet_rede_viaria": dict(
        pasta=r"D:\DL\Modelos\126_RViaria_Resnet50\unet_resnet50_rede_binario_b8_binario_lrlim_treino_20261102_1429",
        referencia=VIAS,
    ),
    "mtre_ferrovia": dict(
        pasta=r"D:\DL\Modelos\126_RViaria\mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209",
        referencia=FERROVIA,
    ),
}

# --- a partir daqui, IGUAL ao 44_avaliar_RViaria.py ---
MARGEM_ESPELHO_M = 32.0
APAGAR_ESPELHO = True
BATCH_INFERENCIA = 2
PADDING_MAX = 128
TOLERANCIAS = [2.0, 3.0, 5.0]
ESPESSURA_MAX = 20.0
TROCO_MIN = 10.0
REFAZER_INFERENCIA = False

LARGURA = {
    "Auto_estrada": 11.0,
    "Estrada_larga": 7.0,
    "Estrada_estreita": 4.0,
    "Arruamento": 5.0,
    "Acesso_auto": 3.0,
    "Caminho_carreteiro": 2.5,
    "Caminho_a_pe_posto": 1.5,
    "Caminho_de_ferro_via_larga_unica": 3.5,
    "Estrada_com_separador": 7.0,   # NOVA na 235 - confirmado: 7 m (como a estrada larga)
}
DESVIO_EIXO = 0.75
LARGURA_MIN = 1.0

VERSOES = ["bruto", "pos", "centerline"]
SIMPLIFICAR_M = 1.0
JANELA_SUAVE = 3
ABERTURA_PX = 1
AREA_MIN_M2 = 50.0
BURACO_MAX_M2 = 8.0
DANGLE_M = 15.0
TRAVESSA_MAX_M = 20.0
ANGULO_TRAVESSA = 30.0
COLINEAR_MIN = 150.0
SUAVE_M = 12.0
EXTENDER_M = 20.0

EIXOS_REF = None   # definido no main (eixos da 235 no sistema das ortos)


# ---------------------------------------------------------------------------
# FUNCOES COPIADAS DO 44_avaliar_RViaria.py (sem alteracoes)
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


def args_inferencia(emd):
    """Argumentos de inferência adequados ao tamanho de chip do modelo."""
    chip = None
    try:
        with open(emd, encoding="utf-8") as fh:
            d = json.load(fh)
        chip = int(d.get("ImageHeight") or d.get("ImageWidth") or 0) or None
    except Exception:
        pass
    padding = min(PADDING_MAX, (chip // 4) if chip else 64)
    return f"padding {padding};batch_size {BATCH_INFERENCIA}", chip


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
    args, chip = args_inferencia(emd)
    log(f"    inferência: chip {chip} | {args}")
    pred = arcpy.ia.ClassifyPixelsUsingDeepLearning(
        in_raster=espelho,
        in_model_definition=emd,
        arguments=args,
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


def arrays_pixel(ras_class, ref_pol):
    """Máscaras booleanas (prevista, referência) na grelha do raster e tamanho do píxel."""
    r = arcpy.Raster(ras_class)
    ext, ncol, nlin, cel = r.extent, r.width, r.height, r.meanCellWidth
    pred = arcpy.RasterToNumPyArray(ras_class, ext.lowerLeft, ncol, nlin,
                                    nodata_to_value=0) == 1
    ref = np.zeros_like(pred)
    if int(arcpy.management.GetCount(ref_pol)[0]) > 0:
        with arcpy.EnvManager(snapRaster=ras_class, extent=ext, cellSize=cel):
            ref_ras = "memory/ref_ras_t"
            arcpy.conversion.PolygonToRaster(ref_pol, "VAL", ref_ras, "CELL_CENTER", "", cel)
            ref = arcpy.RasterToNumPyArray(ref_ras, ext.lowerLeft, ncol, nlin,
                                           nodata_to_value=0) == 1
            arcpy.management.Delete(ref_ras)
    return pred, ref, cel


def contagens_tolerancia(pred, ref, cel, tolerancias):
    """Contagens para precisão/recall relaxados, por tolerância (metros)."""
    from scipy.ndimage import distance_transform_edt
    n_pred, n_ref = int(pred.sum()), int(ref.sum())
    d_ref = distance_transform_edt(~ref) * cel if n_ref else None    # distância à referência
    d_pred = distance_transform_edt(~pred) * cel if n_pred else None  # distância à previsão
    out = {}
    for tol in tolerancias:
        vp_p = int(np.count_nonzero(pred & (d_ref <= tol))) if d_ref is not None else 0
        vp_r = int(np.count_nonzero(ref & (d_pred <= tol))) if d_pred is not None else 0
        out[tol] = dict(n_pred=n_pred, n_ref=n_ref, vp_p=vp_p, vp_r=vp_r)
    return out


def metricas_tolerancia(c):
    p = razao(c["vp_p"], c["n_pred"])
    r = razao(c["vp_r"], c["n_ref"])
    f1 = 2 * p * r / (p + r) if (p is not None and r is not None and p + r > 0) else None
    # IoU relaxado: previsão correta / (previsão + referência não detetada)
    iou = razao(c["vp_p"], c["n_pred"] + (c["n_ref"] - c["vp_r"]))
    return dict(precisao=p, recall=r, f1=f1, iou=iou)


CAMPOS_PIXEL = ["oa", "kappa", "ua_via", "pa_via", "ua_fundo", "pa_fundo",
                "f1_via", "iou_via", "iou_fundo", "miou"]


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def guardar(r, caminho):
    r.save(caminho)
    return caminho


def mascara_pos(raster_class, saida):
    """Fase A: limpa a máscara prevista (via = 1, fundo = 0)."""
    cel = arcpy.Raster(raster_class).meanCellWidth
    t0 = time.time()
    with arcpy.EnvManager(snapRaster=raster_class, extent=raster_class, cellSize=raster_class):
        m = Con(Raster(raster_class) == 1, 1, 0)
        # A1 alisar a borda
        if JANELA_SUAVE > 1:
            media = FocalStatistics(m, NbrRectangle(JANELA_SUAVE, JANELA_SUAVE, "CELL"),
                                    "MEAN", "DATA")
            m = Con(media >= 0.5, 1, 0)
        # A2 abertura morfológica
        if ABERTURA_PX > 0:
            m = Expand(Shrink(m, ABERTURA_PX, [1]), ABERTURA_PX, [1])
        # A3 remover manchas pequenas
        min_px = AREA_MIN_M2 / cel**2
        rg = RegionGroup(SetNull(m != 1, 1), "EIGHT", "WITHIN", "NO_LINK")
        cnt = Lookup(rg, "Count")
        m = Con(IsNull(cnt), 0, Con(cnt >= min_px, 1, 0))
        # A4 preencher buracos pequenos (regiões de fundo pequenas)
        max_px = BURACO_MAX_M2 / cel**2
        rg_f = RegionGroup(SetNull(m != 0, 1), "FOUR", "WITHIN", "NO_LINK")
        cnt_f = Lookup(rg_f, "Count")
        m = Con(m == 1, 1, Con(IsNull(cnt_f), 0, Con(cnt_f <= max_px, 1, 0)))
        guardar(m, saida)
    log(f"      fase A (raster): {contar_pixeis(saida):,} píxeis via ({time.time() - t0:.0f} s)")
    return saida


def esqueleto_linhas(mascara, gdb, nome_fc, dangle_m, prefixo):
    """Thin + Raster to Polyline sobre uma máscara (via = 1). Devolve a fc de linhas brutas."""
    t0 = time.time()
    via = SetNull(Raster(mascara) != 1, 1)
    fino = Thin(via, "NODATA", "FILTER", "ROUND", ESPESSURA_MAX)
    esq = guardar(SetNull(fino != 1, 1), prefixo + "_esqueleto.tif")
    bruto = os.path.join(gdb, nome_fc)
    arcpy.conversion.RasterToPolyline(esq, bruto, "NODATA", dangle_m, "SIMPLIFY")
    log(f"      thin + raster to polyline (dangle {dangle_m:.0f} m): "
        f"{int(arcpy.management.GetCount(bruto)[0]):,} linhas, "
        f"{comprimento(bruto) / 1000:.1f} km ({time.time() - t0:.0f} s)")
    return bruto


def _direcao(p_no, p_perto):
    dx, dy = p_perto.X - p_no.X, p_perto.Y - p_no.Y
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 0 else None


def _angulo(u, v):
    c = max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))
    return math.degrees(math.acos(c))


def remover_travessas(fc, gdb, tag):
    """Apaga, no próprio fc, os troços curtos que ligam duas faixas contínuas.
    Grava os troços removidos em <gdb>/travessas_<tag>. Devolve o número removido."""
    def chave(p):
        return (round(p.X, 1), round(p.Y, 1))

    troços, nos = {}, {}
    with arcpy.da.SearchCursor(fc, ["OID@", "SHAPE@"]) as cur:
        for oid, g in cur:
            if g is None or g.length == 0:
                continue
            d = min(5.0, g.length / 2)
            q0 = g.positionAlongLine(d).firstPoint
            q1 = g.positionAlongLine(g.length - d).firstPoint
            pontas = []
            for p, q in ((g.firstPoint, q0), (g.lastPoint, q1)):
                u = _direcao(p, q)
                k = chave(p)
                nos.setdefault(k, []).append((oid, u))
                pontas.append((k, u))
            troços[oid] = (g.length, pontas)

    def sai_de_faixa(oid, k, u):
        """Na ponta k, existe uma faixa contínua (2 outras linhas quase colineares)
        da qual o troço oid sai com ângulo >= ANGULO_TRAVESSA?"""
        outros = [v for (o, v) in nos[k] if o != oid and v is not None]
        if u is None or len(outros) < 2:
            return False
        for i in range(len(outros)):
            for j in range(i + 1, len(outros)):
                a, b = outros[i], outros[j]
                if _angulo(a, b) >= COLINEAR_MIN:
                    if min(_angulo(u, a), _angulo(u, b)) >= ANGULO_TRAVESSA:
                        return True
        return False

    remover = set()
    for oid, (comp, pontas) in troços.items():
        if comp > TRAVESSA_MAX_M:
            continue
        (k0, u0), (k1, u1) = pontas
        if k0 == k1 or len(nos[k0]) < 3 or len(nos[k1]) < 3:
            continue
        if sai_de_faixa(oid, k0, u0) and sai_de_faixa(oid, k1, u1):
            remover.add(oid)

    if remover:
        lista = ",".join(str(o) for o in remover)
        oid_campo = arcpy.Describe(fc).OIDFieldName
        arcpy.analysis.Select(fc, os.path.join(gdb, f"travessas_{tag}"),
                              f"{oid_campo} IN ({lista})")
        with arcpy.da.UpdateCursor(fc, ["OID@"]) as cur:
            for (oid,) in cur:
                if oid in remover:
                    cur.deleteRow()
    return len(remover)


def filtrar_curtos(fc, saida):
    arcpy.management.CopyFeatures(fc, saida)
    with arcpy.da.UpdateCursor(saida, ["SHAPE@LENGTH"]) as cur:
        for (l,) in cur:
            if l < TROCO_MIN:
                cur.deleteRow()


def polygon_to_centerline(poligonos, saida):
    """Polygon to Centerline (a localização da ferramenta varia com a versão do Pro)."""
    for modulo in ("topographic", "management", "cartography"):
        ferramenta = getattr(getattr(arcpy, modulo, None), "PolygonToCenterline", None)
        if ferramenta:
            ferramenta(poligonos, saida)
            return modulo
    raise SystemExit("Polygon to Centerline não encontrado no arcpy desta instalação.")


def fase_b(linhas, gdb, tag, versao, saida):
    """B1b travessas -> B2 alisar -> B3 fechar falhas -> B4 troços curtos."""
    t0 = time.time()
    n_trav = remover_travessas(linhas, gdb, f"{tag}_{versao}")
    log(f"      travessas removidas: {n_trav} (ver travessas_{tag}_{versao} em avaliacao.gdb)")
    alisadas = os.path.join(gdb, f"alisadas_{tag}_{versao}")
    arcpy.cartography.SmoothLine(linhas, alisadas, "PAEK", f"{SUAVE_M} Meters",
                                 "FIXED_CLOSED_ENDPOINT", "NO_CHECK")
    estendidas = os.path.join(gdb, f"estendidas_{tag}_{versao}")
    arcpy.management.CopyFeatures(alisadas, estendidas)
    try:
        arcpy.edit.ExtendLine(estendidas, f"{EXTENDER_M} Meters", "FEATURE")
    except arcpy.ExecuteError:
        log(f"      AVISO: Extend Line falhou; falhas não fechadas: {arcpy.GetMessages(2)}")
    filtrar_curtos(estendidas, saida)
    log(f"      fase B (vetor): {time.time() - t0:.0f} s")


def vetorizar_versao(versao, raster_class, pasta_m, folha):
    """Devolve (eixos.shp, máscara usada nas métricas por píxel)."""
    gdb = os.path.join(pasta_m, "avaliacao.gdb")
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(pasta_m, "avaliacao.gdb")
    tag = folha.replace("-", "_")
    prefixo = os.path.join(pasta_m, f"{folha}_{versao}")
    saida = os.path.join(pasta_m, f"{folha}_eixos_{versao}.shp")
    mascara_limpa = os.path.join(pasta_m, f"{folha}_mascara_pos.tif")

    if versao == "bruto":
        linhas = esqueleto_linhas(raster_class, gdb, f"brutas_{tag}_bruto", 0, prefixo)
        filtrar_curtos(linhas, saida)
        mascara = raster_class

    elif versao == "pos":
        mascara = mascara_pos(raster_class, mascara_limpa)
        linhas = esqueleto_linhas(mascara, gdb, f"brutas_{tag}_pos", DANGLE_M, prefixo)
        fase_b(linhas, gdb, tag, versao, saida)

    elif versao == "centerline":
        mascara = (mascara_limpa if arcpy.Exists(mascara_limpa)
                   else mascara_pos(raster_class, mascara_limpa))
        t0 = time.time()
        pol = os.path.join(gdb, f"poligonos_{tag}")
        arcpy.conversion.RasterToPolygon(SetNull(Raster(mascara) != 1, 1), pol,
                                         "NO_SIMPLIFY", "Value")
        pol_s = os.path.join(gdb, f"poligonos_simpl_{tag}")
        arcpy.cartography.SimplifyPolygon(pol, pol_s, "POINT_REMOVE", f"{SIMPLIFICAR_M} Meters",
                                          f"{AREA_MIN_M2} SquareMeters", "NO_CHECK", "NO_KEEP")
        log(f"      polígonos: {int(arcpy.management.GetCount(pol_s)[0]):,} "
            f"(Simplify {SIMPLIFICAR_M} m) ({time.time() - t0:.0f} s)")
        t0 = time.time()
        centro = os.path.join(gdb, f"centerline_{tag}")
        modulo = polygon_to_centerline(pol_s, centro)
        log(f"      polygon to centerline (arcpy.{modulo}): "
            f"{int(arcpy.management.GetCount(centro)[0]):,} linhas, "
            f"{comprimento(centro) / 1000:.1f} km ({time.time() - t0:.0f} s)")
        # partir nos cruzamentos, para a deteção de travessas ver os nós
        linhas = os.path.join(gdb, f"centerline_nos_{tag}")
        try:
            arcpy.management.FeatureToLine(centro, linhas)
        except arcpy.ExecuteError:
            log("      AVISO: Feature To Line indisponível; travessas avaliadas sem partir nos nós")
            arcpy.management.CopyFeatures(centro, linhas)
        # B1 podar esporões (Trim Line, como no artigo)
        try:
            arcpy.edit.TrimLine(linhas, f"{DANGLE_M} Meters", "DELETE_SHORT")
        except arcpy.ExecuteError:
            log(f"      AVISO: Trim Line falhou; esporões não podados: {arcpy.GetMessages(2)}")
        fase_b(linhas, gdb, tag, versao, saida)

    log(f"      {versao}: {int(arcpy.management.GetCount(saida)[0]):,} linhas, "
        f"{comprimento(saida) / 1000:.1f} km -> {os.path.basename(saida)}")
    return saida, mascara


# ---------------------------------------------------------------------------
# NOVO (folha 235): blocos, recortes e resumo com a radiometria
# ---------------------------------------------------------------------------
def recortar(ras, ext, saida, grelha):
    """Recorta 'ras' a extensao ext (xmin, ymin, xmax, ymax), na grelha de 'grelha'."""
    with arcpy.EnvManager(snapRaster=grelha, cellSize=grelha):
        arcpy.management.Clip(ras, f"{ext[0]} {ext[1]} {ext[2]} {ext[3]}", saida,
                              maintain_clipping_extent="MAINTAIN_EXTENT")
    return saida


def poligono_bloco(b, sr, saida):
    arcpy.management.CopyFeatures([comum.poligono_bloco(b, sr)], saida)
    return saida


def resumo_blocos(linhas_t, linhas_m, rad, caminho):
    """Desempenho por bloco (versao pos, tolerancia 3 m) vs radiometria (EMD vs 126)."""
    L = ["REDE VIARIA 235 - DESEMPENHO POR BLOCO vs RADIOMETRIA (versao 'pos', tolerancia 3 m)",
         "rho = Spearman entre o EMD medio do bloco (vs 126) e a metrica; negativo = blocos",
         "radiometricamente mais diferentes da 126 tem pior desempenho. Com 16 blocos e indicativo.",
         ""]
    num = lambda s: float(s) if s not in ("", None) else None
    for modelo in MODELOS:
        f1 = {l[2]: num(l[6]) for l in linhas_t if l[0] == modelo and l[1] == "pos" and l[3] == 3.0
              and l[2] != "TOTAL"}
        ql = {l[2]: num(l[8]) for l in linhas_m if l[0] == modelo and l[1] == "pos" and l[3] == 3.0
              and l[2] != "TOTAL"}
        if not f1:
            continue
        L.append(f"MODELO {modelo}")
        L.append(f"  {'bloco':<20}{'EMD':>7}{'fora126':>9}{'F1pix':>8}{'quality':>9}")
        for b in sorted(f1):
            r = rad.get(b, {})
            emd = r.get("emd_medio")
            L.append(f"  {b:<20}{(f'{emd:.1f}' if emd is not None else '-'):>7}{r.get('fora_126', '-'):>9}"
                     f"{(f'{f1[b]:.3f}' if f1[b] is not None else '-'):>8}"
                     f"{(f'{ql.get(b):.3f}' if ql.get(b) is not None else '-'):>9}")
        if rad:
            blocos = sorted(f1)
            emd = [rad.get(b, {}).get("emd_medio") for b in blocos]
            r1 = comum.spearman(emd, [f1[b] for b in blocos])
            r2 = comum.spearman(emd, [ql.get(b) for b in blocos])
            fora = [f1[b] for b in blocos if rad.get(b, {}).get("fora_126") == "sim" and f1[b] is not None]
            dentro = [f1[b] for b in blocos if rad.get(b, {}).get("fora_126") == "nao" and f1[b] is not None]
            f = lambda v: f"{v:.3f}" if v is not None else "-"
            L.append(f"  rho(EMD, F1pix) = {f(r1)} | rho(EMD, quality) = {f(r2)} | F1pix medio: "
                     f"dentro da variacao da 126 {f(float(np.mean(dentro)) if dentro else None)} "
                     f"({len(dentro)}), fora {f(float(np.mean(fora)) if fora else None)} ({len(fora)})")
        L.append("")
    with open(caminho, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("\n".join(L))


# ---------------------------------------------------------------------------
def main():
    global EIXOS_REF
    corrida = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(PASTA_AVAL, exist_ok=True)
    parametros = dict(
        corrida=corrida, folha="235", ortomosaico=ORTOMOSAICO, eixos=EIXOS_ORIGINAL,
        transformacao=TRANSFORMACAO, modo_teste=MODO_TESTE, versoes=VERSOES, tolerancias=TOLERANCIAS,
        espessura_max=ESPESSURA_MAX, troco_min=TROCO_MIN, margem_espelho_m=MARGEM_ESPELHO_M,
        janela_suave=JANELA_SUAVE, abertura_px=ABERTURA_PX, area_min_m2=AREA_MIN_M2,
        buraco_max_m2=BURACO_MAX_M2, dangle_m=DANGLE_M, travessa_max_m=TRAVESSA_MAX_M,
        angulo_travessa=ANGULO_TRAVESSA, colinear_min=COLINEAR_MIN, suave_m=SUAVE_M,
        extender_m=EXTENDER_M, simplificar_m=SIMPLIFICAR_M,
        batch_inferencia=BATCH_INFERENCIA, padding_max=PADDING_MAX, larguras=LARGURA,
    )
    with open(os.path.join(PASTA_AVAL, f"parametros_{corrida}.json"), "w", encoding="utf-8") as fh:
        json.dump(parametros, fh, indent=2, ensure_ascii=False)

    sr = arcpy.Describe(ORTOMOSAICO).spatialReference
    arcpy.env.outputCoordinateSystem = sr
    r_mos = arcpy.Raster(ORTOMOSAICO)
    log(f"Ortomosaico {ORTOMOSAICO}: {r_mos.bandCount} bandas | pixel {r_mos.meanCellWidth} m | {sr.name}")

    # referencia no sistema das ortos (mesma transformacao da 126)
    EIXOS_REF = comum.projetar_se_preciso(EIXOS_ORIGINAL, ORTOMOSAICO,
                                          os.path.join(PASTA_AVAL, "l_vias_235_IGeoE.shp"),
                                          TRANSFORMACAO, log)
    fora = sorted({v for (v,) in arcpy.da.SearchCursor(EIXOS_REF, [CAMPO_SOURCE])
                   if classe_origem(v, VIAS + FERROVIA) is None})
    if fora:
        log(f"   classes de 'source' que NAO entram na referencia: {fora}")

    # blocos (mesma grelha do 0_verificar_dominio)
    blocos = comum.blocos_raster(ORTOMOSAICO)
    comum.criar_fc_blocos(blocos, sr, os.path.join(PASTA_AVAL, "grelha_blocos.shp"))
    modelos = MODELOS
    if MODO_TESTE:
        ext = comum.janela_teste(ORTOMOSAICO, TESTE_LADO_M, TESTE_CENTRO)
        blocos = [dict(nome="teste_janela", ext=ext)]
        modelos = dict(list(MODELOS.items())[:1])
        log(f"MODO TESTE: janela de {TESTE_LADO_M:g} m {tuple(round(v) for v in ext)} | "
            f"modelo {list(modelos)}")
    rad = comum.ler_radiometria(PASTA_VERIFICACAO)
    log(f"{len(blocos)} blocos | radiometria: {'sim' if rad else 'nao'}")

    livre, total = gpu_livre_gb()
    if livre is not None:
        log(f"GPU: {livre:.1f} GB livres de {total:.1f} GB")
        if livre < MIN_GPU_LIVRE_GB:
            raise SystemExit(f"Menos de {MIN_GPU_LIVRE_GB} GB livres na GPU.")
    log(f"Extensoes para o Polygon to Centerline: {ESTADO_EXTENSOES}")

    linhas_m, linhas_c, linhas_p, linhas_t = [], [], [], []
    campos_tol = ["precisao", "recall", "f1", "iou"]
    log(f"Avaliacao 235 | corrida {corrida} | modo teste: {MODO_TESTE} | versoes: {VERSOES} | "
        f"tolerancias: {TOLERANCIAS} m")

    for nome, cfg in modelos.items():
        emd = modelo_emd(cfg["pasta"])
        pasta_m = os.path.join(PASTA_AVAL, nome)
        pasta_c = os.path.join(pasta_m, f"corrida_{corrida}")
        os.makedirs(pasta_c, exist_ok=True)
        log(f"=== {nome} | {emd}")

        # 1. inferencia (uma vez por modelo): mosaico inteiro, ou so o bloco de teste
        if MODO_TESTE:
            for f in ("teste_janela_orto.tif", "teste_janela_inferencia.tif",
                      "teste_janela_inferencia_espelho.tif", "teste_janela_inferencia_espelho_classificado.tif",
                      "teste_janela_classificado.tif"):
                if arcpy.Exists(os.path.join(pasta_m, f)):
                    arcpy.management.Delete(os.path.join(pasta_m, f))   # refazer sempre
            entrada = recortar(ORTOMOSAICO, blocos[0]["ext"],
                               os.path.join(pasta_m, "teste_janela_orto.tif"), ORTOMOSAICO)
            ras_total = os.path.join(pasta_m, "teste_janela_inferencia.tif")
        else:
            entrada = ORTOMOSAICO
            ras_total = os.path.join(pasta_m, "235_classificado.tif")
        t0 = time.time()
        try:
            inferir(emd, entrada, ras_total)
        except arcpy.ExecuteError:
            log(f"    ERRO na inferencia; modelo saltado:\n    {arcpy.GetMessages(2)}")
            continue
        km2 = arcpy.Raster(entrada).width * arcpy.Raster(entrada).height * 0.25 / 1e6
        log(f"  inferencia: {(time.time() - t0) / 60:.1f} min para {km2:.1f} km2")

        acum = {(v, tol): [] for v in VERSOES for tol in TOLERANCIAS}
        acum_px = {v: [] for v in VERSOES}
        acum_tol = {(v, t): [] for v in VERSOES for t in TOLERANCIAS}

        # 2. cada bloco, como uma orto no 44
        for b in blocos:
            folha = b["nome"]
            log(f"  {folha}")
            ras = os.path.join(pasta_m, f"{folha}_classificado.tif")
            if not arcpy.Exists(ras) or REFAZER_INFERENCIA or \
                    os.path.getmtime(ras) < os.path.getmtime(ras_total):
                recortar(ras_total, b["ext"], ras, ORTOMOSAICO)
            frac, valores = fracao_via(ras)
            log(f"    pixeis via: {frac:.1%} | valores no raster: {valores}")
            if frac > MAX_FRACAO_VIA:
                log(f"    ATENCAO: mais de {MAX_FRACAO_VIA:.0%} classificado como via: bloco saltado")
                continue

            area = poligono_bloco(b, sr, "memory/area")
            ref = os.path.join(pasta_c, f"{folha}_referencia.shp")
            referencia(cfg["referencia"], area, ref)
            ref_pol = os.path.join(pasta_c, f"{folha}_referencia_pol.shp")
            referencia_poligonos(ref, cfg["referencia"], area, ref_pol)
            arcpy.management.Delete(area)
            log(f"    referencia {comprimento(ref) / 1000:.1f} km")

            for versao in VERSOES:
                log(f"    --- versao {versao}")
                try:
                    ext, mascara = vetorizar_versao(versao, ras, pasta_c, folha)
                except arcpy.ExecuteError:
                    log(f"      AVISO: versao '{versao}' falhou e foi saltada:\n      {arcpy.GetMessages(2)}")
                    continue

                cm = matriz_confusao(mascara, ref_pol)
                pred_a, ref_a, cel = arrays_pixel(mascara, ref_pol)
                ct = contagens_tolerancia(pred_a, ref_a, cel, TOLERANCIAS)
                del pred_a, ref_a
                for tol, c in ct.items():
                    acum_tol[(versao, tol)].append(c)
                    mt = metricas_tolerancia(c)
                    linhas_t.append([nome, versao, folha, tol] + [fmt(mt[k]) for k in campos_tol])
                    if tol == 3.0:
                        log(f"      pixel tol 3 m: precisao {fmt(mt['precisao'])} | recall "
                            f"{fmt(mt['recall'])} | F1 {fmt(mt['f1'])} | IoU {fmt(mt['iou'])}")
                acum_px[versao].append(cm)
                mp = metricas_pixel(cm)
                linhas_p.append([nome, versao, folha, cm["vp"], cm["fp"], cm["fn"], cm["vn"]]
                                + [fmt(mp[k]) for k in CAMPOS_PIXEL])
                log(f"      pixel: OA {fmt(mp['oa'])} | F1 via {fmt(mp['f1_via'])} | "
                    f"IoU via {fmt(mp['iou_via'])} | Kappa {fmt(mp['kappa'])}")

                for tol in TOLERANCIAS:
                    m = medir(ref, ext, tol, cfg["referencia"])
                    acum[(versao, tol)].append(m)
                    r = metricas(m)
                    linhas_m.append([nome, versao, folha, tol, f"{m['l_ref'] / 1000:.3f}",
                                     f"{m['l_ext'] / 1000:.3f}", fmt(r["completeness"]),
                                     fmt(r["correctness"]), fmt(r["quality"])])
                    for c, l in m["ref_classe"].items():
                        linhas_c.append([nome, versao, folha, tol, c, f"{l / 1000:.3f}",
                                         f"{m['ok_classe'].get(c, 0) / 1000:.3f}",
                                         fmt(razao(m["ok_classe"].get(c, 0), l))])
                    if tol == 3.0:
                        log(f"      eixos tol 3 m: completeness {fmt(r['completeness'])} | "
                            f"correctness {fmt(r['correctness'])} | quality {fmt(r['quality'])}")

        # Totais: soma das matrizes e dos comprimentos (nao a media das razoes), como no 44
        for versao in VERSOES:
            if not acum_px[versao]:
                continue
            for tol in TOLERANCIAS:
                tt = {k: sum(c[k] for c in acum_tol[(versao, tol)])
                      for k in ("n_pred", "n_ref", "vp_p", "vp_r")}
                mt = metricas_tolerancia(tt)
                linhas_t.append([nome, versao, "TOTAL", tol] + [fmt(mt[k]) for k in campos_tol])
            tot_cm = {k: sum(c[k] for c in acum_px[versao]) for k in ("vp", "fp", "fn", "vn")}
            mp = metricas_pixel(tot_cm)
            linhas_p.append([nome, versao, "TOTAL", tot_cm["vp"], tot_cm["fp"], tot_cm["fn"],
                             tot_cm["vn"]] + [fmt(mp[k]) for k in CAMPOS_PIXEL])
            for tol in TOLERANCIAS:
                lista = acum[(versao, tol)]
                if not lista:
                    continue
                tot = {k: sum(m[k] for m in lista) for k in ("l_ref", "l_ext", "l_ref_ok", "l_ext_ok")}
                r = metricas(tot)
                linhas_m.append([nome, versao, "TOTAL", tol, f"{tot['l_ref'] / 1000:.3f}",
                                 f"{tot['l_ext'] / 1000:.3f}", fmt(r["completeness"]),
                                 fmt(r["correctness"]), fmt(r["quality"])])
                for c in sorted({c for m in lista for c in m["ref_classe"]}):
                    l = sum(m["ref_classe"].get(c, 0) for m in lista)
                    ok = sum(m["ok_classe"].get(c, 0) for m in lista)
                    linhas_c.append([nome, versao, "TOTAL", tol, c, f"{l / 1000:.3f}",
                                     f"{ok / 1000:.3f}", fmt(razao(ok, l))])
                if tol == 3.0:
                    log(f"  TOTAL {versao} | pixel F1 {fmt(mp['f1_via'])} | eixos tol 3 m: "
                        f"completeness {fmt(r['completeness'])} | correctness "
                        f"{fmt(r['correctness'])} | quality {fmt(r['quality'])}")

    def gravar(nome_csv, cabecalho, linhas):
        with open(os.path.join(PASTA_AVAL, nome_csv), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(cabecalho)
            w.writerows(linhas)

    gravar(f"matriz_confusao_{corrida}.csv", ["modelo", "versao", "folha", "VP", "FP", "FN", "VN"]
           + CAMPOS_PIXEL, linhas_p)
    gravar(f"pixel_tolerancia_{corrida}.csv", ["modelo", "versao", "folha", "tolerancia_m"]
           + campos_tol, linhas_t)
    gravar(f"metricas_eixos_{corrida}.csv", ["modelo", "versao", "folha", "tolerancia_m", "ref_km",
                                             "extracao_km", "completeness", "correctness", "quality"], linhas_m)
    gravar(f"completeness_classes_{corrida}.csv", ["modelo", "versao", "folha", "tolerancia_m", "classe",
                                                   "ref_km", "detetado_km", "completeness"], linhas_c)
    resumo_blocos(linhas_t, linhas_m, rad, os.path.join(PASTA_AVAL, f"resumo_blocos_{corrida}.txt"))
    log(f"Concluido -> {PASTA_AVAL}")


if __name__ == "__main__":
    main()
