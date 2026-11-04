# -*- coding: utf-8 -*-
"""
44_v4 - Avaliação dos modelos da rede viária (Multi-Task Road Extractor e U-Net) com
pós-processamento da vetorização. Cada ortofoto é avaliada em TRÊS versões,
com as mesmas métricas, para medir o contributo do pós-processamento:

  bruto      : máscara prevista -> Thin -> Raster to Polyline (como na v1)
  centerline : máscara limpa (fase A) -> polígonos -> Simplify Polygon
               -> Polygon to Centerline -> fase B (como em Kramm et al., 2025)
  pos   : máscara prevista limpa (fase A) -> Thin -> Raster to Polyline
          -> poda de esporões, alisamento e fecho de falhas (fase B)

Fase A, no raster (antes do Thin):
  A1 alisar a borda: média focal (JANELA_SUAVE x JANELA_SUAVE) e limiar 0,5
  A2 abertura morfológica (Shrink + Expand): corta pontes finas entre manchas,
     ex.: contactos entre as duas faixas da autoestrada
  A3 remover manchas com área < AREA_MIN_M2
  A4 preencher buracos com área <= BURACO_MAX_M2 (carros, sombras)
Fase B, no vetor:
  B1 podar esporões < DANGLE_M (minimum_dangle_length do Raster to Polyline)
  B1b remover travessas: troços curtos (<= TRAVESSA_MAX_M) entre dois nós, que
      saem com ângulo >= ANGULO_TRAVESSA de uma faixa contínua em cada ponta
      (ex.: ligações entre as duas faixas da autoestrada)
  B2 alisar linhas: Smooth Line PAEK com tolerância SUAVE_M
  B3 fechar falhas: Extend Line até EXTENDER_M (pontas soltas até outra linha)
  B4 remover troços isolados < TROCO_MIN

Para cada modelo, ortofoto e versão:
  - matriz de confusão por píxel (bruto: máscara prevista; pos/centerline: máscara limpa)
  - métricas por píxel COM TOLERÂNCIA (relaxed precision/recall): um píxel previsto
    é correto se houver referência a <= tol; um píxel de referência é detetado se
    houver previsão a <= tol (mesmas tolerâncias das métricas sobre os eixos)
  - métricas sobre os eixos (completeness, correctness, quality) por tolerância
  - completeness por classe de origem

Saídas:
  PASTA_AVAL/<modelo>/<folha>_classificado.tif   (inferência, partilhada entre corridas)
  PASTA_AVAL/<modelo>/corrida_<data_hora>/       (uma pasta por corrida: sem bloqueios
      <folha>_mascara_pos.tif, <folha>_eixos_bruto.shp,     de ficheiros abertos no
      <folha>_eixos_pos.shp, <folha>_referencia.shp,        ArcGIS e corridas com
      avaliacao.gdb (linhas brutas, travessas)              parâmetros diferentes
                                                            ficam todas guardadas)
  PASTA_AVAL/parametros_<corrida>.json, matriz_confusao_<corrida>.csv,
  metricas_eixos_<corrida>.csv, completeness_classes_<corrida>.csv
  (CSV com a coluna 'versao')

Executar na linha de comandos (arcgispro-py3). Requer Image Analyst e Spatial Analyst;
a versão 'centerline' requer a extensão do Polygon to Centerline (Topographic Production).
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

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.CheckOutExtension("Spatial")
# Polygon to Centerline (Topographic Production) precisa de uma extensão, que num
# script tem de ser pedida explicitamente (no ArcGIS Pro aberto é automática).
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
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
MODO_TESTE = False           # True = só ORTO_TESTE; False = as 3 ortos de teste
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
    # U-Net binário (chips 256, backbone congelado): mesma tarefa e referência do MTRE
    "unet_rede_viaria": dict(
        pasta=r"D:\DL\Modelos\126_RViaria_Resnet50\unet_resnet50_rede_binario_b8_binario_lrlim_treino_20261102_1429",
        referencia=VIAS,
    ),
    "mtre_ferrovia": dict(
        pasta=r"D:\DL\Modelos\126_RViaria\mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209",
        referencia=FERROVIA,
    ),
}

MARGEM_ESPELHO_M = 32.0  # metros de imagem espelhada à volta da orto (usar o mesmo dos edifícios)
APAGAR_ESPELHO = True   # apagar a orto espelhada temporária depois da inferência
# Argumentos de inferência: o padding é calculado por modelo (1/4 do chip do .emd,
# máximo 128), porque tem de ser menor do que metade do chip (256 -> 64, 512 -> 128)
BATCH_INFERENCIA = 2      # batch pequeno: a GPU pode estar partilhada com um treino
PADDING_MAX = 128
TOLERANCIAS = [2.0, 3.0, 5.0]   # metros; 3 m é o valor principal, 2 e 5 para sensibilidade
ESPESSURA_MAX = 20.0            # metros: espessura máxima das manchas para o Thin
TROCO_MIN = 10.0                # metros: remove esporões e troços isolados mais curtos
REFAZER_INFERENCIA = False      # True para repetir a inferência mesmo que o raster exista

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

# ---- Pós-processamento (versão 'pos') -------------------------------------
VERSOES = ["bruto", "pos", "centerline"]
SIMPLIFICAR_M = 1.0     # centerline: tolerância do Simplify Polygon (Point Remove)
# Fase A (raster)
JANELA_SUAVE = 3        # píxeis: janela da média focal (5 fechava o separador da autoestrada)
ABERTURA_PX = 1         # píxeis: Shrink + Expand; corta pontes finas entre as faixas
AREA_MIN_M2 = 50.0      # manchas mais pequenas são removidas
BURACO_MAX_M2 = 8.0     # buracos até esta área são preenchidos (carros); 20 fechava o separador
# Fase B (vetor)
DANGLE_M = 15.0         # esporões (pontas soltas) mais curtos são podados
TRAVESSA_MAX_M = 20.0   # travessas entre faixas paralelas: troços até este comprimento...
ANGULO_TRAVESSA = 30.0  # ...que saem da faixa com pelo menos este ângulo (graus)
COLINEAR_MIN = 150.0    # duas linhas num nó formam uma faixa contínua se o ângulo >= isto
SUAVE_M = 12.0          # tolerância do Smooth Line (PAEK)
EXTENDER_M = 20.0       # pontas soltas prolongadas até outra linha


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
CORRIDA = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(PASTA_AVAL, exist_ok=True)
PARAMETROS = dict(
    corrida=CORRIDA, modo_teste=MODO_TESTE, versoes=VERSOES, tolerancias=TOLERANCIAS,
    espessura_max=ESPESSURA_MAX, troco_min=TROCO_MIN, margem_espelho_m=MARGEM_ESPELHO_M,
    janela_suave=JANELA_SUAVE, abertura_px=ABERTURA_PX, area_min_m2=AREA_MIN_M2,
    buraco_max_m2=BURACO_MAX_M2, dangle_m=DANGLE_M, travessa_max_m=TRAVESSA_MAX_M,
    angulo_travessa=ANGULO_TRAVESSA, colinear_min=COLINEAR_MIN, suave_m=SUAVE_M,
    extender_m=EXTENDER_M, simplificar_m=SIMPLIFICAR_M,
    batch_inferencia=BATCH_INFERENCIA, padding_max=PADDING_MAX,
)
with open(os.path.join(PASTA_AVAL, f"parametros_{CORRIDA}.json"), "w", encoding="utf-8") as fh:
    json.dump(PARAMETROS, fh, indent=2, ensure_ascii=False)
arcpy.env.outputCoordinateSystem = arcpy.Describe(
    os.path.join(PASTA_ORTOS, TESTE[0] + ".tif")).spatialReference
FOLHAS = [ORTO_TESTE] if MODO_TESTE else TESTE

livre, total = gpu_livre_gb()
if livre is not None:
    log(f"GPU: {livre:.1f} GB livres de {total:.1f} GB")
    if livre < MIN_GPU_LIVRE_GB and REFAZER_INFERENCIA:
        raise SystemExit(f"Menos de {MIN_GPU_LIVRE_GB} GB livres na GPU: não arrancar "
                         "para não pôr em risco o treino em curso.")

log(f"Extensões para o Polygon to Centerline: {ESTADO_EXTENSOES}")
linhas_m, linhas_c, linhas_p, linhas_t = [], [], [], []
CAMPOS_TOL = ["precisao", "recall", "f1", "iou"]
log(f"Avaliação v4 | corrida {CORRIDA} | modo teste: {MODO_TESTE} | ortos: {', '.join(FOLHAS)} "
    f"| versões: {VERSOES} | tolerâncias: {TOLERANCIAS} m")

for nome, cfg in MODELOS.items():
    emd = modelo_emd(cfg["pasta"])
    pasta_m = os.path.join(PASTA_AVAL, nome)            # rasters classificados (partilhados)
    pasta_c = os.path.join(pasta_m, f"corrida_{CORRIDA}")  # saídas desta corrida
    os.makedirs(pasta_c, exist_ok=True)
    log(f"=== {nome} | {emd}")
    log(f"  saídas desta corrida: {pasta_c}")
    acum = {(v, tol): [] for v in VERSOES for tol in TOLERANCIAS}
    acum_px = {v: [] for v in VERSOES}
    acum_tol = {(v, t): [] for v in VERSOES for t in TOLERANCIAS}

    for folha in FOLHAS:
        tif = os.path.join(PASTA_ORTOS, folha + ".tif")
        log(f"  {folha}")
        ras = os.path.join(pasta_m, f"{folha}_classificado.tif")
        try:
            inferir(emd, tif, ras)
        except arcpy.ExecuteError:
            log(f"    ERRO na inferência; orto saltada:\n    {arcpy.GetMessages(2)}")
            continue
        frac, valores = fracao_via(ras)
        log(f"    píxeis via: {frac:.1%} | valores no raster: {valores}")
        if frac > MAX_FRACAO_VIA:
            log(f"    ATENÇÃO: mais de {MAX_FRACAO_VIA:.0%} classificado como via: orto saltada")
            continue

        area = poligono_orto(tif, "memory/area")
        ref = os.path.join(pasta_c, f"{folha}_referencia.shp")
        referencia(cfg["referencia"], area, ref)
        ref_pol = os.path.join(pasta_c, f"{folha}_referencia_pol.shp")
        referencia_poligonos(ref, cfg["referencia"], area, ref_pol)
        arcpy.management.Delete(area)
        log(f"    referência {comprimento(ref) / 1000:.1f} km")

        for versao in VERSOES:
            log(f"    --- versão {versao}")
            try:
                ext, mascara = vetorizar_versao(versao, ras, pasta_c, folha)
            except arcpy.ExecuteError:
                log(f"      AVISO: versão '{versao}' falhou e foi saltada:\n"
                    f"      {arcpy.GetMessages(2)}")
                continue

            cm = matriz_confusao(mascara, ref_pol)
            pred_a, ref_a, cel = arrays_pixel(mascara, ref_pol)
            ct = contagens_tolerancia(pred_a, ref_a, cel, TOLERANCIAS)
            del pred_a, ref_a
            for tol, c in ct.items():
                acum_tol[(versao, tol)].append(c)
                mt = metricas_tolerancia(c)
                linhas_t.append([nome, versao, folha, tol] + [fmt(mt[k]) for k in CAMPOS_TOL])
                if tol == 3.0:
                    log(f"      píxel tol 3 m: precisão {fmt(mt['precisao'])} | recall "
                        f"{fmt(mt['recall'])} | F1 {fmt(mt['f1'])} | IoU {fmt(mt['iou'])}")
            acum_px[versao].append(cm)
            mp = metricas_pixel(cm)
            linhas_p.append([nome, versao, folha, cm["vp"], cm["fp"], cm["fn"], cm["vn"]]
                            + [fmt(mp[k]) for k in CAMPOS_PIXEL])
            log(f"      píxel: OA {fmt(mp['oa'])} | F1 via {fmt(mp['f1_via'])} | "
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

    # Totais: soma das matrizes e dos comprimentos (não a média das razões)
    for versao in VERSOES:
        if not acum_px[versao]:
            continue
        for tol in TOLERANCIAS:
            tt = {k: sum(c[k] for c in acum_tol[(versao, tol)])
                  for k in ("n_pred", "n_ref", "vp_p", "vp_r")}
            mt = metricas_tolerancia(tt)
            linhas_t.append([nome, versao, "TOTAL", tol] + [fmt(mt[k]) for k in CAMPOS_TOL])
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
                log(f"  TOTAL {versao} | píxel F1 {fmt(mp['f1_via'])} | eixos tol 3 m: "
                    f"completeness {fmt(r['completeness'])} | correctness "
                    f"{fmt(r['correctness'])} | quality {fmt(r['quality'])}")


def gravar(nome_csv, cabecalho, linhas):
    with open(os.path.join(PASTA_AVAL, nome_csv), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(cabecalho)
        w.writerows(linhas)


gravar(f"matriz_confusao_{CORRIDA}.csv", ["modelo", "versao", "folha", "VP", "FP", "FN", "VN"] + CAMPOS_PIXEL,
       linhas_p)
gravar(f"pixel_tolerancia_{CORRIDA}.csv", ["modelo", "versao", "folha", "tolerancia_m"]
       + CAMPOS_TOL, linhas_t)
gravar(f"metricas_eixos_{CORRIDA}.csv", ["modelo", "versao", "folha", "tolerancia_m", "ref_km",
                              "extracao_km", "completeness", "correctness", "quality"], linhas_m)
gravar(f"completeness_classes_{CORRIDA}.csv", ["modelo", "versao", "folha", "tolerancia_m", "classe",
                                    "ref_km", "detetado_km", "completeness"], linhas_c)
log(f"Concluído -> {PASTA_AVAL}")