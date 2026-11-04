# -*- coding: utf-8 -*-
r"""
45_figuras_RViaria.py  (ArcGIS Pro 3.6)

Figuras de comparação visual da rede viária, a partir dos resultados do
44_avaliar_RViaria.py. Regra: comparar sempre o mesmo tipo de objeto
(áreas com áreas, linhas com linhas) e mostrar o que as métricas medem.

  A  fig_A_<recorte>.png        DO MODELO AO RESULTADO FINAL (uma linha por modelo)
       referência (buffer das etiquetas) | previsão (máscara do modelo) |
       erros da área (tolerância TOL) | resultado final: eixos extraídos
       (correto / falso positivo / omitido, tolerância TOL)

  B  fig_B_rede_<recorte>.png   REDE EM ÁREA MAIOR (lado x FATOR_B)
       referência (eixos) | previsão (máscara) | resultado de cada modelo
       (correto / falso positivo / omitido)
  B2 fig_B2_classes_<recorte>.png
       igual, com as vias omitidas coloridas pela classe de origem

  C  fig_C_etapas_<modelo>_<recorte>.png   METODOLOGIA (vetorização)
       referência (buffer) | máscara prevista | eixos bruto | eixos pos | eixos
       centerline, com o buffer da referência por baixo; a ciano, o que cada
       versão removeu em relação ao bruto (esporões, travessas, troços soltos)

Cores (iguais em todas as figuras): amarelo = referência | magenta = previsão |
verde = correto | vermelho = falso positivo | azul = omitido | ciano = removido.

Os eixos são lidos ponto a ponto dos ficheiros (sem recortes do ArcGIS) e
reprojetados para o sistema das ortos se for diferente; o ecrã mostra, para
cada ficheiro, o número de troços lidos.

Recortes: (título, orto, x, y, lado em m, modelos).
  x, y = None        -> janela automática: a que tem mais vias corretas do primeiro
                        modelo, desde que tenha também omitidos e falsos positivos
  x = "autoestrada"  -> janela centrada nos eixos de referência da classe Autoestrada
As coordenadas escolhidas são escritas no ecrã para se poderem fixar ou ajustar.

Correr na linha de comandos (arcgispro-py3):
    python "D:\Users\Cap_Almeida\Projeto\Python\45_figuras_RViaria.py"
"""
import os
import glob
import numpy as np
import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, binary_dilation

arcpy.env.overwriteOutput = True

# ============================================================
# CONFIGURAÇÃO
# ============================================================
PASTA_AVAL    = r"D:\DL\Avaliacao\126_RViaria"
PASTA_ORTOS   = r"D:\Users\Cap_Almeida\Projeto\M888_Ortos\126_Ortofotos"
PASTA_FIGURAS = r"D:\DL\Avaliacao\126_RViaria_figuras"
CORRIDA       = "20261103_101943"   # None = a mais recente com os resultados de cada orto
CAMPO_SOURCE  = "source"

NOMES_MODELOS = {
    "mtre_rede_viaria": "Multi-Task Road Extractor",
    "unet_rede_viaria": "U-Net ResNet-50",
    "mtre_ferrovia":    "Multi-Task Road Extractor (ferrovia)",
}
MODELOS_REDE = ["mtre_rede_viaria", "unet_rede_viaria"]

# (título, orto, x, y, lado em metros, modelos)
RECORTES = [
    ("Zona urbana",    "126-I-2",   None, None, 200, MODELOS_REDE),
    ("Zona mista",     "126-III-4", None, None, 200, MODELOS_REDE),
    ("Zona dispersa",  "126-IV-1",  None, None, 200, MODELOS_REDE),
    ("Autoestrada",    "126-III-4", "autoestrada", None, 250, MODELOS_REDE),
    ("Ferrovia",       "126-I-2",   None, None, 300, ["mtre_ferrovia"]),
    # Zonas escolhidas à mão (coordenadas do centro lidas no ArcGIS Pro), por exemplo:
    # ("Caminhos em floresta", "126-IV-1", 230000.0, 461000.0, 200, MODELOS_REDE),
]

TOL = 3.0                     # tolerância principal das métricas (m)
VERSAO_EIXOS = "bruto"        # eixos do resultado final (A e B): "bruto", "pos" ou "centerline"
FATOR_B = 2.0                 # a opção B usa uma janela com lado x FATOR_B (rede em área maior)
FIG_A, FIG_B, FIG_B2, FIG_C = True, True, True, True
MODELOS_C = None              # None = todos os modelos de cada recorte
VERSOES_C = ["bruto", "pos", "centerline"]

DPI = 300
ESCALA_M = 25                 # barra de escala (m); na opção B usa o dobro
ESBATER = 0.25                # 0 = orto com as cores originais; 1 = branco
# ============================================================

COR_REF, COR_PRED = "#FFE600", "#FF00FF"
COR_TP, COR_FP, COR_FN = "#00CC00", "#EB0000", "#0059FF"
COR_TRAV = "#00E5FF"
CORES_CLASSES = {
    "Auto_estrada":       "#7A00E6",
    "Estrada_larga":      "#E00000",
    "Estrada_estreita":   "#FF8C00",
    "Arruamento":         "#8B4513",
    "Acesso_auto":        "#FF4FA3",
    "Caminho_carreteiro": "#0047FF",
    "Caminho_a_pe_posto": "#00D5FF",
    "Caminho_de_ferro_via_larga_unica": "#000000",
}
NOMES_CLASSES = {
    "Auto_estrada": "Autoestrada", "Estrada_larga": "Estrada larga",
    "Estrada_estreita": "Estrada estreita", "Arruamento": "Arruamento",
    "Acesso_auto": "Acesso auto", "Caminho_carreteiro": "Caminho carreteiro",
    "Caminho_a_pe_posto": "Caminho de pé posto",
    "Caminho_de_ferro_via_larga_unica": "Ferrovia",
}
LISTA_CLASSES = list(CORES_CLASSES)


# ------------------------------------------------------------------ utilitários
def rgb_cor(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def nome_seguro(txt):
    return "".join(ch if ch.isalnum() else "_" for ch in txt)


def normalizar(valor):
    base = os.path.splitext(os.path.basename(str(valor or "").strip()))[0]
    return base.lower().replace(" ", "_").replace("-", "_")


def classe_origem(valor):
    n = normalizar(valor)
    norm = {normalizar(k): k for k in LISTA_CLASSES}
    if n in norm:
        return norm[n]
    for k in sorted(norm, key=len, reverse=True):
        if k in n:
            return norm[k]
    return None


_CORRIDAS = {}


def pasta_corrida(modelo, folha):
    chave = (modelo, folha)
    if chave not in _CORRIDAS:
        base = os.path.join(PASTA_AVAL, modelo)
        cands = ([os.path.join(base, f"corrida_{CORRIDA}")] if CORRIDA
                 else sorted(glob.glob(os.path.join(base, "corrida_*")), reverse=True))
        precisos = [f"eixos_{v}.shp" for v in set(VERSOES_C + [VERSAO_EIXOS])] + \
                   ["referencia.shp", "referencia_pol.shp"]
        escolhida = next((p for p in cands if all(
            os.path.exists(os.path.join(p, f"{folha}_{f}")) for f in precisos)), None)
        if not escolhida:
            raise FileNotFoundError(f"Nenhuma corrida de {modelo} com todos os resultados de {folha}")
        _CORRIDAS[chave] = escolhida
        print(f"   {modelo} | {folha}: {os.path.basename(escolhida)}")
    return _CORRIDAS[chave]


def f_classificado(modelo, folha):
    return os.path.join(PASTA_AVAL, modelo, f"{folha}_classificado.tif")


def f_corrida(modelo, folha, nome):
    return os.path.join(pasta_corrida(modelo, folha), f"{folha}_{nome}")


def f_travessas(modelo, folha, versao):
    return os.path.join(pasta_corrida(modelo, folha), "avaliacao.gdb",
                        f"travessas_{folha.replace('-', '_')}_{versao}")


def contar(fc):
    return int(arcpy.management.GetCount(fc)[0]) if fc and arcpy.Exists(fc) else 0


# ------------------------------------------------------------------ janela e leitura
def janela_grelha(folha, x, y, lado):
    r = arcpy.Raster(os.path.join(PASTA_ORTOS, folha + ".tif"))
    cs, e = r.meanCellWidth, r.extent
    n = int(round(lado / cs))
    xmin = e.XMin + round((x - lado / 2 - e.XMin) / cs) * cs
    ymin = e.YMin + round((y - lado / 2 - e.YMin) / cs) * cs
    xmin = min(max(xmin, e.XMin), e.XMax - n * cs)
    ymin = min(max(ymin, e.YMin), e.YMax - n * cs)
    return dict(xmin=xmin, ymin=ymin, xmax=xmin + n * cs, ymax=ymin + n * cs,
                n=n, cs=cs, sr=r.spatialReference, folha=folha)


def ler_raster(ras, jan, m=0):
    cs = jan["cs"]
    return arcpy.RasterToNumPyArray(ras, arcpy.Point(jan["xmin"] - m * cs, jan["ymin"] - m * cs),
                                    jan["n"] + 2 * m, jan["n"] + 2 * m, nodata_to_value=0)


def extensao(jan, m=0):
    cs = jan["cs"]
    return arcpy.Extent(jan["xmin"] - m * cs, jan["ymin"] - m * cs,
                        jan["xmax"] + m * cs, jan["ymax"] + m * cs)


def rasterizar(fc, jan, m, campo=None, poligonos=False):
    """Rasteriza linhas ou polígonos na grelha da janela (com margem m píxeis).
    Devolve o array de valores (0 = vazio)."""
    tam = jan["n"] + 2 * m
    if contar(fc) == 0:
        return np.zeros((tam, tam), np.int32)
    ras_ref = f_classificado(next(iter(MODELO_GRELHA)), jan["folha"])
    if campo is None:
        campo = arcpy.Describe(fc).OIDFieldName
    ext = extensao(jan, m)
    with arcpy.EnvManager(snapRaster=ras_ref, extent=ext, cellSize=jan["cs"]):
        if poligonos:
            arcpy.conversion.PolygonToRaster(fc, campo, "memory/rz", "CELL_CENTER", "", jan["cs"])
        else:
            arcpy.conversion.PolylineToRaster(fc, campo, "memory/rz", "MAXIMUM_LENGTH", "NONE",
                                              jan["cs"])
    a = arcpy.RasterToNumPyArray("memory/rz", ext.lowerLeft, tam, tam, nodata_to_value=0)
    arcpy.management.Delete("memory/rz")
    return a.astype(np.int32)


MODELO_GRELHA = {}            # modelo usado para alinhar a grelha (preenchido em main)


def recortar_janela(fc, jan, m, nome):
    if not fc or not arcpy.Exists(fc):
        return None
    ext = extensao(jan, m)
    pts = arcpy.Array([arcpy.Point(ext.XMin, ext.YMin), arcpy.Point(ext.XMin, ext.YMax),
                       arcpy.Point(ext.XMax, ext.YMax), arcpy.Point(ext.XMax, ext.YMin),
                       arcpy.Point(ext.XMin, ext.YMin)])
    arcpy.management.CopyFeatures([arcpy.Polygon(pts, jan["sr"])], "memory/jan_poly")
    out = f"memory/{nome}"
    arcpy.analysis.PairwiseClip(fc, "memory/jan_poly", out)
    return out


_GEOM = {}


def ler_linhas(fc, sr, campo=None):
    """Todas as partes de linha do ficheiro: [(array Nx2 de XY, valor do campo)].
    Lidas ponto a ponto; reprojetadas para sr se o sistema for diferente."""
    chave = (fc, campo)
    if chave in _GEOM:
        return _GEOM[chave]
    partes = []
    if fc and arcpy.Exists(fc):
        sr_fc = arcpy.Describe(fc).spatialReference
        projetar = bool(sr_fc and sr_fc.name not in ("Unknown", "") and sr_fc.name != sr.name)
        campos = ["SHAPE@"] + ([campo] if campo else [])
        with arcpy.da.SearchCursor(fc, campos) as cur:
            for row in cur:
                g = row[0]
                if g is None:
                    continue
                if projetar:
                    g = g.projectAs(sr)
                for parte in g:
                    pts = [(p.X, p.Y) for p in parte if p is not None]
                    if len(pts) >= 2:
                        partes.append((np.array(pts, dtype=float), row[1] if campo else None))
        nota = f" (reprojetado de {sr_fc.name})" if projetar else ""
        print(f"     lido {os.path.basename(fc)}: {len(partes)} partes{nota}")
    else:
        print(f"     AVISO: não existe {fc}")
    _GEOM[chave] = partes
    return partes


def raster_linhas(partes, jan, m=0, valor=None):
    """Rasteriza as linhas na grelha da janela (margem m píxeis). valor(v) -> inteiro."""
    cs = jan["cs"]
    tam = jan["n"] + 2 * m
    x0, y1 = jan["xmin"] - m * cs, jan["ymax"] + m * cs
    x1, y0 = x0 + tam * cs, y1 - tam * cs
    a = np.zeros((tam, tam), np.int32)
    for pts, v in partes:
        if (pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or
                pts[:, 1].max() < y0 or pts[:, 1].min() > y1):
            continue
        val = valor(v) if valor else 1
        if not val:
            continue
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            k = int(max(abs(xb - xa), abs(yb - ya)) / (cs / 2)) + 2
            xs, ys = np.linspace(xa, xb, k), np.linspace(ya, yb, k)
            c = np.floor((xs - x0) / cs).astype(int)
            r = np.floor((y1 - ys) / cs).astype(int)
            dentro = (c >= 0) & (c < tam) & (r >= 0) & (r < tam)
            a[r[dentro], c[dentro]] = val
    return a


def linhas_px(partes, jan):
    """Linhas em coordenadas de píxel da janela (só as que a intersetam)."""
    cs = jan["cs"]
    out = []
    for pts, _ in partes:
        if (pts[:, 0].max() < jan["xmin"] or pts[:, 0].min() > jan["xmax"] or
                pts[:, 1].max() < jan["ymin"] or pts[:, 1].min() > jan["ymax"]):
            continue
        out.append(((pts[:, 0] - jan["xmin"]) / cs - 0.5, (jan["ymax"] - pts[:, 1]) / cs - 0.5))
    return out


def codigo_classe(v):
    c = classe_origem(v)
    return LISTA_CLASSES.index(c) + 1 if c else 0


def ref_linhas_classes(mod, jan, m):
    """Eixos de referência rasterizados com o código da classe (1..8; 0 = vazio)."""
    partes = ler_linhas(f_corrida(mod, jan["folha"], "referencia.shp"), jan["sr"], CAMPO_SOURCE)
    return raster_linhas(partes, jan, m, codigo_classe)


def eixos_raster(mod, jan, m, versao):
    partes = ler_linhas(f_corrida(mod, jan["folha"], f"eixos_{versao}.shp"), jan["sr"])
    return raster_linhas(partes, jan, m) > 0


# ------------------------------------------------------------------ cálculo
def erros_area(pred, ref, tol, cs):
    """Áreas: previsto correto (até tol da referência), falso positivo e omitido."""
    if tol <= 0:
        return pred & ref, pred & ~ref, ref & ~pred
    d_ref = distance_transform_edt(~ref) * cs if ref.any() else np.full(ref.shape, np.inf)
    d_pred = distance_transform_edt(~pred) * cs if pred.any() else np.full(pred.shape, np.inf)
    ok = pred & (d_ref <= tol)
    return ok, pred & ~ok, ref & ~(d_pred <= tol)


def estado_linhas(ext_l, ref_l, tol, cs):
    """Linhas: extraído correto, falso positivo e referência omitida (por distância)."""
    d_ref = distance_transform_edt(~ref_l) * cs if ref_l.any() else np.full(ref_l.shape, np.inf)
    d_ext = distance_transform_edt(~ext_l) * cs if ext_l.any() else np.full(ext_l.shape, np.inf)
    ok = ext_l & (d_ref <= tol)
    return ok, ext_l & ~ok, ref_l & ~(d_ext <= tol)


# ------------------------------------------------------------------ escolha automática
def escolher_centro(modelo, folha, lado):
    """Janela com mais vias corretas, desde que tenha também omitidos e falsos positivos."""
    ras = f_classificado(modelo, folha)
    r = arcpy.Raster(ras)
    cs = r.meanCellWidth
    with arcpy.EnvManager(snapRaster=ras, extent=ras, cellSize=ras):
        arcpy.conversion.PolygonToRaster(f_corrida(modelo, folha, "referencia_pol.shp"), "VAL",
                                         "memory/ref_full", "CELL_CENTER", "", cs)
    ref = arcpy.RasterToNumPyArray("memory/ref_full", r.extent.lowerLeft, r.width, r.height,
                                   nodata_to_value=0) == 1
    arcpy.management.Delete("memory/ref_full")
    pred = arcpy.RasterToNumPyArray(ras, r.extent.lowerLeft, r.width, r.height,
                                    nodata_to_value=0) == 1
    tp, fp, fn = erros_area(pred, ref, TOL, cs)
    del pred, ref
    n = int(lado / cs)
    passo = max(n // 4, 1)
    li = np.arange(0, tp.shape[0] - n + 1, passo)
    co = np.arange(0, tp.shape[1] - n + 1, passo)
    I, J = np.meshgrid(li, co, indexing="ij")

    def somas(m):
        s_ = np.pad(m.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return s_[I + n, J + n] - s_[I, J + n] - s_[I + n, J] + s_[I, J]

    s_tp, s_fp, s_fn = somas(tp), somas(fp), somas(fn)
    valido = (s_fn >= 0.10 * s_tp) & (s_fp >= 0.05 * s_tp) & (s_tp > 0)
    pont = np.where(valido, s_tp, -1)
    if pont.max() < 0:                       # nenhuma janela cumpre: a com mais corretos
        pont = s_tp
    i, j = np.unravel_index(np.argmax(pont), pont.shape)
    i, j = li[i], co[j]
    return r.extent.XMin + (j + n / 2) * cs, r.extent.YMax - (i + n / 2) * cs


def centro_autoestrada(modelo, folha, lado):
    """Centro da janela com mais eixos de referência da classe Autoestrada."""
    sr = arcpy.Describe(os.path.join(PASTA_ORTOS, folha + ".tif")).spatialReference
    partes = ler_linhas(f_corrida(modelo, folha, "referencia.shp"), sr, CAMPO_SOURCE)
    pts = [pt for xy, v in partes if classe_origem(v) == "Auto_estrada" for pt in xy]
    if not pts:
        print(f"   sem autoestrada na referência de {folha}: janela automática normal")
        return escolher_centro(modelo, folha, lado)
    p = np.array(pts)
    viz = [(np.abs(p - q).max(axis=1) <= lado / 2).sum() for q in p[:: max(len(p) // 500, 1)]]
    q = p[:: max(len(p) // 500, 1)][int(np.argmax(viz))]
    perto = p[np.abs(p - q).max(axis=1) <= lado / 2]
    cx, cy = perto.mean(axis=0)
    return float(cx), float(cy)


# ------------------------------------------------------------------ desenho
def orto_cor(img):
    a = np.moveaxis(img[:3].astype(np.float32), 0, -1)
    for k in range(3):
        lo, hi = np.percentile(a[..., k], (2, 98))
        a[..., k] = np.clip((a[..., k] - lo) / max(hi - lo, 1), 0, 1)
    return a * (1 - ESBATER) + ESBATER


def pintar(base, mascara, cor, alfa=0.6):
    out = base.copy()
    out[mascara] = (1 - alfa) * base[mascara] + alfa * rgb_cor(cor)
    return out


def engrossar(m, px=1):
    return binary_dilation(m, iterations=px) if px > 0 and m.any() else m


def preparar(ax, jan, titulo=None, escala=ESCALA_M):
    n = jan["n"]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    px = escala / jan["cs"]
    x0, y0 = n * 0.05, n * 0.94
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=5, solid_capstyle="butt")
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - n * 0.025, f"{escala:g} m", color="white", ha="center",
            va="bottom", fontsize=9, fontweight="bold")
    if titulo:
        ax.set_title(titulo, fontsize=11)


def finalizar(fig, nome, titulo, itens, ncol=4):
    """Título e legenda com espaço fixo em polegadas (não se sobrepõem às imagens)."""
    alt = fig.get_size_inches()[1]
    linhas_leg = int(np.ceil(len(itens) / ncol)) if itens else 0
    baixo = (0.15 + 0.32 * linhas_leg) / alt
    cima = 1 - 0.55 / alt
    fig.suptitle(titulo, fontsize=13, y=1 - 0.15 / alt)
    if itens:
        fig.legend(handles=itens, loc="lower center", ncol=min(len(itens), ncol), fontsize=10,
                   frameon=False, bbox_to_anchor=(0.5, 0.05 / alt))
    fig.tight_layout(rect=(0, baixo, 1, cima))
    caminho = os.path.join(PASTA_FIGURAS, nome)
    fig.savefig(caminho, dpi=DPI)
    plt.close(fig)
    print(f"   {caminho}")


def pintar_estado(base, ok, fp, fn, px=2):
    im = pintar(base, engrossar(fn, px), COR_FN, 0.95)
    im = pintar(im, engrossar(fp, px), COR_FP, 0.95)
    return pintar(im, engrossar(ok, px), COR_TP, 0.95)


def estado_eixos(mod, jan, m, versao=None):
    """Linhas extraídas vs referência: (correto, falso positivo, omitido, classes), sem margem."""
    versao = versao or VERSAO_EIXOS
    folha, sl = jan["folha"], (slice(m, -m), slice(m, -m))
    cls_m = ref_linhas_classes(mod, jan, m)
    ext_l = eixos_raster(mod, jan, m, versao)
    ok, fp, fn = [x[sl] for x in estado_linhas(ext_l, cls_m > 0, TOL, jan["cs"])]
    cs = jan["cs"]
    print(f"     {mod}: eixos extraídos ~{ext_l[sl].sum() * cs:.0f} m (corretos ~{ok.sum() * cs:.0f} m,"
          f" falsos positivos ~{fp.sum() * cs:.0f} m) | referência omitida ~{fn.sum() * cs:.0f} m")
    return ok, fp, fn, cls_m[sl]


ITENS_ESTADO = [Patch(color=COR_TP, label=f"Extraído correto (≤ {TOL:g} m)"),
                Patch(color=COR_FP, label="Falso positivo"),
                Patch(color=COR_FN, label="Omitido")]


# ------------------------------------------------------------------ opção A
def figura_A(rec, jan, img):
    titulo, folha, _, _, _, modelos = rec
    base = orto_cor(img)
    m = int(np.ceil(TOL / jan["cs"])) + 2
    sl = (slice(m, -m), slice(m, -m))
    fig, axs = plt.subplots(len(modelos), 4, figsize=(18, 4.7 * len(modelos) + 1.0), squeeze=False)
    for i, mod in enumerate(modelos):
        ref_m = rasterizar(f_corrida(mod, folha, "referencia_pol.shp"), jan, m, "VAL",
                           poligonos=True) > 0
        pred_m = ler_raster(f_classificado(mod, folha), jan, m) == 1
        ref, pred = ref_m[sl], pred_m[sl]
        tp, fp, fn = [x[sl] for x in erros_area(pred_m, ref_m, TOL, jan["cs"])]
        ok, fpl, fnl, _ = estado_eixos(mod, jan, m)
        a = axs[i]
        a[0].imshow(pintar(base, ref, COR_REF, 0.6))
        a[1].imshow(pintar(base, pred, COR_PRED, 0.55))
        a[2].imshow(pintar(pintar(pintar(base, fn, COR_FN, .7), fp, COR_FP, .7), tp, COR_TP, .7))
        a[3].imshow(pintar_estado(base, ok, fpl, fnl))
        for k, t in enumerate(("Referência (buffer das etiquetas)", "Previsão do modelo (máscara)",
                               f"Erros da área (tolerância {TOL:g} m)",
                               f"Resultado final: eixos ({VERSAO_EIXOS})")):
            preparar(a[k], jan, t if i == 0 else None)
        a[0].set_ylabel(NOMES_MODELOS.get(mod, mod), fontsize=12, fontweight="bold")
    itens = [Patch(color=COR_REF, label="Referência"), Patch(color=COR_PRED, label="Previsão")] \
        + ITENS_ESTADO
    finalizar(fig, f"fig_A_{nome_seguro(titulo)}.png",
              f"Do modelo ao resultado final - {titulo} ({folha})", itens, ncol=5)


# ------------------------------------------------------------------ opção B
def figura_B(rec, jan, img, por_classe=False):
    titulo, folha, _, _, _, modelos = rec
    base = orto_cor(img)
    m = int(np.ceil(TOL / jan["cs"])) + 2
    escala = ESCALA_M * 2
    # referência | previsão de cada modelo | resultado de cada modelo
    n_col = 1 + 2 * len(modelos)
    fig, axs = plt.subplots(1, n_col, figsize=(5.6 * n_col, 6.9), squeeze=False)
    ref_l = ref_linhas_classes(modelos[0], jan, 0) > 0
    axs[0][0].imshow(pintar(base, engrossar(ref_l, 2), COR_REF, 0.95))
    preparar(axs[0][0], jan, "Referência (eixos)", escala)
    for k, mod in enumerate(modelos):
        pred = ler_raster(f_classificado(mod, folha), jan) == 1
        axs[0][1 + k].imshow(pintar(base, pred, COR_PRED, 0.55))
        preparar(axs[0][1 + k], jan, f"Previsão (máscara)\n{NOMES_MODELOS.get(mod, mod)}", escala)
    classes_vistas = set()
    for k, mod in enumerate(modelos, start=1 + len(modelos)):
        ok, fp, fn, cls = estado_eixos(mod, jan, m)
        if por_classe:
            im = base.copy()
            for c_idx, c in enumerate(LISTA_CLASSES, start=1):
                mk = fn & (cls == c_idx)
                if mk.any():
                    classes_vistas.add(c)
                    im = pintar(im, engrossar(mk, 2), CORES_CLASSES[c], 0.95)
            im = pintar(im, engrossar(ok, 2), COR_TP, 0.95)
        else:
            im = pintar_estado(base, ok, fp, fn)
        axs[0][k].imshow(im)
        preparar(axs[0][k], jan, f"Resultado (eixos {VERSAO_EIXOS})\n{NOMES_MODELOS.get(mod, mod)}",
                 escala)
    if por_classe:
        itens = [Patch(color=COR_REF, label="Referência"), Patch(color=COR_PRED, label="Previsão"),
                 Patch(color=COR_TP, label=f"Extraído correto (≤ {TOL:g} m)")]
        itens += [Patch(color=CORES_CLASSES[c], label=f"Omitido: {NOMES_CLASSES[c]}")
                  for c in LISTA_CLASSES if c in classes_vistas]
        nome, sub = f"fig_B2_classes_{nome_seguro(titulo)}.png", "vias omitidas por classe"
    else:
        itens = [Patch(color=COR_REF, label="Referência"), Patch(color=COR_PRED, label="Previsão")] \
            + ITENS_ESTADO
        nome, sub = f"fig_B_rede_{nome_seguro(titulo)}.png", f"eixos {VERSAO_EIXOS}"
    finalizar(fig, nome, f"Rede extraída - {titulo} ({folha}) - {sub}", itens)


# ------------------------------------------------------------------ opção C
DIST_REMOVIDO = 2.0           # m: parte do bruto a mais desta distância da versão = removida


def figura_C(rec, jan, img, mod):
    titulo, folha, _, _, _, _ = rec
    base = orto_cor(img)
    cs = jan["cs"]
    ref = rasterizar(f_corrida(mod, folha, "referencia_pol.shp"), jan, 0, "VAL", poligonos=True) > 0
    pred = ler_raster(f_classificado(mod, folha), jan) == 1
    com_ref = pintar(base, ref, COR_REF, 0.45)
    bruto_l = eixos_raster(mod, jan, 0, "bruto")
    paineis = [("Referência (buffer)", pintar(base, ref, COR_REF, 0.6), None),
               ("Máscara prevista", pintar(base, pred, COR_PRED, 0.55), None)]
    paineis += [(f"Eixos {v}", com_ref, v) for v in VERSOES_C]
    fig, axs = plt.subplots(1, len(paineis), figsize=(4.8 * len(paineis), 6.3), squeeze=False)
    ha_removido = False
    for k, (tit, imagem, versao) in enumerate(paineis):
        ax = axs[0][k]
        if versao and versao != "bruto":
            ver_l = eixos_raster(mod, jan, 0, versao)
            d = distance_transform_edt(~ver_l) * cs if ver_l.any() else np.full(ver_l.shape, np.inf)
            removido = bruto_l & (d > DIST_REMOVIDO)
            if removido.any():
                ha_removido = True
                imagem = pintar(imagem, engrossar(removido, 1), COR_TRAV, 0.95)
        ax.imshow(imagem)
        if versao:
            partes = ler_linhas(f_corrida(mod, folha, f"eixos_{versao}.shp"), jan["sr"])
            for xs, ys in linhas_px(partes, jan):
                ax.plot(xs, ys, color=COR_PRED, lw=2.0, zorder=3, solid_capstyle="round")
        preparar(ax, jan, tit)
    itens = [Patch(color=COR_REF, alpha=0.6, label="Referência (buffer)"),
             Patch(color=COR_PRED, alpha=0.6, label="Máscara prevista"),
             Line2D([0], [0], color=COR_PRED, lw=2.5, label="Eixos extraídos")]
    if ha_removido:
        itens.append(Patch(color=COR_TRAV, label="Removido em relação ao bruto"))
    finalizar(fig, f"fig_C_etapas_{mod}_{nome_seguro(titulo)}.png",
              f"Etapas da vetorização - {NOMES_MODELOS.get(mod, mod)} - {titulo} ({folha})", itens)


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    print("Corridas usadas:")
    recortes = []
    for titulo, folha, x, y, lado, modelos in RECORTES:
        for mod in modelos:
            pasta_corrida(mod, folha)
        if x == "autoestrada":
            x, y = centro_autoestrada(modelos[0], folha, lado)
            print(f"   recorte '{titulo}' ({folha}) na autoestrada: centro {x:.1f} {y:.1f}")
        elif x is None or y is None:
            x, y = escolher_centro(modelos[0], folha, lado)
            print(f"   recorte automático '{titulo}' ({folha}): centro {x:.1f} {y:.1f}")
        recortes.append((titulo, folha, x, y, lado, modelos))

    for rec in recortes:
        titulo, folha, x, y, lado, modelos = rec
        MODELO_GRELHA.clear()
        MODELO_GRELHA[modelos[0]] = True
        orto = os.path.join(PASTA_ORTOS, folha + ".tif")
        print(f"\nRecorte '{titulo}' ({folha}) | centro {x:.1f} {y:.1f} | {lado} m")
        jan = janela_grelha(folha, x, y, lado)
        img = ler_raster(orto, jan)
        if FIG_A:
            figura_A(rec, jan, img)
        if FIG_C:
            for mod in (MODELOS_C or modelos):
                if mod in modelos:
                    figura_C(rec, jan, img, mod)
        if FIG_B or FIG_B2:
            jan_b = janela_grelha(folha, x, y, lado * FATOR_B)
            img_b = ler_raster(orto, jan_b)
            print(f"   rede em área maior: {lado * FATOR_B:g} m")
            if FIG_B:
                figura_B(rec, jan_b, img_b)
            if FIG_B2:
                figura_B(rec, jan_b, img_b, por_classe=True)
    arcpy.management.Delete("memory")
    print("\nConcluído.")


if __name__ == "__main__":
    main()