# -*- coding: utf-8 -*-
r"""
11_figuras_2.py  (ArcGIS Pro 3.6)

Gera as figuras da tese que precisam dos rasters e dos vetores (as restantes
foram feitas a partir dos resultados ou ja existem na pasta "figuras"):

  Figura 2  - desvio entre os telhados na ortofotografia e os vetores de edificios
  Figura 3  - ortofotografia e nDSM normalizado, com os contornos dos vetores
  Figura 4  - divisao da folha 126 em ortofotos de treino e de teste
  Figura 5  - exemplos de chips de treino (RGB, nDSM e rotulo)
  Figura 7  - etapas do pos-processamento (classificacao, limpeza, regularizacao)
  Figura 13 - erros tipicos do melhor modelo (fusao, omissao, falso positivo)

Usa as saidas do 06_separar_treino_teste.py, do 07_exportar_chips_nDSM.py e do
09_avaliar_modelos.py (pastas de avaliacao das fases A e B). Nao usa a GPU.

Os recortes sao escolhidos automaticamente (ver cada funcao), e o centro de cada
um fica escrito no ecra; para usar outro sitio, indica as coordenadas em RECORTES
ou em CASOS_ERROS (sistema do ortomosaico, lidas no ArcGIS Pro).

Correr no Python Command Prompt do ArcGIS Pro:
    python 11_figuras_2.py            (todas as figuras)
    python 11_figuras_2.py 4 13       (so as figuras 4 e 13; "2" faz a 2 e a 3)
"""
import os
import sys
import glob
import numpy as np
import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURACAO
# ============================================================
SHP = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf"
EDIFICIOS   = os.path.join(SHP, "a_constr_Limpa.shp")        # referencia (posicao original)
ZONA_TESTE  = os.path.join(SHP, "zona_teste.shp")
GRELHA      = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\Orto\ortos_grelha.shp"
CAMPO_NOME  = "NOME"
CHIPS_NDSM  = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\Training\original1_nDSM_256_s128"

AVAL_B = r"D:\DL\Avaliacao\126_Construcao_ndsm"    # avaliacao da fase B
GRUPO_B = "faseB_ndsm"
MODELO_B = "unet_resnet50"                       # inicio do nome do modelo das figuras 7 e 13

PASTA_FIGURAS = r"D:\DL\Avaliacao\126_Construcao_figuras\figuras_2"

ORTO_URBANA = "126-I-3"
TIPOS_ORTOS = {"126-I-3": "urbana", "126-II-3": "mista", "126-IV-1": "dispersa"}
# Recortes: (x do centro, y do centro, lado em m). None = escolha automatica.
RECORTES = {"fig2": None, "fig7": None}
LADO_FIG2, LADO_FIG7, LADO_ERRO = 70, 80, 40
# Casos da figura 13: lista de (titulo, orto, x, y) ou None = escolha automatica
CASOS_ERROS = None

BANDA_NDSM, NDSM_MAX_M = 5, 40
DPI = 300
# ============================================================

C_REF, C_PRED, C_LIMPO = "#FFD400", "#FF2A2A", "#00E5FF"
C_TP, C_FP, C_FN = (0.15, 0.75, 0.25), (0.90, 0.15, 0.15), (0.15, 0.45, 0.95)
plt.rcParams.update({"font.size": 10})


# ------------------------------------------------------------------ leitura
def ns(txt):
    return txt.replace("-", "_").replace(" ", "_").replace(".", "_")


def ortos_dir(aval, grupo):
    return os.path.join(aval, "ortos_teste", grupo)


def orto_tif(aval, grupo, orto):
    return os.path.join(ortos_dir(aval, grupo), f"orto_{ns(orto)}.tif")


def modelo_pasta(aval, grupo, pref):
    pasta = os.path.join(aval, "classificacoes", grupo)
    c = sorted(d for d in os.listdir(pasta) if d.lower().startswith(pref.lower()))
    if not c:
        raise FileNotFoundError(f"Nenhum modelo '{pref}*' em {pasta}")
    return c[-1]


def janela(raster, xmin, ymin, n):
    return arcpy.RasterToNumPyArray(raster, arcpy.Point(xmin, ymin), n, n, nodata_to_value=0)


def ler_todo(raster):
    r = arcpy.Raster(raster)
    return arcpy.RasterToNumPyArray(raster, nodata_to_value=0), r


def rgb(img):
    a = np.moveaxis(img[:3].astype(np.float32), 0, -1)
    for k in range(3):
        lo, hi = np.percentile(a[..., k], (2, 98))
        a[..., k] = np.clip((a[..., k] - lo) / max(hi - lo, 1), 0, 1)
    return a


def cinzento(img):
    g = rgb(img).mean(axis=-1)
    return np.dstack([g, g, g]) * 0.8 + 0.1


def mapa_erros(img, ref, pred):
    base = cinzento(img)
    out = base.copy()
    for m, cor in ((ref & pred, C_TP), (pred & ~ref, C_FP), (ref & ~pred, C_FN)):
        out[m] = 0.35 * base[m] + 0.65 * np.array(cor)
    return out


def geometrias(fc, ext=None):
    """Geometrias de fc (opcionalmente so as que tocam a extensao ext)."""
    gs = []
    with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as sc:
        for (g,) in sc:
            if g is None:
                continue
            if ext is not None and g.extent.disjoint(ext):
                continue
            gs.append(g)
    return gs


def desenhar(ax, gs, xmin, ymax, cs, cor, lw=1.4, ls="-"):
    """Desenha os contornos das geometrias em coordenadas de pixel da janela."""
    for g in gs:
        for parte in g:
            xs, ys = [], []
            for p in list(parte) + [None]:
                if p is None:
                    if len(xs) > 1:
                        ax.plot(xs, ys, color=cor, lw=lw, ls=ls)
                    xs, ys = [], []
                else:
                    xs.append((p.X - xmin) / cs - 0.5)
                    ys.append((ymax - p.Y) / cs - 0.5)


def extensao(xmin, ymin, lado, sr):
    return arcpy.Extent(xmin, ymin, xmin + lado, ymin + lado, spatial_reference=sr)


def barra(ax, cs, n, m=20):
    px = m / cs
    x0, y0 = n * 0.05, n * 0.93
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=4, solid_capstyle="butt")
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=2, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - n * 0.03, f"{m} m", color="white", ha="center", va="bottom",
            fontsize=8, fontweight="bold")


def limpar(ax):
    ax.set_xticks([]); ax.set_yticks([])


def gravar(fig, nome):
    saida = os.path.join(PASTA_FIGURAS, nome)
    fig.savefig(saida, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   {saida}")


def melhor_janela(peso, msk, n, passo=None):
    """(linha, coluna) da janela n x n inteiramente na mascara com maior soma de 'peso'."""
    passo = passo or max(n // 4, 1)
    melhor, pos = -1, (0, 0)
    for i in range(0, peso.shape[0] - n, passo):
        for j in range(0, peso.shape[1] - n, passo):
            if not msk[i:i + n, j:j + n].all():
                continue
            v = peso[i:i + n, j:j + n].sum()
            if v > melhor:
                melhor, pos = v, (i, j)
    return pos


def centro_janela(raster_ref, pos, n):
    r = arcpy.Raster(raster_ref)
    cs = r.meanCellWidth
    i, j = pos
    return r.extent.XMin + (j + n / 2) * cs, r.extent.YMax - (i + n / 2) * cs


# ------------------------------------------------------------------ figuras 2 e 3
def recorte_edificios_altos(lado):
    """Janela da orto urbana com mais area edificada ponderada pela altura (nDSM)."""
    if RECORTES.get("fig2"):
        return RECORTES["fig2"][:2]
    oB = orto_tif(AVAL_B, GRUPO_B, ORTO_URBANA)
    d = ortos_dir(AVAL_B, GRUPO_B)
    g = arcpy.Raster(oB)
    cs = g.meanCellWidth
    canto = arcpy.Point(g.extent.XMin, g.extent.YMin)
    ref = arcpy.RasterToNumPyArray(os.path.join(d, f"ref_edicao_{ns(ORTO_URBANA)}.tif"), canto,
                                   g.width, g.height, nodata_to_value=0) == 1
    msk = arcpy.RasterToNumPyArray(os.path.join(d, f"mascara_{ns(ORTO_URBANA)}.tif"), canto,
                                   g.width, g.height, nodata_to_value=0) == 1
    ndsm = arcpy.RasterToNumPyArray(oB, canto, g.width, g.height, nodata_to_value=0)[BANDA_NDSM - 1]
    n = int(lado / cs)
    pos = melhor_janela(ref * ndsm.astype(np.float32), msk, n)
    x, y = centro_janela(oB, pos, n)
    print(f"   recorte automatico (edificios altos): centro {x:.1f} {y:.1f}")
    return x, y


def figuras_2_3():
    x, y = recorte_edificios_altos(LADO_FIG2)
    oB = orto_tif(AVAL_B, GRUPO_B, ORTO_URBANA)
    cs = arcpy.Raster(oB).meanCellWidth
    n = int(LADO_FIG2 / cs)
    xmin, ymin = x - LADO_FIG2 / 2, y - LADO_FIG2 / 2
    ymax = ymin + LADO_FIG2
    img = janela(oB, xmin, ymin, n)
    sr = arcpy.Describe(EDIFICIOS).spatialReference
    gs = geometrias(EDIFICIOS, extensao(xmin, ymin, LADO_FIG2, sr))

    # figura 2
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(rgb(img))
    desenhar(ax, gs, xmin, ymax, cs, C_REF, lw=1.6)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    barra(ax, cs, n, 10); limpar(ax)
    ax.legend(handles=[Line2D([0], [0], color=C_REF, lw=2, label="Edifícios (restituição, posição original)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=9)
    gravar(fig, "fig02_desvio_telhados.png")

    # figura 3
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    axs[0].imshow(rgb(img))
    im = axs[1].imshow(img[BANDA_NDSM - 1].astype(float) * NDSM_MAX_M / 255, cmap="gray", vmin=0, vmax=NDSM_MAX_M)
    for a, t in zip(axs, ("Ortofotografia", f"nDSM normalizado (0–{NDSM_MAX_M} m)")):
        desenhar(a, gs, xmin, ymax, cs, C_REF, lw=1.3)
        a.set_xlim(-0.5, n - 0.5); a.set_ylim(n - 0.5, -0.5)
        barra(a, cs, n, 10); limpar(a); a.set_title(t)
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02)
    cb.set_label("Altura acima do terreno (m)")
    fig.legend(handles=[Line2D([0], [0], color=C_REF, lw=2, label="Edifícios (restituição, posição original)")],
               loc="lower center", frameon=False, fontsize=9)
    gravar(fig, "fig03_ndsm.png")


# ------------------------------------------------------------------ figura 4
def seta_norte(ax, x=0.95, y=0.865, tam=0.045):
    """Seta do norte DENTRO do mapa (canto superior direito), com o N por cima da seta.
    x, y: centro da seta em fracao do eixo; tam: meio comprimento da seta."""
    ax.annotate("", xy=(x, y + tam), xytext=(x, y - tam), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", color="black", lw=1.4),
                zorder=6)
    ax.text(x, y + tam + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))


def valores_unicos(vals, tol=50.0):
    """Valores ordenados sem repeticoes (junta os que diferem menos de tol)."""
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def regua_escala(ax, comp_m=2000, n_seg=2, y_rel=-0.098, margem_rel=0.0):
    """Regua de escala grafica pequena, FORA do mapa, por baixo do canto inferior direito
    (ao lado da legenda). O comprimento esta em metros do mapa (x em coordenadas do mapa,
    y em fracao da altura do eixo)."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    x0, x1 = ax.get_xlim()
    xd = x1 - (x1 - x0) * margem_rel - comp_m    # alinhada a direita com o mapa
    alt = 0.008                                  # altura da barra (fracao do eixo)
    seg = comp_m / n_seg
    for k in range(n_seg):
        ax.add_patch(Rectangle((xd + k * seg, y_rel), seg, alt, transform=tr, clip_on=False,
                               fc="black" if k % 2 == 0 else "white", ec="black", lw=0.6))
    for k in range(n_seg + 1):
        ax.text(xd + k * seg, y_rel + alt + 0.004, f"{k * seg / 1000:g}", transform=tr,
                ha="center", va="bottom", fontsize=6)
    ax.text(xd + comp_m + seg * 0.12, y_rel + alt / 2, "km", transform=tr,
            ha="left", va="center", fontsize=6)


def figura_4():
    """Grelha das ortofotos (treino / teste) com os edificios ao fundo.
    As coordenadas so aparecem nos limites das ortofotos, sem linhas de grelha extra."""
    from matplotlib.collections import PolyCollection
    fig, ax = plt.subplots(figsize=(9, 7.5))

    # 1) Ortofotos: preenchimento claro (por baixo de tudo)
    ortos, xs, ys = [], [], []
    with arcpy.da.SearchCursor(GRELHA, [CAMPO_NOME, "SHAPE@"]) as sc:
        for nome, g in sc:
            nome = str(nome)
            teste = nome in TIPOS_ORTOS
            for parte in g:
                pts = [(p.X, p.Y) for p in parte if p is not None]
                ax.add_patch(MplPolygon(pts, closed=True, fc="#F2A93B" if teste else "#DCE8F5",
                                        ec="none", alpha=0.55 if teste else 0.6, zorder=1))
            ortos.append((nome, teste, g))
            xs += [g.extent.XMin, g.extent.XMax]; ys += [g.extent.YMin, g.extent.YMax]

    # 2) Edificios ao fundo (todos, num so PolyCollection para ser rapido)
    sr = arcpy.Describe(GRELHA).spatialReference
    verts = []
    with arcpy.da.SearchCursor(EDIFICIOS, ["SHAPE@"], spatial_reference=sr) as sc:
        for (g,) in sc:
            if g is None:
                continue
            for parte in g:
                pts = [(p.X, p.Y) for p in parte if p is not None]
                if len(pts) > 2:
                    verts.append(pts)
    ax.add_collection(PolyCollection(verts, facecolors="#555555", edgecolors="#555555",
                                     linewidths=0.1, zorder=2))
    print(f"   {len(verts)} poligonos de edificios desenhados")

    # 3) Contornos e nomes das ortofotos (por cima dos edificios)
    for nome, teste, g in ortos:
        for parte in g:
            pts = [(p.X, p.Y) for p in parte if p is not None]
            ax.add_patch(MplPolygon(pts, closed=True, fc="none", ec="#B25E00" if teste else "#333",
                                    lw=2.2 if teste else 0.9, zorder=3))
        c = g.centroid
        txt = f"{nome}\n({TIPOS_ORTOS[nome]})" if teste else nome
        t = ax.text(c.X, c.Y, txt, ha="center", va="center", fontsize=8, zorder=4,
                    fontweight="bold" if teste else "normal")
        t.set_bbox(dict(fc="white", ec="none", alpha=0.7, pad=1.2))

    # 4) Coordenadas apenas nos limites das ortofotos; sem linhas de grelha
    ax.set_xlim(min(xs), max(xs)); ax.set_ylim(min(ys), max(ys)); ax.set_aspect("equal")
    ax.set_xticks(valores_unicos(xs)); ax.set_yticks(valores_unicos(ys))
    fmt = lambda v, _: f"{v:,.0f}".replace(",", " ")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    ax.tick_params(labelsize=7, length=3)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    # nos cantos, os rotulos ficam virados para dentro para nao se sobreporem
    xl, yl = ax.get_xticklabels(), ax.get_yticklabels()
    xl[0].set_ha("left"); xl[-1].set_ha("right")
    yl[0].set_va("bottom"); yl[-1].set_va("top")
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_linewidth(0.8)

    seta_norte(ax)
    regua_escala(ax, 2000)
    ax.legend(handles=[Patch(fc="#DCE8F5", ec="#333", label="Ortofoto de treino (13)"),
                       Patch(fc="#F2A93B", ec="#B25E00", lw=2, label="Ortofoto de teste (3)"),
                       Patch(fc="#555555", label="Edifícios")],
              loc="upper left", bbox_to_anchor=(0.0, -0.05), ncol=3, frameon=False, fontsize=9)
    gravar(fig, "fig04_treino_teste.png")


# ------------------------------------------------------------------ figura 5
def figura_5(n_chips=3):
    labs = sorted(glob.glob(os.path.join(CHIPS_NDSM, "labels", "*.tif")))
    if not labs:
        raise FileNotFoundError(f"Sem rotulos em {CHIPS_NDSM}\\labels")
    escolhidos = []
    for f in labs[::max(len(labs) // 400, 1)]:          # amostra de ~400 chips
        a = arcpy.RasterToNumPyArray(f)
        a = a[0] if a.ndim == 3 else a
        frac = np.mean(a == 1)
        if 0.10 <= frac <= 0.30:
            escolhidos.append(f)
    if len(escolhidos) < n_chips:
        raise RuntimeError("Poucos chips com 10-30 % de edificios na amostra")
    sel = [escolhidos[int(k)] for k in np.linspace(0, len(escolhidos) - 1, n_chips)]
    fig, axs = plt.subplots(n_chips, 3, figsize=(10, 3.4 * n_chips), squeeze=False)
    for i, f in enumerate(sel):
        img = arcpy.RasterToNumPyArray(os.path.join(CHIPS_NDSM, "images", os.path.basename(f)))
        lab = arcpy.RasterToNumPyArray(f)
        lab = lab[0] if lab.ndim == 3 else lab
        axs[i][0].imshow(rgb(img))
        axs[i][1].imshow(img[BANDA_NDSM - 1].astype(float) * NDSM_MAX_M / 255, cmap="gray",
                         vmin=0, vmax=NDSM_MAX_M)
        axs[i][2].imshow(lab == 1, cmap="gray", vmin=0, vmax=1)
        axs[i][0].set_ylabel(os.path.splitext(os.path.basename(f))[0], fontsize=8)
        for a in axs[i]:
            limpar(a)
    for j, t in enumerate(("Imagem (RGB)", f"nDSM (0–{NDSM_MAX_M} m)", "Rótulo (branco = edifício)")):
        axs[0][j].set_title(t)
    fig.tight_layout()
    gravar(fig, "fig05_chips.png")


# ------------------------------------------------------------------ figura 7
def figura_7():
    mod = modelo_pasta(AVAL_B, GRUPO_B, MODELO_B)
    so = ns(ORTO_URBANA)
    oB = orto_tif(AVAL_B, GRUPO_B, ORTO_URBANA)
    cl = os.path.join(AVAL_B, "classificacoes", GRUPO_B, mod, f"class_{so}.tif")
    gdb = os.path.join(AVAL_B, "poligonos.gdb")
    base = ns(f"{GRUPO_B}_{mod}_{ORTO_URBANA}")
    limpo, reg = os.path.join(gdb, base + "_limpo"), os.path.join(gdb, base + "_reg")
    g = arcpy.Raster(oB)
    cs = g.meanCellWidth
    n = int(LADO_FIG7 / cs)
    if RECORTES.get("fig7"):
        x, y = RECORTES["fig7"][:2]
    else:
        d = ortos_dir(AVAL_B, GRUPO_B)
        canto = arcpy.Point(g.extent.XMin, g.extent.YMin)
        ref = arcpy.RasterToNumPyArray(os.path.join(d, f"ref_edicao_{so}.tif"), canto,
                                       g.width, g.height, nodata_to_value=0) == 1
        msk = arcpy.RasterToNumPyArray(os.path.join(d, f"mascara_{so}.tif"), canto,
                                       g.width, g.height, nodata_to_value=0) == 1
        x, y = centro_janela(oB, melhor_janela(ref.astype(np.float32), msk, n), n)
        print(f"   recorte automatico (figura 7): centro {x:.1f} {y:.1f}")
    xmin, ymin = x - LADO_FIG7 / 2, y - LADO_FIG7 / 2
    ymax = ymin + LADO_FIG7
    img = janela(oB, xmin, ymin, n)
    cla = janela(cl, xmin, ymin, n) == 1
    sr = arcpy.Describe(limpo).spatialReference
    ext = extensao(xmin, ymin, LADO_FIG7, sr)
    g_lim, g_reg = geometrias(limpo, ext), geometrias(reg, ext)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.8))
    base_img = rgb(img)
    over = base_img.copy()
    over[cla] = 0.45 * over[cla] + 0.55 * np.array([1.0, 0.1, 0.6])
    axs[0].imshow(over)
    axs[1].imshow(base_img); desenhar(axs[1], g_lim, xmin, ymax, cs, C_LIMPO, lw=1.3)
    axs[2].imshow(base_img); desenhar(axs[2], g_reg, xmin, ymax, cs, C_PRED, lw=1.5)
    for a, t in zip(axs, ("(a) Classificação em bruto", "(b) Polígonos após a limpeza",
                          "(c) Polígonos regularizados")):
        a.set_xlim(-0.5, n - 0.5); a.set_ylim(n - 0.5, -0.5)
        barra(a, cs, n, 20); limpar(a); a.set_title(t)
    fig.tight_layout()
    gravar(fig, "fig07_pos_processamento.png")


# ------------------------------------------------------------------ figura 13
def procurar_casos():
    """Um exemplo de fusao, de omissao de um edificio pequeno e de falso positivo."""
    mod = modelo_pasta(AVAL_B, GRUPO_B, MODELO_B)
    gdb = os.path.join(AVAL_B, "poligonos.gdb")
    sr = arcpy.Describe(EDIFICIOS).spatialReference
    fusao, omissao, falso = [], [], []
    for orto in TIPOS_ORTOS:
        zona = [gz for nm, gz in arcpy.da.SearchCursor(ZONA_TESTE, [CAMPO_NOME, "SHAPE@"]) if nm == orto][0]
        zona_int = zona.buffer(-LADO_ERRO)               # para a janela caber na orto
        refs = [g for g in geometrias(EDIFICIOS, zona.extent) if zona.contains(g)]
        preds = geometrias(os.path.join(gdb, ns(f"{GRUPO_B}_{mod}_{orto}") + "_reg"))
        for p in preds:
            if not zona_int.contains(p.centroid):
                continue
            viz = [r for r in refs if not p.extent.disjoint(r.extent)]
            cobertos = [r for r in viz if not p.disjoint(r) and p.intersect(r, 4).area >= 0.5 * r.area]
            if len(cobertos) >= 2:
                fusao.append((len(cobertos), p.area, orto, p.centroid))
            if not any(not p.disjoint(r) for r in viz) and p.area >= 30:
                falso.append((p.area, orto, p.centroid))
        for r in refs:
            if 12 <= r.area < 25 and zona_int.contains(r.centroid):
                perto = [p for p in preds if p.extent.XMin < r.extent.XMax + 3 and p.extent.XMax > r.extent.XMin - 3
                         and p.extent.YMin < r.extent.YMax + 3 and p.extent.YMax > r.extent.YMin - 3]
                if all(p.distanceTo(r) > 2 for p in perto):
                    omissao.append((r.area, orto, r.centroid))
    print(f"   candidatos: {len(fusao)} fusoes, {len(omissao)} omissoes, {len(falso)} falsos positivos")
    casos = []
    if fusao:
        fusao.sort(key=lambda t: (-min(t[0], 3), t[1]))  # 2-3 edificios, poligono pequeno
        _, _, o, c = fusao[0]
        casos.append(("Fusão de edifícios contíguos", o, c.X, c.Y))
    if omissao:
        omissao.sort(key=lambda t: -t[0])
        _, o, c = omissao[len(omissao) // 2]
        casos.append(("Omissão de um edifício pequeno", o, c.X, c.Y))
    if falso:
        falso.sort(key=lambda t: -t[0])
        _, o, c = falso[len(falso) // 4]
        casos.append(("Falso positivo", o, c.X, c.Y))
    for t, o, x, y in casos:
        print(f"   {t}: {o}, centro {x:.1f} {y:.1f}")
    return casos


def figura_13():
    casos = CASOS_ERROS or procurar_casos()
    mod = modelo_pasta(AVAL_B, GRUPO_B, MODELO_B)
    gdb = os.path.join(AVAL_B, "poligonos.gdb")
    sr = arcpy.Describe(EDIFICIOS).spatialReference
    fig, axs = plt.subplots(len(casos), 3, figsize=(13, 4.4 * len(casos)), squeeze=False)
    for i, (titulo, orto, x, y) in enumerate(casos):
        so = ns(orto)
        oB = orto_tif(AVAL_B, GRUPO_B, orto)
        d = ortos_dir(AVAL_B, GRUPO_B)
        cs = arcpy.Raster(oB).meanCellWidth
        n = int(LADO_ERRO / cs)
        xmin, ymin = x - LADO_ERRO / 2, y - LADO_ERRO / 2
        ymax = ymin + LADO_ERRO
        img = janela(oB, xmin, ymin, n)
        ref = janela(os.path.join(d, f"ref_edicao_{so}.tif"), xmin, ymin, n) == 1
        pred = janela(os.path.join(AVAL_B, "classificacoes", GRUPO_B, mod, f"poligonos_{so}.tif"),
                      xmin, ymin, n) == 1
        ext = extensao(xmin, ymin, LADO_ERRO, sr)
        g_ref = geometrias(EDIFICIOS, ext)
        g_pred = geometrias(os.path.join(gdb, ns(f"{GRUPO_B}_{mod}_{orto}") + "_reg"), ext)
        a0, a1, a2 = axs[i]
        a0.imshow(rgb(img))
        desenhar(a0, g_ref, xmin, ymax, cs, C_REF, lw=1.6)
        desenhar(a0, g_pred, xmin, ymax, cs, C_PRED, lw=1.6, ls="--")
        a1.imshow(mapa_erros(img, ref, pred))
        a2.imshow(img[BANDA_NDSM - 1].astype(float) * NDSM_MAX_M / 255, cmap="gray", vmin=0, vmax=NDSM_MAX_M)
        desenhar(a2, g_ref, xmin, ymax, cs, C_REF, lw=1.2)
        for a in axs[i]:
            a.set_xlim(-0.5, n - 0.5); a.set_ylim(n - 0.5, -0.5)
            barra(a, cs, n, 10); limpar(a)
        a0.set_ylabel(f"{titulo}\n({orto})", fontsize=10)
    for j, t in enumerate(("Ortofotografia e contornos", "Mapa de erros", f"nDSM (0–{NDSM_MAX_M} m)")):
        axs[0][j].set_title(t)
    fig.legend(handles=[Line2D([0], [0], color=C_REF, lw=2, label="Referência"),
                        Line2D([0], [0], color=C_PRED, lw=2, ls="--", label="Previsão (regularizada)"),
                        Patch(color=C_TP, label="Correto"), Patch(color=C_FP, label="Falso positivo"),
                        Patch(color=C_FN, label="Omitido")],
               loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    gravar(fig, "fig13_erros_tipicos.png")


# ------------------------------------------------------------------ principal
def localizar_avaliacao():
    """Se AVAL_B nao tiver a avaliacao da fase B, procura-a em D:\\DL\\Avaliacao."""
    global AVAL_B
    alvo = os.path.join("ortos_teste", GRUPO_B, f"orto_{ns(ORTO_URBANA)}.tif")
    if os.path.exists(os.path.join(AVAL_B, alvo)):
        return
    raiz = os.path.dirname(AVAL_B)
    print(f"A avaliacao da fase B nao esta em {AVAL_B}; a procurar em {raiz} ...")
    for d, subs, _ in os.walk(raiz):
        if os.path.exists(os.path.join(d, alvo)) and os.path.isdir(os.path.join(d, "classificacoes", GRUPO_B)):
            AVAL_B = d
            print(f"   encontrada: {AVAL_B}")
            return
        subs[:] = [x for x in subs if x not in ("classificacoes", "poligonos.gdb", "teste_ndsm_zero")]
    print("   nao encontrada: indica a pasta certa em AVAL_B")


def main():
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    localizar_avaliacao()
    todas = {"2": figuras_2_3, "4": figura_4, "5": figura_5, "7": figura_7, "13": figura_13}
    # sem argumentos faz todas; com argumentos so as indicadas (ex.: python 11_figuras_2.py 4)
    pedidas = sys.argv[1:] or list(todas)
    for n in pedidas:
        if n not in todas:
            print(f"Figura {n} nao existe neste script (opcoes: {', '.join(todas)})")
            continue
        print("Figuras 2 e 3" if n == "2" else f"Figura {n}")
        try:
            todas[n]()
        except Exception as e:                           # uma figura que falha nao para as outras
            print(f"   ERRO: {e}")
    print("Concluido.")


if __name__ == "__main__":
    main()