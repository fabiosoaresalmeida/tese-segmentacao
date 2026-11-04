# -*- coding: utf-8 -*-
r"""
10_figuras.py  (ArcGIS Pro 3.6)

Gera as figuras de comparacao visual a partir dos resultados da avaliacao
(ortos de teste, referencias rasterizadas, classificacoes e poligonos regularizados),
para a fase A (sem nDSM) e a fase B (com nDSM), cada uma na sua pasta de avaliacao.

Figura 1 - comparacao de modelos (uma figura por recorte):
   uma linha por modelo, com duas colunas:
     (a) ortofotografia com os contornos da referencia (amarelo) e dos poligonos
         regularizados previstos (vermelho);
     (b) mapa de erros dos poligonos regularizados sobre a ortofotografia em tons
         de cinzento: verde = correto (TP), vermelho = falso positivo (FP),
         azul = omitido (FN).

Figura 2 - fase A vs fase B (uma figura por modelo, com todos os recortes):
   uma linha por recorte, com quatro colunas:
     ortofotografia com a referencia | nDSM | mapa de erros fase A | mapa de erros fase B

Os recortes podem ser indicados por coordenadas (centro e lado, em metros) ou
escolhidos automaticamente (a janela com mais area edificada na referencia).

Correr no Python Command Prompt do ArcGIS Pro:
    python "D:\Users\Cap_Almeida\Projeto\Python\22_figuras.py"
"""

import os
import numpy as np
import arcpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURACAO
# ============================================================
# Pasta de avaliacao de cada grupo (as do 20_avaliar_modelos.py / 09_avaliar_modelos.py)
PASTAS_GRUPOS = {
    "faseA_4bandas": r"D:\DL\Avaliacao\126_original",        # RGB + NIR
    "faseB_ndsm":    r"D:\DL\Avaliacao\126_original_ndsm",   # RGB + NIR + nDSM
}
PASTA_FIGURAS = r"D:\DL\Avaliacao\126_figuras_faseA_faseB"
REFERENCIA    = "edicao"                                    # nome da referencia na avaliacao

# Recortes: (titulo, orto, x do centro, y do centro, lado em metros)
# Com x e y = None, o recorte e escolhido automaticamente (janela com mais area edificada).
# Os recortes sao escolhidos UMA vez e usados em todas as figuras, para serem comparaveis.
# As coordenadas sao no sistema do ortomosaico (podem ser lidas no ArcGIS Pro).
RECORTES = [
    ("Zona urbana",    "126-I-3",  None, None, 150),
    ("Zona mista",     "126-II-3", None, None, 150),
    ("Zona dispersa",  "126-IV-1", None, None, 150),
]

# Figura 1 (comparacao de modelos), para cada grupo e cada recorte:
#   - uma figura por ARQUITETURA, com todos os modelos dessa arquitetura no grupo;
#   - uma figura com os MELHORES modelos de cada arquitetura (inicio do nome da pasta).
FIG1_GRUPOS = ["faseA_4bandas", "faseB_ndsm"]
FIG1_POR_ARQUITETURA = True
FIG1_MELHORES = {                                           # lista vazia = nao gera
    "faseA_4bandas": ["unet_resnet152", "deeplab_resnet101", "samlora_vit_l"],
    "faseB_ndsm":    ["unet_resnet50", "deeplab_resnet101", "samlora_vit_l"],
}
ARQUITETURAS = [("U-Net", "unet"), ("DeepLabV3", "deeplab"), ("SAMLoRA", "samlora")]

# Figura 2 (fase A vs fase B): (grupo A, grupo B, inicio do nome do modelo)
FIG2_PARES = []
# True = gera a figura 2 para TODOS os modelos que existam nos dois grupos de FIG2_GRUPOS
FIG2_TODOS  = True
FIG2_GRUPOS = ("faseA_4bandas", "faseB_ndsm")
BANDA_NDSM = 5            # banda do nDSM no ortomosaico da fase B
NDSM_MAX_M = 40           # altura correspondente ao valor 255 (para a legenda)

DPI = 300
ESCALA_M = 20             # comprimento da barra de escala (m)
NOMES_GRUPOS = {"faseA_4bandas": "Sem nDSM (fase A)", "faseB_ndsm": "Com nDSM (fase B)"}
# ============================================================

COR_REF, COR_PRED = "#FFD400", "#FF2A2A"
COR_TP, COR_FP, COR_FN = (0.15, 0.75, 0.25), (0.90, 0.15, 0.15), (0.15, 0.45, 0.95)


def nome_seguro(txt):
    return txt.replace("-", "_").replace(" ", "_").replace(".", "_")


def nome_modelo(pasta):
    """'unet_resnet152_b8_...' -> 'U-Net ResNet-152'."""
    p = pasta.lower()
    arq = "U-Net" if p.startswith("unet") else "DeepLabV3" if p.startswith("deeplab") else \
          "SAMLoRA" if p.startswith("samlora") else pasta
    for cod, txt in (("resnet152", "ResNet-152"), ("resnet101", "ResNet-101"), ("resnet50", "ResNet-50"),
                     ("resnet34", "ResNet-34"), ("vit_l", "ViT-L"), ("vit_b", "ViT-B"), ("vit_h", "ViT-H")):
        if cod in p:
            return f"{arq} {txt}"
    return arq


ORDEM = ["unet_resnet34", "unet_resnet50", "unet_resnet101", "unet_resnet152",
         "deeplab_resnet34", "deeplab_resnet50", "deeplab_resnet101",
         "samlora_vit_b", "samlora_vit_l", "samlora_vit_h"]


def todos_modelos(grupo):
    """Pastas de modelos com classificacao no grupo, por ordem de arquitetura e codificador."""
    pasta = os.path.join(PASTAS_GRUPOS[grupo], "classificacoes", grupo)
    dirs = sorted(d for d in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, d)))
    chave = lambda d: next((i for i, o in enumerate(ORDEM) if d.lower().startswith(o + "_")), len(ORDEM))
    return sorted(dirs, key=lambda d: (chave(d), d))


def prefixo_de(pasta):
    return next((o for o in ORDEM if pasta.lower().startswith(o + "_")), pasta)


def encontrar_modelo(grupo, prefixo):
    pasta = os.path.join(PASTAS_GRUPOS[grupo], "classificacoes", grupo)
    cands = sorted(d for d in os.listdir(pasta) if d.lower().startswith(prefixo.lower()))
    if not cands:
        raise FileNotFoundError(f"Nenhum modelo '{prefixo}*' em {pasta}")
    return cands[-1]


# ------------------------------------------------------------------ leitura
def caminhos(grupo, orto):
    so = nome_seguro(orto)
    base = os.path.join(PASTAS_GRUPOS[grupo], "ortos_teste", grupo)
    return dict(orto=os.path.join(base, f"orto_{so}.tif"),
                ref=os.path.join(base, f"ref_{REFERENCIA}_{so}.tif"),
                msk=os.path.join(base, f"mascara_{so}.tif"))


def janela(raster, xmin, ymin, n):
    return arcpy.RasterToNumPyArray(raster, arcpy.Point(xmin, ymin), n, n, nodata_to_value=0)


def escolher_janela(grupo, orto, lado):
    """Centro da janela com mais area edificada, dentro da orto de teste."""
    c = caminhos(grupo, orto)
    g = arcpy.Raster(c["orto"])
    cs = g.meanCellWidth
    ref = arcpy.RasterToNumPyArray(c["ref"], arcpy.Point(g.extent.XMin, g.extent.YMin),
                                   g.width, g.height, nodata_to_value=0) == 1
    msk = arcpy.RasterToNumPyArray(c["msk"], arcpy.Point(g.extent.XMin, g.extent.YMin),
                                   g.width, g.height, nodata_to_value=0) == 1
    n = int(lado / cs)
    passo = max(n // 2, 1)
    melhor, pos = -1, (0, 0)
    for i in range(0, ref.shape[0] - n, passo):
        for j in range(0, ref.shape[1] - n, passo):
            if not msk[i:i + n, j:j + n].all():
                continue
            v = ref[i:i + n, j:j + n].sum()
            if v > melhor:
                melhor, pos = v, (i, j)
    i, j = pos
    x = g.extent.XMin + (j + n / 2) * cs
    y = g.extent.YMax - (i + n / 2) * cs
    return x, y


def preparar_recortes(grupo):
    rec = []
    for titulo, orto, x, y, lado in RECORTES:
        if x is None or y is None:
            x, y = escolher_janela(grupo, orto, lado)
            print(f"   recorte automatico '{titulo}' ({orto}): centro {x:.1f} {y:.1f}")
        rec.append((titulo, orto, x, y, lado))
    return rec


def ler_recorte(grupo, orto, x, y, lado, modelo=None):
    c = caminhos(grupo, orto)
    cs = arcpy.Raster(c["orto"]).meanCellWidth
    n = int(lado / cs)
    xmin, ymin = x - lado / 2, y - lado / 2
    img = janela(c["orto"], xmin, ymin, n)
    d = dict(img=img, ref=janela(c["ref"], xmin, ymin, n) == 1, cs=cs)
    if modelo:
        pg = os.path.join(PASTAS_GRUPOS[grupo], "classificacoes", grupo, modelo,
                          f"poligonos_{nome_seguro(orto)}.tif")
        d["pred"] = janela(pg, xmin, ymin, n) == 1
    return d


# ------------------------------------------------------------------ desenho
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
    for m, cor in ((ref & pred, COR_TP), (pred & ~ref, COR_FP), (ref & ~pred, COR_FN)):
        out[m] = 0.35 * base[m] + 0.65 * np.array(cor)
    return out


def contornos(ax, arr, cor, lw=1.2):
    if arr.any():
        ax.contour(arr.astype(float), levels=[0.5], colors=[cor], linewidths=lw)


def barra_escala(ax, cs, n):
    px = ESCALA_M / cs
    x0, y0 = n * 0.05, n * 0.93
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=4, solid_capstyle="butt")
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=2, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - n * 0.03, f"{ESCALA_M} m", color="white", ha="center", va="bottom",
            fontsize=8, fontweight="bold")


def limpar(ax):
    ax.set_xticks([]); ax.set_yticks([])


def legenda(fig, com_contornos=True):
    itens = []
    if com_contornos:
        itens += [Line2D([0], [0], color=COR_REF, lw=2, label="Referência"),
                  Line2D([0], [0], color=COR_PRED, lw=2, label="Previsão (polígonos regularizados)")]
    itens += [Patch(color=COR_TP, label="Correto (TP)"), Patch(color=COR_FP, label="Falso positivo (FP)"),
              Patch(color=COR_FN, label="Omitido (FN)")]
    fig.legend(handles=itens, loc="lower center", ncol=min(len(itens), 3), fontsize=9, frameon=False)


# ------------------------------------------------------------------ figuras
def figura_modelos(grupo, recorte, modelos, titulo_fig, saida):
    """Uma linha por modelo: (a) ortofotografia com contornos, (b) mapa de erros."""
    titulo, orto, x, y, lado = recorte
    fig, axs = plt.subplots(len(modelos), 2, figsize=(9, 4.3 * len(modelos)), squeeze=False)
    for i, mod in enumerate(modelos):
        d = ler_recorte(grupo, orto, x, y, lado, mod)
        n = d["ref"].shape[0]
        a1, a2 = axs[i]
        a1.imshow(rgb(d["img"]))
        contornos(a1, d["ref"], COR_REF)
        contornos(a1, d["pred"], COR_PRED)
        a2.imshow(mapa_erros(d["img"], d["ref"], d["pred"]))
        for a in (a1, a2):
            barra_escala(a, d["cs"], n)
            limpar(a)
        a1.set_ylabel(nome_modelo(mod), fontsize=12, fontweight="bold")
    axs[0][0].set_title("Ortofotografia e contornos", fontsize=12)
    axs[0][1].set_title("Mapa de erros", fontsize=12)
    fig.suptitle(f"{titulo_fig}\n{titulo} ({orto})", fontsize=13)
    legenda(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=1.5)
    fig.savefig(saida, dpi=DPI)
    plt.close(fig)
    print(f"   {saida}")


def figura1(recs):
    for grupo in FIG1_GRUPOS:
        todos = todos_modelos(grupo)
        ng = NOMES_GRUPOS.get(grupo, grupo)
        for recorte in recs:
            tag = f"{grupo}_{nome_seguro(recorte[0])}"
            if FIG1_POR_ARQUITETURA:
                for nome_arq, pref in ARQUITETURAS:
                    mods = [m for m in todos if m.lower().startswith(pref)]
                    if mods:
                        figura_modelos(grupo, recorte, mods, f"{nome_arq} - {ng}",
                                       os.path.join(PASTA_FIGURAS, f"fig_{pref}_{tag}.png"))
            melhores = FIG1_MELHORES.get(grupo, [])
            if melhores:
                mods = [encontrar_modelo(grupo, p) for p in melhores]
                figura_modelos(grupo, recorte, mods, f"Melhor modelo de cada arquitetura - {ng}",
                               os.path.join(PASTA_FIGURAS, f"fig_melhores_{tag}.png"))


def figura2(recs, grupo_a, grupo_b, prefixo):
    mod_a, mod_b = encontrar_modelo(grupo_a, prefixo), encontrar_modelo(grupo_b, prefixo)
    fig, axs = plt.subplots(len(recs), 4, figsize=(16, 4 * len(recs)), squeeze=False)
    for i, (titulo, orto, x, y, lado) in enumerate(recs):
        da = ler_recorte(grupo_a, orto, x, y, lado, mod_a)
        db = ler_recorte(grupo_b, orto, x, y, lado, mod_b)
        n = da["ref"].shape[0]
        ax = axs[i][0]
        ax.imshow(rgb(da["img"]))
        contornos(ax, da["ref"], COR_REF)
        ax.set_ylabel(f"{titulo}\n({orto})", fontsize=11)
        ax = axs[i][1]
        im = ax.imshow(db["img"][BANDA_NDSM - 1], cmap="viridis", vmin=0, vmax=255)
        contornos(ax, da["ref"], COR_REF)
        axs[i][2].imshow(mapa_erros(da["img"], da["ref"], da["pred"]))
        axs[i][3].imshow(mapa_erros(db["img"], db["ref"], db["pred"]))
        for a in axs[i]:
            barra_escala(a, da["cs"], n)
            limpar(a)
    for j, t in enumerate(("Ortofotografia e referência", f"nDSM (0–{NDSM_MAX_M} m)",
                           "Erros - sem nDSM (fase A)", "Erros - com nDSM (fase B)")):
        axs[0][j].set_title(t, fontsize=11)
    fig.suptitle(f"{nome_modelo(mod_a)}: efeito do nDSM", fontsize=13)
    legenda(fig, com_contornos=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    saida = os.path.join(PASTA_FIGURAS, f"fig_faseA_faseB_{nome_seguro(prefixo)}.png")
    fig.savefig(saida, dpi=DPI)
    plt.close(fig)
    print(f"   {saida}")


def main():
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    for g, p in PASTAS_GRUPOS.items():
        if not os.path.isdir(os.path.join(p, "classificacoes", g)):
            raise FileNotFoundError(f"Nao encontrei as classificacoes do grupo {g} em {p}")
    # recortes escolhidos uma vez (na grelha do primeiro grupo) e usados em todas as figuras
    recs = preparar_recortes(next(iter(PASTAS_GRUPOS)))
    print("Figura 1 - comparacao de modelos")
    figura1(recs)
    pares = list(FIG2_PARES)
    if FIG2_TODOS:
        ga, gb = FIG2_GRUPOS
        comuns = sorted(set(map(prefixo_de, todos_modelos(ga))) & set(map(prefixo_de, todos_modelos(gb))),
                        key=lambda p: ORDEM.index(p) if p in ORDEM else len(ORDEM))
        pares += [(ga, gb, p) for p in comuns if (ga, gb, p) not in pares]
    for ga, gb, pref in pares:
        print(f"Figura 2 - fase A vs fase B ({pref})")
        figura2(recs, ga, gb, pref)
    print("Concluido.")


if __name__ == "__main__":
    main()