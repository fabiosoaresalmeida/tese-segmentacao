# -*- coding: utf-8 -*-
r"""
12_figuras_sem_arcpy.py

Gera as figuras da tese que NAO precisam do ArcGIS (so matplotlib):

  Figura 1  - fluxo de trabalho (esquema das etapas e dos scripts)
  Figura 6  - perda de validacao por epoca (lida dos log_treinos.txt das fases A e B)
  Figura 8  - criterio de detecao por objeto (detetado, deslocado, fusao)
  Figura 9  - areas VP / FP / FN e formulas das metricas ao pixel
  Figura 12 - detecao e cobertura por classe de area (lidas dos metricas_por_area.csv)

As figuras 1, 8 e 9 sao esquemas e nao leem dados. A 6 e a 12 leem os ficheiros
indicados na CONFIGURACAO; se algum faltar, essa figura e saltada e as outras continuam.

Corre em qualquer Python com matplotlib (incluindo o do ArcGIS Pro):
    python 12_figuras_sem_arcpy.py                 (grava em PASTA_FIGURAS)
    python 12_figuras_sem_arcpy.py D:\outra\pasta  (grava na pasta indicada)
"""
import os
import re
import csv
import sys
import matplotlib
matplotlib.use("Agg")                       # sem janela: so grava os PNG
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURACAO
# ============================================================
PASTA_FIGURAS = r"D:\DL\Avaliacao\126_Construcao_figuras\figuras_2"

# Figura 6: registos de treino do 08_treinar_modelos.py (fase A) e do
# 08_treinar_modelos_nDSM.py (fase B), e nomes das pastas dos modelos a mostrar
LOG_FASE_A = r"D:\DL\Modelos\126_construcoes\log_treinos.txt"
LOG_FASE_B = r"D:\DL\Modelos\126_construcoes_ndsm\log_treinos.txt"
CURVAS = [  # (legenda, fase, nome do modelo no log, cor, tipo de linha)
    ("U-Net ResNet-50 – fase A",      "A", "unet_resnet50_b8_20261022_112403",     "#1F5FA8", "--"),
    ("U-Net ResNet-50 – fase B",      "B", "unet_resnet50_b8_20261030_000624",     "#1F5FA8", "-"),
    ("DeepLabV3 ResNet-101 – fase A", "A", "deeplab_resnet101_b8_20261021_231023", "#C8102E", "--"),
    ("DeepLabV3 ResNet-101 – fase B", "B", "deeplab_resnet101_b8_20261029_175624", "#C8102E", "-"),
]
N_EPOCAS = 20

# Figura 12: metricas por classe de area do 09_avaliar_modelos.py
AREA_FASE_A = r"D:\DL\Avaliacao\126_Construcao\metricas_por_area.csv"
AREA_FASE_B = r"D:\DL\Avaliacao\126_Construcao_ndsm\metricas_por_area.csv"
MODELO_AREA = "unet_resnet152"           # inicio do nome do modelo
REFERENCIA = "edicao"

DPI = 300
# ============================================================

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
# Cores comuns a todas as figuras da tese
C_REF, C_PRED = "#E8A400", "#C8102E"                    # referencia / previsao
C_TP, C_FP, C_FN = "#2E9E4A", "#D63A3A", "#2F6FD6"      # correto / falso positivo / omitido


def gravar(fig, nome, **kw):
    """Grava a figura em PASTA_FIGURAS e fecha-a."""
    saida = os.path.join(PASTA_FIGURAS, nome)
    fig.savefig(saida, dpi=DPI, **kw)
    plt.close(fig)
    print(f"   {saida}")


# ------------------------------------------------------------------ Figura 1
def fig1():
    """Esquema do fluxo de trabalho: dados de entrada a esquerda, etapas (scripts) a direita."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11); ax.set_ylim(-0.1, 6.2); ax.axis("off")

    def caixa(x, y, w, h, txt, cor, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=cor, ec="#444", lw=1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def seta(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2))

    # Coluna dos dados de entrada (texto, y da caixa)
    dados = [("Ortofotografia\n4 bandas, 0,5 m", 5.0), ("Edifícios\n(edição cartográfica)", 3.6),
             ("MDS / MDT LiDAR\nDGT 2024", 2.2), ("Edifícios\n(fotogrametria)", 0.8)]
    ax.text(1.05, 5.95, "Dados", ha="center", fontweight="bold")
    for t, y in dados:
        caixa(0.1, y, 1.9, 0.9, t, "#EEF2F7")

    # Etapas (texto, x, y). Linha de cima: preparacao e treino; a seguir desce pela direita
    # (classificacao) e volta para a esquerda em baixo (avaliacao e figuras).
    et = [("1. Limpeza e projeção\ndos vetores\n(03)", 2.5, 3.6),
          ("2. Separação\ntreino / teste\n(06)", 4.7, 3.6),
          ("3. Exportação\ndos chips\n256 px (07)", 6.9, 3.6),
          ("4. Treino dos modelos\nU-Net, DeepLabV3,\nSAMLoRA (08)", 9.1, 3.6),
          ("nDSM =\nmáx(0, MDS − MDT)\n0–40 m → 8 bits; 5.ª banda", 2.5, 2.0),
          ("5. Classificação e\nregularização\n(09)", 9.1, 2.0),
          ("6. Avaliação ao píxel\ne por objeto\n(09)", 6.9, 0.4),
          ("7. Figuras e\ncomparação\n(10, 11)", 4.7, 0.4)]
    ax.text(6.1, 5.95, "Processamento", ha="center", fontweight="bold")
    for t, x, y in et:
        ndsm = "nDSM" in t                               # a caixa do nDSM so existe na fase B
        caixa(x, y, 1.8, 1.05, t, "#FFF4DD" if ndsm else "#E6F2E6", fs=7.6 if ndsm else 8.5)

    # Ligacoes entre caixas
    seta(2.0, 4.05, 2.5, 4.1)                            # edificios -> limpeza
    seta(2.0, 2.65, 2.5, 2.55)                           # LiDAR -> nDSM
    seta(4.3, 4.1, 4.7, 4.1); seta(6.5, 4.1, 6.9, 4.1); seta(8.7, 4.1, 9.1, 4.1)
    seta(10.0, 3.6, 10.0, 3.05)                          # treino -> classificacao
    seta(4.3, 2.5, 6.9, 3.65)                            # nDSM -> chips (so fase B)
    ax.text(5.7, 2.75, "fase B", fontsize=8, color="#8A6100", rotation=24)
    seta(9.1, 2.3, 8.7, 1.1); seta(6.9, 0.92, 6.5, 0.92)
    # Fotogrametria -> avaliacao: linha por baixo das caixas, para nao as cruzar
    ax.plot([1.05, 1.05, 7.8], [0.8, 0.12, 0.12], color="#444", lw=1.2)
    seta(7.8, 0.12, 7.8, 0.4)
    ax.text(4.4, 0.18, "referência complementar", fontsize=8, color="#555", ha="center")
    # Ortofotografia -> chips: linha por cima das caixas
    ax.plot([2.0, 7.8], [5.45, 5.45], color="#444", lw=1.2)
    seta(7.8, 5.45, 7.8, 4.65)
    ax.text(4.9, 5.52, "imagem", fontsize=8, color="#555", ha="center")

    ax.legend(handles=[Patch(fc="#EEF2F7", ec="#444", label="Dados de entrada"),
                       Patch(fc="#E6F2E6", ec="#444", label="Etapa (script)"),
                       Patch(fc="#FFF4DD", ec="#444", label="Só na fase B")],
              loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 1.02))
    gravar(fig, "fig01_fluxo_trabalho.png", bbox_inches="tight")


# ------------------------------------------------------------------ Figura 6
def ler_perdas(log, modelo):
    """Perda de validacao por epoca de um modelo, lida do log_treinos.txt.

    No log, as linhas de cada epoca do treino (tabela do fastai) aparecem antes da linha
    'REGISTADO <modelo>'. Procura-se o ultimo 'Start Time' antes dessa linha (inicio do
    treino desse modelo) e le-se a tabela: epoca, perda de treino, perda de validacao, ...
    """
    txt = open(log, encoding="utf-8", errors="ignore").read()
    i = txt.find(f"REGISTADO {modelo}")
    if i < 0:
        raise ValueError(f"'{modelo}' nao encontrado em {log}")
    bloco = txt[:i]
    j = bloco.rfind("Start Time")
    vals = {}
    for m in re.finditer(r"\]\s+(\d+)\s+([\d.e-]+)\s+([\d.e-]+)\s+[\d.]+\s+[\d.]+\s+\d", bloco[j:]):
        vals[int(m.group(1))] = float(m.group(3))       # grupo 3 = perda de validacao
    return [vals[k] for k in sorted(vals)]


def fig6():
    """Perda de validacao por epoca (escala logaritmica), fase A a tracejado e fase B a cheio."""
    logs = {"A": LOG_FASE_A, "B": LOG_FASE_B}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for legenda, fase, modelo, cor, ls in CURVAS:
        v = ler_perdas(logs[fase], modelo)
        if len(v) != N_EPOCAS:
            print(f"   aviso: {legenda} tem {len(v)} epocas no log (esperadas {N_EPOCAS})")
        ax.plot(range(1, len(v) + 1), v, ls, color=cor, marker="o", ms=3.5, lw=1.6, label=legenda)
    ax.set_yscale("log")                                 # as perdas variam em ordens de grandeza
    ax.set_xlabel("Época"); ax.set_ylabel("Perda de validação (escala log.)")
    ax.set_xticks(range(1, N_EPOCAS + 1)); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    gravar(fig, "fig06_perda_validacao.png")


# ------------------------------------------------------------------ Figura 8
def fig8():
    """Tres casos do criterio de detecao por objeto (IoU >= 0,5, correspondencia 1:1).

    Retangulos (x, y, largura, altura): referencia a cheio, previsao a tracejado.
    Caso 1: IoU = 25,2 / 36 = 0,70 -> detetado. Caso 2: IoU = 8 / 24 = 0,33 -> falso positivo
    e omissao. Caso 3: um so poligono cobre dois edificios -> fusao.
    """
    fig, axs = plt.subplots(1, 3, figsize=(11, 4))
    casos = [("Detetado (IoU = 0,70)", [(1, 1, 6, 6)], [(1.6, 1.4, 6, 5.8)]),
             ("Deslocado: falso positivo\ne omissão (IoU = 0,33)", [(1, 1, 4, 4)], [(3, 1, 4, 4)]),
             ("Fusão: um polígono\ncobre dois edifícios", [(1, 1.5, 3.2, 4.5), (4.6, 1.5, 3.2, 4.5)],
              [(0.8, 1.3, 7.2, 4.9)])]
    for ax, (titulo, refs, preds) in zip(axs, casos):
        for x, y, w, h in refs:
            ax.add_patch(Rectangle((x, y), w, h, fc="#F5D88A", ec=C_REF, lw=2.2, alpha=.85))
        for x, y, w, h in preds:
            ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=C_PRED, lw=2.2, ls="--"))
        ax.set_xlim(0, 9); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(titulo, fontsize=10)
    fig.legend(handles=[Patch(fc="#F5D88A", ec=C_REF, label="Edifício de referência"),
                        Line2D([0], [0], color=C_PRED, ls="--", lw=2, label="Polígono previsto")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    gravar(fig, "fig08_criterio_deteccao.png", bbox_inches="tight")


# ------------------------------------------------------------------ Figura 9
def fig9():
    """Referencia e previsao deslocadas: as areas VP, FP e FN e as formulas ao pixel."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 20); ax.set_ylim(0, 9.6); ax.axis("off")
    rx, ry, rw, rh = 1.5, 1.5, 6, 6          # referencia
    px, py = 3.3, 0.6                        # previsao, com o mesmo tamanho mas deslocada
    # Pinta-se a referencia a azul (FN), a previsao a vermelho (FP) e por cima a intersecao a verde (VP)
    ax.add_patch(Rectangle((rx, ry), rw, rh, fc=C_FN, alpha=.75, lw=0))
    ax.add_patch(Rectangle((px, py), rw, rh, fc=C_FP, alpha=.75, lw=0))
    ix, iy = max(rx, px), max(ry, py)
    ax.add_patch(Rectangle((ix, iy), rx + rw - ix, py + rh - iy, fc=C_TP, lw=0))
    # Contornos: referencia a cheio, previsao a tracejado
    ax.add_patch(Rectangle((rx, ry), rw, rh, fc="none", ec="black", lw=2))
    ax.add_patch(Rectangle((px, py), rw, rh, fc="none", ec="black", lw=1.5, ls="--"))
    ax.text(5.4, 4.5, "VP", ha="center", va="center", fontsize=16, color="white", fontweight="bold")
    ax.text(2.4, 6.5, "FN", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax.text(8.4, 1.3, "FP", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax.text(rx, ry + rh + .3, "Referência", fontsize=9)
    ax.text(px + rw - 1.6, py - .5, "Previsão", fontsize=9)
    # Formulas (mathtext do matplotlib) a direita
    formulas = [r"$UA = \dfrac{VP}{VP+FP}$", r"$PA = \dfrac{VP}{VP+FN}$",
                r"$F1 = \dfrac{2\,VP}{2\,VP+FP+FN}$", r"$IoU = \dfrac{VP}{VP+FP+FN}$"]
    for k, t in enumerate(formulas):
        ax.text(11.5, 8.0 - k * 2.0, t, fontsize=15, va="center")
    ax.legend(handles=[Patch(color=C_TP, label="Verdadeiro positivo (VP)"),
                       Patch(color=C_FP, label="Falso positivo (FP)"),
                       Patch(color=C_FN, label="Falso negativo (FN)")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8.5, frameon=False)
    gravar(fig, "fig09_areas_metricas.png", bbox_inches="tight")


# ------------------------------------------------------------------ Figura 12
def ler_area(f, modelo_pref):
    """{classe de area: (taxa de detecao, taxa de cobertura, n.o de edificios)} do TOTAL das ortos."""
    d = {}
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["referencia"] == REFERENCIA and r["orto"] == "TOTAL" and r["modelo"].startswith(modelo_pref):
                d[r["classe_area_m2"]] = (float(r["taxa_detecao"]), float(r["taxa_cobertura"]), int(r["n_ref"]))
    if not d:
        raise ValueError(f"Sem linhas de '{modelo_pref}' / {REFERENCIA} / TOTAL em {f}")
    return d


def fig12():
    """Barras: % de edificios detetados (IoU >= 0,5) e cobertos (>= 50 %) por classe de area, fases A e B."""
    a, b = ler_area(AREA_FASE_A, MODELO_AREA), ler_area(AREA_FASE_B, MODELO_AREA)
    cl = ["0-25", "25-50", "50-100", "100-200", "200-inf"]          # classes no CSV
    rot = ["< 25", "25–50", "50–100", "100–200", "> 200"]           # rotulos no grafico
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w, x = .2, range(len(cl))
    # (dados, indice: 0 = detecao, 1 = cobertura, legenda, cor, padrao)
    series = [(a, 0, "Detetados – fase A", "#9DB9DE", None), (b, 0, "Detetados – fase B", "#1F5FA8", None),
              (a, 1, "Cobertos – fase A", "#F2C29B", "//"), (b, 1, "Cobertos – fase B", "#D96B1C", "//")]
    for k, (d, i, legenda, cor, padrao) in enumerate(series):
        vals = [d[c][i] * 100 for c in cl]
        barras = ax.bar([xi + (k - 1.5) * w for xi in x], vals, w, label=legenda, color=cor, hatch=padrao, ec="white")
        for bb, v in zip(barras, vals):                              # valor por cima de cada barra
            ax.text(bb.get_x() + w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}\n(n = {a[c][2]})" for r, c in zip(rot, cl)])  # n = edificios de referencia
    ax.set_xlabel("Área do edifício (m²)"); ax.set_ylabel("Edifícios (%)"); ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    gravar(fig, "fig12_deteccao_por_area.png")


# ------------------------------------------------------------------ principal
def main():
    global PASTA_FIGURAS
    if len(sys.argv) > 1:                                # pasta de saida opcional na linha de comandos
        PASTA_FIGURAS = sys.argv[1]
    os.makedirs(PASTA_FIGURAS, exist_ok=True)
    for f in (fig1, fig6, fig8, fig9, fig12):
        print(f.__doc__.strip().splitlines()[0])
        try:
            f()
        except Exception as e:                           # uma figura que falha nao para as outras
            print(f"   ERRO: {e}")
    print("Concluido.")


if __name__ == "__main__":
    main()