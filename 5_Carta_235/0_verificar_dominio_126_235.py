# -*- coding: utf-8 -*-
"""
0_verificar_dominio_126_235.py

Verificações antes de decidir calibrar as ortos da 235:

  1. RADIOMETRIA - histogramas e estatísticas por banda (R, G, B e NIR) das ortos da 126
     e da 235, e distâncias entre histogramas.
     Critério: compara a diferença 235 vs 126 com a VARIAÇÃO INTERNA da 126
     (cada orto da 126 vs restantes ortos da 126). Se a 235 cair dentro dessa
     variação, a diferença radiométrica é da mesma ordem da que o modelo já
     viu no treino.

  2. PAISAGEM - composição da paisagem em cada folha a partir da cartografia
     de referência (área por classe nos polígonos, comprimento nas linhas,
     contagem nos pontos), em % da área da folha.

Corre no ambiente arcgispro-py3 (arcpy + numpy + matplotlib). Não precisa de
torch nem de skimage.

Saídas (PASTA_SAIDA):
  radiometria_por_orto.csv        estatísticas por orto e banda
  radiometria_por_folha.csv       estatísticas por folha (mosaico) e banda
  radiometria_distancias.csv      distâncias entre histogramas (todas as comparações)
  histogramas_126_235.png         histogramas por banda
  composicao_por_folha.csv        área/comprimento/contagem por classe e folha
  composicao_comparacao.csv       126 vs 235 lado a lado
  resumo.txt                      resumo e leitura dos resultados
"""

import os
import csv
import glob
import math

import numpy as np
import arcpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURAÇÃO - editar
# =============================================================================

# Ortos de cada folha (padrão glob). Pode ser uma lista de padrões.
ORTOS = {
    "126": [r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"],
    "235": [r"D:\Users\Cap_Almeida\Projeto\235\Ortomosaico\235.tif"],
}

# Bandas a analisar (1 = primeira banda do raster) e respetivos nomes
BANDAS = [1, 2, 3]                    # a 235 so tem RGB (sem NIR)
NOMES_BANDAS = ["R", "G", "B"]

# Valor de NoData: pixels com TODAS as bandas iguais a este valor são ignorados
# (os NoData definidos no raster também são convertidos para este valor)
NODATA = 0

# Leitura por blocos (pixels) e subamostragem (1 = todos os pixels;
# 2 = um em cada 2 linhas/colunas, ~4x mais rápido, estatísticas quase iguais)
BLOCO = 4096
PASSO = 1

# Cartografia de referência por folha.
#   tema   : nome do tema (construcoes, rede_viaria, hidrografia, vegetacao, ...)
#   caminho: shapefile / feature class
#   campo  : campo com a classe ("source"); None -> usa o nome do tema
# Se a mesma shapefile cobrir as duas folhas, repetir o caminho em ambas.
CAMADAS = {
    "126": [
        {"tema": "construcoes", "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_constr.shp", "campo": "source"},
        {"tema": "rede_viaria", "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\l_vias.shp", "campo": "source"},
        {"tema": "hidrografia", "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_hidro.shp", "campo": "source"},
        {"tema": "vegetacao",   "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_vegetacao.shp", "campo": "source"},
    ],
    "235": [
        {"tema": "construcoes", "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\235_2020_5\a_constr.shp", "campo": "source"},
        {"tema": "rede_viaria", "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\235_2020_5\l_vias.shp", "campo": "source"},
        {"tema": "hidrografia", "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\235_2020_5\a_hidro.shp", "campo": "source"},
        {"tema": "vegetacao",   "caminho": r"D:\Users\Cap_Almeida\Projeto\M888_SHP\235_2020_5\a_vegetacao.shp", "campo": "source"},
    ],
}

# Transformação de datum quando a camada está noutro sistema (ex.: Datum Lisboa
# -> WGS84 Militares IGeoE). None = a primeira da lista do ArcGIS (a que o mapa
# assume por defeito).
TRANSFORMACAO = None

PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\235\Verificacoes_dominio"

# Quando uma folha tem um só raster (mosaico), a variação interna é calculada
# dividindo-o numa grelha GRELHA x GRELHA de blocos (4 -> 16 blocos, à escala
# das 16 ortos que compõem a folha). Blocos com menos de MIN_VALIDOS da média
# de pixels válidos (ex.: bordos com NoData) são ignorados.
GRELHA = 4
MIN_VALIDOS = 0.25

# CSV para Excel em português: separador ";" e vírgula decimal
SEPARADOR = ";"
VIRGULA_DECIMAL = True

# Folhas a comparar: (referência, alvo)
REF, ALVO = "126", "235"


# =============================================================================
# Utilitários
# =============================================================================

def fmt(v, casas=4):
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return ""
        s = "{:.{c}f}".format(float(v), c=casas)
        return s.replace(".", ",") if VIRGULA_DECIMAL else s
    return str(v)


def escrever_csv(caminho, cabecalho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=SEPARADOR)
        w.writerow(cabecalho)
        for l in linhas:
            w.writerow([fmt(v) for v in l])
    print("  -> " + caminho)


def listar_ortos(padroes):
    ficheiros = []
    for p in padroes:
        ficheiros.extend(glob.glob(p))
    ficheiros = sorted(set(ficheiros))
    if not ficheiros:
        raise RuntimeError("Nenhuma orto encontrada em: {}".format(padroes))
    return ficheiros


def nome_orto(caminho):
    return os.path.splitext(os.path.basename(caminho))[0]


# =============================================================================
# 1. RADIOMETRIA
# =============================================================================

def histogramas_orto(caminho):
    """Histogramas (contagens) por banda de uma orto, ignorando NoData.
    Devolve (lista de arrays np.int64 por banda,
             dict {(linha, coluna) da grelha: lista de arrays por banda})."""
    r = arcpy.Raster(caminho)
    nb = r.bandCount
    for b in BANDAS:
        if b > nb:
            raise RuntimeError("{} tem {} bandas; pedida a banda {}".format(caminho, nb, b))

    cw, ch = r.meanCellWidth, r.meanCellHeight
    largura, altura = r.width, r.height
    xmin, ymin = r.extent.XMin, r.extent.YMin

    hists = [np.zeros(256, dtype=np.int64) for _ in BANDAS]
    grelha = {}

    # Blocos de leitura não maiores que uma célula da grelha
    bloco = max(256, min(BLOCO, largura // GRELHA, altura // GRELHA))

    # Leitura por blocos (padrão da documentação do RasterToNumPyArray)
    for x in range(0, largura, bloco):
        for y in range(0, altura, bloco):
            canto = arcpy.Point(xmin + x * cw, ymin + y * ch)
            nc = min(x + bloco, largura) - x
            nr = min(y + bloco, altura) - y
            # célula da grelha pelo centro do bloco (y conta a partir de baixo)
            gc = min(GRELHA - 1, int((x + nc / 2.0) / largura * GRELHA))
            gl = min(GRELHA - 1, int((1 - (y + nr / 2.0) / altura) * GRELHA))
            g = grelha.setdefault((gl + 1, gc + 1), [np.zeros(256, dtype=np.int64) for _ in BANDAS])
            arr = arcpy.RasterToNumPyArray(caminho, canto, nc, nr, nodata_to_value=NODATA)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if PASSO > 1:
                arr = arr[:, ::PASSO, ::PASSO]

            sel = arr[[b - 1 for b in BANDAS]]
            validos = ~np.all(sel == NODATA, axis=0)
            if not validos.any():
                continue

            for i in range(len(BANDAS)):
                v = sel[i][validos]
                v = np.clip(v, 0, None).astype(np.int64)
                h = np.bincount(v.ravel())
                hists[i] = igualar(hists[i], h)[0]  # rasters com mais de 8 bits
                hists[i][:h.size] += h
                g[i] = igualar(g[i], h)[0]
                g[i][:h.size] += h
    return hists, grelha


def igualar(a, b):
    """Iguala o comprimento de dois histogramas."""
    n = max(a.size, b.size)
    if a.size < n:
        a = np.pad(a, (0, n - a.size))
    if b.size < n:
        b = np.pad(b, (0, n - b.size))
    return a, b


def estatisticas(h):
    n = h.sum()
    if n == 0:
        return dict(n=0, media=np.nan, desvio=np.nan, p2=np.nan, p50=np.nan, p98=np.nan)
    v = np.arange(h.size, dtype=np.float64)
    media = (v * h).sum() / n
    desvio = math.sqrt(((v - media) ** 2 * h).sum() / n)
    cdf = np.cumsum(h) / n

    def pct(q):
        return int(np.searchsorted(cdf, q))

    return dict(n=int(n), media=media, desvio=desvio,
                p2=pct(0.02), p50=pct(0.50), p98=pct(0.98))


def distancias(ha, hb):
    """Distâncias entre dois histogramas (contagens)."""
    ha, hb = igualar(ha, hb)
    pa = ha / ha.sum()
    pb = hb / hb.sum()
    ea, eb = estatisticas(ha), estatisticas(hb)
    emd = np.abs(np.cumsum(pa) - np.cumsum(pb)).sum()  # Wasserstein-1, em ND
    inter = np.minimum(pa, pb).sum()                     # 1 = iguais
    bc = np.sqrt(pa * pb).sum()
    bhatt = -math.log(bc) if bc > 0 else np.inf          # 0 = iguais
    return dict(
        dif_media=ea["media"] - eb["media"],
        razao_desvio=ea["desvio"] / eb["desvio"] if eb["desvio"] else np.nan,
        emd=emd,
        intersecao=inter,
        bhattacharyya=bhatt,
    )


def somar(lista):
    tot = [np.zeros(1, dtype=np.int64) for _ in BANDAS]
    for hs in lista:
        for i in range(len(BANDAS)):
            a, b = igualar(tot[i], hs[i])
            tot[i] = a + b
    return tot


def subtrair(total, parte):
    res = []
    for i in range(len(BANDAS)):
        a, b = igualar(total[i], parte[i])
        res.append(a - b)
    return res


def analise_radiometrica():
    print("\n=== 1. Radiometria ===")
    hist_orto = {}   # folha -> {orto ou bloco: [hist por banda]}
    hist_folha = {}
    for folha, padroes in ORTOS.items():
        hist_orto[folha] = {}
        ficheiros = listar_ortos(padroes)
        grelhas = {}
        for c in ficheiros:
            print("  a ler {} ({})".format(nome_orto(c), folha))
            hs, gr = histogramas_orto(c)
            hist_orto[folha][nome_orto(c)] = hs
            grelhas[nome_orto(c)] = gr
        hist_folha[folha] = somar(list(hist_orto[folha].values()))

        # Um só raster: usar os blocos da grelha como unidades de comparação
        if len(ficheiros) == 1:
            nome = nome_orto(ficheiros[0])
            gr = grelhas[nome]
            media = np.mean([b[0].sum() for b in gr.values()])
            hist_orto[folha] = {
                "{}_bloco_L{}C{}".format(nome, l, c): hs
                for (l, c), hs in sorted(gr.items())
                if hs[0].sum() >= MIN_VALIDOS * media
            }
            print("  {}: um só raster -> {} blocos de uma grelha {}x{} (ignorados {})".format(
                folha, len(hist_orto[folha]), GRELHA, GRELHA, len(gr) - len(hist_orto[folha])))

    # --- estatísticas por orto e por folha
    cab = ["folha", "orto", "banda", "n_pixels", "media", "desvio", "p2", "p50", "p98"]
    linhas = []
    for folha, d in hist_orto.items():
        for orto, hs in d.items():
            for i, nb in enumerate(NOMES_BANDAS):
                e = estatisticas(hs[i])
                linhas.append([folha, orto, nb, e["n"], e["media"], e["desvio"], e["p2"], e["p50"], e["p98"]])
    escrever_csv(os.path.join(PASTA_SAIDA, "radiometria_por_orto.csv"), cab, linhas)

    linhas = []
    for folha, hs in hist_folha.items():
        for i, nb in enumerate(NOMES_BANDAS):
            e = estatisticas(hs[i])
            linhas.append([folha, "MOSAICO", nb, e["n"], e["media"], e["desvio"], e["p2"], e["p50"], e["p98"]])
    escrever_csv(os.path.join(PASTA_SAIDA, "radiometria_por_folha.csv"), cab, linhas)

    # --- distâncias
    # tipo:
    #   interna_REF  : cada orto da REF vs restantes ortos da REF (variação natural)
    #   interna_ALVO : cada orto do ALVO vs restantes ortos do ALVO
    #   orto_ALVO_vs_REF : cada orto do ALVO vs mosaico da REF
    #   folha_ALVO_vs_REF: mosaico do ALVO vs mosaico da REF
    comps = []
    for folha in (REF, ALVO):
        tot = hist_folha[folha]
        for orto, hs in hist_orto[folha].items():
            resto = subtrair(tot, hs)
            if all(r.sum() == 0 for r in resto):
                continue  # folha com uma só orto
            comps.append(("interna_" + folha, orto, "restantes_" + folha, hs, resto))
    for orto, hs in hist_orto[ALVO].items():
        comps.append(("orto_{}_vs_{}".format(ALVO, REF), orto, "mosaico_" + REF, hs, hist_folha[REF]))
    comps.append(("folha_{}_vs_{}".format(ALVO, REF), "mosaico_" + ALVO, "mosaico_" + REF,
                  hist_folha[ALVO], hist_folha[REF]))

    cab = ["tipo", "a", "b", "banda", "dif_media", "razao_desvio", "emd", "intersecao", "bhattacharyya"]
    linhas, registos = [], []
    for tipo, a, b, ha, hb in comps:
        for i, nb in enumerate(NOMES_BANDAS):
            d = distancias(ha[i], hb[i])
            linhas.append([tipo, a, b, nb, d["dif_media"], d["razao_desvio"], d["emd"], d["intersecao"], d["bhattacharyya"]])
            registos.append((tipo, a, nb, d))
    escrever_csv(os.path.join(PASTA_SAIDA, "radiometria_distancias.csv"), cab, linhas)

    grafico_histogramas(hist_orto, hist_folha)
    return registos, hist_folha


def grafico_histogramas(hist_orto, hist_folha):
    cores = {REF: "#1f77b4", ALVO: "#d62728"}
    fig, eixos = plt.subplots(1, len(BANDAS), figsize=(5 * len(BANDAS), 4), sharey=True)
    if len(BANDAS) == 1:
        eixos = [eixos]
    for i, (ax, nb) in enumerate(zip(eixos, NOMES_BANDAS)):
        for folha in (REF, ALVO):
            for hs in hist_orto[folha].values():
                h = hs[i]
                if h.sum():
                    ax.plot(h / h.sum(), color=cores[folha], alpha=0.15, lw=0.7)
            h = hist_folha[folha][i]
            ax.plot(h / h.sum(), color=cores[folha], lw=2, label="folha {} (mosaico)".format(folha))
        ax.set_title("Banda " + nb)
        ax.set_xlabel("Nível digital")
        ax.set_xlim(0, 255)
    eixos[0].set_ylabel("Frequência relativa")
    eixos[0].legend()
    fig.suptitle("Histogramas das ortos (linhas finas = ortos individuais)")
    fig.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, "histogramas_{}_{}.png".format(REF, ALVO))
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print("  -> " + caminho)


# =============================================================================
# 2. COMPOSIÇÃO DA PAISAGEM
# =============================================================================

def limite_folha(padroes):
    """Limite da folha = união das extensões das ortos."""
    geom = None
    sr = None
    for c in listar_ortos(padroes):
        r = arcpy.Raster(c)
        sr = r.spatialReference
        p = r.extent.polygon
        p = arcpy.Polygon(p.getPart(0), sr)
        geom = p if geom is None else geom.union(p)
    return geom, sr


def mesmo_sr(a, b):
    return a.exportToString() == b.exportToString()


def projetar(geom, sr_orig, sr_dest):
    if mesmo_sr(sr_orig, sr_dest):
        return geom
    trans = TRANSFORMACAO
    if trans is None:
        lista = arcpy.ListTransformations(sr_orig, sr_dest, geom.extent)
        trans = lista[0] if lista else None
    if trans:
        print("    transformação: " + trans)
        return geom.projectAs(sr_dest, trans)
    return geom.projectAs(sr_dest)


def cursor(caminho, campos, filtro):
    try:
        return arcpy.da.SearchCursor(caminho, campos, spatial_filter=filtro,
                                     spatial_relationship="INTERSECTS")
    except TypeError:  # versões sem spatial_filter
        return arcpy.da.SearchCursor(caminho, campos)


def composicao_folha(folha):
    limite, sr_orto = limite_folha(ORTOS[folha])
    resultados = []  # (tema, classe, geometria, valor, unidade, pct_ou_densidade)
    for cam in CAMADAS.get(folha, []):
        caminho, tema, campo = cam["caminho"], cam["tema"], cam.get("campo")
        if not arcpy.Exists(caminho):
            print("  AVISO: não existe " + caminho)
            continue
        desc = arcpy.Describe(caminho)
        sr = desc.spatialReference
        tipo = desc.shapeType
        mpu = sr.metersPerUnit if sr.type == "Projected" else 1.0
        if sr.type != "Projected":
            print("  AVISO: {} não está num sistema projetado; áreas sem significado".format(caminho))
        print("  {} / {} ({})".format(folha, tema, tipo))

        lim = projetar(limite, sr_orto, sr)
        area_folha_m2 = lim.area * mpu ** 2

        campos = ["SHAPE@"] + ([campo] if campo else [])
        acum = {}
        with cursor(caminho, campos, lim) as cur:
            for row in cur:
                g = row[0]
                if g is None:
                    continue
                classe = row[1] if campo else tema
                if tipo == "Polygon":
                    v = g.intersect(lim, 4).area * mpu ** 2
                elif tipo == "Polyline":
                    v = g.intersect(lim, 2).length * mpu
                else:
                    v = 1 if not lim.disjoint(g) else 0
                acum[classe] = acum.get(classe, 0) + v

        for classe, v in sorted(acum.items(), key=lambda kv: str(kv[0])):
            if tipo == "Polygon":
                resultados.append((tema, classe, "poligono", v / 1e6, "km2",
                                   100.0 * v / area_folha_m2, "% area folha"))
            elif tipo == "Polyline":
                resultados.append((tema, classe, "linha", v / 1e3, "km",
                                   (v / 1e3) / (area_folha_m2 / 1e6), "km/km2"))
            else:
                resultados.append((tema, classe, "ponto", v, "n",
                                   v / (area_folha_m2 / 1e6), "n/km2"))
    area_km2 = limite.area * (sr_orto.metersPerUnit ** 2) / 1e6
    return resultados, area_km2


def analise_paisagem():
    print("\n=== 2. Composição da paisagem ===")
    comp, areas = {}, {}
    for folha in (REF, ALVO):
        comp[folha], areas[folha] = composicao_folha(folha)

    cab = ["folha", "area_folha_km2", "tema", "classe", "geometria", "valor", "unidade", "relativo", "unidade_relativo"]
    linhas = []
    for folha in (REF, ALVO):
        for t, c, g, v, u, rel, ur in comp[folha]:
            linhas.append([folha, areas[folha], t, c, g, v, u, rel, ur])
    escrever_csv(os.path.join(PASTA_SAIDA, "composicao_por_folha.csv"), cab, linhas)

    # lado a lado
    chave = lambda r: (r[0], str(r[1]), r[2])
    a = {chave(r): r for r in comp[REF]}
    b = {chave(r): r for r in comp[ALVO]}
    cab = ["tema", "classe", "geometria", "unidade_relativo",
           "relativo_" + REF, "relativo_" + ALVO, "diferenca_{}_menos_{}".format(ALVO, REF)]
    linhas = []
    for k in sorted(set(a) | set(b)):
        ra, rb = a.get(k), b.get(k)
        va = ra[5] if ra else 0.0
        vb = rb[5] if rb else 0.0
        ur = (ra or rb)[6]
        linhas.append([k[0], k[1], k[2], ur, va, vb, vb - va])
    escrever_csv(os.path.join(PASTA_SAIDA, "composicao_comparacao.csv"), cab, linhas)
    return linhas


# =============================================================================
# RESUMO
# =============================================================================

def resumo(registos, comp_linhas):
    L = []
    L.append("RESUMO - verificações {} (referência) vs {} (alvo)".format(REF, ALVO))
    L.append("=" * 70)
    L.append("")
    L.append("1. RADIOMETRIA")
    L.append("EMD = distância entre histogramas em níveis digitais (0 = iguais).")
    L.append("Critério: a diferença {a} vs {r} está dentro da variação interna da {r}?".format(a=ALVO, r=REF))
    L.append("")
    for nb in NOMES_BANDAS:
        interna = [d["emd"] for t, a, b, d in registos if t == "interna_" + REF and b == nb]
        folha = [d for t, a, b, d in registos if t.startswith("folha_") and b == nb]
        ortos = [d["emd"] for t, a, b, d in registos if t.startswith("orto_") and b == nb]
        if not interna or not folha:
            continue
        lo, hi = min(interna), max(interna)
        f = folha[0]
        dentro = sum(1 for e in ortos if e <= hi)
        L.append("Banda {}:".format(nb))
        L.append("  variação interna {}: EMD {:.1f} a {:.1f} (mediana {:.1f})".format(
            REF, lo, hi, float(np.median(interna))))
        L.append("  mosaico {} vs {}: EMD {:.1f} | dif. média {:+.1f} ND | razão desvio {:.2f} -> {}".format(
            ALVO, REF, f["emd"], f["dif_media"], f["razao_desvio"],
            "DENTRO da variação interna" if f["emd"] <= hi else "FORA da variação interna"))
        L.append("  ortos/blocos {} dentro da variação interna da {}: {}/{}".format(ALVO, REF, dentro, len(ortos)))
    interna_alvo = [d["emd"] for t, a, b, d in registos if t == "interna_" + ALVO]
    if interna_alvo:
        L.append("")
        L.append("Variação interna da {} (todas as bandas): EMD máx {:.1f}. Se for muito maior".format(
            ALVO, max(interna_alvo)))
        L.append("que a da {}, convém uniformizar as ortos da {} entre si antes de outra coisa.".format(REF, ALVO))

    L.append("")
    L.append("2. COMPOSIÇÃO DA PAISAGEM (maiores diferenças)")
    L.append("Polígonos em % da área da folha; linhas em km/km2; pontos em n/km2.")
    ordenadas = sorted(comp_linhas, key=lambda r: abs(r[6]), reverse=True)[:10]
    for t, c, g, ur, va, vb, dif in ordenadas:
        L.append("  {:<12} {:<30} {:>8.2f} -> {:>8.2f}  ({:+.2f} {})".format(t, str(c)[:30], va, vb, dif, ur))

    L.append("")
    L.append("LEITURA")
    L.append("- Radiometria dentro da variação interna: calibrar deve ter pouco efeito.")
    L.append("- Radiometria fora + paisagem semelhante: histogram matching é defensável.")
    L.append("- Radiometria fora + paisagem diferente: preferir PIF (alvos pseudo-invariantes).")
    L.append("- Resultados piores na {} com radiometria semelhante: o problema não é radiométrico.".format(ALVO))

    texto = "\n".join(L)
    caminho = os.path.join(PASTA_SAIDA, "resumo.txt")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    print("\n" + texto)
    print("\n  -> " + caminho)


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    registos, _ = analise_radiometrica()
    comp_linhas = analise_paisagem()
    resumo(registos, comp_linhas)


if __name__ == "__main__":
    main()