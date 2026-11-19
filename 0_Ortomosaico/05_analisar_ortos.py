# -*- coding: utf-8 -*-
"""
05_analisar_ortos.py  (ArcGIS Pro 3.6)

Análise das ortofotos da folha 126 (série M888) e sugestão das áreas de teste.

1) Para cada ortofoto conta os edifícios (atribuídos pelo centróide),
   a área edificada e a densidade de edifícios.
2) Divide as ortofotos em 3 estratos de densidade (disperso, misto, denso),
   pelos tercis da densidade, e sugere 3 ortofotos para TESTE - uma de cada
   estrato - que, em conjunto, tenham entre PROP_MIN e PROP_MAX dos edifícios.

Critério de escolha da combinação de teste (por esta ordem):
  (1) proporção de edifícios dentro do intervalo PROP_MIN-PROP_MAX;
  (2) pontuação = afastamento à proporção-alvo (PROP_ALVO)
                  + representatividade (densidade próxima da mediana do estrato);
  (3) preferência por ortofotos na margem da carta, segundo MODO_MARGEM:
        "penalizacao" - cada ortofoto interior soma PESO_MARGEM à pontuação.
                        A margem pesa na escolha, mesmo sem empate, mas uma
                        combinação claramente melhor continua a prevalecer.
                        Indicado quando se quer favorecer ativamente áreas de
                        teste na periferia, com menos vizinhas de treino.
        "desempate"   - a margem só é usada para desempatar combinações com a
                        mesma pontuação (arredondada a CASAS_EMPATE casas
                        decimais). Indicado quando a proporção e a
                        representatividade devem ser os únicos critérios.

Saídas:
  <SAIDA_GRELHA>                      shapefile com os limites das ortofotos e estatísticas
  <SAIDA_GRELHA>_relatorio.txt        relatório completo
  <SAIDA_GRELHA>_estatisticas.csv     tabela (abre no Excel, separador ;)

Atribuição dos edifícios: cada edifício é atribuído à ortofoto que contém o
seu centróide. Na folha 126 as 16 ortofotos são contíguas e sem sobreposição;
um centróide só pode cair em mais de uma ortofoto se estiver exatamente sobre
um limite comum. Nesse caso (e em folhas com sobreposição) é atribuído à
ortofoto cujo centro está mais próximo, para que cada edifício conte uma só vez.
"""
import os
import csv
import math
import time
import datetime
import itertools
import statistics
import traceback
import arcpy

# =============================== PARÂMETROS ===============================
# --- dados de entrada e saída (mesma raiz que os scripts 01 e 02)
EDIFICIOS = r"D:\Users\Cap Almeida\Projeto\126\126_Serta\SHP\a_constr_fil3.shp"

PASTA_ORTOS = r"D:\Users\Cap Almeida\Projeto\M888_Ortos\126_Ortofotos"   # pasta com as 16 ortofotos
EXTENSOES = (".tif", ".tiff", ".ecw", ".jp2", ".img")
GRELHA_ORTOS = None        # alternativa: shapefile com os limites das ortofotos
CAMPO_NOME = "NOME"        # campo com o nome de cada ortofoto (só com GRELHA_ORTOS)

SAIDA_GRELHA = r"D:\Users\Cap Almeida\Projeto\126\126_Serta\Orto\ortos_grelha.shp"

# --- proporção de edifícios no conjunto de teste
PROP_ALVO = 0.15           # proporção pretendida
PROP_MIN, PROP_MAX = 0.10, 0.20   # intervalo aceite

# --- preferência por ortofotos na margem da carta (ver cabeçalho)
MODO_MARGEM = "penalizacao"   # "penalizacao" ou "desempate"
PESO_MARGEM = 0.1             # só "penalizacao": soma por cada ortofoto interior
CASAS_EMPATE = 3              # só "desempate": casas decimais para considerar empate

# --- outros
MIN_EDIF_ORTO = 100        # ortofotos com menos edifícios não são sugeridas
AREA_PEQUENO = 30.0        # m2; para a estatística "% edifícios pequenos"
N_COMBINACOES = 5          # n.º de combinações alternativas no relatório
ADICIONAR_AO_MAPA = True
# ==========================================================================

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False
TOL_MARGEM = 1.0           # m; tolerância para considerar uma ortofoto na margem da carta
MODOS_MARGEM = ("penalizacao", "desempate")


class Relatorio:
    def __init__(self):
        self.linhas = []

    def __call__(self, txt=""):
        print(txt)
        self.linhas.append(str(txt))

    def gravar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas) + "\n")


def mapa_atual():
    try:
        return arcpy.mp.ArcGISProject("CURRENT").activeMap
    except Exception:
        return None


def _norm(c):
    return os.path.normcase(os.path.splitext(os.path.abspath(c))[0])


def remover_camadas(mapa, caminhos):
    if mapa is None:
        return
    alvo = {_norm(c) for c in caminhos}
    for lyr in mapa.listLayers():
        try:
            if lyr.supports("DATASOURCE"):
                ds = lyr.dataSource
                if ds.lower().startswith("memory") or _norm(ds) in alvo:
                    mapa.removeLayer(lyr)
        except Exception:
            pass


def apagar_se_existe(caminho):
    if arcpy.Exists(caminho):
        arcpy.management.Delete(caminho)


def retangulo(ext, sr):
    return arcpy.Polygon(arcpy.Array([
        arcpy.Point(ext.XMin, ext.YMin), arcpy.Point(ext.XMin, ext.YMax),
        arcpy.Point(ext.XMax, ext.YMax), arcpy.Point(ext.XMax, ext.YMin),
        arcpy.Point(ext.XMin, ext.YMin)]), sr)


def ler_ortos(sr, rel):
    ortos = []
    if GRELHA_ORTOS:
        with arcpy.da.SearchCursor(GRELHA_ORTOS, [CAMPO_NOME, "SHAPE@"],
                                   spatial_reference=sr) as sc:
            for nome, g in sc:
                if g is not None:
                    ortos.append({"nome": str(nome), "geom": g, "retangulo": False,
                                  "origem": GRELHA_ORTOS})
        return ortos

    ficheiros = sorted(f for f in os.listdir(PASTA_ORTOS) if f.lower().endswith(EXTENSOES))
    for f in ficheiros:
        caminho = os.path.join(PASTA_ORTOS, f)
        d = arcpy.Describe(caminho)
        g = retangulo(d.extent, d.spatialReference)
        if d.spatialReference.name != sr.name:
            rel(f"AVISO: {f} está em {d.spatialReference.name}; extensão reprojetada.")
            g = g.projectAs(sr)
            ret = False
        else:
            ret = True
        ortos.append({"nome": os.path.splitext(f)[0], "geom": g, "retangulo": ret,
                      "origem": caminho})
    return ortos


def texto_criterio_margem():
    """Descrição do critério da margem, tal como é aplicado, para o relatório."""
    if MODO_MARGEM == "penalizacao":
        return (f"(4) preferência pela margem da carta: cada ortofoto interior soma "
                f"{PESO_MARGEM:g} à pontuação (modo 'penalizacao').")
    return (f"(4) desempate: entre combinações com a mesma pontuação (a {CASAS_EMPATE} "
            "casas decimais), prefere-se a que tem mais ortofotos na margem da carta "
            "(modo 'desempate').")


def main(rel):
    t0 = time.time()
    if MODO_MARGEM not in MODOS_MARGEM:
        raise ValueError(f"MODO_MARGEM inválido: '{MODO_MARGEM}'. "
                         f"Usa um de: {', '.join(MODOS_MARGEM)}.")
    desc = arcpy.Describe(EDIFICIOS)
    sr = desc.spatialReference
    arcpy.env.outputCoordinateSystem = sr
    base = os.path.splitext(SAIDA_GRELHA)[0]
    saida_txt = base + "_relatorio.txt"
    saida_csv = base + "_estatisticas.csv"

    mapa = mapa_atual()
    remover_camadas(mapa, [SAIDA_GRELHA])
    apagar_se_existe(SAIDA_GRELHA)

    rel("=" * 78)
    rel("RELATÓRIO - ANÁLISE DOS ORTOFOTOS E SUGESTÃO DE ÁREAS DE TESTE")
    rel("=" * 78)
    rel(f"Data/hora:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rel(f"ArcGIS Pro:  {arcpy.GetInstallInfo().get('Version', '?')}")
    rel()
    rel("PARÂMETROS")
    nomes_param = ["EDIFICIOS", "PASTA_ORTOS", "GRELHA_ORTOS", "SAIDA_GRELHA", "PROP_ALVO",
                   "PROP_MIN", "PROP_MAX", "MODO_MARGEM"]
    nomes_param.append("PESO_MARGEM" if MODO_MARGEM == "penalizacao" else "CASAS_EMPATE")
    nomes_param += ["MIN_EDIF_ORTO", "AREA_PEQUENO"]
    for nome in nomes_param:
        rel(f"  {nome:<14} = {globals()[nome]}")
    rel(f"  Sistema de coordenadas: {sr.name}")
    rel()

    # --- ortofotos
    ortos = ler_ortos(sr, rel)
    if len(ortos) < 3:
        raise ValueError(f"Encontradas só {len(ortos)} ortofotos - verifica PASTA_ORTOS.")
    for o in ortos:
        e = o["geom"].extent
        o.update(bbox=(e.XMin, e.YMin, e.XMax, e.YMax),
                 cx=(e.XMin + e.XMax) / 2, cy=(e.YMin + e.YMax) / 2,
                 area=o["geom"].area, n=0, area_ed=0.0, areas=[], sugerido=0)
    gx0 = min(o["bbox"][0] for o in ortos)
    gy0 = min(o["bbox"][1] for o in ortos)
    gx1 = max(o["bbox"][2] for o in ortos)
    gy1 = max(o["bbox"][3] for o in ortos)
    for o in ortos:
        x0, y0, x1, y1 = o["bbox"]
        o["margem"] = int(abs(x0 - gx0) <= TOL_MARGEM or abs(y0 - gy0) <= TOL_MARGEM or
                          abs(x1 - gx1) <= TOL_MARGEM or abs(y1 - gy1) <= TOL_MARGEM)
    rel(f"Ortofotos encontradas: {len(ortos)}")
    rel()

    # --- atribuição dos edifícios pelo centróide
    print("A atribuir edifícios aos ortofotos...")
    n_total = n_fora = n_sobrep = 0
    with arcpy.da.SearchCursor(EDIFICIOS, ["SHAPE@"]) as sc:
        for (g,) in sc:
            if g is None:
                continue
            n_total += 1
            c = g.centroid
            cand = []
            for o in ortos:
                x0, y0, x1, y1 = o["bbox"]
                if x0 <= c.X <= x1 and y0 <= c.Y <= y1:
                    if o["retangulo"] or o["geom"].contains(arcpy.PointGeometry(c, sr)):
                        cand.append(o)
            if not cand:
                n_fora += 1
                continue
            if len(cand) > 1:
                n_sobrep += 1
            o = min(cand, key=lambda k: (k["cx"] - c.X) ** 2 + (k["cy"] - c.Y) ** 2)
            a = g.area
            o["n"] += 1
            o["area_ed"] += a
            o["areas"].append(a)

    n_atrib = sum(o["n"] for o in ortos)
    for o in ortos:
        km2 = o["area"] / 1e6
        o["km2"] = km2
        o["dens"] = o["n"] / km2 if km2 > 0 else 0.0
        o["pct_ed"] = o["area_ed"] / o["area"] if o["area"] > 0 else 0.0
        o["pct_tot"] = o["n"] / n_atrib if n_atrib else 0.0
        o["mediana"] = statistics.median(o["areas"]) if o["areas"] else 0.0
        o["pct_peq"] = (sum(1 for a in o["areas"] if a < AREA_PEQUENO) / o["n"]
                        if o["n"] else 0.0)

    # --- estratos de densidade
    validos = sorted([o for o in ortos if o["n"] >= MIN_EDIF_ORTO], key=lambda o: o["dens"])
    for o in ortos:
        o["estrato"] = "insuficiente"
    if len(validos) < 3:
        raise ValueError("Menos de 3 ortofotos com edifícios suficientes (MIN_EDIF_ORTO).")
    nomes_estr = ["disperso", "misto", "denso"]
    estratos = {nm: [] for nm in nomes_estr}
    k = len(validos)
    for i, o in enumerate(validos):
        nm = nomes_estr[min(2, i * 3 // k)]
        o["estrato"] = nm
        estratos[nm].append(o)
    med = {nm: statistics.median([o["dens"] for o in L]) for nm, L in estratos.items()}

    # --- combinações (uma ortofoto de cada estrato)
    # pontuação base: afastamento à proporção-alvo (normalizado pela largura do
    # intervalo aceite) + representatividade (desvio logarítmico médio da
    # densidade de cada ortofoto em relação à mediana do seu estrato; 0 = ideal)
    combos = []
    for trio in itertools.product(estratos["disperso"], estratos["misto"], estratos["denso"]):
        prop = sum(o["n"] for o in trio) / n_atrib
        rep = statistics.mean(abs(math.log(o["dens"] / med[o["estrato"]])) for o in trio)
        interiores = sum(1 for o in trio if not o["margem"])
        pontuacao = abs(prop - PROP_ALVO) / (PROP_MAX - PROP_MIN) + rep
        if MODO_MARGEM == "penalizacao":
            pontuacao += PESO_MARGEM * interiores   # a margem entra na pontuação
        dentro = PROP_MIN <= prop <= PROP_MAX
        combos.append((not dentro, pontuacao, prop, rep, interiores, trio))

    # ordenação: primeiro as combinações dentro do intervalo, depois a pontuação;
    # no modo "desempate", o n.º de ortofotos interiores só decide entre
    # combinações com a mesma pontuação arredondada
    if MODO_MARGEM == "penalizacao":
        combos.sort(key=lambda x: (x[0], x[1]))
    else:
        combos.sort(key=lambda x: (x[0], round(x[1], CASAS_EMPATE), x[4]))
    sugerido = combos[0][5]
    for o in sugerido:
        o["sugerido"] = 1

    # --- shapefile da grelha
    pasta, nome = os.path.split(SAIDA_GRELHA)
    arcpy.management.CreateFeatureclass(pasta, nome, "POLYGON", spatial_reference=sr)
    campos = [("NOME", "TEXT", 50), ("N_EDIF", "LONG", None), ("AREA_KM2", "DOUBLE", None),
              ("AREA_ED", "DOUBLE", None), ("DENS_KM2", "DOUBLE", None),
              ("PCT_ED", "DOUBLE", None), ("PCT_TOT", "DOUBLE", None),
              ("MED_M2", "DOUBLE", None), ("PCT_PEQ", "DOUBLE", None),
              ("ESTRATO", "TEXT", 12), ("MARGEM", "SHORT", None), ("SUGERIDO", "SHORT", None)]
    for nm, tp, ln in campos:
        if ln:
            arcpy.management.AddField(SAIDA_GRELHA, nm, tp, field_length=ln)
        else:
            arcpy.management.AddField(SAIDA_GRELHA, nm, tp)
    with arcpy.da.InsertCursor(SAIDA_GRELHA, ["SHAPE@"] + [c[0] for c in campos]) as ic:
        for o in ortos:
            ic.insertRow([o["geom"], o["nome"][:50], o["n"], round(o["km2"], 4),
                          round(o["area_ed"], 1), round(o["dens"], 1),
                          round(100 * o["pct_ed"], 2), round(100 * o["pct_tot"], 2),
                          round(o["mediana"], 1), round(100 * o["pct_peq"], 1),
                          o["estrato"], o["margem"], o["sugerido"]])

    # --- CSV
    with open(saida_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["ortofoto", "n_edificios", "area_km2", "area_edificada_m2",
                    "densidade_edif_km2", "pct_area_edificada", "pct_dos_edificios",
                    "mediana_area_m2", f"pct_edif_menores_{AREA_PEQUENO:g}m2",
                    "estrato", "margem_carta", "sugerido_teste"])
        for o in sorted(ortos, key=lambda o: o["nome"]):
            w.writerow([o["nome"], o["n"], f"{o['km2']:.3f}".replace(".", ","),
                        f"{o['area_ed']:.1f}".replace(".", ","),
                        f"{o['dens']:.1f}".replace(".", ","),
                        f"{100 * o['pct_ed']:.2f}".replace(".", ","),
                        f"{100 * o['pct_tot']:.2f}".replace(".", ","),
                        f"{o['mediana']:.1f}".replace(".", ","),
                        f"{100 * o['pct_peq']:.1f}".replace(".", ","),
                        o["estrato"], "sim" if o["margem"] else "não",
                        "sim" if o["sugerido"] else "não"])

    # --- relatório
    rel("RESUMO DOS EDIFÍCIOS")
    rel(f"  Edifícios lidos:                    {n_total}")
    rel(f"  Atribuídos a uma ortofoto:          {n_atrib}")
    rel(f"  Fora de todas as ortofotos:         {n_fora}")
    rel(f"  Em limite comum / sobreposição:     {n_sobrep} "
        f"(atribuídos à ortofoto de centro mais próximo)")
    rel()
    rel("ESTATÍSTICAS POR ORTOFOTO")
    rel(f"  {'ortofoto':<22}{'edif.':>7}{'km2':>7}{'edif/km2':>10}{'%área ed.':>10}"
        f"{'%edif.':>8}{'mediana':>9}{'%peq.':>7}  {'estrato':<12}{'margem':<7}")
    for o in sorted(ortos, key=lambda o: o["nome"]):
        marca = "  <- SUGERIDO" if o["sugerido"] else ""
        rel(f"  {o['nome'][:21]:<22}{o['n']:>7}{o['km2']:>7.2f}{o['dens']:>10.1f}"
            f"{100 * o['pct_ed']:>9.2f}%{100 * o['pct_tot']:>7.2f}%{o['mediana']:>8.1f} "
            f"{100 * o['pct_peq']:>6.1f}%  {o['estrato']:<12}{'sim' if o['margem'] else 'não':<7}"
            f"{marca}")
    rel(f"  (mediana = área mediana dos edifícios em m2; %peq. = edifícios < {AREA_PEQUENO:g} m2)")
    rel()
    rel("ESTRATOS DE DENSIDADE (tercis das ortofotos com >= "
        f"{MIN_EDIF_ORTO} edifícios, por edifícios/km2)")
    for nm in nomes_estr:
        L = estratos[nm]
        rel(f"  {nm:<9} {len(L):>2} ortofotos  densidade {min(o['dens'] for o in L):.1f} a "
            f"{max(o['dens'] for o in L):.1f} edif/km2 (mediana {med[nm]:.1f})")
    insuf = [o["nome"] for o in ortos if o["estrato"] == "insuficiente"]
    if insuf:
        rel(f"  excluídos (< {MIN_EDIF_ORTO} edifícios): {', '.join(insuf)}")
    rel()
    rel("CRITÉRIO DE SUGESTÃO")
    rel("  Uma ortofoto por estrato. Ordenação por: (1) proporção de edifícios dentro de "
        f"{PROP_MIN:.0%}-{PROP_MAX:.0%};")
    rel(f"  (2) proximidade a {PROP_ALVO:.0%}; (3) densidade próxima da mediana do estrato "
        "(representatividade);")
    rel(f"  {texto_criterio_margem()}")
    rel()
    rel(f"MELHORES COMBINAÇÕES (de {len(combos)} possíveis)")
    for i, (fora, pont, prop, rep, interiores, trio) in enumerate(combos[:N_COMBINACOES], 1):
        n = sum(o["n"] for o in trio)
        rel(f"  {i}. {' + '.join(o['nome'] for o in trio)}")
        rel(f"     {n} edifícios ({prop:.1%}){'  FORA DO INTERVALO' if fora else ''} | "
            f"representatividade {rep:.2f} (0 = ideal) | {3 - interiores}/3 na margem | "
            f"pontuação {pont:.3f}")
    rel()
    rel("SUGESTÃO PARA TESTE")
    for o in sugerido:
        rel(f"  {o['estrato']:<9} {o['nome']}  ({o['n']} edifícios, {o['dens']:.1f} edif/km2)")
    n_sug = sum(o["n"] for o in sugerido)
    rel(f"  Total: {n_sug} edifícios = {n_sug / n_atrib:.1%} dos edifícios; "
        f"{sum(o['km2'] for o in sugerido):.2f} km2")
    rel()
    rel("FICHEIROS DE SAÍDA")
    rel(f"  Grelha:     {SAIDA_GRELHA}  (campo SUGERIDO = 1)")
    rel(f"  Tabela:     {saida_csv}")
    rel(f"  Relatório:  {saida_txt}")

    if ADICIONAR_AO_MAPA and mapa is not None:
        lyr = mapa.addDataFromPath(SAIDA_GRELHA)
        try:
            lc = lyr.listLabelClasses()[0]
            lc.expression = "$feature.NOME + TextFormatting.NewLine + $feature.ESTRATO"
            lyr.showLabels = True
        except Exception:
            pass
        rel(f"Camada aberta no mapa '{mapa.name}': {os.path.basename(SAIDA_GRELHA)}")
    rel(f"Tempo de execução: {time.time() - t0:.1f} s")
    rel.gravar(saida_txt)


if __name__ == "__main__":
    _rel = Relatorio()
    try:
        main(_rel)
    except Exception:
        _rel()
        _rel("ERRO - execução interrompida")
        _rel(traceback.format_exc())
        _rel.gravar(os.path.splitext(SAIDA_GRELHA)[0] + "_relatorio.txt")
        raise
