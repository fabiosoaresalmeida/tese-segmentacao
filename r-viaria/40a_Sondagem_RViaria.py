# -*- coding: utf-8 -*-
"""
40a_Sondagem_RViaria.py - Sondagem da rede viária por ortofoto.

0. Projeta os eixos (Datum Lisboa) para o sistema de coordenadas das ortofotos
   -> l_vias_IGeoE.shp, usado depois pelo 41, 44, 45 e 46
1. Cria um polígono por ortofoto (extensão de cada TIF) -> folhas_ortos.shp
2. Mede o comprimento de eixos por classe em cada ortofoto -> sondagem_vias.csv
3. Avalia todas as combinações de N_TESTE ortos de teste e ordena-as pela
   diversidade dos perfis viários (soma das distâncias entre os perfis normalizados),
   com restrições de representatividade (MAX_TESTE). Mostra também a posição da
   combinação usada nos edifícios e da escolhida para as vias.

Executar na janela Python ou num Notebook do ArcGIS Pro 3.6.
"""
import os
import csv
import math
import itertools
import arcpy

arcpy.env.overwriteOutput = True

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
PASTA_ORTOS = r"D:\Users\Cap_Almeida\Projeto\M888_Ortos\126_Ortofotos"     # as 16 ortofotos individuais
EXTENSOES = (".tif", ".tiff")
EIXOS = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\l_vias.shp"    # original (Datum Lisboa)
CAMPO_SOURCE = "source"
PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"

# Eixos projetados para o sistema das ortofotos (usado pelo 41, 44, 45 e 46)
EIXOS_PROJ = os.path.join(PASTA_SAIDA, "l_vias_IGeoE.shp")
# Transformação de datum Lisboa -> WGS84: a mesma que o mapa usa
# (Map Properties > Transformation). Se ficar None e os sistemas forem
# diferentes, o script para e lista as transformações possíveis.
TRANSFORMACAO = "Lisbon_Lisbon_To_WGS_1984_2"

ATUAL = ["126-I-3", "126-II-3", "126-IV-1"]   # teste dos edifícios (nomes = nome do TIF)
ESCOLHIDA = ["126-I-2", "126-III-4", "126-IV-1"]   # teste escolhido para as vias
N_TESTE = 3
MAX_TESTE = 0.35     # fração máxima de cada classe (pavimentada, não pavimentada) em teste
TOP = 10             # número de combinações mostradas

CLASSE = {
    "Auto_estrada": 1,
    "Estrada_larga": 1,
    "Estrada_estreita": 1,
    "Arruamento": 1,
    "Acesso_auto": 2,
    "Caminho_carreteiro": 2,
    "Caminho_a_pe_posto": 2,
    "Caminho_de_ferro_via_larga_unica": 3,
}
NOMES = {1: "Pavimentada", 2: "Nao_pavimentada", 3: "Ferrovia"}
CLASSES_RESTRITAS = [1, 2]   # a ferrovia é só reportada (pode existir numa única orto)


def normalizar(valor):
    """Tolera caminho, extensão, maiúsculas, espaços e hífens no valor de 'source'."""
    base = os.path.splitext(os.path.basename(str(valor or "").strip()))[0]
    return base.lower().replace(" ", "_").replace("-", "_")


CHAVES = {normalizar(k): k for k in CLASSE}


def classe_origem(valor):
    """Devolve a chave de CLASSE correspondente ao valor, ou None."""
    n = normalizar(valor)
    if n in CHAVES:
        return CHAVES[n]
    # valor com prefixo/sufixo (ex.: 'L_Estrada_larga'): ganha a chave mais longa
    for k in sorted(CHAVES, key=len, reverse=True):
        if k in n:
            return CHAVES[k]
    return None


# ---------------------------------------------------------------------------
os.makedirs(PASTA_SAIDA, exist_ok=True)

ortos = sorted(f for f in os.listdir(PASTA_ORTOS) if f.lower().endswith(EXTENSOES))
if not ortos:
    raise SystemExit(f"Nenhum TIF em {PASTA_ORTOS}")
sr = arcpy.Describe(os.path.join(PASTA_ORTOS, ortos[0])).spatialReference

print("0. Sistema de coordenadas dos eixos")
sr_eixos = arcpy.Describe(EIXOS).spatialReference
print(f"  ortofotos: {sr.name}")
print(f"  eixos:     {sr_eixos.name}")
if sr_eixos.exportToString() != sr.exportToString():
    if not TRANSFORMACAO:
        raise SystemExit("Sistemas diferentes: definir TRANSFORMACAO. Possíveis:\n  "
                         + "\n  ".join(arcpy.ListTransformations(sr_eixos, sr)))
    arcpy.management.Project(EIXOS, EIXOS_PROJ, sr, TRANSFORMACAO)
    print(f"  projetados com '{TRANSFORMACAO}' -> {EIXOS_PROJ}")
else:
    arcpy.management.CopyFeatures(EIXOS, EIXOS_PROJ)
    print(f"  já no sistema das ortofotos, copiados -> {EIXOS_PROJ}")
# todas as operações seguintes ficam no sistema das ortofotos
arcpy.env.outputCoordinateSystem = sr

print("1. Polígonos das ortofotos")
folhas = os.path.join(PASTA_SAIDA, "folhas_ortos.shp")
arcpy.management.CreateFeatureclass(PASTA_SAIDA, "folhas_ortos.shp", "POLYGON",
                                    spatial_reference=sr)
arcpy.management.AddField(folhas, "FOLHA", "TEXT", field_length=30)
arcpy.management.AddField(folhas, "AREA_KM2", "DOUBLE")

area = {}
with arcpy.da.InsertCursor(folhas, ["SHAPE@", "FOLHA", "AREA_KM2"]) as cur:
    for f in ortos:
        desc = arcpy.Describe(os.path.join(PASTA_ORTOS, f))
        if desc.spatialReference.name != sr.name:
            print(f"  ATENÇÃO: {f} tem outro sistema de coordenadas")
        poli = desc.extent.polygon
        nome = os.path.splitext(f)[0]
        area[nome] = poli.area / 1e6
        cur.insertRow([poli, nome, area[nome]])
print(f"  {len(ortos)} ortofotos -> {folhas}")

for a in ATUAL + ESCOLHIDA:
    if a not in area:
        print(f"  ATENÇÃO: '{a}' não corresponde a nenhum TIF")

print("2. Comprimento por classe e por ortofoto")
inter = "memory/inter"
arcpy.analysis.PairwiseIntersect([EIXOS_PROJ, folhas], inter)
print(f"  troços cruzados com as ortos: {arcpy.management.GetCount(inter)[0]}")
km = {f: {s: 0.0 for s in CLASSE} for f in area}
desconhecidos = {}
with arcpy.da.SearchCursor(inter, ["FOLHA", CAMPO_SOURCE, "SHAPE@LENGTH"]) as cur:
    for folha, valor, comp in cur:
        src = classe_origem(valor)
        if src:
            km[folha][src] += comp / 1000.0
        else:
            desconhecidos[valor] = desconhecidos.get(valor, 0) + 1
arcpy.management.Delete(inter)

if desconhecidos:
    print("  ATENÇÃO, valores de source não reconhecidos (ignorados):")
    for v, n in sorted(desconhecidos.items(), key=lambda x: -x[1]):
        print(f"    {v!r}: {n}")
total_km = sum(sum(d.values()) for d in km.values())
if total_km == 0:
    raise SystemExit("Comprimento total = 0: nenhum troço reconhecido. "
                     "Ver os valores acima e ajustar o dicionário CLASSE.")
for s_ in CLASSE:
    print(f"  {s_}: {sum(km[f][s_] for f in area):.1f} km")

km_cls = {f: {c: sum(v for s, v in km[f].items() if CLASSE[s] == c) for c in NOMES}
          for f in area}

csv_path = os.path.join(PASTA_SAIDA, "sondagem_vias.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["FOLHA", "AREA_KM2", "TOTAL_KM", "DENS_KM_KM2", "PCT_PAVIMENTADA"]
               + [f"KM_{NOMES[c]}" for c in NOMES] + [f"KM_{s}" for s in CLASSE])
    for f in sorted(area):
        tot = sum(km[f].values())
        pct = 100 * km_cls[f][1] / tot if tot else 0
        w.writerow([f, f"{area[f]:.2f}", f"{tot:.2f}", f"{tot / area[f]:.2f}", f"{pct:.1f}"]
                   + [f"{km_cls[f][c]:.2f}" for c in NOMES]
                   + [f"{km[f][s]:.2f}" for s in CLASSE])
print(f"  -> {csv_path}")

print("3. Combinações de teste")
# Perfil de cada orto: densidade (km/km²) de cada classe de origem, normalizada (z-score)
presentes = [s for s in CLASSE if sum(km[f][s] for f in area) > 0]
dens = {f: [km[f][s] / area[f] for s in presentes] for f in area}
z = {f: [] for f in area}
for i in range(len(presentes)):
    col = [dens[f][i] for f in area]
    media = sum(col) / len(col)
    desvio = math.sqrt(sum((x - media) ** 2 for x in col) / len(col))
    for f in area:
        z[f].append((dens[f][i] - media) / desvio if desvio > 0 else 0.0)

tot_cls = {c: sum(km_cls[f][c] for f in area) for c in NOMES}


def avaliar(trio):
    partes = {c: (sum(km_cls[f][c] for f in trio) / tot_cls[c] if tot_cls[c] else 0)
              for c in NOMES}
    valido = all(0 < partes[c] <= MAX_TESTE for c in CLASSES_RESTRITAS if tot_cls[c] > 0)
    diversidade = sum(math.dist(z[a], z[b]) for a, b in itertools.combinations(trio, 2))
    ausentes = [s for s in presentes if sum(km[f][s] for f in trio) == 0]
    return dict(trio=tuple(sorted(trio)), div=diversidade, valido=valido,
                partes=partes, ausentes=ausentes)


resultados = [avaliar(t) for t in itertools.combinations(sorted(area), N_TESTE)]
validos = sorted((r for r in resultados if r["valido"]), key=lambda r: -r["div"])
print(f"  {len(resultados)} combinações, {len(validos)} válidas "
      f"(cada classe com 0 < fração em teste <= {MAX_TESTE:.0%})")


def mostrar(i, r):
    p = ", ".join(f"{NOMES[c]} {r['partes'][c]:.0%}" for c in NOMES if tot_cls[c] > 0)
    aus = f" | sem: {', '.join(r['ausentes'])}" if r["ausentes"] else ""
    print(f"  {i:>3}. {' + '.join(r['trio'])}  div={r['div']:.2f}  [{p}]{aus}")


print(f"\nMelhores {TOP}:")
for i, r in enumerate(validos[:TOP], 1):
    mostrar(i, r)

for titulo, combo in (("Combinação dos edifícios", ATUAL), ("Combinação escolhida (vias)", ESCOLHIDA)):
    alvo = tuple(sorted(combo))
    r_alvo = next((r for r in resultados if r["trio"] == alvo), None)
    if not r_alvo:
        continue
    print(f"\n{titulo}:")
    if r_alvo["valido"]:
        pos = [r["trio"] for r in validos].index(alvo) + 1
        mostrar(pos, r_alvo)
        print(f"  -> posição {pos} de {len(validos)} válidas")
    else:
        mostrar(0, r_alvo)
        print("  -> NÃO cumpre as restrições de representatividade")
