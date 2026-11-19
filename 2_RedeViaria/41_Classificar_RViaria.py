# -*- coding: utf-8 -*-
"""
41_Classificar_RViaria.py - Etiquetas de treino da rede viária (Classified Tiles).
Executar na janela Python ou num Notebook do ArcGIS Pro 3.6.

Entrada: l_vias_IGeoE.shp (eixos já projetados para o sistema das ortos pelo 40a),
com o campo 'source' (nome da classe). O shapefile NÃO é alterado: tudo é copiado
para memória. Se os eixos ainda não estiverem no sistema da orto, são projetados
com TRANSFORMACAO (a mesma do 40a).
Passos: classe e largura por troço -> buffer -> dissolução por classe -> sobreposições
resolvidas por PRIORIDADE -> recorte à área de treino (40b).
Saída: etiquetas_vias.shp (3 classes), usada pelo 41b e pela exportação multiclasse.
"""
import os
import arcpy

arcpy.env.overwriteOutput = True

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
EIXOS = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP\l_vias_IGeoE.shp"   # projetado pelo 40a
CAMPO_SOURCE = "source"

# Verificação de segurança: os eixos têm de estar no sistema da orto. Com o
# l_vias_IGeoE.shp do 40a já estão; se se usar o original (Datum Lisboa), são
# projetados com a MESMA transformação do 40a.
ORTO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"
TRANSFORMACAO = "Lisbon_Lisbon_To_WGS_1984_2"
CAMPO_MANUAL = "CLS_MANUAL"   # inteiro opcional: 1, 2 ou 3 sobrepõe a classe automática
PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"               # pasta dos shapefiles criados
AREA_TREINO = os.path.join(PASTA_SAIDA, "area_treino.shp")  # criado pelo 40b; None = não recortar
SAIDA = "etiquetas_vias.shp"

# source -> classe de treino por defeito
# 1 = Via pavimentada, 2 = Via não pavimentada, 3 = Ferrovia
CLASSE = {
    "Auto_estrada": 1,
    "Estrada_larga": 1,
    "Estrada_estreita": 1,
    "Arruamento": 1,          # de terra -> CLS_MANUAL = 2
    "Acesso_auto": 2,         # de piso duro -> CLS_MANUAL = 1
    "Caminho_carreteiro": 2,
    "Caminho_a_pe_posto": 2,
    "Caminho_de_ferro_via_larga_unica": 3,
}

# source -> LARGURA TOTAL em metros (mediana das medições na ortofoto)
LARGURA = {
    "Auto_estrada": 11.0,        # UMA faixa de rodagem com bermas (um eixo por sentido)
    "Estrada_larga": 7.0,
    "Estrada_estreita": 4.0,
    "Arruamento": 5.0,
    "Acesso_auto": 3.0,
    "Caminho_carreteiro": 2.5,
    "Caminho_a_pe_posto": 1.5,
    "Caminho_de_ferro_via_larga_unica": 3.5,
}

# Desvio típico do eixo restituído em relação ao centro real da via (m).
# A largura do buffer passa a ser LARGURA - 2 x DESVIO_EIXO (mínimo LARGURA_MIN),
# para a etiqueta ficar dentro da via mesmo com o eixo descentrado.
DESVIO_EIXO = 0.75     # substituir pela mediana medida na ortofoto
LARGURA_MIN = 1.0      # 2 píxeis a 0,5 m

# Separador central da autoestrada (m), que fica como fundo:
#  - eixo único no centro: LARGURA["Auto_estrada"] = largura total bordo a bordo
#    (com separador) e SEPARADOR_AE = largura do separador
#  - um eixo por sentido: LARGURA["Auto_estrada"] = largura de UMA faixa de
#    rodagem com bermas, e SEPARADOR_AE = 0
SEPARADOR_AE = 0.0

NOMES = {1: "Via_pavimentada", 2: "Via_nao_pavimentada", 3: "Ferrovia"}
PRIORIDADE = [1, 3, 2]        # em sobreposições (cruzamentos), ganha a primeira


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
    # valor com prefixo/sufixo (ex.: '126_Estrada_larga'): ganha a chave mais longa
    for k in sorted(CHAVES, key=len, reverse=True):
        if k in n:
            return CHAVES[k]
    return None


# ---------------------------------------------------------------------------
os.makedirs(PASTA_SAIDA, exist_ok=True)

print("1. Copiar eixos e atribuir classes")
tmp = "memory/eixos"
sr_orto = arcpy.Describe(ORTO).spatialReference
sr_eixos = arcpy.Describe(EIXOS).spatialReference
if sr_eixos.exportToString() != sr_orto.exportToString():
    if not TRANSFORMACAO:
        raise SystemExit(
            f"Eixos em '{sr_eixos.name}' e orto em '{sr_orto.name}': definir TRANSFORMACAO.\n"
            f"Possíveis: {arcpy.ListTransformations(sr_eixos, sr_orto)}")
    print(f"  projetar eixos: {sr_eixos.name} -> {sr_orto.name} ({TRANSFORMACAO})")
    arcpy.management.Project(EIXOS, tmp, sr_orto, TRANSFORMACAO)
else:
    arcpy.management.CopyFeatures(EIXOS, tmp)
# todas as operações seguintes ficam no sistema da orto
arcpy.env.outputCoordinateSystem = sr_orto
arcpy.management.AddField(tmp, "Classvalue", "SHORT")
arcpy.management.AddField(tmp, "BUF_M", "DOUBLE")    # metade da largura
arcpy.management.AddField(tmp, "AE", "SHORT")         # 1 = autoestrada

campos = [f.name for f in arcpy.ListFields(tmp)]
usa_manual = CAMPO_MANUAL in campos
cur_campos = [CAMPO_SOURCE, "Classvalue", "BUF_M", "AE"] + ([CAMPO_MANUAL] if usa_manual else [])

desconhecidos, contagem, manuais = {}, {}, 0
with arcpy.da.UpdateCursor(tmp, cur_campos) as cur:
    for row in cur:
        src = classe_origem(row[0])
        if src is None:
            desconhecidos[row[0]] = desconhecidos.get(row[0], 0) + 1
            cur.deleteRow()
            continue
        classe = CLASSE[src]
        if usa_manual and row[4] in (1, 2, 3):
            classe = row[4]
            manuais += 1
        row[1] = classe
        row[2] = max(LARGURA_MIN, LARGURA[src] - 2 * DESVIO_EIXO) / 2.0
        row[3] = 1 if src == "Auto_estrada" else 0
        cur.updateRow(row)
        contagem[src] = contagem.get(src, 0) + 1

if not contagem:
    raise SystemExit("Nenhum troço reconhecido: verificar os valores de source.")
for src, n in sorted(contagem.items()):
    efetiva = max(LARGURA_MIN, LARGURA[src] - 2 * DESVIO_EIXO)
    print(f"  {src}: {n} troços, largura {LARGURA[src]:.1f} m -> buffer {efetiva:.1f} m")
print(f"  correções manuais aplicadas: {manuais}")
if desconhecidos:
    print("  ATENÇÃO, valores de source não reconhecidos (ignorados):")
    for src, n in desconhecidos.items():
        print(f"    '{src}': {n}")

print("2. Buffer e dissolução por classe de treino")
buf = "memory/buffers"
# ROUND nas extremidades evita falhas nos cruzamentos e mantém a conectividade
arcpy.analysis.Buffer(tmp, buf, "BUF_M", "FULL", "ROUND", "NONE")

if SEPARADOR_AE > 0:
    eixos_ae = "memory/eixos_ae"
    arcpy.analysis.Select(tmp, eixos_ae, "AE = 1")
    if int(arcpy.management.GetCount(eixos_ae)[0]) > 0:
        print(f"  remover separador central da autoestrada ({SEPARADOR_AE} m)")
        sep = "memory/separador"
        arcpy.analysis.Buffer(eixos_ae, sep, f"{SEPARADOR_AE / 2.0} Meters",
                              "FULL", "FLAT", "ALL")
        buf_ae, buf_resto = "memory/buf_ae", "memory/buf_resto"
        arcpy.analysis.Select(buf, buf_ae, "AE = 1")
        arcpy.analysis.Select(buf, buf_resto, "AE = 0")
        buf_ae_sem = "memory/buf_ae_sem"
        arcpy.analysis.Erase(buf_ae, sep, buf_ae_sem)   # só a autoestrada perde o separador
        buf = "memory/buffers_sep"
        arcpy.management.Merge([buf_resto, buf_ae_sem], buf)
dissolvido = "memory/dissolvido"
arcpy.management.Dissolve(buf, dissolvido, ["Classvalue"])

print("3. Resolver sobreposições por prioridade")
camadas, ocupado = [], None
for c in PRIORIDADE:
    lyr = f"memory/c{c}"
    arcpy.analysis.Select(dissolvido, lyr, f"Classvalue = {c}")
    if int(arcpy.management.GetCount(lyr)[0]) == 0:
        print(f"  {NOMES[c]}: sem elementos")
        continue
    if ocupado:
        apagado = f"memory/c{c}_e"
        arcpy.analysis.Erase(lyr, ocupado, apagado)
        lyr = apagado
        novo = f"memory/ocupado_{c}"
        arcpy.management.Merge([ocupado, lyr], novo)
        ocupado = novo
    else:
        ocupado = lyr
    camadas.append(lyr)

print("4. Juntar, recortar e gravar")
final = "memory/final"
arcpy.management.Merge(camadas, final)
if AREA_TREINO:
    recortado = "memory/recortado"
    arcpy.analysis.Clip(final, AREA_TREINO, recortado)
    final = recortado

saida = os.path.join(PASTA_SAIDA, SAIDA)
arcpy.management.MultipartToSinglepart(final, saida)
arcpy.management.AddField(saida, "Classname", "TEXT", field_length=30)
with arcpy.da.UpdateCursor(saida, ["Classvalue", "Classname"]) as cur:
    for row in cur:
        row[1] = NOMES[row[0]]
        cur.updateRow(row)

print("5. Área por classe (ha)")
areas = {}
with arcpy.da.SearchCursor(saida, ["Classname", "SHAPE@AREA"]) as cur:
    for nome, area in cur:
        areas[nome] = areas.get(nome, 0) + area
for nome, area in sorted(areas.items()):
    print(f"  {nome}: {area / 10000:.1f} ha")

arcpy.management.Delete("memory")
print(f"Concluído: {saida}")
