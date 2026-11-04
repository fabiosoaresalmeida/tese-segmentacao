# -*- coding: utf-8 -*-
"""
03_Limpar_Constr_Projt.py  (ArcGIS Pro 3.6)

Prepara a camada de construções (a_constr) da edição cartográfica para servir de
referência ao treino e à avaliação dos modelos de segmentação de edifícios.
Primeiro passo da cadeia; a saída (a_constr_Limpa.shp) é a entrada do
06_separar_treino_teste.py e a referência ao píxel do 09_avaliar_modelos.py.

0. Verifica se o sistema de coordenadas da shapefile é o mesmo do ortomosaico.
   Se não for, avisa e a shapefile de saída é PROJETADA para o sistema do
   ortomosaico, com a transformação geográfica mais adequada quando os datums são
   diferentes (por exemplo, Lisboa -> WGS 84). A projeção não altera a posição dos
   edifícios no terreno: os vetores mantêm a posição original da restituição.
1. Elimina os polígonos que não estejam totalmente dentro da área do ortomosaico
   (.tif), incluindo os que intersetam o seu limite.
2. Elimina os polígonos cujo campo "source" contenha alguma das palavras
   "campo", "limite", "piscina" ou "electrica" (sem distinguir maiúsculas,
   minúsculas e acentos). As ruínas são mantidas.

No fim apresenta um relatório com os sistemas de coordenadas (entrada,
ortomosaico e saída) e a transformação usada, o número inicial de polígonos, os
removidos por estarem fora ou no limite do ortomosaico, os removidos por
palavra-chave (por palavra e por valor de "source") e o número final.

Pode correr na janela Python do ArcGIS Pro ou no Python Command Prompt:
    python "D:\\Users\\Cap_Almeida\\Projeto\\Python\\03_Limpar_Constr_Projt.py"
Os parâmetros estão na secção "PARÂMETROS".
"""

import arcpy
import os
import unicodedata
from collections import Counter


def normalizar(texto):
    """Remove acentos e espaços nas extremidades e converte para minúsculas,
    para que a comparação 'contém a palavra' funcione mesmo com acentos
    ou espaços extra."""
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto


def mapa_ativo():
    """Devolve o mapa ativo do ArcGIS Pro, ou None se o script correr fora do Pro."""
    try:
        return arcpy.mp.ArcGISProject("CURRENT").activeMap
    except OSError:
        return None


def remover_layers_do_mapa(nomes):
    """Remove do mapa ativo as layers com os nomes indicados.
    Só funciona dentro do ArcGIS Pro; em modo standalone não faz nada."""
    mapa = mapa_ativo()
    if mapa is None:
        return
    for lyr in list(mapa.listLayers()):
        if lyr.supports("NAME") and lyr.name in nomes:
            nome_lyr = lyr.name  # guarda antes de remover
            mapa.removeLayer(lyr)
            print(f"Layer '{nome_lyr}' removida do mapa.")


def descrever_sr(sr):
    """Texto curto com o nome, o código EPSG (se existir) e o datum."""
    codigo = sr.factoryCode if sr.factoryCode else "sem código EPSG"
    datum = sr.GCS.datumName if sr.GCS else "?"
    return f"{sr.name} ({codigo}; datum {datum})"


def mesmo_sr(a, b):
    """Compara dois sistemas de coordenadas.
    Com código EPSG nos dois, compara o código. Sem código (sistemas personalizados,
    como o WGS84 Militares IGeoE), compara o nome do sistema projetado, o datum e
    os parâmetros da projeção."""
    if a.factoryCode and b.factoryCode:
        return a.factoryCode == b.factoryCode
    if a.type != b.type or a.name != b.name:
        return False
    if (a.GCS.datumName if a.GCS else None) != (b.GCS.datumName if b.GCS else None):
        return False
    if a.type == "Projected":
        parametros = ("falseEasting", "falseNorthing", "centralMeridian",
                      "scaleFactor", "latitudeOfOrigin")
        for p in parametros:
            va, vb = getattr(a, p, None), getattr(b, p, None)
            if va is not None and vb is not None and abs(va - vb) > 1e-6:
                return False
    return True


# ============================================================
# PARÂMETROS - AJUSTAR CONFORME NECESSÁRIO
# ============================================================

shp_path = r"D:\Users\Cap_Almeida\Projeto\126\SHP\a_constr.shp"   # shapefile de entrada
tif_path = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"                         # ortofoto (.tif)
output_shp = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf\a_constr_Limpa.shp"  # shapefile de saída

# Campo a verificar para as palavras a excluir
campo_source = "source"

# Palavras que, se encontradas no campo "source", levam à eliminação do polígono
# (a comparação ignora maiúsculas/minúsculas e acentos)
palavras_excluir = ["campo", "limite", "piscina", "electrica"]

# Transformação geográfica a usar se for preciso projetar e os datums forem
# diferentes. None = escolhida automaticamente (a primeira proposta pelo ArcGIS
# para a extensão dos dados, normalmente a mais adequada). Para forçar uma
# transformação, indicar o nome, por exemplo "Lisbon_Lisbon_To_WGS_1984_2".
transformacao = None

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False  # não adiciona ao mapa os outputs temporários

lyr_name = "lyr_poligonos"
raster_domain_fc = r"memory\raster_domain"

# Remove do mapa ativo qualquer layer antiga com o mesmo nome do output,
# para evitar locks de esquema em execuções repetidas
remover_layers_do_mapa([os.path.splitext(os.path.basename(output_shp))[0]])


# ============================================================
# PASSO 0 - Verificar os sistemas de coordenadas da shapefile e do ortomosaico
# ============================================================

print("A verificar os sistemas de coordenadas...")

sr_shp = arcpy.Describe(shp_path).spatialReference
sr_tif = arcpy.Describe(tif_path).spatialReference
print(f"  Shapefile:   {descrever_sr(sr_shp)}")
print(f"  Ortomosaico: {descrever_sr(sr_tif)}")

if sr_shp.name == "Unknown":
    raise RuntimeError("A shapefile não tem sistema de coordenadas definido (.prj em falta). "
                       "Define-o primeiro com Define Projection.")
if sr_tif.name == "Unknown":
    raise RuntimeError("O ortomosaico não tem sistema de coordenadas definido.")

precisa_projetar = not mesmo_sr(sr_shp, sr_tif)
transformacao_usada = None

if precisa_projetar:
    print("  AVISO: os sistemas de coordenadas são DIFERENTES. A shapefile de saída "
          "vai ser projetada para o sistema do ortomosaico.")
    datum_shp = sr_shp.GCS.datumName if sr_shp.GCS else None
    datum_tif = sr_tif.GCS.datumName if sr_tif.GCS else None
    if datum_shp != datum_tif:
        # Datums diferentes: é preciso uma transformação geográfica.
        # Sem ela, os dados podem ficar desfasados dezenas de metros.
        if transformacao:
            transformacao_usada = transformacao
        else:
            ext = arcpy.Describe(shp_path).extent
            candidatas = arcpy.ListTransformations(sr_shp, sr_tif, ext)
            if not candidatas:
                raise RuntimeError(f"Não foi encontrada nenhuma transformação geográfica entre "
                                   f"{datum_shp} e {datum_tif}. Indica-a em 'transformacao'.")
            transformacao_usada = candidatas[0]
            print(f"  Transformações disponíveis: {candidatas}")
        print(f"  Datums diferentes ({datum_shp} -> {datum_tif}); "
              f"transformação geográfica: {transformacao_usada}")
    else:
        print("  Mesmo datum: não é necessária transformação geográfica.")
else:
    print("  OK: a shapefile e o ortomosaico estão no mesmo sistema de coordenadas.")


# ============================================================
# PASSO 1 - Criar um polígono a partir da extensão/área real do raster
# ============================================================

print("A gerar o domínio (footprint) do raster...")

# Garante que a extensão 3D Analyst está disponível (RasterDomain precisa dela)
if arcpy.CheckExtension("3D") == "Available":
    arcpy.CheckOutExtension("3D")
else:
    raise RuntimeError("Extensão 3D Analyst não disponível/licenciada.")

# "memory" em vez de "in_memory" - RasterDomain (3D Analyst) é pouco fiável com in_memory
arcpy.ddd.RasterDomain(tif_path, raster_domain_fc, "POLYGON")

# Verificação imediata - falha aqui é muito mais fácil de diagnosticar
if not arcpy.Exists(raster_domain_fc):
    raise RuntimeError(f"RasterDomain não criou {raster_domain_fc} - verifica o .tif e a licença 3D Analyst.")


# ============================================================
# PASSO 2 - Criar a shapefile de saída (o original fica intacto):
#   - mesmo sistema de coordenadas: cópia simples;
#   - sistema diferente: projeção para o sistema do ortomosaico.
# O Z e todos os atributos são mantidos.
# ============================================================

if precisa_projetar:
    print("A projetar a shapefile para o sistema de coordenadas do ortomosaico...")
    arcpy.management.Project(shp_path, output_shp, sr_tif, transformacao_usada)
else:
    print("A copiar shapefile de entrada...")
    arcpy.management.CopyFeatures(shp_path, output_shp)

sr_saida = arcpy.Describe(output_shp).spatialReference
if not mesmo_sr(sr_saida, sr_tif):
    print(f"ATENÇÃO: a saída ficou em {descrever_sr(sr_saida)}, diferente do ortomosaico.")

count_inicial = int(arcpy.management.GetCount(output_shp)[0])
print(f"Número inicial de polígonos: {count_inicial}")


# ============================================================
# PASSO 3 - Eliminar polígonos que NÃO estejam totalmente dentro
# da área da ortofoto (usa seleção espacial "COMPLETELY_WITHIN" e inverte)
# ============================================================

print("A identificar polígonos fora da área da ortofoto...")

arcpy.management.MakeFeatureLayer(output_shp, lyr_name)

# Seleciona os que ESTÃO completamente dentro
arcpy.management.SelectLayerByLocation(
    lyr_name,
    "COMPLETELY_WITHIN",
    raster_domain_fc,
    selection_type="NEW_SELECTION"
)

# Inverte a seleção -> fica com os que NÃO estão completamente dentro
arcpy.management.SelectLayerByAttribute(lyr_name, "SWITCH_SELECTION")

count_fora = int(arcpy.management.GetCount(lyr_name)[0])
print(f"Polígonos fora da área da ortofoto a eliminar: {count_fora}")

if count_fora > 0:
    arcpy.management.DeleteFeatures(lyr_name)

# Limpar seleção
arcpy.management.SelectLayerByAttribute(lyr_name, "CLEAR_SELECTION")


# ============================================================
# PASSO 4 - Eliminar polígonos cujo campo "source" contenha
# alguma das palavras-chave (ignora maiúsculas/minúsculas e acentos)
# ============================================================

print(f"A verificar o campo '{campo_source}' para eliminar por palavra-chave...")

campos_disponiveis = [f.name for f in arcpy.ListFields(output_shp)]
if campo_source not in campos_disponiveis:
    raise ValueError(
        f"O campo '{campo_source}' não existe no shapefile. "
        f"Campos disponíveis: {campos_disponiveis}"
    )

# Normaliza também a lista de palavras a excluir (por consistência)
palavras_excluir_norm = [normalizar(p) for p in palavras_excluir]

eliminados_por_palavra = Counter()   # quantos polígonos cada palavra eliminou
eliminados_por_valor = Counter()     # quantos polígonos de cada valor de "source"
palavra_do_valor = {}                # palavra que levou à eliminação de cada valor

total_eliminados_source = 0

with arcpy.da.UpdateCursor(output_shp, [campo_source]) as cursor:
    for row in cursor:
        valor = row[0]
        if valor is None:
            continue

        valor_norm = normalizar(valor)  # ex: "A_Campo_de_Futebol" -> "a_campo_de_futebol"

        match = next((p for p in palavras_excluir_norm if p in valor_norm), None)

        if match:
            cursor.deleteRow()
            eliminados_por_palavra[match] += 1
            eliminados_por_valor[valor] += 1
            palavra_do_valor[valor] = match
            total_eliminados_source += 1

count_palavras = sum(eliminados_por_palavra.values())
print(f"Polígonos eliminados por palavra-chave: {count_palavras}")


# ============================================================
# LIMPEZA
# ============================================================

# Remove do Contents (TOC) qualquer layer temporária que tenha sido adicionada,
# independentemente do addOutputsToMap (só dentro do ArcGIS Pro)
mapa = mapa_ativo()
if mapa is not None:
    for lyr in list(mapa.listLayers()):
        if lyr.supports("NAME") and lyr.name in ("raster_domain", lyr_name):
            nome_lyr = lyr.name          # guarda antes de remover
            mapa.removeLayer(lyr)
            print(f"Layer '{nome_lyr}' removida do mapa.")

arcpy.management.Delete(lyr_name)
print(f"Layer temporária '{lyr_name}' apagada.")

arcpy.management.Delete(raster_domain_fc)
print(f"Feature class temporária '{raster_domain_fc}' apagada.")

arcpy.CheckInExtension("3D")


# ============================================================
# RELATÓRIO FINAL
# ============================================================

count_final = int(arcpy.management.GetCount(output_shp)[0])

print()
print("=" * 70)
print("RELATÓRIO DE FILTRAGEM")
print("=" * 70)
print(f"Shapefile de entrada: {shp_path}")
print(f"Shapefile de saída:   {output_shp}")
print()
print("SISTEMAS DE COORDENADAS")
print(f"  Entrada:     {descrever_sr(sr_shp)}")
print(f"  Ortomosaico: {descrever_sr(sr_tif)}")
print(f"  Saída:       {descrever_sr(sr_saida)}")
if precisa_projetar:
    print(f"  A saída foi PROJETADA para o sistema do ortomosaico.")
    print(f"  Transformação geográfica: {transformacao_usada or 'não necessária (mesmo datum)'}")
else:
    print("  Sem projeção (mesmo sistema de coordenadas).")
print()
print(f"{'Polígonos iniciais':<45}{count_inicial:>10}")
print(f"{'Removidos (fora ou no limite do ortomosaico)':<45}{count_fora:>10}")
print(f"{'Removidos (palavra-chave no campo source)':<45}{count_palavras:>10}")
print(f"{'Polígonos finais':<45}{count_final:>10}")

# Verificação de consistência
esperado = count_inicial - count_fora - count_palavras
if esperado != count_final:
    print(f"ATENÇÃO: contagem final ({count_final}) diferente da esperada ({esperado}).")

print()
print("Removidos por palavra-chave:")
for p in palavras_excluir_norm:
    print(f"  {p:<20}{eliminados_por_palavra[p]:>10}")

print()
print(f"Removidos por valor do campo '{campo_source}':")
print(f"  {'Valor':<45}{'Palavra':<15}{'N.º':>8}")
for valor, n in sorted(eliminados_por_valor.items(), key=lambda x: (-x[1], x[0])):
    print(f"  {str(valor):<45}{palavra_do_valor[valor]:<15}{n:>8}")
