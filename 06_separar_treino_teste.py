# -*- coding: utf-8 -*-
r"""
06_separar_treino_teste.py  (ArcGIS Pro 3.6)

A partir dos ortofotos escolhidos para teste (grelha criada por analisar_ortos.py):
  - zona_teste.shp       área dos ortofotos de teste, com UM POLÍGONO POR ORTOFOTO
                         e o campo NOME (o mesmo da grelha), pronta a usar no
                         09_avaliar_modelos.py com CAMPO_NOME = "NOME"
  - mascara_treino.shp   resto da carta, num só polígono (usar em Input Mask Polygons
                         do Export Training Data e em AREA_ESTUDO do criar_rotulos_borda.py)
  - mascara_treino_ortos.shp  a mesma área, com UM POLÍGONO POR ORTOFOTO de treino e o
                         campo NOME (para análises por ortofoto; não usar na exportação)
  - <PREFIXO>_teste.shp  edifícios de TESTE: só os que estão COMPLETAMENTE dentro
                         da zona de teste
  - <PREFIXO>_treino.shp edifícios de TREINO: todos os outros, incluindo os das
                         margens que atravessam o limite para um ortofoto de treino
  - <PREFIXO>_margens_teste.shp  edifícios que atravessam o limite da zona de teste

Os edifícios das margens (cortados pelo limite da zona de teste) ficam só no
treino: assim a parte que está do lado do treino é rotulada como edifício e não
como fundo. As tiles de treino ficam dentro da máscara, por isso o modelo nunca
vê a parte que está na zona de teste.
Edifícios que atravessam o limite entre dois ortofotos de TESTE contíguos ficam
no teste, porque estão inteiramente dentro da zona de teste (a seleção é feita
sobre a união dos ortofotos de teste, e não sobre cada polígono isolado).

Os ficheiros originais NÃO são alterados. O Z e todos os atributos são mantidos.

Pode correr:
  - na janela Python do ArcGIS Pro (as camadas finais são abertas no mapa ativo);
  - no Python Command Prompt:
        python "D:\Users\Cap_Almeida\Projeto\60_separar_treino_teste.py"
    (sem mapa: as camadas não são abertas, mas os ficheiros são criados igual).
    Neste caso, fechar antes o ArcGIS Pro ou remover do mapa as camadas de saída,
    para não haver ficheiros bloqueados.
"""
import os
import time
import datetime
import functools
import traceback
import arcpy

# =============================== PARÂMETROS ===============================
EDIFICIOS = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf\a_constr_Limpa.shp"   # TODOS os edifícios
GRELHA = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\Orto\ortos_grelha.shp"          # do analisar_ortos.py
CAMPO_NOME = "NOME"

ORTOS_TESTE = ["126-I-3", "126-II-3", "126-IV-1"]   # nomes do campo NOME

PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf"
PREFIXO = "a_constr"
ADICIONAR_AO_MAPA = True    # só tem efeito dentro do ArcGIS Pro
# ==========================================================================

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False


class Relatorio:
    def __init__(self):
        self.linhas = []

    def __call__(self, txt=""):
        print(txt)
        self.linhas.append(str(txt))

    def gravar(self, caminho):
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas) + "\n")


def mapa_atual():
    """Mapa ativo do ArcGIS Pro, ou None se o script correr no Python Command Prompt."""
    try:
        return arcpy.mp.ArcGISProject("CURRENT").activeMap
    except Exception:
        return None


def _norm(c):
    return os.path.normcase(os.path.splitext(os.path.abspath(c))[0])


def remover_camadas(mapa, caminhos):
    """Remove do mapa as camadas que apontam para as saídas, para não as bloquear."""
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


def contar(lyr):
    return int(arcpy.management.GetCount(lyr)[0])


def descrever_sr(sr):
    codigo = sr.factoryCode if sr.factoryCode else "sem código EPSG"
    datum = sr.GCS.datumName if sr.GCS else "?"
    return f"{sr.name} ({codigo}; datum {datum})"


def validar_entradas(rel):
    """Confirma que as entradas existem e que estão em sistemas de coordenadas compatíveis."""
    for nome, caminho in (("EDIFICIOS", EDIFICIOS), ("GRELHA", GRELHA)):
        if not arcpy.Exists(caminho):
            raise FileNotFoundError(f"{nome} não existe: {caminho}")
    if not os.path.isdir(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA, exist_ok=True)

    campos = [f.name for f in arcpy.ListFields(GRELHA)]
    if CAMPO_NOME not in campos:
        raise ValueError(f"O campo {CAMPO_NOME} não existe na grelha. Campos: {campos}")

    sr_e = arcpy.Describe(EDIFICIOS).spatialReference
    sr_g = arcpy.Describe(GRELHA).spatialReference
    rel("SISTEMAS DE COORDENADAS")
    rel(f"  Edifícios: {descrever_sr(sr_e)}")
    rel(f"  Grelha:    {descrever_sr(sr_g)}")
    if sr_e.name == "Unknown" or sr_g.name == "Unknown":
        raise RuntimeError("Os edifícios ou a grelha não têm sistema de coordenadas definido.")
    datum_e = sr_e.GCS.datumName if sr_e.GCS else None
    datum_g = sr_g.GCS.datumName if sr_g.GCS else None
    if datum_e != datum_g:
        # A grelha é lida já no sistema dos edifícios; com datums diferentes, essa
        # conversão precisa de uma transformação geográfica, ou fica desfasada.
        raise RuntimeError(f"A grelha ({datum_g}) e os edifícios ({datum_e}) estão em datums "
                           "diferentes. Projeta a grelha para o sistema dos edifícios "
                           "(com a transformação adequada) antes de correr este script.")
    if sr_e.name != sr_g.name:
        rel("  AVISO: sistemas diferentes, mas com o mesmo datum: a grelha é convertida "
            "para o sistema dos edifícios.")
    else:
        rel("  OK: mesmo sistema de coordenadas.")
    rel()
    return sr_e


def criar_zona_por_orto(caminho, ortos, nomes, sr):
    """Cria uma feature class com um polígono por ortofoto e o campo NOME."""
    pasta, nome = os.path.split(caminho)
    arcpy.management.CreateFeatureclass(pasta, nome, "POLYGON", spatial_reference=sr)
    arcpy.management.AddField(caminho, CAMPO_NOME, "TEXT", field_length=50)
    with arcpy.da.InsertCursor(caminho, ["SHAPE@", CAMPO_NOME]) as ic:
        for n in nomes:
            ic.insertRow([ortos[n][1], n])


def main(rel):
    t0 = time.time()
    saida_zona = os.path.join(PASTA_SAIDA, "zona_teste.shp")
    saida_mascara = os.path.join(PASTA_SAIDA, "mascara_treino.shp")
    saida_mascara_ortos = os.path.join(PASTA_SAIDA, "mascara_treino_ortos.shp")
    saida_teste = os.path.join(PASTA_SAIDA, f"{PREFIXO}_teste.shp")
    saida_treino = os.path.join(PASTA_SAIDA, f"{PREFIXO}_treino.shp")
    saida_margem = os.path.join(PASTA_SAIDA, f"{PREFIXO}_margens_teste.shp")
    saida_txt = os.path.join(PASTA_SAIDA, "separacao_treino_teste_relatorio.txt")
    saidas = [saida_zona, saida_mascara, saida_mascara_ortos, saida_teste, saida_treino]
    antigas = saidas + [saida_margem]

    rel("=" * 78)
    rel("RELATÓRIO - SEPARAÇÃO DE EDIFÍCIOS EM TREINO E TESTE")
    rel("=" * 78)
    rel(f"Data/hora:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rel(f"ArcGIS Pro:  {arcpy.GetInstallInfo().get('Version', '?')}")
    mapa = mapa_atual()
    rel(f"Execução:    {'janela Python do ArcGIS Pro' if mapa is not None else 'Python Command Prompt (sem mapa)'}")
    rel()
    rel("PARÂMETROS")
    for nome in ("EDIFICIOS", "GRELHA", "CAMPO_NOME", "ORTOS_TESTE", "PASTA_SAIDA", "PREFIXO"):
        rel(f"  {nome:<12} = {globals()[nome]}")
    rel()

    sr = validar_entradas(rel)
    arcpy.env.outputCoordinateSystem = sr

    remover_camadas(mapa, antigas)
    for s in antigas:
        apagar_se_existe(s)

    # --- geometria da carta e da zona de teste
    ortos = {}
    campos_extra = [f.name for f in arcpy.ListFields(GRELHA)]
    usar_n = "N_EDIF" in campos_extra and "ESTRATO" in campos_extra
    cols = [CAMPO_NOME, "SHAPE@"] + (["N_EDIF", "ESTRATO"] if usar_n else [])
    with arcpy.da.SearchCursor(GRELHA, cols, spatial_reference=sr) as sc:
        for row in sc:
            ortos[str(row[0])] = row
    em_falta = [n for n in ORTOS_TESTE if n not in ortos]
    if em_falta:
        raise ValueError(f"Ortofotos não encontrados na grelha: {em_falta}. "
                         f"Disponíveis: {sorted(ortos)}")
    if len(set(ORTOS_TESTE)) != len(ORTOS_TESTE):
        raise ValueError("ORTOS_TESTE tem nomes repetidos.")

    carta = functools.reduce(lambda a, b: a.union(b), [r[1] for r in ortos.values()])
    teste = functools.reduce(lambda a, b: a.union(b), [ortos[n][1] for n in ORTOS_TESTE])
    mascara = carta.difference(teste)
    ortos_treino = sorted(n for n in ortos if n not in ORTOS_TESTE)

    # zona de teste: um polígono por ortofoto, com o campo NOME
    criar_zona_por_orto(saida_zona, ortos, ORTOS_TESTE, sr)
    arcpy.management.CopyFeatures([mascara], saida_mascara)
    # máscara de treino: também uma versão com um polígono por ortofoto e o campo NOME
    criar_zona_por_orto(saida_mascara_ortos, ortos, ortos_treino, sr)

    # união da zona de teste, usada só para as seleções espaciais
    # (assim, um edifício que atravessa dois ortofotos de teste contíguos
    #  continua a contar como "completamente dentro")
    zona_uniao = os.path.join("memory", "zona_teste_uniao")
    apagar_se_existe(zona_uniao)
    arcpy.management.CopyFeatures([teste], zona_uniao)

    # --- seleção dos edifícios
    print("A selecionar edifícios...")
    lyr = arcpy.management.MakeFeatureLayer(EDIFICIOS, "edif_sep")[0]
    n_total = contar(lyr)

    # teste: só edifícios inteiramente dentro da zona de teste
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", zona_uniao,
                                           selection_type="NEW_SELECTION")
    n_teste = contar(lyr)
    arcpy.management.CopyFeatures(lyr, saida_teste)

    # margens: tocam a zona de teste mas não estão inteiramente dentro
    arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", zona_uniao,
                                           selection_type="NEW_SELECTION")
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", zona_uniao,
                                           selection_type="REMOVE_FROM_SELECTION")
    n_margem = contar(lyr)
    if n_margem:
        arcpy.management.CopyFeatures(lyr, saida_margem)

    # treino: todos os que não estão inteiramente dentro da zona de teste
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", zona_uniao,
                                           selection_type="NEW_SELECTION",
                                           invert_spatial_relationship="INVERT")
    n_treino = contar(lyr)
    arcpy.management.CopyFeatures(lyr, saida_treino)
    arcpy.management.Delete(lyr)
    arcpy.management.Delete(zona_uniao)

    # --- relatório
    rel("ORTOFOTOS DE TESTE")
    for n in ORTOS_TESTE:
        r = ortos[n]
        extra = f"  ({r[2]} edifícios, estrato {r[3]})" if usar_n else ""
        rel(f"  {n}{extra}")
    rel()
    rel("ÁREAS")
    rel(f"  Carta ({len(ortos)} ortofotos):   {carta.area / 1e6:10.2f} km2")
    rel(f"  Zona de teste:          {teste.area / 1e6:10.2f} km2  "
        f"({teste.area / carta.area:.1%})")
    for n in ORTOS_TESTE:
        rel(f"    {n:<20} {ortos[n][1].area / 1e6:10.2f} km2")
    rel(f"  Máscara de treino:      {mascara.area / 1e6:10.2f} km2  "
        f"({mascara.area / carta.area:.1%})  -  {len(ortos_treino)} ortofotos")
    for n in ortos_treino:
        rel(f"    {n:<20} {ortos[n][1].area / 1e6:10.2f} km2")
    rel()
    rel("EDIFÍCIOS")
    rel(f"  Total:                              {n_total}")
    rel(f"  TESTE (inteiramente na zona de teste):  {n_teste}  ({n_teste / n_total:.1%})")
    rel(f"  TREINO (todos os restantes):            {n_treino}  ({n_treino / n_total:.1%})")
    rel(f"    dos quais nas margens da zona de teste: {n_margem}  "
        f"(excluídos do teste, incluídos no treino)")
    rel(f"  Verificação: teste + treino = {n_teste + n_treino} "
        f"({'OK' if n_teste + n_treino == n_total else 'DIFERENTE do total!'})")
    rel()
    rel("PRÓXIMOS PASSOS")
    rel(f"  1. criar_rotulos_borda.py com ENTRADA = {saida_treino}")
    rel(f"     e AREA_ESTUDO = {saida_mascara}")
    rel(f"  2. Export Training Data for Deep Learning com Input Mask Polygons = "
        f"{saida_mascara}")
    rel(f"     (usar a máscara num só polígono, e não a {os.path.basename(saida_mascara_ortos)})")
    rel(f"  3. Avaliação: ZONA_TESTE = {saida_zona}, CAMPO_NOME = \"{CAMPO_NOME}\",")
    rel(f"     REFERENCIA = {saida_teste}, MARGENS = {saida_margem}")
    rel()
    rel("FICHEIROS DE SAÍDA")
    for s in saidas + ([saida_margem] if n_margem else []) + [saida_txt]:
        rel(f"  {s}")

    if ADICIONAR_AO_MAPA and mapa is not None:
        abrir = [saida_mascara, saida_zona, saida_treino, saida_teste]
        if n_margem:
            abrir.append(saida_margem)
        for s in abrir:
            mapa.addDataFromPath(s)
        rel(f"Camadas abertas no mapa '{mapa.name}': "
            f"{', '.join(os.path.basename(s) for s in abrir)}")
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
        _rel.gravar(os.path.join(PASTA_SAIDA, "separacao_treino_teste_relatorio.txt"))
        raise