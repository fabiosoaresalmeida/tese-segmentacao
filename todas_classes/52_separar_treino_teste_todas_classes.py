# -*- coding: utf-8 -*-
r"""
52_separar_treino_teste_todas_classes.py  (ArcGIS Pro 3.6)

Separação em treino e teste para o modelo com TODAS AS CLASSES (construções,
rede viária, água, vegetação e sem representação), a partir dos rótulos do
51_rotulos_todas_classes.py. Usa as MESMAS ortos de teste da ocupação do solo.
Adaptado do 22_separar_treino_teste_ocupacao.py.

1. Cria a grelha dos ortofotos (ortos_grelha.shp) a partir da extensão de cada
   ortofoto da pasta ORTO_FOLDER, com o campo NOME (nome do ficheiro sem extensão).
2. A partir dos ortofotos escolhidos para teste cria:
  - zona_teste.shp          área de teste, com UM POLÍGONO POR ORTOFOTO e o campo NOME
  - mascara_treino.shp      resto da carta, num só polígono (usar em Input Mask
                            Polygons do Export Training Data)
  - mascara_treino_ortos.shp  a mesma área, um polígono por ortofoto de treino
                            (para análises por ortofoto; não usar na exportação)
  - rotulos_teste.shp       rótulos RECORTADOS pela zona de teste
                            (referência para a avaliação)
  - rotulos_treino.shp      rótulos RECORTADOS pela máscara de treino
                            (usar no Export Training Data)

Todos os rótulos (incluindo construções e vias) são RECORTADOS pelo limite das
ortofotos (e não selecionados por "completamente dentro"): cada parte fica do seu
lado, e nenhum píxel da zona de teste entra no treino. Uma construção cortada pelo
limite fica em duas partes; na avaliação por objeto dos edifícios ter isto em conta.

O sem representação tem polígonos próprios (classe 9). A "classe 0" do relatório
é só a área residual sem nenhum polígono, que deve ser praticamente nula.
Não há classe Ignorar (8): no treino NÃO usar ignore_classes.

Os ficheiros originais NÃO são alterados.

Pode correr:
  - na janela Python do ArcGIS Pro (as camadas finais são abertas no mapa ativo);
  - no Python Command Prompt:
        python "D:\Users\Cap_Almeida\Projeto\Python\4_TodasClasses\52_separar_treino_teste_todas_classes.py"
    Neste caso, fechar antes o ArcGIS Pro ou remover do mapa as camadas de saída.
"""
import os
import time
import datetime
import functools
import traceback
from collections import defaultdict
import arcpy

# =============================== PARÂMETROS ===============================
ORTO_FOLDER = r"D:\Users\Cap_Almeida\Projeto\M888_Ortos\126_Ortofotos"   # ortofotos individuais
ORTO_EXT = (".tif", ".tiff", ".jp2", ".ecw", ".img")

ROTULOS = r"D:\Users\Cap_Almeida\Projeto\126\126_TodasClasses\51_rotulos_todas_classes\rotulos_todas_classes.shp"  # do 51
CAMPO_CLASSE = "classe"
CAMPO_NOME_CLASSE = "nome"
IGNORE = None                               # sem classe Ignorar neste modelo (era 8 na ocupação do solo)

CAMPO_NOME = "NOME"
ORTOS_TESTE = ["126-I-3", "126-II-4", "126-IV-3"]   # as mesmas da ocupação do solo

PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_TodasClasses\52_separacao"
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


def descrever_sr(sr):
    codigo = sr.factoryCode if sr.factoryCode else "sem código EPSG"
    datum = sr.GCS.datumName if sr.GCS else "?"
    return f"{sr.name} ({codigo}; datum {datum})"


def retangulo(ext, sr):
    pts = [(ext.XMin, ext.YMin), (ext.XMin, ext.YMax), (ext.XMax, ext.YMax),
           (ext.XMax, ext.YMin), (ext.XMin, ext.YMin)]
    return arcpy.Polygon(arcpy.Array([arcpy.Point(x, y) for x, y in pts]), sr)


# ------------------------------------------------------------- grelha
def criar_grelha(caminho, rel):
    """Grelha com um polígono por ortofoto (extensão do raster) e o campo NOME."""
    rasters = sorted(f for f in os.listdir(ORTO_FOLDER) if f.lower().endswith(ORTO_EXT))
    if not rasters:
        raise FileNotFoundError(f"Nenhum ortofoto encontrado em {ORTO_FOLDER}")
    sr = arcpy.Describe(os.path.join(ORTO_FOLDER, rasters[0])).spatialReference
    pasta, nome = os.path.split(caminho)
    arcpy.management.CreateFeatureclass(pasta, nome, "POLYGON", spatial_reference=sr)
    arcpy.management.AddField(caminho, CAMPO_NOME, "TEXT", field_length=50)
    arcpy.management.AddField(caminho, "AREA_HA", "DOUBLE")
    geoms = []
    with arcpy.da.InsertCursor(caminho, ["SHAPE@", CAMPO_NOME, "AREA_HA"]) as ic:
        for rf in rasters:
            d = arcpy.Describe(os.path.join(ORTO_FOLDER, rf))
            if d.spatialReference.name != sr.name:
                rel(f"  AVISO: {rf} tem outro sistema ({d.spatialReference.name})")
            g = retangulo(d.extent, sr)
            geoms.append(g)
            ic.insertRow([g, os.path.splitext(rf)[0], g.area / 10000])
    soma = sum(g.area for g in geoms)
    uniao = functools.reduce(lambda a, b: a.union(b), geoms).area
    rel("GRELHA DOS ORTOFOTOS")
    rel(f"  {len(rasters)} ortofotos | sistema: {descrever_sr(sr)}")
    if abs(soma - uniao) > 1.0:
        rel(f"  AVISO: os ortofotos sobrepõem-se ({(soma - uniao) / 10000:.2f} ha). "
            "A área sobreposta com o teste fica na zona de teste e sai da máscara de treino.")
    else:
        rel("  OK: sem sobreposições entre ortofotos.")
    rel()
    return sr


def validar_rotulos(sr_grelha, rel):
    if not arcpy.Exists(ROTULOS):
        raise FileNotFoundError(f"ROTULOS não existe: {ROTULOS}")
    campos = [f.name.lower() for f in arcpy.ListFields(ROTULOS)]
    if CAMPO_CLASSE.lower() not in campos:
        raise ValueError(f"O campo {CAMPO_CLASSE} não existe em {ROTULOS}")
    sr_r = arcpy.Describe(ROTULOS).spatialReference
    rel("SISTEMAS DE COORDENADAS")
    rel(f"  Rótulos: {descrever_sr(sr_r)}")
    rel(f"  Grelha:  {descrever_sr(sr_grelha)}")
    if sr_r.name == "Unknown":
        raise RuntimeError("Os rótulos não têm sistema de coordenadas definido.")
    dr = sr_r.GCS.datumName if sr_r.GCS else None
    dg = sr_grelha.GCS.datumName if sr_grelha.GCS else None
    if dr != dg:
        raise RuntimeError(f"Rótulos ({dr}) e grelha ({dg}) em datums diferentes. "
                           "Os rótulos devem estar no sistema da orto (51_rotulos_todas_classes.py).")
    rel("  OK: mesmo sistema de coordenadas." if sr_r.name == sr_grelha.name else
        "  AVISO: sistemas diferentes com o mesmo datum.")
    rel()


def criar_zona_por_orto(caminho, ortos, nomes, sr):
    pasta, nome = os.path.split(caminho)
    arcpy.management.CreateFeatureclass(pasta, nome, "POLYGON", spatial_reference=sr)
    arcpy.management.AddField(caminho, CAMPO_NOME, "TEXT", field_length=50)
    with arcpy.da.InsertCursor(caminho, ["SHAPE@", CAMPO_NOME]) as ic:
        for n in nomes:
            ic.insertRow([ortos[n], n])


def areas_por_classe(fc):
    campos = [CAMPO_CLASSE, "SHAPE@AREA"]
    tem_nome = CAMPO_NOME_CLASSE.lower() in [f.name.lower() for f in arcpy.ListFields(fc)]
    if tem_nome:
        campos.append(CAMPO_NOME_CLASSE)
    area, nomes, n = defaultdict(float), {}, defaultdict(int)
    with arcpy.da.SearchCursor(fc, campos) as sc:
        for row in sc:
            area[row[0]] += row[1] or 0.0
            n[row[0]] += 1
            if tem_nome:
                nomes[row[0]] = row[2]
    return area, nomes, n


def main(rel):
    t0 = time.time()
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    saida_grelha = os.path.join(PASTA_SAIDA, "ortos_grelha.shp")
    saida_zona = os.path.join(PASTA_SAIDA, "zona_teste.shp")
    saida_mascara = os.path.join(PASTA_SAIDA, "mascara_treino.shp")
    saida_mascara_ortos = os.path.join(PASTA_SAIDA, "mascara_treino_ortos.shp")
    saida_teste = os.path.join(PASTA_SAIDA, "rotulos_teste.shp")
    saida_treino = os.path.join(PASTA_SAIDA, "rotulos_treino.shp")
    saida_txt = os.path.join(PASTA_SAIDA, "separacao_treino_teste_relatorio.txt")
    saidas = [saida_grelha, saida_zona, saida_mascara, saida_mascara_ortos,
              saida_teste, saida_treino]

    rel("=" * 78)
    rel("RELATÓRIO - SEPARAÇÃO EM TREINO E TESTE - TODAS AS CLASSES")
    rel("=" * 78)
    rel(f"Data/hora:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rel(f"ArcGIS Pro:  {arcpy.GetInstallInfo().get('Version', '?')}")
    mapa = mapa_atual()
    rel(f"Execução:    {'janela Python do ArcGIS Pro' if mapa is not None else 'Python Command Prompt (sem mapa)'}")
    rel()
    rel("PARÂMETROS")
    for nome in ("ORTO_FOLDER", "ROTULOS", "CAMPO_CLASSE", "IGNORE", "ORTOS_TESTE", "PASTA_SAIDA"):
        rel(f"  {nome:<12} = {globals()[nome]}")
    rel()

    remover_camadas(mapa, saidas)
    for s in saidas:
        apagar_se_existe(s)

    # --- grelha
    sr = criar_grelha(saida_grelha, rel)
    validar_rotulos(sr, rel)
    arcpy.env.outputCoordinateSystem = sr

    ortos = {str(n): g for n, g in arcpy.da.SearchCursor(saida_grelha, [CAMPO_NOME, "SHAPE@"])}
    em_falta = [n for n in ORTOS_TESTE if n not in ortos]
    if em_falta:
        raise ValueError(f"Ortofotos não encontrados na grelha: {em_falta}. "
                         f"Disponíveis: {sorted(ortos)}")
    if len(set(ORTOS_TESTE)) != len(ORTOS_TESTE):
        raise ValueError("ORTOS_TESTE tem nomes repetidos.")

    carta = functools.reduce(lambda a, b: a.union(b), list(ortos.values()))
    teste = functools.reduce(lambda a, b: a.union(b), [ortos[n] for n in ORTOS_TESTE])
    mascara = carta.difference(teste)
    ortos_treino = sorted(n for n in ortos if n not in ORTOS_TESTE)

    criar_zona_por_orto(saida_zona, ortos, ORTOS_TESTE, sr)
    arcpy.management.CopyFeatures([mascara], saida_mascara)
    criar_zona_por_orto(saida_mascara_ortos, ortos, ortos_treino, sr)

    # --- recortar os rótulos
    print("A recortar os rótulos...")
    zona_uniao = os.path.join("memory", "zona_teste_uniao")
    apagar_se_existe(zona_uniao)
    arcpy.management.CopyFeatures([teste], zona_uniao)
    arcpy.analysis.PairwiseClip(ROTULOS, zona_uniao, saida_teste)
    arcpy.analysis.PairwiseClip(ROTULOS, saida_mascara, saida_treino)
    arcpy.management.Delete(zona_uniao)

    # --- estatísticas
    # (os totais são treino + teste, para contar só o que está dentro da carta)
    a_te, nomes_te, n_te = areas_por_classe(saida_teste)
    a_tr, nomes_tr, n_tr = areas_por_classe(saida_treino)
    nomes = {**nomes_tr, **nomes_te}

    # classe 0 = área sem polígonos
    a_te[0] = teste.area - sum(a_te.values())
    a_tr[0] = mascara.area - sum(a_tr.values())
    nomes[0] = "Sem_poligono (residual)"
    a_tot = {c: a_tr.get(c, 0.0) + a_te.get(c, 0.0) for c in set(a_tr) | set(a_te)}

    rel("ÁREAS")
    rel(f"  Carta ({len(ortos)} ortofotos):   {carta.area / 1e6:10.2f} km2")
    rel(f"  Zona de teste:          {teste.area / 1e6:10.2f} km2  ({teste.area / carta.area:.1%})")
    for n in ORTOS_TESTE:
        rel(f"    {n:<20} {ortos[n].area / 1e6:10.2f} km2")
    rel(f"  Máscara de treino:      {mascara.area / 1e6:10.2f} km2  "
        f"({mascara.area / carta.area:.1%})  -  {len(ortos_treino)} ortofotos")
    rel(f"    {', '.join(ortos_treino)}")
    rel()

    rel("CLASSES (área em ha; % = parte do total da classe que fica no teste)")
    rel(f"  {'classe':>6}  {'nome':<20} {'total':>10} {'treino':>10} {'teste':>10} {'% teste':>8}  pol. teste")
    alertas = []
    for c in sorted(a_tot):
        tot, tr, te = a_tot[c] / 1e4, a_tr.get(c, 0) / 1e4, a_te.get(c, 0) / 1e4
        pct = 100 * te / tot if tot else 0
        rel(f"  {c:>6}  {nomes.get(c, '?'):<20} {tot:>10.1f} {tr:>10.1f} {te:>10.1f} "
            f"{pct:>7.1f}%  {n_te.get(c, '-') if c != 0 else '-'}")
        if c not in (IGNORE, 0) and tot >= 1 and (pct < 10 or pct > 30):
            alertas.append(f"{nomes.get(c, c)}: {pct:.1f}% no teste")
    fora = sum(areas_por_classe(ROTULOS)[0].values()) - (sum(a_tot.values()) - a_tot[0])
    rel(f"  Rótulos fora da carta (não usados): {fora / 1e4:.2f} ha")
    if IGNORE is not None:
        rel(f"  (classe {IGNORE} = Ignorar: não conta para a loss nem para as métricas)")
    if a_tot.get(0, 0) > 0.01 * sum(a_tot.values()):
        rel("  AVISO: a área residual sem polígono (classe 0) passa de 1 % - verificar o 51")
    if alertas:
        rel("  AVISO - classes fora do intervalo 10-30% no teste: " + "; ".join(alertas))
    rel()

    rel("PRÓXIMOS PASSOS")
    rel("  1. Export Training Data For Deep Learning (Classified Tiles):")
    rel(f"     Input Feature Class = {saida_treino}  |  Class Value Field = {CAMPO_CLASSE}")
    rel(f"     Input Mask Polygons = {saida_mascara}")
    rel(f"     (usar a máscara num só polígono, e não a {os.path.basename(saida_mascara_ortos)})")
    if IGNORE is None:
        rel("  2. Treino: class_balancing=True, SEM ignore_classes")
    else:
        rel(f"  2. Treino: ignore_classes=[{IGNORE}] e class_balancing=True")
    rel(f"  3. Avaliação: ZONA_TESTE = {saida_zona}, CAMPO_NOME = \"{CAMPO_NOME}\",")
    rel(f"     REFERENCIA = {saida_teste}")
    rel()
    rel("FICHEIROS DE SAÍDA")
    for s in saidas + [saida_txt]:
        rel(f"  {s}")

    if ADICIONAR_AO_MAPA and mapa is not None:
        for s in [saida_grelha, saida_mascara, saida_zona, saida_treino, saida_teste]:
            mapa.addDataFromPath(s)
        rel(f"Camadas abertas no mapa '{mapa.name}'")
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