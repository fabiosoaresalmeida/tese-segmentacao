# -*- coding: utf-8 -*-
"""
SHAPE de rótulos para o modelo com TODAS AS CLASSES (construções + rede viária +
água + vegetação + sem representação), sem sobreposições, pronta para o
Export Training Data (Classified Tiles, campo 'classe').

Objetivo: comparar um modelo único multiclasse com os modelos separados
(edifícios, rede viária, ocupação do solo) e ver se há ganhos em separar as classes.

Classes (campo 'classe'). Os códigos 1-7 e 9 são os MESMOS do 21_rotulos_ocupacao.py,
para a avaliação poder ser comparada diretamente com o modelo de ocupação do solo:
   1  Vinha
   2  Pomar_vinha
   3  Pomar
   4  Arvoredo_denso
   5  Arvoredo_esparso
   6  Mato_denso
   7  Água (A_Albufeira, A_Rio_de_duas_margens, A_Lagoa)
   8  Ignorar -> NÃO é usado neste modelo (só aparece se VIAS_COMO = "ignorar")
   9  Sem representação (sobretudo solo nu): polígonos explícitos (não se usa o 0,
      que alguns modelos do ArcGIS tratam como NoData)
  10  Construção (a_constr, todas as construções numa só classe)
  11  Via pavimentada
  12  Via não pavimentada
  13  Ferrovia

Precedência nas sobreposições (a de cima ganha):
  Construção > Via pavimentada > Ferrovia > Via não pavimentada > Água > Vegetação > Sem repr.
  - Construção x via: ganha a construção (o contorno restituído é mais exato do que
    o buffer da via, que é uma largura nominal)
  - Via x água (pontes): ganha a via (o tabuleiro é via; no modelo de ocupação era Ignorar)
  - Água x vegetação: ganha a água (como no 21)

No treino: SEM ignore_classes (não há classe 8), class_balancing=True.
Correr com o python do ambiente arcgispro-py3.
"""
import os
import csv
from collections import defaultdict
import arcpy

# ============================ CONFIGURAÇÃO ============================
MOSAIC = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"

VEGETACAO = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_vegetacao.shp"
HIDRO_POL = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_hidro.shp"
CONSTR    = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_constr.shp"
ESTRADAS  = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\126_2018_4\l_vias.shp"   # eixos
CLASS_FIELD = "source"
ESTRADAS_CLASS_FIELD = "source"

# "classes"  -> vias como classes 11/12/13 (modelo com todas as classes)
# "ignorar"  -> vias como Ignorar (8), só para testar construções + ocupação do solo
VIAS_COMO = "classes"

# LARGURA TOTAL em metros por 'source' (as mesmas do 41 e do 21); buffer = largura/2
ESTRADAS_LARGURA = {
    "Auto_estrada": 11.0,
    "Estrada_larga": 7.0,
    "Estrada_estreita": 4.0,
    "Arruamento": 5.0,
    "Acesso_auto": 3.0,
    "Caminho_carreteiro": 2.5,
    "Caminho_a_pe_posto": 1.5,
    "Caminho_de_ferro_via_larga_unica": 3.5,
}
ESTRADAS_LARGURA_DEFEITO = 3.0

# 'source' -> classe da via. CONFIRMAR que é a MESMA divisão do 41 (etiquetas_vias.shp),
# em especial o Acesso_auto (piso duro / piso mole). 'source' fora do mapa -> excluído.
VIAS_MAP = {
    "Auto_estrada": 11,
    "Estrada_larga": 11,
    "Estrada_estreita": 11,
    "Arruamento": 11,
    "Acesso_auto": 12,
    "Caminho_carreteiro": 12,
    "Caminho_a_pe_posto": 12,
    "Caminho_de_ferro_via_larga_unica": 13,
}

OUT_FOLDER = r"D:\Users\Cap_Almeida\Projeto\126\126_TodasClasses\51_rotulos_todas_classes"
GDB_NAME = "rotulos.gdb"
OUT_SHP = "rotulos_todas_classes.shp"

VEG_MAP = {
    "A_Vinha": 1,
    "A_Pomar_vinha": 2,
    "A_Pomar": 3,
    "A_Arvoredo_denso": 4,
    "A_Arvoredo_esparso": 5,
    "A_Mato_denso_ou_arbustos": 6,
}
AGUA_MAP = {"A_Albufeira": 7, "A_Rio_de_duas_margens": 7, "A_Lagoa": 7}
IGNORE = 8
SEM_REPR = 9
CONSTRUCAO = 10

NOMES = {1: "Vinha", 2: "Pomar_vinha", 3: "Pomar", 4: "Arvoredo_denso",
         5: "Arvoredo_esparso", 6: "Mato_denso", 7: "Agua", IGNORE: "Ignorar",
         SEM_REPR: "Sem_representacao", CONSTRUCAO: "Construcao",
         11: "Via_pavimentada", 12: "Via_nao_pavimentada", 13: "Ferrovia"}

# Ordem de precedência (primeiro = ganha nas sobreposições)
PRECEDENCIA = [CONSTRUCAO, 11, 13, 12, IGNORE, 7, 1, 2, 3, 4, 5, 6]
# =====================================================================

arcpy.env.overwriteOutput = True


def log(msg):
    print(msg)


def find_field(fc, wanted):
    for f in arcpy.ListFields(fc):
        if f.name.lower() == wanted.lower():
            return f.name
    raise RuntimeError(f"Campo '{wanted}' não encontrado em {fc}")


def sem_prefixo(c):
    """Valor de 'source' sem o prefixo L_/A_ (os eixos podem vir com ou sem ele)."""
    c = "" if c is None else str(c)
    return c[2:] if c[:2] in ("L_", "A_") else c


def largura_de(c):
    return ESTRADAS_LARGURA.get(c, ESTRADAS_LARGURA.get(sem_prefixo(c)))


def prepare(gdb, name, path, sr):
    d = arcpy.Describe(path)
    out = os.path.join(gdb, name)
    if d.spatialReference.name == sr.name:
        arcpy.management.CopyFeatures(path, out)
    else:
        trans = arcpy.ListTransformations(d.spatialReference, sr, d.extent)
        t = trans[0] if trans else ""
        log(f"{name}: {d.spatialReference.name} -> {sr.name} | transformação: {t or 'nenhuma'}")
        arcpy.management.Project(path, out, sr, t)
    arcpy.management.RepairGeometry(out)
    return out


def set_class(fc, mapping=None, constant=None, field=CLASS_FIELD):
    """Campo 'classe'. Com mapping, apaga os elementos com source fora do mapa."""
    arcpy.management.AddField(fc, "classe", "SHORT")
    fields = ["classe"] + ([find_field(fc, field)] if mapping is not None else [])
    fora = defaultdict(int)
    with arcpy.da.UpdateCursor(fc, fields) as cur:
        for row in cur:
            if mapping is None:
                cur.updateRow([constant])
                continue
            v = mapping.get(row[1], mapping.get(sem_prefixo(row[1])))
            if v is None:
                fora[row[1]] += 1
                cur.deleteRow()
            else:
                cur.updateRow([v, row[1]])
    for k, n in fora.items():
        log(f"  AVISO: '{k}' ({n} elementos) sem classe no mapeamento -> excluído")


def buffer_estradas(gdb, fc):
    """Buffer dos eixos com a LARGURA TOTAL de cada 'source' (distância = largura / 2)."""
    src = find_field(fc, ESTRADAS_CLASS_FIELD)
    arcpy.management.AddField(fc, "buf_m", "DOUBLE")
    sem = set()
    with arcpy.da.UpdateCursor(fc, [src, "buf_m"]) as cur:
        for c, _ in cur:
            w = largura_de(c)
            if w is None:
                sem.add(c)
                w = ESTRADAS_LARGURA_DEFEITO
            cur.updateRow([c, w / 2.0])
    if sem:
        log(f"  AVISO: vias sem largura definida (usa {ESTRADAS_LARGURA_DEFEITO} m): {sorted(sem)}")
    out = os.path.join(gdb, "vias_buffer")
    arcpy.analysis.PairwiseBuffer(fc, out, "buf_m", "NONE")
    return out


def erase(in_fc, erase_fc, out_fc):
    try:
        arcpy.analysis.PairwiseErase(in_fc, erase_fc, out_fc)
    except Exception as e:
        log(f"  Pairwise Erase falhou ({e}); a tentar Erase")
        arcpy.analysis.Erase(in_fc, erase_fc, out_fc)
    return out_fc


def keep_fields(fc, keep=("classe",)):
    drop = [f.name for f in arcpy.ListFields(fc)
            if not f.required and f.name.lower() not in [k.lower() for k in keep]]
    if drop:
        arcpy.management.DeleteField(fc, drop)


def n_feat(fc):
    return int(arcpy.management.GetCount(fc)[0])


def main():
    if VIAS_COMO not in ("classes", "ignorar"):
        raise ValueError("VIAS_COMO tem de ser 'classes' ou 'ignorar'")
    os.makedirs(OUT_FOLDER, exist_ok=True)
    gdb = os.path.join(OUT_FOLDER, GDB_NAME)
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(OUT_FOLDER, GDB_NAME)

    if not arcpy.Exists(MOSAIC):
        raise RuntimeError(f"MOSAIC não encontrado: {MOSAIC}")
    sr = arcpy.Raster(MOSAIC).spatialReference
    arcpy.env.outputCoordinateSystem = sr
    log(f"Sistema de referência: {sr.name} | vias como: {VIAS_COMO}")

    # --- preparar e classificar
    veg = prepare(gdb, "veg", VEGETACAO, sr)
    set_class(veg, VEG_MAP)
    agua = prepare(gdb, "agua", HIDRO_POL, sr)
    set_class(agua, AGUA_MAP)
    constr = prepare(gdb, "constr", CONSTR, sr)
    set_class(constr, constant=CONSTRUCAO)

    eixos = prepare(gdb, "vias_eixos", ESTRADAS, sr)
    if arcpy.Describe(eixos).shapeType != "Polyline":
        raise RuntimeError("ESTRADAS tem de ser a shape de eixos (linhas)")
    if VIAS_COMO == "classes":
        set_class(eixos, VIAS_MAP, field=ESTRADAS_CLASS_FIELD)   # antes do buffer
    vias = buffer_estradas(gdb, eixos)
    if VIAS_COMO == "ignorar":
        set_class(vias, constant=IGNORE)      # o buffer mantém o campo 'classe' no modo "classes"

    for fc in (veg, agua, constr, vias):
        keep_fields(fc)

    # --- uma camada dissolvida por classe (sem sobreposições dentro da classe)
    todas_m = os.path.join(gdb, "todas_merge")
    arcpy.management.Merge([constr, vias, agua, veg], todas_m)
    todas_d = os.path.join(gdb, "todas_dissolve_classe")
    arcpy.analysis.PairwiseDissolve(todas_m, todas_d, "classe", multi_part="MULTI_PART")
    presentes = sorted({c for (c,) in arcpy.da.SearchCursor(todas_d, ["classe"])})
    log(f"Classes presentes: {presentes}")

    # --- precedência: cada classe perde para a união das que estão acima dela
    log("A recortar por precedência...")
    partes = []
    acima = None          # união dissolvida das classes já processadas
    ordem = [c for c in PRECEDENCIA if c in presentes] + \
            [c for c in presentes if c not in PRECEDENCIA]
    for c in ordem:
        lyr = arcpy.management.MakeFeatureLayer(todas_d, f"lyr_{c}", f"classe = {c}")
        cls_fc = os.path.join(gdb, f"c{c}_orig")
        arcpy.management.CopyFeatures(lyr, cls_fc)
        arcpy.management.Delete(lyr)
        if acima is None:
            ok = cls_fc
        else:
            ok = erase(cls_fc, acima, os.path.join(gdb, f"c{c}_ok"))
        if n_feat(ok):
            partes.append(ok)
        # atualizar a união das classes acima
        novo = os.path.join(gdb, f"acima_{c}")
        if acima is None:
            arcpy.analysis.PairwiseDissolve(cls_fc, novo, multi_part="MULTI_PART")
        else:
            m = os.path.join(gdb, f"acima_{c}_m")
            arcpy.management.Merge([acima, cls_fc], m)
            arcpy.analysis.PairwiseDissolve(m, novo, multi_part="MULTI_PART")
        acima = novo
        log(f"  {c:>3} {NOMES.get(c, '?'):<20} ok")

    # --- sem representação (9): área do mosaico sem nenhum outro rótulo
    log("A criar os polígonos de sem representação...")
    ext = arcpy.Raster(MOSAIC).extent
    folha = os.path.join(gdb, "folha")
    arcpy.management.CreateFeatureclass(gdb, "folha", "POLYGON", spatial_reference=sr)
    pts = [(ext.XMin, ext.YMin), (ext.XMin, ext.YMax), (ext.XMax, ext.YMax),
           (ext.XMax, ext.YMin), (ext.XMin, ext.YMin)]
    with arcpy.da.InsertCursor(folha, ["SHAPE@"]) as cur:
        cur.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(x, y) for x, y in pts]), sr)])
    sem = erase(folha, acima, os.path.join(gdb, "sem_representacao"))
    set_class(sem, constant=SEM_REPR)
    keep_fields(sem)
    partes.append(sem)

    # --- juntar
    final_m = os.path.join(gdb, "rotulos_todas")
    arcpy.management.Merge(partes, final_m)
    final_gdb = final_m + "_sp"
    arcpy.management.MultipartToSinglepart(final_m, final_gdb)
    keep_fields(final_gdb)
    arcpy.management.AddField(final_gdb, "nome", "TEXT", field_length=24)
    with arcpy.da.UpdateCursor(final_gdb, ["classe", "nome"]) as cur:
        for c, _ in cur:
            cur.updateRow([c, NOMES.get(c, "?")])

    # --- verificação: não pode haver sobreposições
    chk = os.path.join(gdb, "verif_sobreposicoes")
    arcpy.analysis.CountOverlappingFeatures(final_gdb, chk, 2)
    area_sob = sum(a or 0 for (a,) in arcpy.da.SearchCursor(chk, ["SHAPE@AREA"]))
    if area_sob > 1.0:
        log(f"AVISO: ainda há {area_sob:.1f} m² de sobreposições (ver {chk})")
    else:
        log("Sem sobreposições.")

    # --- exportar shapefile
    out_shp = os.path.join(OUT_FOLDER, OUT_SHP)
    if arcpy.Exists(out_shp):
        arcpy.management.Delete(out_shp)
    arcpy.conversion.ExportFeatures(final_gdb, out_shp)

    # --- resumo
    agg = defaultdict(lambda: [0, 0.0])
    for c, a in arcpy.da.SearchCursor(final_gdb, ["classe", "SHAPE@AREA"]):
        agg[c][0] += 1
        agg[c][1] += a or 0.0
    total = sum(a for _, a in agg.values()) or 1.0
    csv_path = os.path.join(OUT_FOLDER, "resumo_rotulos.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["classe", "nome", "n_poligonos", "area_ha", "pct_area"])
        log("\nclasse  nome                  polígonos   área (ha)   % área")
        for c in sorted(agg):
            n, a = agg[c]
            w.writerow([c, NOMES.get(c, "?"), n, round(a / 10000, 2), round(100 * a / total, 2)])
            log(f"{c:>6}  {NOMES.get(c, '?'):<20} {n:>9} {a / 10000:>11.1f} {100 * a / total:>8.2f}")
    log(f"\nShape: {out_shp}\nResumo: {csv_path}")
    if VIAS_COMO == "classes":
        log("Treino: SEM ignore_classes (não há classe 8); class_balancing=True.")


if __name__ == "__main__":
    main()
