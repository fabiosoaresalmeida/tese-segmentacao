# -*- coding: utf-8 -*-
"""
41b - Etiquetas binárias para o Multi-Task Road Extractor (modelo só binário).
Parte das etiquetas multiclasse criadas pelo 41 e gera duas shapes:

  etiquetas_rede_viaria_binario.shp : Via pavimentada (1) + Via não pavimentada (2) -> 1
  etiquetas_ferrovia_binario.shp    : Ferrovia (3) -> 1

Em cada shape, tudo o resto fica como fundo.
Executar na janela Python ou num Notebook do ArcGIS Pro 3.6.
"""
import os
import arcpy

arcpy.env.overwriteOutput = True

PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"
ENTRADA = os.path.join(PASTA_SAIDA, "etiquetas_vias.shp")       # criado pelo 41

# nome da shape -> (classes de origem, nome da classe binária)
BINARIOS = {
    "etiquetas_rede_viaria_binario.shp": ((1, 2), "Rede_viaria"),
    "etiquetas_ferrovia_binario.shp":    ((3,), "Ferrovia"),
}

arcpy.env.outputCoordinateSystem = arcpy.Describe(ENTRADA).spatialReference

for nome, (classes, classname) in BINARIOS.items():
    saida = os.path.join(PASTA_SAIDA, nome)
    lista = ", ".join(str(c) for c in classes)

    sel, diss = "memory/sel", "memory/diss"
    arcpy.analysis.Select(ENTRADA, sel, f"Classvalue IN ({lista})")
    n = int(arcpy.management.GetCount(sel)[0])
    if n == 0:
        print(f"{nome}: sem elementos das classes {lista}, ignorado")
        continue
    arcpy.management.Dissolve(sel, diss)
    arcpy.management.MultipartToSinglepart(diss, saida)

    arcpy.management.AddField(saida, "Classvalue", "SHORT")
    arcpy.management.AddField(saida, "Classname", "TEXT", field_length=30)
    area, partes = 0.0, 0
    with arcpy.da.UpdateCursor(saida, ["Classvalue", "Classname", "SHAPE@AREA"]) as cur:
        for row in cur:
            row[0], row[1] = 1, classname
            area += row[2]
            partes += 1
            cur.updateRow(row)

    arcpy.management.Delete("memory")
    print(f"{classname}: {partes} polígonos, {area / 10000:.1f} ha -> {saida}")