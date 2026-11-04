import arcpy
import os

pasta_tifs = r"D:\Users\Cap Almeida\Projeto\M888_Ortos\126_Ortofotos"
pasta_saida = r"D:\Users\Cap Almeida\Projeto\126\Ortomosaico"

# lista todos os .tif da pasta
lista_tifs = [os.path.join(pasta_tifs, f) for f in os.listdir(pasta_tifs) if f.lower().endswith(".tif")]

print(f"{len(lista_tifs)} tifs encontrados:")
for f in lista_tifs:
    print(" -", f)

for f in lista_tifs:
    r = arcpy.Raster(f)
    print(os.path.basename(f), r.bandCount, r.pixelType, r.spatialReference.name, r.meanCellWidth)

# usa o primeiro tif da lista como referência para Snap Raster
original = lista_tifs[0]

# sistema de referência
sr = arcpy.Describe(original).spatialReference

print("Nome:", sr.name)
print("Código:", sr.factoryCode)
print("Datum:", sr.GCS.datumName)
print("Projeção:", sr.projectionName)
print("Meridiano central:", sr.centralMeridian)
print("Latitude de origem:", sr.latitudeOfOrigin)
print("Falso Este / Norte:", sr.falseEasting, sr.falseNorthing)
print("Fator de escala:", sr.scaleFactor)

for f in lista_tifs:
    e = arcpy.Describe(f).extent
    print(os.path.basename(f), round(e.XMin, 2), round(e.YMin, 2), round(e.XMax, 2), round(e.YMax, 2))

arcpy.env.snapRaster = original
arcpy.env.cellSize = 0.5

arcpy.management.MosaicToNewRaster(
    input_rasters=lista_tifs,
    output_location=pasta_saida,
    raster_dataset_name_with_extension="126.tif", #nome de saida
    coordinate_system_for_the_raster=sr,
    pixel_type="8_BIT_UNSIGNED",
    number_of_bands=4,
    mosaic_method="LAST"
)

print("Mosaico criado com sucesso.")
