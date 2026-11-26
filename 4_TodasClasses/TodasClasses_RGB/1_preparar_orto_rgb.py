# -*- coding: utf-8 -*-
r"""
1_preparar_orto_rgb.py - confirma (ou cria, se faltar) o ortomosaico RGB da 126:

  126_RGB = bandas 1-3 do 126.tif (8 bits, mesma grelha, pixel e sistema)

E o mesmo ficheiro do 6_RGB_vs_NIR e do MTRE_RGB. Se ja existir, so o verifica:
  - 3 bandas de 8 bits, mesmo pixel, sistema e extensao do 126.tif;
  - bandas 1-3 iguais as do 126.tif numa amostra central de 500 x 500 px.

Correr no Python Command Prompt do ArcGIS Pro:
    python 1_preparar_orto_rgb.py
"""
import os
import time

print("a carregar o arcpy...", flush=True)
import arcpy
import numpy as np

import comum_todas_rgb as C


def criar_rgb(src, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    bandas = ";".join(os.path.join(src, f"Band_{i}") for i in (1, 2, 3))
    with arcpy.EnvManager(snapRaster=src, cellSize=src, extent=src, outputCoordinateSystem=src):
        arcpy.management.CompositeBands(bandas, out)
    print(f"RGB criado em {(time.time() - t0) / 60:.1f} min", flush=True)


def verificar(src, out):
    r4, r3 = arcpy.Raster(src), arcpy.Raster(out)
    print(f"   126.tif : {r4.bandCount} bandas | {r4.pixelType} | {r4.width} x {r4.height} | "
          f"pixel {r4.meanCellWidth} | {r4.spatialReference.name}")
    print(f"   126 RGB : {r3.bandCount} bandas | {r3.pixelType} | {r3.width} x {r3.height} | "
          f"pixel {r3.meanCellWidth} | {r3.spatialReference.name}")
    erros = []
    if r3.bandCount != 3:
        erros.append(f"tem {r3.bandCount} bandas (esperadas 3)")
    if r3.pixelType != "U8":
        erros.append(f"pixel {r3.pixelType} (esperado U8)")
    if abs(r3.meanCellWidth - r4.meanCellWidth) > 1e-6:
        erros.append("pixel diferente do 126.tif")
    if r3.spatialReference.name != r4.spatialReference.name:
        erros.append("sistema de coordenadas diferente do 126.tif")
    if (r3.width, r3.height) != (r4.width, r4.height) or \
            abs(r3.extent.XMin - r4.extent.XMin) > 0.01 or abs(r3.extent.YMin - r4.extent.YMin) > 0.01:
        erros.append("extensao ou grelha diferente do 126.tif")
    if not erros:
        n = 500
        x = r4.extent.XMin + (r4.width // 2 - n // 2) * r4.meanCellWidth
        y = r4.extent.YMin + (r4.height // 2 - n // 2) * r4.meanCellHeight
        a4 = arcpy.RasterToNumPyArray(src, arcpy.Point(x, y), n, n)
        a3 = arcpy.RasterToNumPyArray(out, arcpy.Point(x, y), n, n)
        if not np.array_equal(a4[:3], a3):
            erros.append("as bandas 1-3 nao sao iguais as do 126.tif na amostra central")
        else:
            print("   amostra central 500 x 500: bandas 1-3 iguais as do 126.tif")
    if erros:
        raise RuntimeError(f"{out}: " + "; ".join(erros))
    print("   OK")


def main():
    arcpy.env.overwriteOutput = True
    if not arcpy.Exists(C.ORTO_126):
        raise FileNotFoundError(C.ORTO_126)
    if arcpy.Exists(C.ORTO_126_RGB):
        print(f"ja existe: {C.ORTO_126_RGB}")
    else:
        criar_rgb(C.ORTO_126, C.ORTO_126_RGB)
    verificar(C.ORTO_126, C.ORTO_126_RGB)
    print("Ortomosaico RGB pronto.")


if __name__ == "__main__":
    main()
