# -*- coding: utf-8 -*-
r"""
1_preparar_ortos.py - cria os ortomosaicos da experiencia (so se ainda nao existirem):

  126_RGB       = bandas 1-3 do 126.tif                       -> chips RGB e avaliacao B na 126
  126_NIR_medio = bandas 1-3 do 126.tif + NIR constante       -> avaliacao A na 126
  235_NIR_medio = 235.tif (RGB) + NIR constante               -> avaliacao A na 235

NIR constante = media do NIR no treino do U-Net R152 de 4 bandas (band_mean_values[3] do
.emd, ~125); os pixeis NoData (R=G=B=0) ficam com NIR = 0. Todos em 8 bits, com a grelha,
o pixel e o sistema do original.

Correr no Python Command Prompt do ArcGIS Pro:
    python 1_preparar_ortos.py
"""
import os
import time

print("a carregar o arcpy...", flush=True)
import arcpy
from arcpy.sa import Con, Raster

import comum_rgb as C


def bandas(src, n=3):
    return [os.path.join(src, f"Band_{i}") for i in range(1, n + 1)]


def verificar(out, n_bandas):
    r = arcpy.Raster(out)
    ok = r.bandCount == n_bandas and r.pixelType == "U8"
    print(f"   {out}: {r.bandCount} bandas | {r.pixelType} | {r.width} x {r.height} | pixel {r.meanCellWidth} | "
          f"{r.spatialReference.name} -> {'OK' if ok else 'VERIFICAR'}", flush=True)
    if not ok:
        raise RuntimeError(f"{out}: esperadas {n_bandas} bandas de 8 bits")


def criar_rgb(src, out):
    if arcpy.Exists(out):
        print(f"ja existe: {out}")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    with arcpy.EnvManager(snapRaster=src, cellSize=src, extent=src, outputCoordinateSystem=src):
        arcpy.management.CompositeBands(";".join(bandas(src, 3)), out)
    print(f"RGB criado em {(time.time() - t0) / 60:.1f} min")
    verificar(out, 3)


def criar_nir_medio(src, out, valor):
    if arcpy.Exists(out):
        print(f"ja existe: {out}")
        verificar_nir(out, valor)
        return
    pasta = os.path.dirname(out)
    os.makedirs(pasta, exist_ok=True)
    t0 = time.time()
    tmp1, tmp2 = os.path.join(pasta, "_nir_tmp.tif"), os.path.join(pasta, "_nir_tmp_u8.tif")
    b1, b2, b3 = (Raster(b) for b in bandas(src, 3))
    with arcpy.EnvManager(snapRaster=src, cellSize=src, extent=src, outputCoordinateSystem=src):
        nir = Con((b1 == 0) & (b2 == 0) & (b3 == 0), 0, valor)
        nir.save(tmp1)
        arcpy.management.CopyRaster(tmp1, tmp2, pixel_type="8_BIT_UNSIGNED")
        arcpy.management.CompositeBands(";".join(bandas(src, 3) + [tmp2]), out)
    for t in (tmp1, tmp2):
        arcpy.management.Delete(t)
    print(f"NIR medio ({valor}) criado em {(time.time() - t0) / 60:.1f} min")
    verificar_nir(out, valor)


def verificar_nir(out, valor):
    """Amostra de 500 x 500 px no centro: banda 4 tem de ser o valor (ou 0 sem imagem)."""
    verificar(out, 4)
    r = arcpy.Raster(out)
    n = 500
    x = r.extent.XMin + (r.width // 2 - n // 2) * r.meanCellWidth
    y = r.extent.YMin + (r.height // 2 - n // 2) * r.meanCellHeight
    a = arcpy.RasterToNumPyArray(out, arcpy.Point(x, y), n, n)
    vals = sorted(set(int(v) for v in a[3].ravel()))
    ok = set(vals) <= {0, valor} and valor in vals
    print(f"   banda 4 (amostra central): valores {vals[:10]} -> {'OK' if ok else 'VERIFICAR'} (esperado {valor})")
    if not ok:
        raise RuntimeError(f"{out}: a banda 4 nao e constante = {valor}")


def main():
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")
    v = int(round(C.nir_medio()))
    print(f"NIR medio do U-Net R152 (4 bandas): {v}")
    for f in (C.ORTO_126, C.ORTO_235):
        if not arcpy.Exists(f):
            raise FileNotFoundError(f)
    criar_rgb(C.ORTO_126, C.ORTO_126_RGB)
    criar_nir_medio(C.ORTO_126, C.ORTO_126_NIRM, v)
    criar_nir_medio(C.ORTO_235, C.ORTO_235_NIRM, v)
    print("Ortomosaicos prontos.")


if __name__ == "__main__":
    main()