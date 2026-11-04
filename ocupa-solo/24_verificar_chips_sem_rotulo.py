# -*- coding: utf-8 -*-
r"""
verificar_chips_sem_rotulo.py  (ArcGIS Pro 3.6)

Encontra os chips com imagem mas sem rótulo (ou vice-versa) numa exportação
Classified Tiles e verifica se têm polígonos de rótulo por baixo.

MODO:
  "relatorio"     só analisa e cria chips_sem_rotulo.shp para ver no ArcGIS (por defeito)
  "rotulos_zero"  cria um rótulo todo a 0 (sem representação) para os chips SEM nenhum
                  polígono por baixo; os que têm polígonos são movidos para fora
  "mover"         move todas as imagens sem rótulo para a pasta images_sem_rotulo

Correr primeiro em "relatorio", ver o resultado e depois escolher o modo.
"""
import os
import glob
import shutil
import arcpy

# =============================== PARÂMETROS ===============================
CHIPS = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\23_Training\chips_ocupacao_256_128"
ROTULOS_TREINO = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\22_separacao\rotulos_treino.shp"
MODO = "rotulos_zero"        # "relatorio" | "rotulos_zero" | "mover"
# ==========================================================================

arcpy.env.overwriteOutput = True


def log(m):
    print(m)


def stems(pasta):
    return {os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(pasta, "*.tif"))}


def mover_chip(stem, origem, destino):
    """Move o .tif e os ficheiros auxiliares com o mesmo nome (.aux.xml, .tfw, ...)."""
    os.makedirs(destino, exist_ok=True)
    for f in glob.glob(os.path.join(origem, stem + ".*")):
        shutil.move(f, os.path.join(destino, os.path.basename(f)))


def fracao_sem_dados(img):
    """Fração de píxeis com 0 em todas as bandas (NoData/fora da orto)."""
    try:
        a = arcpy.RasterToNumPyArray(img)
        if a.ndim == 3:
            return float((a == 0).all(axis=0).mean())
        return float((a == 0).mean())
    except Exception:
        return float("nan")


def main():
    p_img, p_lab = os.path.join(CHIPS, "images"), os.path.join(CHIPS, "labels")
    imgs, labs = stems(p_img), stems(p_lab)
    so_img = sorted(set(imgs) - set(labs))
    so_lab = sorted(set(labs) - set(imgs))
    log(f"imagens: {len(imgs)} | rótulos: {len(labs)}")
    log(f"imagens sem rótulo: {len(so_img)} | rótulos sem imagem: {len(so_lab)}")
    if so_lab:
        log(f"  AVISO: rótulos sem imagem (ex.: {so_lab[:5]}) - estes não são tratados aqui")
    if not so_img:
        log("Nada a fazer.")
        return

    # --- extensões dos chips sem rótulo
    sr = arcpy.Describe(imgs[so_img[0]]).spatialReference
    fc = os.path.join(CHIPS, "chips_sem_rotulo.shp")
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)
    arcpy.management.CreateFeatureclass(CHIPS, "chips_sem_rotulo.shp", "POLYGON",
                                        spatial_reference=sr)
    arcpy.management.AddField(fc, "chip", "TEXT", field_length=50)
    arcpy.management.AddField(fc, "sem_dados", "DOUBLE")
    arcpy.management.AddField(fc, "tem_rot", "SHORT")
    print("A ler as extensões e os dados dos chips...")
    with arcpy.da.InsertCursor(fc, ["SHAPE@", "chip", "sem_dados", "tem_rot"]) as ic:
        for s in so_img:
            e = arcpy.Describe(imgs[s]).extent
            pts = [(e.XMin, e.YMin), (e.XMin, e.YMax), (e.XMax, e.YMax), (e.XMax, e.YMin)]
            poly = arcpy.Polygon(arcpy.Array([arcpy.Point(x, y) for x, y in pts]), sr)
            ic.insertRow([poly, s, fracao_sem_dados(imgs[s]), 0])

    # --- têm polígonos de rótulo por baixo?
    lyr = arcpy.management.MakeFeatureLayer(fc, "chips_sem_rot")[0]
    arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", ROTULOS_TREINO)
    arcpy.management.CalculateField(lyr, "tem_rot", "1", "PYTHON3")
    arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")
    arcpy.management.Delete(lyr)

    linhas = list(arcpy.da.SearchCursor(fc, ["chip", "sem_dados", "tem_rot"]))
    com_pol = [c for c, _, t in linhas if t == 1]
    sem_pol = [c for c, _, t in linhas if t == 0]
    vazios = [c for c, f, _ in linhas if f == f and f > 0.99]
    log(f"  sem nenhum polígono por baixo (só sem representação): {len(sem_pol)}")
    log(f"  COM polígonos por baixo (inesperado): {len(com_pol)}")
    log(f"  imagem quase toda sem dados (>99% a 0): {len(vazios)}")
    log(f"  ver no ArcGIS: {fc}")

    if MODO == "relatorio":
        log("\nModo relatório: nada foi alterado.")
        return

    destino = os.path.join(CHIPS, "images_sem_rotulo")
    if MODO == "mover":
        for s in so_img:
            mover_chip(s, p_img, destino)
        log(f"\n{len(so_img)} imagens movidas para {destino}")
        return

    if MODO == "rotulos_zero":
        import numpy as np
        # rótulos a 0 só para os chips sem polígonos e com imagem válida
        criar = [s for s in sem_pol if s not in vazios]
        mover = sorted(set(so_img) - set(criar))
        ref = arcpy.Raster(next(iter(labs.values())))
        tipos = {"U8": np.uint8, "U16": np.uint16, "S16": np.int16,
                 "U32": np.uint32, "S32": np.int32}
        dtype = tipos.get(ref.pixelType, np.uint16)
        nodata_ref = ref.noDataValue
        log(f"\nRótulo de referência: {os.path.basename(ref.catalogPath)} "
            f"({ref.pixelType}, {ref.width} x {ref.height}, NoData = {nodata_ref})")
        if nodata_ref == 0:
            raise RuntimeError("Os rótulos exportados têm NoData = 0; criar rótulos a 0 "
                               "seria criar rótulos vazios. Usa MODO = 'mover'.")
        arcpy.env.outputCoordinateSystem = sr
        for i, s in enumerate(criar, 1):
            r_img = arcpy.Raster(imgs[s])
            e = r_img.extent
            arr = np.zeros((r_img.height, r_img.width), dtype=dtype)
            r = arcpy.NumPyArrayToRaster(arr, arcpy.Point(e.XMin, e.YMin),
                                         r_img.meanCellWidth, r_img.meanCellHeight,
                                         nodata_ref)
            r.save(os.path.join(p_lab, s + ".tif"))
            if i % 100 == 0:
                print(f"  {i}/{len(criar)}")
        novo = arcpy.Raster(os.path.join(p_lab, criar[0] + ".tif"))
        log(f"Verificação do 1.º rótulo criado ({criar[0]}): {novo.pixelType}, "
            f"{novo.width} x {novo.height}, NoData = {novo.noDataValue}, "
            f"máx = {novo.maximum}")
        for s in mover:
            mover_chip(s, p_img, destino)
        log(f"Rótulos a 0 criados: {len(criar)}")
        log(f"Imagens movidas para {destino}: {len(mover)} "
            "(com polígonos por baixo ou sem dados)")
        imgs2, labs2 = stems(p_img), stems(p_lab)
        log(f"Agora: imagens {len(imgs2)} | rótulos {len(labs2)} "
            f"({'OK' if set(imgs2) == set(labs2) else 'ainda diferentes'})")
        return

    raise ValueError(f"MODO desconhecido: {MODO}")


if __name__ == "__main__":
    main()