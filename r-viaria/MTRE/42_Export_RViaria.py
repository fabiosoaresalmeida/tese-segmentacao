# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
ORTO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"          # ortomosaico das 16 ortos
ETIQUETAS = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP\etiquetas_vias.shp"     # criado pelo 41
AREA_TREINO = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP\area_treino.shp"     # criado pelo 40
PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\Training\vias_chip512"

# -*- coding: utf-8 -*-
"""
Export Training Data For Deep Learning - rede viária (Classified Tiles).
Três conjuntos:
  - multiclasse: Via pavimentada / Via não pavimentada / Ferrovia (U-Net)
  - rede_viaria: binário, via pavimentada + não pavimentada (Multi-Task Road Extractor)
  - ferrovia:    binário, ferrovia (Multi-Task Road Extractor)
Executar na janela Python ou num Notebook do ArcGIS Pro 3.6.
Requer a extensão Image Analyst.
"""
import os
import time
import shutil
import arcpy

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.env.overwriteOutput = True

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
ORTO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"          # ortomosaico das 16 ortos
SHP = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"
TRAINING = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\Training"
AREA_TREINO = os.path.join(SHP, "area_treino.shp")                     # criado pelo 40

CONJUNTOS = {
    # 3 classes: U-Net
    "multiclasse": (os.path.join(SHP, "etiquetas_vias.shp"),          # criado pelo 41
                    os.path.join(TRAINING, "vias_chip512_unet")),
    # binários: Multi-Task Road Extractor
    "rede_viaria": (os.path.join(SHP, "etiquetas_rede_viaria_binario.shp"),   # criado pelo 41b
                    os.path.join(TRAINING, "rede_viaria_chip512_binario")),
    "ferrovia":    (os.path.join(SHP, "etiquetas_ferrovia_binario.shp"),      # criado pelo 41b
                    os.path.join(TRAINING, "ferrovia_chip512_binario")),
}
EXPORTAR = ["multiclasse", "rede_viaria", "ferrovia"]

# Pastas já existentes:
#  - válidas (todos os pares do map.txt existem): saltadas, salvo REFAZER = True
#  - inconsistentes: apagadas e exportadas de novo
REFAZER = False

CHIP = 512        # mais contexto que nos edifícios: ajuda a continuidade das vias
STRIDE = 256      # sobreposição de 50 %


def validar_pasta(pasta):
    """Confirma que todos os pares do map.txt existem. Devolve (n_pares, em_falta)."""
    mapa = os.path.join(pasta, "map.txt")
    if not os.path.exists(mapa):
        return 0, ["map.txt inexistente"]
    pares, falta = 0, []
    with open(mapa, encoding="utf-8") as fh:
        for linha in fh:
            partes = linha.split()
            if not partes:
                continue
            pares += 1
            for rel in partes[:2]:
                rel = rel.replace("\\", os.sep).replace("/", os.sep)
                if not os.path.exists(os.path.join(pasta, rel)):
                    falta.append(rel)
    return pares, falta


# ---------------------------------------------------------------------------
for nome in EXPORTAR:
    etiquetas, pasta = CONJUNTOS[nome]
    print(f"\n=== {nome} -> {pasta}")
    if os.path.exists(pasta):
        pares, falta = validar_pasta(pasta)
        if pares and not falta and not REFAZER:
            print(f"  já exportado e válido ({pares} pares): saltado")
            continue
        motivo = "REFAZER = True" if (pares and not falta) else "exportação inconsistente"
        print(f"  a apagar a exportação anterior ({motivo})")
        shutil.rmtree(pasta)
    os.makedirs(os.path.dirname(pasta), exist_ok=True)
    t0 = time.time()

    arcpy.ia.ExportTrainingDataForDeepLearning(
        in_raster=ORTO,
        out_folder=pasta,
        in_class_data=etiquetas,
        image_chip_format="TIFF",
        tile_size_x=CHIP,
        tile_size_y=CHIP,
        stride_x=STRIDE,
        stride_y=STRIDE,
        output_nofeature_tiles="ONLY_TILES_WITH_FEATURES",
        metadata_format="Classified_Tiles",
        start_index=0,
        class_value_field="Classvalue",
        buffer_radius=0,
        # A máscara é essencial: sem ela, os chips que cruzam o limite da área
        # de treino têm vias reais fora da área etiquetadas como fundo.
        in_mask_polygons=AREA_TREINO,
        rotation_angle=0,
        reference_system="MAP_SPACE",
        processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
        blacken_around_feature="NO_BLACKEN",
        crop_mode="FIXED_SIZE",
    )

    print(arcpy.GetMessages())
    n_chips = len(os.listdir(os.path.join(pasta, "images")))
    pares, falta = validar_pasta(pasta)
    print(f"Chips exportados: {n_chips} | pares no map.txt: {pares} | "
          f"tempo: {(time.time() - t0) / 60:.1f} min")
    if falta:
        print(f"  ATENÇÃO: {len(falta)} ficheiros do map.txt em falta, ex.: {falta[:5]}")
    else:
        print("  verificação OK: todos os pares imagem/label existem")
