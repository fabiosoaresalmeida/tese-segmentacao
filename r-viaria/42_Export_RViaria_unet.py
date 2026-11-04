# -*- coding: utf-8 -*-
"""
Export Training Data For Deep Learning - rede viária (Classified Tiles).
Conjuntos:
  - multiclasse:     Via pavimentada / Via não pavimentada / Ferrovia (U-Net), chips 256
  - rede_viaria_256: binário, via pavimentada + não pavimentada (U-Net), chips 256
Corre só no CPU (a GPU fica livre para os treinos).
Executar na linha de comandos (arcgispro-py3), na janela Python ou num Notebook do
ArcGIS Pro 3.6. Requer a extensão Image Analyst.
"""
import os
# Antes de importar arcpy: esconder a GPU deste processo (não interfere com treinos)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import shutil
import arcpy

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.env.overwriteOutput = True
arcpy.env.processorType = "CPU"
# "0" = sem processos paralelos: funciona mesmo quando o arranque de novos python.exe
# está bloqueado (a exportação com 50 % ficou presa a 0 %). Mais lento (1–3 h).
# Com o Python a arrancar normalmente, "50%" é mais rápido e deixa núcleos ao treino.
PARALELO = "0"
arcpy.env.parallelProcessingFactor = PARALELO

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
                    os.path.join(TRAINING, "vias_chip256_unet")),
    # binário: U-Net (mesmas etiquetas do Multi-Task Road Extractor)
    "rede_viaria_256": (os.path.join(SHP, "etiquetas_rede_viaria_binario.shp"),   # criado pelo 41b
                        os.path.join(TRAINING, "rede_viaria_chip256_unet_binario")),
    # binários: Multi-Task Road Extractor (chips 512, já exportados)
    # "rede_viaria": (os.path.join(SHP, "etiquetas_rede_viaria_binario.shp"),
                    # os.path.join(TRAINING, "rede_viaria_chip512_binario")),
    # "ferrovia":    (os.path.join(SHP, "etiquetas_ferrovia_binario.shp"),
                    # os.path.join(TRAINING, "ferrovia_chip512_binario")),
}
EXPORTAR = ["rede_viaria_256"]

# Pastas já existentes:
#  - válidas (todos os pares do map.txt existem): saltadas, salvo REFAZER = True
#  - inconsistentes: apagadas e exportadas de novo
REFAZER = False

CHIP = 256        # como nos edifícios: permite batch 8 no U-Net ResNet-50
STRIDE = 128      # sobreposição de 50 %


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
print(f"Processador: {arcpy.env.processorType} | paralelo: {arcpy.env.parallelProcessingFactor} "
      f"| GPU visível: {os.environ.get('CUDA_VISIBLE_DEVICES') or 'nenhuma'}")

for nome in EXPORTAR:
    if nome not in CONJUNTOS:
        print(f"\n=== {nome}: não está definido em CONJUNTOS, saltado")
        continue
    etiquetas, pasta = CONJUNTOS[nome]
    print(f"\n=== {nome} -> {pasta}")
    if not arcpy.Exists(etiquetas):
        print(f"  ERRO: etiquetas inexistentes ({etiquetas}): conjunto saltado")
        continue
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
    n_chips = sum(1 for f in os.listdir(os.path.join(pasta, "images"))
                  if f.lower().endswith((".tif", ".tiff")))
    pares, falta = validar_pasta(pasta)
    print(f"Chips exportados: {n_chips} | pares no map.txt: {pares} | "
          f"tempo: {(time.time() - t0) / 60:.1f} min")
    if falta:
        print(f"  ATENÇÃO: {len(falta)} ficheiros do map.txt em falta, ex.: {falta[:5]}")
    else:
        print("  verificação OK: todos os pares imagem/label existem")