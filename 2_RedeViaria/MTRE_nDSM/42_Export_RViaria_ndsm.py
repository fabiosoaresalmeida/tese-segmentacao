# -*- coding: utf-8 -*-
r"""
MTRE_nDSM/42_Export_RViaria_ndsm.py
Export Training Data For Deep Learning - rede viária (Classified Tiles), chips 512,
a partir do ortomosaico com 5 bandas (R, G, B, NIR, nDSM normalizado), para treinar
o Multi-Task Road Extractor com o nDSM e comparar com o MTRE de 4 bandas.

Tudo o resto é IGUAL ao MTRE/42_Export_RViaria.py (etiquetas, máscara, chips 512,
passo 256), para que a única diferença entre os dois modelos seja a 5.ª banda.

Antes de exportar, verifica o ortomosaico de 5 bandas contra o de 4 bandas:
  - número de bandas, tamanho da célula e sistema de coordenadas;
  - alinhamento da grelha (a origem tem de coincidir com a do 126.tif);
  - numa janela de amostra, se as bandas 1-4 são iguais às do 126.tif (confirma que
    o ortomosaico é o mesmo e que não há deslocamento);
  - valores da banda 5 (nDSM) na amostra.
Com VERIFICAR_E_PARAR = True só faz as verificações (não exporta).

Executar no Python Command Prompt do ArcGIS Pro (arcgispro-py3):
    python 42_Export_RViaria_ndsm.py
Requer a extensão Image Analyst.
"""
import os
import time
import shutil
import numpy as np
import arcpy

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.env.overwriteOutput = True

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
ORTO_NDSM = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126_ndsm\126_nDSM_5bandas_v2.tif"
ORTO_4B = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"     # só para as verificações
N_BANDAS = 5
BANDA_NDSM = 5

SHP = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"
TRAINING = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\Training"
AREA_TREINO = os.path.join(SHP, "area_treino.shp")                          # criado pelo 40b
ETIQUETAS = os.path.join(SHP, "etiquetas_rede_viaria_binario.shp")         # criado pelo 41b
PASTA_SAIDA = os.path.join(TRAINING, "rede_viaria_chip512_binario_ndsm")

CHIP = 512        # iguais ao MTRE de 4 bandas
STRIDE = 256      # sobreposição de 50 %

# Pasta já existente: válida (todos os pares do map.txt existem) -> saltada, salvo
# REFAZER = True; inconsistente -> apagada e exportada de novo
REFAZER = False
VERIFICAR_E_PARAR = False   # True = só as verificações do ortomosaico

# Núcleos na exportação. "0" = sem processos paralelos (funciona mesmo quando o arranque
# de novos python.exe está bloqueado, mas demora várias horas); "50%" é mais rápido.
PARALELO = "50%"

JANELA_AMOSTRA_M = 500.0    # lado da janela de amostra das verificações (centro da área de treino)


# ---------------------------------------------------------------------------
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


def verificar_ortomosaico():
    """Compara o ortomosaico de 5 bandas com o de 4 bandas. Devolve a lista de problemas."""
    problemas = []
    r5, r4 = arcpy.Raster(ORTO_NDSM), arcpy.Raster(ORTO_4B)
    d5 = arcpy.Describe(ORTO_NDSM)
    sr5, sr4 = d5.spatialReference, arcpy.Describe(ORTO_4B).spatialReference
    print(f"Ortomosaico nDSM: {ORTO_NDSM}")
    print(f"  bandas {r5.bandCount} | célula {r5.meanCellWidth:.3f} m | tipo {r5.pixelType} | "
          f"{r5.width} x {r5.height} px | SR '{sr5.name}'")
    print(f"  extensão X {r5.extent.XMin:.1f}-{r5.extent.XMax:.1f} | Y {r5.extent.YMin:.1f}-{r5.extent.YMax:.1f}")
    print(f"Ortomosaico 4 bandas: bandas {r4.bandCount} | célula {r4.meanCellWidth:.3f} m | "
          f"tipo {r4.pixelType} | {r4.width} x {r4.height} px")

    if r5.bandCount != N_BANDAS:
        problemas.append(f"o ortomosaico tem {r5.bandCount} bandas (esperadas {N_BANDAS})")
    if abs(r5.meanCellWidth - r4.meanCellWidth) > 1e-6:
        problemas.append(f"célula diferente: {r5.meanCellWidth} vs {r4.meanCellWidth}")
    if sr5.exportToString() != sr4.exportToString():
        problemas.append(f"sistema de coordenadas diferente: '{sr5.name}' vs '{sr4.name}'")
    cel = r4.meanCellWidth
    dx = (r5.extent.XMin - r4.extent.XMin) / cel
    dy = (r5.extent.YMin - r4.extent.YMin) / cel
    if abs(dx - round(dx)) > 0.01 or abs(dy - round(dy)) > 0.01:
        problemas.append(f"grelha desalinhada em relação ao 126.tif: desvio de {dx % 1:.2f} / {dy % 1:.2f} células")
    if r5.pixelType not in ("U8", "8_BIT_UNSIGNED"):
        print(f"  NOTA: tipo de píxel {r5.pixelType} (o 126.tif é 8 bits). O treino normaliza as bandas, "
              "mas confirmar que é o mesmo ficheiro usado nos edifícios.")

    # janela de amostra no centro da área de treino
    e = arcpy.Describe(AREA_TREINO).extent
    cx, cy = (e.XMin + e.XMax) / 2, (e.YMin + e.YMax) / 2
    cx = r4.extent.XMin + round((cx - r4.extent.XMin) / cel) * cel
    cy = r4.extent.YMin + round((cy - r4.extent.YMin) / cel) * cel
    n = int(JANELA_AMOSTRA_M / cel)
    canto = arcpy.Point(cx - n * cel / 2, cy - n * cel / 2)
    a5 = arcpy.RasterToNumPyArray(ORTO_NDSM, canto, n, n, nodata_to_value=0)
    a4 = arcpy.RasterToNumPyArray(ORTO_4B, canto, n, n, nodata_to_value=0)
    if a5.ndim == 3 and a4.ndim == 3 and a5.shape[0] >= 4:
        iguais = [float(np.mean(a5[b].astype(np.int32) == a4[b].astype(np.int32))) for b in range(min(4, a4.shape[0]))]
        print("  amostra (%.0f m x %.0f m em %.0f, %.0f): bandas 1-4 iguais ao 126.tif em %s dos píxeis"
              % (JANELA_AMOSTRA_M, JANELA_AMOSTRA_M, cx, cy, ", ".join(f"{v:.1%}" for v in iguais)))
        if min(iguais) < 0.99:
            problemas.append("as bandas 1-4 não coincidem com o 126.tif na amostra "
                             "(ortomosaico diferente ou deslocado)")
        nd = a5[BANDA_NDSM - 1].astype(float)
        print(f"  banda {BANDA_NDSM} (nDSM): mín {nd.min():.1f} | mediana {np.median(nd):.1f} | "
              f"p99 {np.percentile(nd, 99):.1f} | máx {nd.max():.1f} | {np.mean(nd > 0):.1%} dos píxeis > 0")
        if nd.max() == nd.min():
            problemas.append("a banda do nDSM é constante na amostra")
    else:
        problemas.append(f"não foi possível ler a amostra (forma {a5.shape} / {a4.shape})")
    return problemas


# ---------------------------------------------------------------------------
arcpy.env.parallelProcessingFactor = PARALELO
problemas = verificar_ortomosaico()
if problemas:
    print("\nPROBLEMAS no ortomosaico:")
    for p in problemas:
        print(f"  - {p}")
    raise SystemExit("Corrigir antes de exportar.")
print("Verificações do ortomosaico: OK")
if VERIFICAR_E_PARAR:
    raise SystemExit("VERIFICAR_E_PARAR = True: exportação não feita.")

print(f"\n=== rede_viaria (nDSM) -> {PASTA_SAIDA}")
for f in (ETIQUETAS, AREA_TREINO):
    if not arcpy.Exists(f):
        raise SystemExit(f"Inexistente: {f}")
if os.path.exists(PASTA_SAIDA):
    pares, falta = validar_pasta(PASTA_SAIDA)
    if pares and not falta and not REFAZER:
        raise SystemExit(f"Já exportado e válido ({pares} pares): nada a fazer (REFAZER = True para repetir).")
    motivo = "REFAZER = True" if (pares and not falta) else "exportação inconsistente"
    print(f"  a apagar a exportação anterior ({motivo})")
    shutil.rmtree(PASTA_SAIDA)
os.makedirs(os.path.dirname(PASTA_SAIDA), exist_ok=True)
t0 = time.time()

arcpy.ia.ExportTrainingDataForDeepLearning(
    in_raster=ORTO_NDSM,
    out_folder=PASTA_SAIDA,
    in_class_data=ETIQUETAS,
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
imgs = os.path.join(PASTA_SAIDA, "images")
chips = [f for f in os.listdir(imgs) if f.lower().endswith((".tif", ".tiff"))]
pares, falta = validar_pasta(PASTA_SAIDA)
print(f"Chips exportados: {len(chips)} | pares no map.txt: {pares} | "
      f"tempo: {(time.time() - t0) / 60:.1f} min")
if falta:
    print(f"  ATENÇÃO: {len(falta)} ficheiros do map.txt em falta, ex.: {falta[:5]}")
else:
    print("  verificação OK: todos os pares imagem/label existem")
if chips:
    nb = arcpy.Raster(os.path.join(imgs, chips[0])).bandCount
    print(f"  bandas nos chips: {nb}" + ("" if nb == N_BANDAS else f"  ATENÇÃO: esperadas {N_BANDAS}"))
print("Para comparar: o MTRE de 4 bandas foi treinado com 6694 chips (rede_viaria_chip512_binario).")
