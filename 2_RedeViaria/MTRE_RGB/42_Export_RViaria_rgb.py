# -*- coding: utf-8 -*-
r"""
MTRE_RGB/42_Export_RViaria_rgb.py
Export Training Data For Deep Learning (Classified Tiles), chips 512, a partir do
ortomosaico RGB de 3 bandas (bandas 1-3 do 126.tif, criado pelo
6_RGB_vs_NIR/1_preparar_ortos.py), para treinar o Multi-Task Road Extractor só com RGB
e comparar com o MTRE de 4 bandas (RGB + NIR). Dois conjuntos binários:
  - rede_viaria: via pavimentada + não pavimentada
  - ferrovia:    caminho de ferro

Tudo o resto é IGUAL ao MTRE/42_Export_RViaria.py (etiquetas, máscara, chips 512,
passo 256), para que a única diferença entre os dois modelos seja a falta do NIR.

Antes de exportar, verifica o ortomosaico RGB contra o de 4 bandas:
  - número de bandas, tamanho da célula e sistema de coordenadas;
  - alinhamento da grelha (a origem tem de coincidir com a do 126.tif);
  - numa janela de amostra, se as bandas 1-3 são iguais às do 126.tif (confirma que
    o ortomosaico é o mesmo e que não há deslocamento).
Com VERIFICAR_E_PARAR = True só faz as verificações (não exporta).

Executar no Python Command Prompt do ArcGIS Pro (arcgispro-py3):
    python 42_Export_RViaria_rgb.py
Requer a extensão Image Analyst.
"""
import os
import json
import time
import shutil
import numpy as np
import arcpy

arcpy.CheckOutExtension("ImageAnalyst")
arcpy.env.overwriteOutput = True

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
ORTO_RGB = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico_RGB\126.tif"   # 6_RGB_vs_NIR/1_preparar_ortos.py
ORTO_4B = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"     # só para as verificações
N_BANDAS = 3

SHP = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"
TRAINING = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\Training"
AREA_TREINO = os.path.join(SHP, "area_treino.shp")                          # criado pelo 40b
CONJUNTOS = {                                                               # etiquetas do 41b
    "rede_viaria": (os.path.join(SHP, "etiquetas_rede_viaria_binario.shp"),
                    os.path.join(TRAINING, "rede_viaria_chip512_binario_rgb")),
    "ferrovia":    (os.path.join(SHP, "etiquetas_ferrovia_binario.shp"),
                    os.path.join(TRAINING, "ferrovia_chip512_binario_rgb")),
}
EXPORTAR = ["rede_viaria", "ferrovia"]   # os já exportados e válidos são saltados
# Modelos MTRE de 4 bandas equivalentes: o n.º de chips com que foram treinados é lido
# do parametros.json do registo de treino (<modelo>_registo), só para comparação no fim
MODELOS_4B = {
    "rede_viaria": r"D:\DL\Modelos\126_RViaria\mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_1137",
    "ferrovia": r"D:\DL\Modelos\126_RViaria\mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209",
}

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


def chips_4_bandas(nome):
    """N.º de chips do treino do MTRE de 4 bandas equivalente (None se não houver registo)."""
    pasta = MODELOS_4B.get(nome)
    if not pasta:
        return None
    try:
        with open(os.path.join(pasta + "_registo", "parametros.json"), encoding="utf-8") as fh:
            return json.load(fh).get("n_chips")
    except (OSError, ValueError):
        return None


def verificar_ortomosaico():
    """Compara o ortomosaico RGB com o de 4 bandas. Devolve a lista de problemas."""
    problemas = []
    rr, r4 = arcpy.Raster(ORTO_RGB), arcpy.Raster(ORTO_4B)
    dr = arcpy.Describe(ORTO_RGB)
    srr, sr4 = dr.spatialReference, arcpy.Describe(ORTO_4B).spatialReference
    print(f"Ortomosaico RGB: {ORTO_RGB}")
    print(f"  bandas {rr.bandCount} | célula {rr.meanCellWidth:.3f} m | tipo {rr.pixelType} | "
          f"{rr.width} x {rr.height} px | SR '{srr.name}'")
    print(f"  extensão X {rr.extent.XMin:.1f}-{rr.extent.XMax:.1f} | Y {rr.extent.YMin:.1f}-{rr.extent.YMax:.1f}")
    print(f"Ortomosaico 4 bandas: bandas {r4.bandCount} | célula {r4.meanCellWidth:.3f} m | "
          f"tipo {r4.pixelType} | {r4.width} x {r4.height} px")

    if rr.bandCount != N_BANDAS:
        problemas.append(f"o ortomosaico tem {rr.bandCount} bandas (esperadas {N_BANDAS})")
    if abs(rr.meanCellWidth - r4.meanCellWidth) > 1e-6:
        problemas.append(f"célula diferente: {rr.meanCellWidth} vs {r4.meanCellWidth}")
    if srr.exportToString() != sr4.exportToString():
        problemas.append(f"sistema de coordenadas diferente: '{srr.name}' vs '{sr4.name}'")
    cel = r4.meanCellWidth
    dx = (rr.extent.XMin - r4.extent.XMin) / cel
    dy = (rr.extent.YMin - r4.extent.YMin) / cel
    if abs(dx - round(dx)) > 0.01 or abs(dy - round(dy)) > 0.01:
        problemas.append(f"grelha desalinhada em relação ao 126.tif: desvio de {dx % 1:.2f} / {dy % 1:.2f} células")
    if rr.pixelType not in ("U8", "8_BIT_UNSIGNED"):
        print(f"  NOTA: tipo de píxel {rr.pixelType} (o 126.tif é 8 bits). O treino normaliza as bandas, "
              "mas confirmar que é o mesmo ficheiro usado nos edifícios.")

    # janela de amostra no centro da área de treino
    e = arcpy.Describe(AREA_TREINO).extent
    cx, cy = (e.XMin + e.XMax) / 2, (e.YMin + e.YMax) / 2
    cx = r4.extent.XMin + round((cx - r4.extent.XMin) / cel) * cel
    cy = r4.extent.YMin + round((cy - r4.extent.YMin) / cel) * cel
    n = int(JANELA_AMOSTRA_M / cel)
    canto = arcpy.Point(cx - n * cel / 2, cy - n * cel / 2)
    ar = arcpy.RasterToNumPyArray(ORTO_RGB, canto, n, n, nodata_to_value=0)
    a4 = arcpy.RasterToNumPyArray(ORTO_4B, canto, n, n, nodata_to_value=0)
    if ar.ndim == 3 and a4.ndim == 3 and ar.shape[0] >= 3:
        iguais = [float(np.mean(ar[b].astype(np.int32) == a4[b].astype(np.int32))) for b in range(3)]
        print("  amostra (%.0f m x %.0f m em %.0f, %.0f): bandas 1-3 iguais ao 126.tif em %s dos píxeis"
              % (JANELA_AMOSTRA_M, JANELA_AMOSTRA_M, cx, cy, ", ".join(f"{v:.1%}" for v in iguais)))
        if min(iguais) < 0.99:
            problemas.append("as bandas 1-3 não coincidem com o 126.tif na amostra "
                             "(ortomosaico diferente ou deslocado)")
    else:
        problemas.append(f"não foi possível ler a amostra (forma {ar.shape} / {a4.shape})")
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

if not arcpy.Exists(AREA_TREINO):
    raise SystemExit(f"Inexistente: {AREA_TREINO}")


def exportar(nome, etiquetas, pasta_saida):
    print(f"\n=== {nome} (RGB) -> {pasta_saida}")
    if not arcpy.Exists(etiquetas):
        print(f"  ERRO: etiquetas inexistentes ({etiquetas}): conjunto saltado")
        return
    if os.path.exists(pasta_saida):
        pares, falta = validar_pasta(pasta_saida)
        if pares and not falta and not REFAZER:
            print(f"  já exportado e válido ({pares} pares): saltado (REFAZER = True para repetir)")
            return
        motivo = "REFAZER = True" if (pares and not falta) else "exportação inconsistente"
        print(f"  a apagar a exportação anterior ({motivo})")
        shutil.rmtree(pasta_saida)
    os.makedirs(os.path.dirname(pasta_saida), exist_ok=True)
    t0 = time.time()

    arcpy.ia.ExportTrainingDataForDeepLearning(
        in_raster=ORTO_RGB,
        out_folder=pasta_saida,
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
    imgs = os.path.join(pasta_saida, "images")
    chips = [f for f in os.listdir(imgs) if f.lower().endswith((".tif", ".tiff"))]
    pares, falta = validar_pasta(pasta_saida)
    print(f"Chips exportados: {len(chips)} | pares no map.txt: {pares} | "
          f"tempo: {(time.time() - t0) / 60:.1f} min")
    if falta:
        print(f"  ATENÇÃO: {len(falta)} ficheiros do map.txt em falta, ex.: {falta[:5]}")
    else:
        print("  verificação OK: todos os pares imagem/label existem")
    if chips:
        nb = arcpy.Raster(os.path.join(imgs, chips[0])).bandCount
        print(f"  bandas nos chips: {nb}" + ("" if nb == N_BANDAS else f"  ATENÇÃO: esperadas {N_BANDAS}"))
    n4 = chips_4_bandas(nome)
    if n4:
        print(f"Para comparar: o MTRE de 4 bandas foi treinado com {n4} chips"
              + (" (iguais)" if n4 == len(chips) else f" (diferença de {len(chips) - n4:+d})"))
    else:
        print("Para comparar: n.º de chips do MTRE de 4 bandas não encontrado "
              f"({MODELOS_4B.get(nome)}_registo\\parametros.json)")


for nome in EXPORTAR:
    if nome not in CONJUNTOS:
        print(f"\n=== {nome}: não está definido em CONJUNTOS, saltado")
        continue
    exportar(nome, *CONJUNTOS[nome])
