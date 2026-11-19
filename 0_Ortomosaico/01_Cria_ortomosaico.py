#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_Cria_ortomosaico.py

Junta os tiles (blocos) de ortofoto de uma folha M888 num único ortomosaico.

Funcionamento:
  1. Lê todos os .tif da pasta de tiles indicada em CONFIG.
  2. Deteta automaticamente o número da folha a partir do nome dessa pasta
     (ex.: "113_Ortofotos" -> folha 113).
  3. Verifica que todos os tiles têm o mesmo sistema de coordenadas, número de
     bandas, tipo de píxel e resolução (pára com uma mensagem se não tiverem).
  4. Cria a pasta de saída, se não existir, e grava o mosaico com o nome da
     folha (ex.: <pasta de saída>\\113.tif).

Configuração: alterar apenas o bloco CONFIG abaixo.
Correr com o Python do ArcGIS Pro:  python 01_Cria_ortomosaico.py
"""
import os
import re

import arcpy

# ======================================================================
#  CONFIGURAÇÃO — alterar só aqui (use r"..." nos caminhos)
# ======================================================================
CONFIG = {
    # pasta com os tiles da folha (o número da folha é lido do nome desta pasta)
    "pasta_tifs": r"D:\Users\Cap_Almeida\Projeto\M888_Ortos\113_Ortofotos",

    # pasta do projeto; o mosaico é gravado em <pasta_projeto>\<folha>\<subpasta_saida>
    "pasta_projeto": r"D:\Users\Cap_Almeida\Projeto",
    "subpasta_saida": "Ortomosaico",

    # número da folha: None = detetar automaticamente pelo nome da pasta dos tiles
    "folha": None,

    # método para zonas sobrepostas (os tiles M888 não se sobrepõem, por isso é indiferente)
    "metodo_mosaico": "LAST",

    # substituir o mosaico se já existir
    "substituir": True,
}

# tipos de píxel do arcpy.Raster -> valores aceites pelo MosaicToNewRaster
TIPOS_PIXEL = {
    "U1": "1_BIT", "U2": "2_BIT", "U4": "4_BIT",
    "U8": "8_BIT_UNSIGNED", "S8": "8_BIT_SIGNED",
    "U16": "16_BIT_UNSIGNED", "S16": "16_BIT_SIGNED",
    "U32": "32_BIT_UNSIGNED", "S32": "32_BIT_SIGNED",
    "F32": "32_BIT_FLOAT", "F64": "64_BIT",
}


def detetar_folha(pasta_tifs):
    """Número da folha a partir do nome da pasta: '113_Ortofotos' -> '113'."""
    nome = os.path.basename(os.path.normpath(pasta_tifs))
    m = re.match(r"^(\d+)", nome)
    if not m:
        raise SystemExit(f"Não consegui detetar o número da folha no nome da pasta '{nome}'. "
                         "Indique-o em CONFIG['folha'].")
    return m.group(1)


def main():
    cfg = CONFIG
    pasta_tifs = cfg["pasta_tifs"]
    if not os.path.isdir(pasta_tifs):
        raise SystemExit(f"A pasta dos tiles não existe: {pasta_tifs}")

    folha = str(cfg["folha"]) if cfg["folha"] else detetar_folha(pasta_tifs)
    pasta_saida = os.path.join(cfg["pasta_projeto"], folha, cfg["subpasta_saida"])
    nome_saida = f"{folha}.tif"
    caminho_saida = os.path.join(pasta_saida, nome_saida)
    print(f"Folha: {folha}")
    print(f"Saída: {caminho_saida}\n")

    # 1. tiles
    lista_tifs = sorted(os.path.join(pasta_tifs, f) for f in os.listdir(pasta_tifs)
                        if f.lower().endswith(".tif"))
    if not lista_tifs:
        raise SystemExit(f"Nenhum .tif encontrado em {pasta_tifs}")
    print(f"{len(lista_tifs)} tifs encontrados:")

    # 2. verificação de consistência
    ref = arcpy.Raster(lista_tifs[0])
    sr_ref = ref.spatialReference
    esperado = (ref.bandCount, ref.pixelType, sr_ref.name, round(ref.meanCellWidth, 6))
    problemas = []
    for f in lista_tifs:
        r = arcpy.Raster(f)
        e = r.extent
        atual = (r.bandCount, r.pixelType, r.spatialReference.name, round(r.meanCellWidth, 6))
        marca = "" if atual == esperado else "   <-- DIFERENTE"
        print(f" - {os.path.basename(f)}: {r.bandCount} bandas, {r.pixelType}, "
              f"{r.meanCellWidth} m, extensão {e.XMin:.0f} {e.YMin:.0f} {e.XMax:.0f} {e.YMax:.0f}{marca}")
        if marca:
            problemas.append(os.path.basename(f))
    if problemas:
        raise SystemExit(f"\nTiles com características diferentes do primeiro: {problemas}")

    print(f"\nSistema: {sr_ref.name} | {sr_ref.projectionName} | datum {sr_ref.GCS.datumName}")
    print(f"Meridiano central {sr_ref.centralMeridian} | latitude de origem {sr_ref.latitudeOfOrigin} | "
          f"falso E/N {sr_ref.falseEasting} {sr_ref.falseNorthing} | fator {sr_ref.scaleFactor}")

    tipo = TIPOS_PIXEL.get(ref.pixelType)
    if tipo is None:
        raise SystemExit(f"Tipo de píxel não previsto: {ref.pixelType}")

    # 3. saída
    os.makedirs(pasta_saida, exist_ok=True)
    arcpy.env.overwriteOutput = bool(cfg["substituir"])
    if arcpy.Exists(caminho_saida) and not cfg["substituir"]:
        raise SystemExit(f"O mosaico já existe e 'substituir' é False: {caminho_saida}")

    arcpy.env.snapRaster = lista_tifs[0]
    arcpy.env.cellSize = ref.meanCellWidth

    print(f"\nA criar o mosaico ({len(lista_tifs)} tiles, {ref.bandCount} bandas, {tipo})...")
    arcpy.management.MosaicToNewRaster(
        input_rasters=lista_tifs,
        output_location=pasta_saida,
        raster_dataset_name_with_extension=nome_saida,
        coordinate_system_for_the_raster=sr_ref,
        pixel_type=tipo,
        cellsize=ref.meanCellWidth,
        number_of_bands=ref.bandCount,
        mosaic_method=cfg["metodo_mosaico"],
    )

    e = arcpy.Describe(caminho_saida).extent
    print(f"Mosaico criado: {caminho_saida}")
    print(f"Extensão: {e.XMin:.0f} {e.YMin:.0f} {e.XMax:.0f} {e.YMax:.0f} "
          f"({(e.XMax - e.XMin) / 1000:.1f} x {(e.YMax - e.YMin) / 1000:.1f} km)")


if __name__ == "__main__":
    main()
