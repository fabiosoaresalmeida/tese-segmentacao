# -*- coding: utf-8 -*-
r"""
07_exportar_chips_RGB.py  (ArcGIS Pro 3.6) - FASE C (so RGB)

1. Cria o ortomosaico RGB (bandas 1, 2 e 3 do 126.tif, 8 bits, mesma grelha, pixel e
   sistema de coordenadas), se ainda nao existir.
2. Exporta os chips de treino RGB com o 07_exportar_chips.py ORIGINAL, mudando so o
   ortomosaico e a pasta de saida: edificios de treino, mascara, zona de teste, campo
   de classe, 256 px, passo 128, 0,5 m - iguais as fases A e B.

Se o ortomosaico RGB ou os chips ja existirem (por exemplo, da experiencia 6_RGB_vs_NIR),
sao reutilizados e verificados.

Correr no Python Command Prompt do ArcGIS Pro, na pasta 1_Construcoes\3_3bandas_RGB:
    python 07_exportar_chips_RGB.py
"""
import os
import time
import traceback

print("a carregar o arcpy...", flush=True)
import arcpy

import comum_RGB as C


def verificar_orto(f, n_bandas):
    r = arcpy.Raster(f)
    ok = r.bandCount == n_bandas and r.pixelType == "U8"
    print(f"   {f}: {r.bandCount} bandas | {r.pixelType} | {r.width} x {r.height} | pixel {r.meanCellWidth} | "
          f"{r.spatialReference.name} -> {'OK' if ok else 'VERIFICAR'}", flush=True)
    if not ok:
        raise RuntimeError(f"{f}: esperadas {n_bandas} bandas de 8 bits")


def criar_orto_rgb():
    if arcpy.Exists(C.ORTO_RGB):
        print(f"Ortomosaico RGB ja existe: {C.ORTO_RGB}")
        verificar_orto(C.ORTO_RGB, 3)
        return
    if not arcpy.Exists(C.ORTO_4BANDAS):
        raise FileNotFoundError(C.ORTO_4BANDAS)
    verificar_orto(C.ORTO_4BANDAS, 4)
    os.makedirs(os.path.dirname(C.ORTO_RGB), exist_ok=True)
    t0 = time.time()
    bandas = ";".join(os.path.join(C.ORTO_4BANDAS, f"Band_{i}") for i in (1, 2, 3))
    src = C.ORTO_4BANDAS
    with arcpy.EnvManager(snapRaster=src, cellSize=src, extent=src, outputCoordinateSystem=src):
        arcpy.management.CompositeBands(bandas, C.ORTO_RGB)
    print(f"Ortomosaico RGB criado em {(time.time() - t0) / 60:.1f} min")
    verificar_orto(C.ORTO_RGB, 3)


def chips_existem():
    rel = os.path.join(C.CHIPS_RGB, "exportacao_relatorio.txt")
    imgs = glob_count(os.path.join(C.CHIPS_RGB, "images", "*.tif"))
    if os.path.exists(rel) and imgs:
        print(f"Chips RGB ja exportados: {C.CHIPS_RGB} ({imgs} imagens)")
        # confirmar que tem 3 bandas
        import glob
        primeiro = sorted(glob.glob(os.path.join(C.CHIPS_RGB, "images", "*.tif")))[0]   # so .tif (nao .tfw/.aux.xml)
        nb = arcpy.Raster(primeiro).bandCount
        primeiro = os.path.basename(primeiro)
        if nb != 3:
            raise RuntimeError(f"Os chips em {C.CHIPS_RGB} tem {nb} bandas (esperadas 3)")
        print(f"   {primeiro}: {nb} bandas -> OK")
        return True
    return False


def glob_count(padrao):
    import glob
    return len(glob.glob(padrao))


def main():
    arcpy.env.overwriteOutput = True
    criar_orto_rgb()
    if chips_existem():
        return
    m = C.carregar("07_exportar_chips.py")
    m.ORTOMOSAICO = C.ORTO_RGB
    m.PASTA_SAIDA = C.CHIPS_RGB
    m.SCRIPT_TREINO = None
    m.APAGAR_SE_EXISTIR = True          # apaga uma exportacao incompleta anterior
    m.FATOR_PARALELO = "85%"
    m.arcpy.env.parallelProcessingFactor = "85%"   # o 07 aplica-o ao ser importado
    print(f"Exportacao RGB | ortomosaico {m.ORTOMOSAICO} | edificios {m.EDIFICIOS} | "
          f"{m.TILE} px, passo {m.STRIDE} | saida {m.PASTA_SAIDA}", flush=True)
    rel = m.Relatorio()
    try:
        m.main(rel)
    except Exception:
        rel()
        rel("ERRO - execucao interrompida")
        rel(traceback.format_exc())
        destino = m.PASTA_SAIDA if os.path.isdir(m.PASTA_SAIDA) else os.path.dirname(m.PASTA_SAIDA)
        rel.gravar(os.path.join(destino, "exportacao_relatorio_ERRO.txt"))
        raise


if __name__ == "__main__":
    main()
