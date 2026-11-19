# -*- coding: utf-8 -*-
r"""
4_avaliar_126.py - compara as tres versoes do U-Net R152 nas 3 ortos de teste da 126
(126-I-3, 126-II-3, 126-IV-1), com o 09_avaliar_modelos.py ORIGINAL (mesmas metricas
ao pixel e por objeto, mesma regularizacao, mesmas referencias "edicao" e "fotogrametria"):

  grupo "nir_real"   modelo 4 bandas no 126.tif (RGB + NIR verdadeiro)  -> referencia
  grupo "nir_medio"  modelo 4 bandas no 126 com NIR constante           -> solucao A
  grupo "rgb"        modelo treinado so com RGB no 126 RGB               -> solucao B

Resultados em D:\DL\Avaliacao\126_rgb_vs_nir (tabela_comparativa.txt e CSV do 09).

Correr no Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO, GPU livre):
    python 4_avaliar_126.py
"""
import os

import comum_rgb as C


def main():
    rgb = C.modelos_rgb()
    if not rgb:
        raise RuntimeError(f"Ainda nao ha modelo RGB em {C.PASTA_MODELOS_RGB} (correr o 3_treinar_rgb.py)")
    for f in (C.ORTO_126, C.ORTO_126_NIRM, C.ORTO_126_RGB):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} (correr o 1_preparar_ortos.py)")
    m = C.carregar("09_avaliar_modelos.py")
    m.GRUPOS = [
        ("nir_real", C.PASTA_MODELOS_NIR, C.ORTO_126),
        ("nir_medio", C.PASTA_MODELOS_NIR, C.ORTO_126_NIRM),
        ("rgb", C.PASTA_MODELOS_RGB, C.ORTO_126_RGB),
    ]
    m.MODELOS = [C.MODELO_NIR] + rgb
    m.PASTA_AVALIACAO = C.AVAL_126
    m.FICHEIRO_LOG = os.path.join(C.AVAL_126, "log_avaliacao.txt")
    m.GDB = os.path.join(C.AVAL_126, "poligonos.gdb")
    m.MODO_TESTE = False
    print(f"Avaliacao 126 | grupos {[g[0] for g in m.GRUPOS]} | modelos {m.MODELOS} | ortos {m.ORTOS_TESTE}",
          flush=True)
    m.main()


if __name__ == "__main__":
    main()
