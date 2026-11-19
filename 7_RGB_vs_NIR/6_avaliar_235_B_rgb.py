# -*- coding: utf-8 -*-
r"""
6_avaliar_235_B_rgb.py - SOLUCAO B na 235: U-Net R152 treinado so com RGB, no 235.tif
original (3 bandas). Resultados em D:\DL\Avaliacao\235_construcoes_B_rgb.

    python 6_avaliar_235_B_rgb.py
"""
import comum_rgb as C
from avaliar_235_comum import correr

if __name__ == "__main__":
    rgb = C.modelos_rgb()
    if not rgb:
        raise RuntimeError(f"Ainda nao ha modelo RGB em {C.PASTA_MODELOS_RGB} (correr o 3_treinar_rgb.py)")
    correr(C.ORTO_235, C.PASTA_MODELOS_RGB, rgb, C.AVAL_235_B)
