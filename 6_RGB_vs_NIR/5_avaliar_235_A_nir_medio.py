# -*- coding: utf-8 -*-
r"""
5_avaliar_235_A_nir_medio.py - SOLUCAO A na 235: U-Net R152 de 4 bandas, com a banda NIR
substituida pela media do NIR do treino (235 NIR medio, criado pelo 1_preparar_ortos.py).
Resultados em D:\DL\Avaliacao\235_construcoes_A_nir_medio.

    python 5_avaliar_235_A_nir_medio.py
"""
import comum_rgb as C
from avaliar_235_comum import correr

if __name__ == "__main__":
    correr(C.ORTO_235_NIRM, C.PASTA_MODELOS_NIR, [C.MODELO_NIR], C.AVAL_235_A)
