# -*- coding: utf-8 -*-
r"""
3_treinar_rgb.py - treina o U-Net ResNet-152 so com RGB, com o 08_treinar_modelos.py
ORIGINAL, mudando so os chips (RGB), a pasta dos modelos e a lista de experiencias.
Regras iguais a fase A: batch 8 (4 se faltar memoria), 20 epocas, LR fixo 2e-4,
10 % validacao, FREEZE_MODEL, sem paragem antecipada, class_balancing, chip 256.

Se ja existir um modelo RGB com .dlpk em PASTA_MODELOS_RGB, nao treina outra vez.

Correr no Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO, GPU livre):
    python 3_treinar_rgb.py
"""
import os

import comum_rgb as C


def main():
    feitos = C.modelos_rgb()
    if feitos:
        print(f"Ja existe modelo RGB: {feitos} -> nao treina outra vez")
        return
    if not os.path.isdir(os.path.join(C.CHIPS_RGB, "images")):
        raise FileNotFoundError(f"Faltam os chips RGB (correr o 2_exportar_chips_rgb.py): {C.CHIPS_RGB}")
    m = C.carregar("08_treinar_modelos.py")
    m.PASTA_CHIPS = C.CHIPS_RGB
    m.PASTA_MODELOS = C.PASTA_MODELOS_RGB
    m.FICHEIRO_LOG = os.path.join(m.PASTA_MODELOS, "log_treinos.txt")
    m.FICHEIRO_RESUMO = os.path.join(m.PASTA_MODELOS, "resumo_treinos.csv")
    m.FICHEIRO_EPOCAS = os.path.join(m.PASTA_MODELOS, "epocas_treinos.csv")
    m.MODO_TESTE = False
    m.EXPERIENCIAS = [(True, "UNET", "RESNET152", [8, 4], m.ARGS_CNN, m.LR, m.EPOCAS)]
    print(f"Treino RGB | chips {m.PASTA_CHIPS} | modelos {m.PASTA_MODELOS} | "
          f"UNET RESNET152 | epocas {m.EPOCAS} | lr {m.LR} | args '{m.ARGS_CNN}'", flush=True)
    m.main()
    if not C.modelos_rgb():
        raise RuntimeError("O treino terminou sem modelo RGB (.dlpk); ver log_treinos.txt")


if __name__ == "__main__":
    main()
