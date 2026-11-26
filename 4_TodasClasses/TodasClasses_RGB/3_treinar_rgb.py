# -*- coding: utf-8 -*-
r"""
3_treinar_rgb.py - treina os modelos de TODAS AS CLASSES so com RGB, com o
54_treinar_todas_classes.py ORIGINAL, mudando so os chips (RGB) e a pasta dos modelos.
Regras iguais as do modelo de 4 bandas: SAMLoRA (ViT-B), DeepLabV3 e U-Net (ResNet-50),
batch 8 (4 e 2 se faltar memoria), ate 50 epocas com paragem antecipada pela perda de
validacao, LR fixo 2e-4, 10 % de validacao, FREEZE_MODEL, class_balancing,
ignore_classes [0], chip 256.

Nota: com 3 bandas, os pesos pre-treinados do RGB sao usados tal como estao (a 1.a camada
nao e reinicializada, ao contrario do modelo de 4 bandas, em que o NIR comeca aleatorio).

Os modelos ja treinados (pasta com .emd) sao saltados: pode voltar a correr depois de uma
interrupcao. Antes de treinar confirma que os chips tem 3 bandas.

Correr no Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO, GPU livre):
    python 3_treinar_rgb.py
"""
import os

import comum_todas_rgb as C


def main():
    feitos = C.modelos_treinados(C.PASTA_MODELOS_RGB)
    if len(feitos) == len(C.MODELOS):
        print(f"Todos os modelos RGB ja treinados: {feitos} -> nada a fazer")
        return
    if not os.path.isdir(os.path.join(C.CHIPS_RGB, "images")):
        raise FileNotFoundError(f"Faltam os chips RGB (correr o 2_exportar_chips_rgb.py): {C.CHIPS_RGB}")
    nb = C.bandas_chips(C.CHIPS_RGB)
    if nb != 3:
        raise RuntimeError(f"Os chips em {C.CHIPS_RGB} tem {nb} bandas (esperadas 3)")
    m = C.carregar("54_treinar_todas_classes.py")
    m.CHIPS = C.CHIPS_RGB
    m.PASTA_MODELOS = C.PASTA_MODELOS_RGB
    m.ORDEM = [n for n in m.ORDEM if n in C.MODELOS]
    print(f"Treino RGB | chips {m.CHIPS} (3 bandas) | modelos {m.PASTA_MODELOS} | ordem {m.ORDEM} | "
          f"epocas ate {m.EPOCHS} | lr {m.LEARNING_RATE} | args '{m.ARGS_COMUNS}' | ja treinados: {feitos}",
          flush=True)
    m.main()
    feitos = C.modelos_treinados(C.PASTA_MODELOS_RGB)
    faltam = [n for n in m.ORDEM if n not in feitos]
    for n in feitos:
        print(f"   {n}: treinado com {C.bandas_emd(os.path.join(C.PASTA_MODELOS_RGB, n))} bandas")
    if faltam:
        raise RuntimeError(f"O treino terminou sem os modelos {faltam}; ver treino_todas_relatorio.txt "
                           f"em {C.PASTA_MODELOS_RGB}")


if __name__ == "__main__":
    main()
