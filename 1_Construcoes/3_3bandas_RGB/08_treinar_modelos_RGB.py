# -*- coding: utf-8 -*-
r"""
08_treinar_modelos_RGB.py  (ArcGIS Pro 3.6) - FASE C (so RGB)

Treina os MESMOS quatro modelos da fase B (DeepLabV3 ResNet-101, U-Net ResNet-50,
SAMLoRA ViT-L e U-Net ResNet-152) so com R, G e B, usando o 08_treinar_modelos_nDSM.py
ORIGINAL (que tem o SALTAR_CONCLUIDOS) e mudando so os chips, a pasta dos modelos e a
lista de experiencias. Regras das fases A e B: batch 8 (4 ou 2 se faltar memoria), LR
fixo 2e-4, 10 % de validacao, codificador congelado, class_balancing, chip 256. O numero
de epocas e a paragem antecipada vem do comum_RGB.py (EPOCAS, PARAGEM).

Os modelos ja concluidos em PASTA_MODELOS_RGB sao saltados (pode-se correr outra vez
depois de uma interrupcao).

Registos em PASTA_MODELOS_RGB: log_treinos.txt, resumo_treinos.csv, epocas_treinos.csv.

Correr no Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO, GPU livre), na pasta
1_Construcoes\3_3bandas_RGB:
    python 08_treinar_modelos_RGB.py            treino completo
    python 08_treinar_modelos_RGB.py teste      1 epoca de cada tipo de modelo (verificacao)
ANTES DE CORRER: "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback" e
"Modo de Edicao Rapida" desativado (como nas fases A e B).
"""
import os
import sys
import glob

import comum_RGB as C


def main():
    teste = len(sys.argv) > 1 and sys.argv[1].lower() == "teste"
    imgs = glob.glob(os.path.join(C.CHIPS_RGB, "images", "*.tif"))
    if not imgs:
        raise FileNotFoundError(f"Faltam os chips RGB (correr o 07_exportar_chips_RGB.py): {C.CHIPS_RGB}")

    m = C.carregar("08_treinar_modelos_nDSM.py")
    m.PASTA_CHIPS = C.CHIPS_RGB
    m.PASTA_MODELOS = C.PASTA_MODELOS_RGB if not teste else os.path.join(C.PASTA_MODELOS_RGB, "_teste")
    m.FICHEIRO_LOG = os.path.join(m.PASTA_MODELOS, "log_treinos.txt")
    m.FICHEIRO_RESUMO = os.path.join(m.PASTA_MODELOS, "resumo_treinos.csv")
    m.FICHEIRO_EPOCAS = os.path.join(m.PASTA_MODELOS, "epocas_treinos.csv")
    m.MODO_TESTE = teste
    m.SALTAR_CONCLUIDOS = True
    m.EPOCAS = C.EPOCAS
    m.PARAGEM = C.PARAGEM
    m.EXPERIENCIAS = [
        (True, tipo, backbone, batches, m.ARGS_SAMLORA if tipo == "SAMLORA" else m.ARGS_CNN, m.LR, m.EPOCAS)
        for tipo, backbone, batches in C.MODELOS_FASE_B
    ]
    os.makedirs(m.PASTA_MODELOS, exist_ok=True)
    print(f"Treino FASE C (RGB) | {'MODO TESTE | ' if teste else ''}chips {m.PASTA_CHIPS} ({len(imgs)} imagens) | "
          f"modelos {m.PASTA_MODELOS} | ate {m.EPOCAS} epocas | paragem {m.PARAGEM}", flush=True)
    for e in m.EXPERIENCIAS:
        print(f"   {e[1]} {e[2]} | batch {e[3]} | lr {e[5]} | epocas {e[6]} | '{e[4]}'", flush=True)
    m.main()

    if not teste:
        feitos = C.modelos_rgb_da_fase_b()
        print(f"Modelos RGB concluidos ({len(feitos)}/{len(C.MODELOS_FASE_B)}): {feitos}")
        if len(feitos) < len(C.MODELOS_FASE_B):
            print("ATENCAO: faltam modelos; ver log_treinos.txt e correr outra vez (os concluidos sao saltados)")


if __name__ == "__main__":
    main()
