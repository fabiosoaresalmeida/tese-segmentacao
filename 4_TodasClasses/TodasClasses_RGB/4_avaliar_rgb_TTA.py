# -*- coding: utf-8 -*-
r"""
4_avaliar_rgb_TTA.py - avalia os modelos de TODAS AS CLASSES treinados so com RGB, com o
55_avaliar_todas_classes_TTA.py ORIGINAL, mudando so o ortomosaico (126 RGB), os modelos e a
pasta de saida. Mesmas 3 ortos de teste (126-I-3, 126-II-4, 126-IV-3), mesma referencia
(rotulos_teste.shp do 52), mesmo contexto de 256 m, mesma TTA, mesmas metricas e mesmos
agrupamentos (completa, generalizada, rede_viaria, ocupacao sem construcoes e vias...).

Os modelos ainda sem .emd sao saltados com um aviso: pode correr antes de todos os treinos
acabarem e voltar a correr no fim (as classificacoes feitas sao reutilizadas).

Resultados em D:\DL\Avaliacao\126_todas_classes_rgb_TTA (tabela_comparativa.txt, CSV,
matrizes, log). A comparacao com os modelos de 4 bandas e feita pelo 5_comparar_4bandas_rgb.py.

Correr no Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO):
    python 4_avaliar_rgb_TTA.py
    python 4_avaliar_rgb_TTA.py --cpu        (sempre no CPU, ex.: com um treino na GPU)
"""
import os

import comum_todas_rgb as C


def main():
    if not os.path.exists(C.ORTO_126_RGB):
        raise FileNotFoundError(f"{C.ORTO_126_RGB} (correr o 1_preparar_orto_rgb.py)")
    feitos = C.modelos_treinados(C.PASTA_MODELOS_RGB)
    if not feitos:
        raise RuntimeError(f"Ainda nao ha modelos RGB em {C.PASTA_MODELOS_RGB} (correr o 3_treinar_rgb.py)")
    m = C.carregar("55_avaliar_todas_classes_TTA.py")
    m.ORTOMOSAICO = C.ORTO_126_RGB
    m.MODELOS_PASTA = C.PASTA_MODELOS_RGB
    m.MODELOS = [(rot + "_rgb", os.path.join(C.PASTA_MODELOS_RGB, pasta), {}, m.ARGS_V2_TTA)
                 for pasta, rot in C.MODELOS.items()]
    m.REFAZER = []
    m.PASTA_AVALIACAO = C.AVAL_RGB
    m.FICHEIRO_LOG = os.path.join(C.AVAL_RGB, "log_avaliacao.txt")
    m.MODO_TESTE = False
    print(f"Avaliacao RGB (TTA) | ortomosaico {m.ORTOMOSAICO} | modelos {[x[0] for x in m.MODELOS]} | "
          f"treinados: {feitos} | saida {m.PASTA_AVALIACAO}", flush=True)
    m.main()


if __name__ == "__main__":
    main()
