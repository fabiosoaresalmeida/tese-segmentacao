# -*- coding: utf-8 -*-
r"""
avaliar_235_comum.py - corre o 1_avaliar_construcoes_235.py ORIGINAL (mesma limpeza da
referencia, mesmas metricas ao pixel e por objeto, mesma regularizacao, metricas por
bloco cruzadas com a radiometria) com outro ortomosaico e outro modelo.
Usado pelo 5_avaliar_235_A_nir_medio.py e pelo 6_avaliar_235_B_rgb.py.
"""
import os

import comum_rgb as C


def correr(ortomosaico, pasta_modelos, modelos, pasta_avaliacao):
    if not os.path.exists(ortomosaico):
        raise FileNotFoundError(f"{ortomosaico} (correr o 1_preparar_ortos.py)")
    m = C.carregar("1_avaliar_construcoes_235.py")
    m.ORTOMOSAICO_NOVO = ortomosaico
    m.PASTA_MODELOS = pasta_modelos
    m.MODELOS = modelos
    m.PASTA_AVALIACAO = pasta_avaliacao
    m.FICHEIRO_LOG = os.path.join(pasta_avaliacao, "log_avaliacao.txt")
    m.GDB = os.path.join(pasta_avaliacao, "poligonos.gdb")
    m.MODO_TESTE = False
    print(f"Avaliacao 235 | ortomosaico {ortomosaico} | modelos {modelos} em {pasta_modelos} | "
          f"saida {pasta_avaliacao}", flush=True)
    m.main()
