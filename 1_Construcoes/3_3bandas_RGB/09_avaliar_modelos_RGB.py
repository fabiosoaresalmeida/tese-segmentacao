# -*- coding: utf-8 -*-
r"""
09_avaliar_modelos_RGB.py  (ArcGIS Pro 3.6) - FASE C (so RGB)

Avalia os quatro modelos RGB nas 3 ortos de teste da 126 (126-I-3, 126-II-3, 126-IV-1)
com o 09_avaliar_modelos.py ORIGINAL: mesmas ortos com 256 m de contexto e bordas
espelhadas, mesma limpeza e regularizacao, mesmas metricas ao pixel e por objeto e
mesmas referencias ("edicao" e "fotogrametria") das fases A e B.

Grupo "faseC_rgb": modelos de PASTA_MODELOS_RGB no ortomosaico RGB (3 bandas).
Resultados em AVAL_RGB (D:\DL\Avaliacao\126_Construcao_rgb): tabela_comparativa.txt,
metricas_pixel.csv, metricas_objeto.csv, metricas_por_area.csv, log_avaliacao.txt,
classificacoes e poligonos.gdb. As classificacoes ja feitas sao reaproveitadas, por
isso pode-se correr outra vez depois de uma interrupcao.

Correr no Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO, sem treinos na GPU),
na pasta 1_Construcoes\3_3bandas_RGB:
    python 09_avaliar_modelos_RGB.py
"""
import os

import comum_RGB as C


def main():
    modelos = C.modelos_rgb_da_fase_b()
    if not modelos:
        raise RuntimeError(f"Ainda nao ha modelos RGB em {C.PASTA_MODELOS_RGB} (correr o 08_treinar_modelos_RGB.py)")
    if len(modelos) < len(C.MODELOS_FASE_B):
        print(f"ATENCAO: so {len(modelos)} dos {len(C.MODELOS_FASE_B)} modelos estao concluidos: {modelos}")
    if not os.path.exists(C.ORTO_RGB):
        raise FileNotFoundError(f"{C.ORTO_RGB} (correr o 07_exportar_chips_RGB.py)")

    os.makedirs(C.AVAL_RGB, exist_ok=True)
    m = C.carregar("09_avaliar_modelos.py")
    m.GRUPOS = [(C.GRUPO_RGB, C.PASTA_MODELOS_RGB, C.ORTO_RGB)]
    m.MODELOS = modelos
    m.PASTA_AVALIACAO = C.AVAL_RGB
    m.FICHEIRO_LOG = os.path.join(C.AVAL_RGB, "log_avaliacao.txt")
    m.GDB = os.path.join(C.AVAL_RGB, "poligonos.gdb")
    m.MODO_TESTE = False
    print(f"Avaliacao FASE C (RGB) | modelos {m.MODELOS} | ortos {m.ORTOS_TESTE} | saida {m.PASTA_AVALIACAO}",
          flush=True)
    m.main()


if __name__ == "__main__":
    main()
