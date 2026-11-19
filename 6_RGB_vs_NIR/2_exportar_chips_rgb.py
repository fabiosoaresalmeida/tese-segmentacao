# -*- coding: utf-8 -*-
r"""
2_exportar_chips_rgb.py - exporta os chips de treino RGB com o 07_exportar_chips.py
ORIGINAL, mudando so o ortomosaico (126 RGB) e a pasta de saida. Edificios de treino,
mascara, zona de teste, campo de classe, 256 px, passo 128, 0,5 m: iguais a fase A.
Usa todos os nucleos do CPU (parallelProcessingFactor = 100%).

Se a pasta ja tiver uma exportacao completa (exportacao_relatorio.txt), nao faz nada.

Correr no Python Command Prompt do ArcGIS Pro:
    python 2_exportar_chips_rgb.py
"""
import os
import traceback

import comum_rgb as C


def main():
    if os.path.exists(os.path.join(C.CHIPS_RGB, "exportacao_relatorio.txt")):
        print(f"Chips RGB ja exportados: {C.CHIPS_RGB}")
        return
    m = C.carregar("07_exportar_chips.py")
    if not os.path.exists(C.ORTO_126_RGB):
        raise FileNotFoundError(f"Falta o ortomosaico RGB (correr o 1_preparar_ortos.py): {C.ORTO_126_RGB}")
    m.ORTOMOSAICO = C.ORTO_126_RGB
    m.PASTA_SAIDA = C.CHIPS_RGB
    m.SCRIPT_TREINO = None
    m.APAGAR_SE_EXISTIR = True
    # todos os nucleos (o 07 aplica o FATOR_PARALELO ao ser importado, por isso muda-se
    # tambem diretamente no ambiente do ArcGIS)
    m.FATOR_PARALELO = "85%"
    m.arcpy.env.parallelProcessingFactor = "85%"
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