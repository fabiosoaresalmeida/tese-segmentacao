# -*- coding: utf-8 -*-
r"""
2_exportar_chips_rgb.py - exporta os chips de treino RGB com o 53_exportar_chips_todas_classes.py
ORIGINAL, mudando so o ortomosaico (126 RGB) e a pasta de saida. Rotulos de treino, mascara
(13 ortos de treino), campo classe, 256 px, passo 128, ALL_TILES: iguais aos de 4 bandas.

No fim confirma que os chips tem 3 bandas e compara o n.o de chips com o da exportacao de
4 bandas (tem de ser o mesmo: so muda a imagem).

Se a pasta ja tiver uma exportacao completa (exportacao_chips_relatorio.txt), nao faz nada.
Uma exportacao interrompida (pasta sem relatorio) e apagada e feita de novo.

Corre no CPU: pode correr com um treino a decorrer na GPU.
Correr no Python Command Prompt do ArcGIS Pro:
    python 2_exportar_chips_rgb.py
"""
import os
import glob
import traceback

import comum_todas_rgb as C

RELATORIO = "exportacao_chips_relatorio.txt"


def contar_chips(pasta):
    return len(glob.glob(os.path.join(pasta, "images", "*.tif")))


def main():
    if os.path.exists(os.path.join(C.CHIPS_RGB, RELATORIO)):
        print(f"Chips RGB ja exportados: {C.CHIPS_RGB} ({contar_chips(C.CHIPS_RGB)} chips)")
        return
    if not os.path.exists(C.ORTO_126_RGB):
        raise FileNotFoundError(f"Falta o ortomosaico RGB (correr o 1_preparar_orto_rgb.py): {C.ORTO_126_RGB}")
    m = C.carregar("53_exportar_chips_todas_classes.py")
    chips_4b = m.PASTA_SAIDA                     # a exportacao de 4 bandas, para comparar
    m.MOSAICO = C.ORTO_126_RGB
    m.ROTULOS_TREINO = C.ROTULOS_TREINO
    m.MASCARA_TREINO = C.MASCARA_TREINO
    m.PASTA_SAIDA = C.CHIPS_RGB
    m.APAGAR_SAIDA_SE_EXISTIR = True             # so ha pasta aqui se uma exportacao anterior ficou a meio
    print(f"Exportacao RGB | mosaico {m.MOSAICO} | rotulos {m.ROTULOS_TREINO} | "
          f"{m.TILE} px, passo {m.STRIDE} | saida {m.PASTA_SAIDA}", flush=True)
    rel = m.Relatorio()
    try:
        m.main(rel)
    except Exception:
        rel()
        rel("ERRO - execucao interrompida")
        rel(traceback.format_exc())
        destino = m.PASTA_SAIDA if os.path.isdir(m.PASTA_SAIDA) else os.path.dirname(m.PASTA_SAIDA)
        rel.gravar(os.path.join(destino, "exportacao_chips_relatorio_ERRO.txt"))
        raise

    # verificacoes finais
    nb = C.bandas_chips(C.CHIPS_RGB)
    n_rgb, n_4b = contar_chips(C.CHIPS_RGB), contar_chips(chips_4b)
    print(f"Chips RGB: {n_rgb} | bandas por chip: {nb} | chips de 4 bandas ({chips_4b}): {n_4b}")
    if nb != 3:
        raise RuntimeError(f"Os chips exportados tem {nb} bandas (esperadas 3)")
    if n_4b and n_rgb != n_4b:
        print(f"AVISO: n.o de chips diferente do de 4 bandas ({n_rgb} vs {n_4b}) - verificar a mascara")


if __name__ == "__main__":
    main()
