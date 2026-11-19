# -*- coding: utf-8 -*-
r"""
comum_rgb.py - configuracao e funcoes partilhadas pelos scripts da experiencia
"NIR medio vs treino so com RGB" (U-Net ResNet-152 das construcoes).

Pergunta: numa folha sem NIR (235), e melhor
  A) usar o modelo treinado com 4 bandas, com a banda NIR substituida pela media do NIR
     no treino (nao da informacao ao modelo), ou
  B) treinar o mesmo modelo, com as mesmas regras, so com R, G, B?

Os scripts desta pasta NAO alteram os originais: importam o 07, o 08, o 09 e o
1_avaliar_construcoes_235.py, mudam so a configuracao (ortomosaico, pastas, modelo) e
chamam o main() de cada um. Assim os chips, o treino e as metricas sao exatamente os
da fase A.
"""
import os
import sys
import json
import glob
import importlib.util

# ============================================================ CONFIGURACAO
PROJETO = r"D:\Users\Cap_Almeida\Projeto"
PYTHON = os.path.join(PROJETO, "Python")
# onde procurar os scripts originais (por esta ordem)
PASTAS_SCRIPTS = [os.path.join(PYTHON, "1_Construcoes"), PYTHON, os.path.join(PYTHON, "5_Carta_235")]

# modelos
PASTA_MODELOS_NIR = r"D:\DL\Modelos\126_construcoes"                      # fase A, 4 bandas
MODELO_NIR = "unet_resnet152_b8_20261024_050812"                          # U-Net R152, 4 bandas
PASTA_MODELOS_RGB = r"D:\DL\Modelos\126_construcoes_rgb"                  # novo, 3 bandas

# ortomosaicos (os de "NIR medio" e "RGB" sao criados pelo 1_preparar_ortos.py)
ORTO_126 = os.path.join(PROJETO, "126", "Ortomosaico", "126.tif")
ORTO_126_RGB = os.path.join(PROJETO, "126", "Ortomosaico_RGB", "126.tif")
ORTO_126_NIRM = os.path.join(PROJETO, "126", "Ortomosaico_NIR_medio", "126.tif")
ORTO_235 = os.path.join(PROJETO, "235", "Ortomosaico", "235.tif")             # so RGB
ORTO_235_NIRM = os.path.join(PROJETO, "235", "Ortomosaico_NIR_medio", "235.tif")
# (os ficheiros mantem o nome 126.tif / 235.tif para os blocos terem os mesmos nomes
#  da verificacao radiometrica, ex.: 235_bloco_L4C4)

# chips RGB (mesmos parametros dos da fase A)
CHIPS_RGB = os.path.join(PROJETO, "126", "126_Construcoes", "Training", "rgb_256_s128")

# avaliacoes
AVAL_126 = r"D:\DL\Avaliacao\126_rgb_vs_nir"
AVAL_235_A = r"D:\DL\Avaliacao\235_construcoes_A_nir_medio"
AVAL_235_B = r"D:\DL\Avaliacao\235_construcoes_B_rgb"
# ============================================================


def carregar(nome_ficheiro, nome_modulo=None):
    """Importa um script pelo nome do ficheiro (procura em PASTAS_SCRIPTS), sem o executar
    como programa (o bloco if __name__ == '__main__' nao corre)."""
    for d in PASTAS_SCRIPTS:
        caminho = os.path.join(d, nome_ficheiro)
        if os.path.exists(caminho):
            if d not in sys.path:
                sys.path.insert(0, d)
            spec = importlib.util.spec_from_file_location(nome_modulo or "m_" + os.path.splitext(nome_ficheiro)[0],
                                                          caminho)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"   carregado: {caminho}", flush=True)
            return mod
    raise FileNotFoundError(f"{nome_ficheiro} nao encontrado em {PASTAS_SCRIPTS}")


def emd_do_modelo(pasta):
    f = sorted(glob.glob(os.path.join(pasta, "*.emd")))
    if not f:
        raise FileNotFoundError(f"sem .emd em {pasta}")
    with open(f[0], encoding="utf-8") as fh:
        return json.load(fh)


def nir_medio():
    """Media do NIR no treino do U-Net R152 de 4 bandas (band_mean_values[3] do .emd)."""
    d = emd_do_modelo(os.path.join(PASTA_MODELOS_NIR, MODELO_NIR))
    return float(d["NormalizationStats"]["band_mean_values"][3])


def modelos_rgb():
    """Nomes das pastas dos modelos RGB ja treinados (com .dlpk)."""
    if not os.path.isdir(PASTA_MODELOS_RGB):
        return []
    return sorted(os.path.basename(os.path.dirname(f))
                  for f in glob.glob(os.path.join(PASTA_MODELOS_RGB, "*", "*.dlpk")))
