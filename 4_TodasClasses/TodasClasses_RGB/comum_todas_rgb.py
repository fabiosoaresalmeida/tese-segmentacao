# -*- coding: utf-8 -*-
r"""
comum_todas_rgb.py - configuracao e funcoes partilhadas pelos scripts da experiencia
"TODAS AS CLASSES so com RGB" (3 bandas), para comparar com o modelo de todas as classes
de 4 bandas (RGB + NIR) da pasta 4_TodasClasses (esta pasta e uma subpasta dela).

Os scripts desta pasta NAO alteram os originais: importam o 53, o 54 e o 55, mudam so a
configuracao (ortomosaico, pastas, nomes dos modelos) e chamam o main() de cada um.
Assim os rotulos, a mascara de treino, os chips (256/128), os parametros de treino e as
metricas sao exatamente os do modelo de 4 bandas; a unica diferenca e a falta do NIR.
"""
import os
import sys
import glob
import json
import importlib.util

# ============================================================ CONFIGURACAO
PROJETO = r"D:\Users\Cap_Almeida\Projeto"
PYTHON = os.path.join(PROJETO, "Python")
PASTA = os.path.dirname(os.path.abspath(__file__))
# onde procurar os scripts originais 53, 54 e 55 (por esta ordem): a pasta de cima
# (4_TodasClasses, onde esta subpasta fica), depois Python\4_TodasClasses, esta pasta e Python
PASTAS_SCRIPTS = [os.path.dirname(PASTA), os.path.join(PYTHON, "4_TodasClasses"), PASTA, PYTHON]

# ortomosaicos
ORTO_126 = os.path.join(PROJETO, "126", "Ortomosaico", "126.tif")              # 4 bandas
ORTO_126_RGB = os.path.join(PROJETO, "126", "Ortomosaico_RGB", "126.tif")      # bandas 1-3 do 126.tif
# (o mesmo RGB do 6_RGB_vs_NIR e do MTRE_RGB; o 1_preparar_orto_rgb.py so o cria se faltar)

# rotulos e mascara (do 52, os mesmos do modelo de 4 bandas)
SEP = os.path.join(PROJETO, "126", "126_TodasClasses", "52_separacao")
ROTULOS_TREINO = os.path.join(SEP, "rotulos_treino.shp")
MASCARA_TREINO = os.path.join(SEP, "mascara_treino.shp")

# chips RGB (mesmos parametros do 53: 256 px, passo 128, ALL_TILES)
CHIPS_RGB = os.path.join(PROJETO, "126", "126_TodasClasses", "53_Training", "chips_todas_rgb_256_128")

# modelos
PASTA_MODELOS_4B = r"D:\DL\Modelos\126_modelos_todas_classes"      # 4 bandas (54)
PASTA_MODELOS_RGB = r"D:\DL\Modelos\126_todas_classes_rgb"          # novo, 3 bandas

# avaliacoes (com TTA, como no 55)
AVAL_4B = r"D:\DL\Avaliacao\126_todas_classes_TTA"                  # 4 bandas (55)
AVAL_RGB = r"D:\DL\Avaliacao\126_todas_classes_rgb_TTA"             # novo, 3 bandas

# modelos a treinar e a avaliar: pasta do modelo -> rotulo na avaliacao de 4 bandas
# (os rotulos RGB sao estes com o sufixo _rgb)
MODELOS = {
    "samlora":       "samlora_todas",
    "deeplabv3_r50": "deeplabv3_todas",
    "unet_r50":      "unet_todas",
}
# ============================================================


def carregar(nome_ficheiro):
    """Importa um script original pelo nome do ficheiro (procura em PASTAS_SCRIPTS), sem o
    executar como programa (o bloco if __name__ == '__main__' nao corre)."""
    for d in PASTAS_SCRIPTS:
        caminho = os.path.join(d, nome_ficheiro)
        if os.path.exists(caminho):
            if d not in sys.path:
                sys.path.insert(0, d)
            nome = "m_" + os.path.splitext(nome_ficheiro)[0]
            spec = importlib.util.spec_from_file_location(nome, caminho)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"   carregado: {caminho}", flush=True)
            return mod
    raise FileNotFoundError(f"{nome_ficheiro} nao encontrado em {PASTAS_SCRIPTS}")


def encontrar_original(nome_ficheiro):
    for d in PASTAS_SCRIPTS:
        if os.path.exists(os.path.join(d, nome_ficheiro)):
            return os.path.join(d, nome_ficheiro)
    return None


def modelos_treinados(pasta_modelos):
    """Pastas dos modelos ja treinados (com .emd) em pasta_modelos."""
    return sorted(n for n in MODELOS
                  if glob.glob(os.path.join(pasta_modelos, n, "*.emd")))


def bandas_emd(pasta_modelo):
    """N.o de bandas com que o modelo foi treinado (ExtractBands do .emd)."""
    f = sorted(glob.glob(os.path.join(pasta_modelo, "*.emd")))
    if not f:
        return None
    with open(f[0], encoding="utf-8") as fh:
        d = json.load(fh)
    b = d.get("ExtractBands")
    return len(b) if b else None


def bandas_chips(pasta_chips):
    """N.o de bandas das imagens dos chips (le o primeiro chip com o arcpy)."""
    imgs = sorted(glob.glob(os.path.join(pasta_chips, "images", "*.tif")))
    if not imgs:
        return None
    import arcpy
    return arcpy.Raster(imgs[0]).bandCount
