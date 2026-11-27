# -*- coding: utf-8 -*-
r"""
comum_RGB.py - configuracao partilhada da FASE C (so R, G, B) das construcoes.

Fase C: os MESMOS quatro modelos da fase B (U-Net ResNet-50 e ResNet-152, DeepLabV3
ResNet-101, SAMLoRA ViT-L), com as mesmas regras de treino (batch 8, LR fixo 2e-4, 10 %
de validacao, codificador congelado, class_balancing, chips 256 px com passo 128), mas so
com as bandas R, G e B do ortomosaico. O numero de epocas e a paragem antecipada sao
definidos em EPOCAS e PARAGEM (ver abaixo).

Os scripts da fase C NAO alteram os originais: importam o 07, o 08 e o 09 desta pasta,
mudam so a configuracao (ortomosaico, pastas, lista de modelos) e chamam o main() de cada
um. Assim os chips, o treino e as metricas sao exatamente os das fases A e B.

Ordem (na pasta 1_Construcoes\3_3bandas_RGB):
    python 07_exportar_chips_RGB.py    ortomosaico RGB (se faltar) e chips RGB   (CPU, horas)
    python 08_treinar_modelos_RGB.py   treino dos 4 modelos                     (GPU)
    python 09_avaliar_modelos_RGB.py   avaliacao nas 3 ortos de teste            (GPU, ~2-3 h)
e depois, na pasta 1_Construcoes\4_Avaliacao_Figuras:
    python 13_comparar_fases.py        tabela A (4 bandas) x B (nDSM) x C (RGB)

Os scripts originais sao procurados nesta pasta e nas outras subpastas de 1_Construcoes
(07 em 1_4bandas_NIR, 08_nDSM em 2_5bandas_nDSM, 09 em 4_Avaliacao_Figuras).
"""
import os
import sys
import glob
import importlib.util

# ============================================================ CONFIGURACAO
PROJETO = r"D:\Users\Cap_Almeida\Projeto"
PASTA_SCRIPTS = os.path.dirname(os.path.abspath(__file__))   # esta pasta (3_3bandas_RGB)
PASTA_CONSTRUCOES = os.path.dirname(PASTA_SCRIPTS)            # 1_Construcoes, com as subpastas das fases

# Ortomosaico de 4 bandas (R, G, B, NIR) e o RGB criado a partir dele (bandas 1-3).
# O RGB e o mesmo da experiencia 6_RGB_vs_NIR: se ja existir, e reutilizado.
ORTO_4BANDAS = os.path.join(PROJETO, "126", "Ortomosaico", "126.tif")
ORTO_RGB = os.path.join(PROJETO, "126", "Ortomosaico_RGB", "126.tif")

# Chips RGB (mesmos parametros da fase A; tambem partilhados com a experiencia 6)
CHIPS_RGB = os.path.join(PROJETO, "126", "126_Construcoes", "Training", "rgb_256_s128")

# Regime de treino. As fases A e B usaram 20 epocas fixas ("CONTINUE_TRAINING").
# Aqui: ate 50 epocas com paragem antecipada ("STOP_TRAINING": para quando a perda de
# validacao deixa de melhorar). Em ambos os casos e gravado o modelo da melhor epoca.
EPOCAS = 50
PARAGEM = "STOP_TRAINING"          # "CONTINUE_TRAINING" = faz sempre todas as epocas

# Modelos RGB: pasta propria para cada regime, para nao misturar modelos de 20 epocas
# (como o U-Net ResNet-152 RGB da experiencia 6, em 126_construcoes_rgb) com os de 50.
PASTA_MODELOS_RGB = (r"D:\DL\Modelos\126_construcoes_rgb" if (EPOCAS, PARAGEM) == (20, "CONTINUE_TRAINING")
                     else rf"D:\DL\Modelos\126_construcoes_rgb_e{EPOCAS}" + ("_pa" if PARAGEM == "STOP_TRAINING" else ""))

# Avaliacao da fase C e avaliacoes ja feitas das fases A e B (para a comparacao)
GRUPO_RGB = "faseC_rgb"
AVAL_RGB = os.path.join(r"D:\DL\Avaliacao",
                        os.path.basename(PASTA_MODELOS_RGB).replace("126_construcoes", "126_Construcao"))
AVAL_A = r"D:\DL\Avaliacao\126_Construcao"
AVAL_B = r"D:\DL\Avaliacao\126_Construcao_ndsm"

# Os quatro modelos da fase B: (tipo, backbone, batch sizes a tentar por ordem)
MODELOS_FASE_B = [
    ("DEEPLAB", "RESNET101", [8, 4]),       # ~5 h
    ("UNET",    "RESNET50",  [8, 4]),       # ~10 h
    ("SAMLORA", "VIT_L",     [8, 4, 2]),    # ~12 h
    ("UNET",    "RESNET152", [8, 4]),       # ~14 h
]
# ============================================================


def pastas_scripts():
    """Esta pasta, depois as outras subpastas de 1_Construcoes, depois a propria 1_Construcoes."""
    outras = sorted(d for d in glob.glob(os.path.join(PASTA_CONSTRUCOES, "*"))
                    if os.path.isdir(d) and os.path.abspath(d) != PASTA_SCRIPTS)
    return [PASTA_SCRIPTS] + outras + [PASTA_CONSTRUCOES]


def carregar(nome_ficheiro):
    """Importa um script original pelo nome do ficheiro, SEM o executar como programa
    (o bloco if __name__ == '__main__' nao corre). Procura-o em pastas_scripts()."""
    pastas = pastas_scripts()
    caminho = next((os.path.join(d, nome_ficheiro) for d in pastas
                    if os.path.exists(os.path.join(d, nome_ficheiro))), None)
    if caminho is None:
        raise FileNotFoundError(f"Nao encontrei o script original {nome_ficheiro} em {pastas}")
    pasta = os.path.dirname(caminho)
    if pasta not in sys.path:
        sys.path.insert(0, pasta)
    nome_modulo = "m_" + os.path.splitext(nome_ficheiro)[0]
    spec = importlib.util.spec_from_file_location(nome_modulo, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"   carregado: {caminho}", flush=True)
    return mod


def modelos_rgb():
    """Nomes das pastas dos modelos RGB ja concluidos (com .dlpk)."""
    if not os.path.isdir(PASTA_MODELOS_RGB):
        return []
    return sorted(os.path.basename(os.path.dirname(f))
                  for f in glob.glob(os.path.join(PASTA_MODELOS_RGB, "*", "*.dlpk")))


def modelos_rgb_da_fase_b():
    """Dos modelos RGB concluidos, os que correspondem aos 4 da fase B (o ultimo de cada)."""
    feitos = modelos_rgb()
    escolhidos = []
    for tipo, backbone, _ in MODELOS_FASE_B:
        pref = f"{tipo.lower()}_{backbone.lower()}_b"
        c = [m for m in feitos if m.startswith(pref)]
        if c:
            escolhidos.append(c[-1])
    return escolhidos
