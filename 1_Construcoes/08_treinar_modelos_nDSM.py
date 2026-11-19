# -*- coding: utf-8 -*-
r"""
08_treinar_modelos_nDSM.py  (ArcGIS Pro 3.6)

Treino sequencial dos modelos de segmentacao de edificios (U-Net, DeepLabV3 e
SAMLoRA), com os chips do 07_exportar_chips_nDSM.py (vetores na posicao original).
Fase B: ortomosaico com 5 bandas (R, G, B, NIR, nDSM). Os pesos da banda do nDSM
sao inicializados sem pre-treino (os codificadores pre-treinados so conhecem RGB).

Na lista EXPERIENCIAS escolhe-se o que treinar (ATIVO = True/False) e, para cada
modelo, o batch size, os argumentos, o learning rate e o numero de epocas.
Regras comuns a todos os modelos e as duas fases: batch 8, 20 epocas, 10 % de
validacao, learning rate fixo de 2e-4, codificador congelado e sem paragem
antecipada, para que os resultados sejam diretamente comparaveis.

Funcionalidades:
  - SALTAR_CONCLUIDOS: salta as experiencias que ja tem um modelo concluido
    (pasta <tipo>_<backbone>_b*_* com .emd) em PASTA_MODELOS; pastas sem .emd
    (treinos interrompidos) nao contam e sao listadas no registo;
  - se faltar memoria na GPU, tenta o batch seguinte da lista;
  - se um ficheiro estiver bloqueado por outro processo (WinError 32, por exemplo
    o antivirus), espera e tenta de novo com o mesmo batch;
  - apaga as pastas de modelos incompletos (treinos que falharam);
  - regista tudo em PASTA_MODELOS:
      log_treinos.txt     registo completo
      resumo_treinos.csv  uma linha por modelo (lr, epocas, melhor epoca, F1, tempo)
      epocas_treinos.csv  a tabela de epocas de cada modelo
      <modelo>\mensagens.txt  as mensagens completas da ferramenta

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro FECHADO):
    python "D:\Users\Cap_Almeida\Projeto\Python\08_treinar_modelos_nDSM.py"

ANTES DE CORRER:
  - "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback" (Painel NVIDIA)
  - Desativar o "Modo de Edicao Rapida" nas Propriedades da janela de comandos
  - Confirmar que nao ha outros python.exe a correr
  - Correr primeiro com MODO_TESTE = True (1 epoca de cada tipo de modelo)
"""

import arcpy
import os
import csv
import glob
import time
import shutil
import traceback
from datetime import datetime

# ============================================================
# CONFIGURACAO
# ============================================================
PASTA_CHIPS   = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\Training\original1_nDSM_256_s128"
PASTA_MODELOS = r"D:\DL\Modelos\126_original_ndsm"

# Argumentos por arquitetura
ARGS_CNN     = "class_balancing True;mixup False;focal_loss False;chip_size 256"
ARGS_SAMLORA = "class_balancing True"

# Valores comuns (podem ser alterados em cada experiencia)
EPOCAS = 20
LR     = 0.0002        # fixo: o calculo automatico revelou-se pouco reprodutivel

# Experiencias: (ATIVO, tipo de modelo, backbone, batch sizes a tentar por ordem,
#                argumentos, learning rate, epocas)
# Ordenadas do mais rapido para o mais lento, para ter resultados cedo.
# Tempos aproximados por modelo (20 epocas, fase A): entre parenteses.
EXPERIENCIAS = [
#    (True,  "DEEPLAB", "RESNET34",  [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~3 h
#    (True,  "UNET",    "RESNET34",  [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~4 h
#    (True,  "DEEPLAB", "RESNET50",  [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~4 h
    (True,  "DEEPLAB", "RESNET101", [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~5 h
#    (True,  "SAMLORA", "VIT_B",     [8, 4, 2], ARGS_SAMLORA, LR, EPOCAS),   # ~5 h
    (True,  "UNET",    "RESNET50",  [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~10 h
#    (True,  "UNET",    "RESNET101", [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~11 h
    (True,  "SAMLORA", "VIT_L",     [8, 4, 2], ARGS_SAMLORA, LR, EPOCAS),   # ~12 h
    (True,  "UNET",    "RESNET152", [8, 4],    ARGS_CNN,     LR, EPOCAS),   # ~14 h
#    (False, "SAMLORA", "VIT_H",     [4, 2],    ARGS_SAMLORA, LR, EPOCAS),   # muito pesado (~25 h ou mais)
]

VALIDACAO_PCT = 10
PARAGEM       = "CONTINUE_TRAINING"    # faz sempre todas as epocas ("STOP_TRAINING" = paragem antecipada)

TENTATIVAS_BLOQUEIO = 3
ESPERA_BLOQUEIO_S   = 120

# Saltar experiencias ja concluidas (pasta com .emd em PASTA_MODELOS)
SALTAR_CONCLUIDOS = True

# Modo teste: True = so 1 epoca da primeira experiencia ATIVA de cada tipo de modelo
MODO_TESTE = False
# ============================================================

FICHEIRO_LOG    = os.path.join(PASTA_MODELOS, "log_treinos.txt")
FICHEIRO_RESUMO = os.path.join(PASTA_MODELOS, "resumo_treinos.csv")
FICHEIRO_EPOCAS = os.path.join(PASTA_MODELOS, "epocas_treinos.csv")
SINAIS_MEMORIA  = ("out of memory", "cuda error", "cublas", "cudnn")
SINAIS_BLOQUEIO = ("winerror 32", "being used by another process")
SINAIS_PALAVRA_CHAVE = ("000800", "not a member of", "invalid value", "000622", "000735")


# ------------------------------------------------------------------ registo
def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(FICHEIRO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def acrescentar_csv(ficheiro, cabecalho, linhas):
    novo = not os.path.exists(ficheiro)
    with open(ficheiro, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        if novo:
            w.writerow(cabecalho)
        w.writerows(linhas)


def ler_lr(msgs):
    for linha in msgs.splitlines():
        if "Learning Rate" in linha:
            nums = []
            for p in linha.replace("(", " ").replace(")", " ").replace(",", " ").split():
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if nums:
                return max(nums)
    return None


def ler_epocas(msgs):
    epocas = {}
    for linha in msgs.splitlines():
        p = linha.split()
        if len(p) >= 6 and p[0].isdigit():
            try:
                valores = [float(x) for x in p[1:5]]
            except ValueError:
                continue
            epocas[int(p[0])] = [int(p[0])] + valores + [p[5]]
    return [epocas[k] for k in sorted(epocas)]


def ler_metricas(msgs):
    """Precision, recall e F1 da classe edificio (ultima coluna da tabela final)."""
    m = {}
    for linha in msgs.splitlines():
        p = linha.split()
        if len(p) >= 2 and p[0] in ("precision", "recall", "f1"):
            try:
                m[p[0]] = float(p[-1])
            except ValueError:
                pass
    return m


def registar(tipo, backbone, bs, epocas_max, pasta, msgs, horas):
    epocas = ler_epocas(msgs)
    met = ler_metricas(msgs)
    melhor = min(epocas, key=lambda e: e[2]) if epocas else None
    modelo = os.path.basename(pasta)
    acrescentar_csv(
        FICHEIRO_RESUMO,
        ["data", "modelo", "tipo", "backbone", "batch", "lr", "epocas_max", "epocas_feitas",
         "melhor_epoca", "melhor_val_loss", "val_loss_final", "precision", "recall", "f1",
         "duracao_h", "pasta"],
        [[datetime.now().strftime("%Y-%m-%d %H:%M"), modelo, tipo, backbone, bs,
          ler_lr(msgs), epocas_max, len(epocas),
          melhor[0] if melhor else "", round(melhor[2], 5) if melhor else "",
          round(epocas[-1][2], 5) if epocas else "",
          met.get("precision", ""), met.get("recall", ""), met.get("f1", ""),
          round(horas, 2), pasta]])
    acrescentar_csv(
        FICHEIRO_EPOCAS,
        ["modelo", "epoca", "perda_treino", "perda_validacao", "accuracy", "dice", "tempo"],
        [[modelo] + e for e in epocas])
    log(f"   REGISTADO {modelo} | epocas={len(epocas)}/{epocas_max} | "
        f"melhor epoca={melhor[0] if melhor else '?'} | F1 validacao={met.get('f1', '?')}")


# ------------------------------------------------------------------ concluidos
def pastas_da_experiencia(tipo, backbone):
    """Pastas em PASTA_MODELOS deste tipo e backbone, separadas em (concluidas, incompletas).
    Concluida = tem pelo menos um .emd (o modelo final)."""
    padrao = os.path.join(PASTA_MODELOS, f"{tipo.lower()}_{backbone.lower()}_b*_*")
    concluidas, incompletas = [], []
    for p in sorted(glob.glob(padrao)):
        if not os.path.isdir(p):
            continue
        (concluidas if glob.glob(os.path.join(p, "*.emd")) else incompletas).append(p)
    return concluidas, incompletas


# ------------------------------------------------------------------ treino
def treinar(tipo, backbone, bs, args, lr, epocas):
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta = os.path.join(PASTA_MODELOS, f"{tipo.lower()}_{backbone.lower()}_b{bs}_{carimbo}")
    log(f"INICIO {tipo} {backbone} | batch={bs} | epocas={epocas} | lr={lr} | saida={pasta}")
    t0 = time.time()
    try:
        arcpy.ia.TrainDeepLearningModel(
            in_folder=PASTA_CHIPS, out_folder=pasta, max_epochs=epocas, model_type=tipo,
            batch_size=bs, arguments=args if args else None, learning_rate=lr,
            backbone_model=backbone, validation_percentage=VALIDACAO_PCT,
            stop_training=PARAGEM, freeze="FREEZE_MODEL", augmentation="DEFAULT",
            monitor="VALID_LOSS")
    except Exception:
        shutil.rmtree(pasta, ignore_errors=True)      # nao deixar modelos incompletos
        raise
    horas = (time.time() - t0) / 3600
    msgs = arcpy.GetMessages()
    with open(os.path.join(pasta, "mensagens.txt"), "w", encoding="utf-8") as f:
        f.write(msgs)
    for i in range(arcpy.GetMessageCount()):
        log("   " + arcpy.GetMessage(i))
    log(f"FIM {tipo} {backbone} | batch={bs} | duracao={horas:.2f} h")
    return pasta, msgs, horas


def treinar_com_tentativas(tipo, backbone, bs, args, lr, epocas):
    """Repete o treino se falhar por ficheiro bloqueado."""
    for tentativa in range(1, TENTATIVAS_BLOQUEIO + 1):
        try:
            return treinar(tipo, backbone, bs, args, lr, epocas)
        except Exception as e:
            erro = (str(e) + "\n" + arcpy.GetMessages()).lower()
            if any(s in erro for s in SINAIS_BLOQUEIO) and tentativa < TENTATIVAS_BLOQUEIO:
                log(f"   ficheiro bloqueado (tentativa {tentativa}/{TENTATIVAS_BLOQUEIO}); "
                    f"nova tentativa em {ESPERA_BLOQUEIO_S} s")
                time.sleep(ESPERA_BLOQUEIO_S)
                continue
            raise


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    if not os.path.isdir(os.path.join(PASTA_CHIPS, "images")):
        raise FileNotFoundError(f"PASTA_CHIPS sem a subpasta 'images': {PASTA_CHIPS}")

    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.env.processorType = "GPU"
    arcpy.env.gpuId = 0
    arcpy.env.overwriteOutput = True

    ativas = [e[1:] for e in EXPERIENCIAS if e[0]]
    if MODO_TESTE:
        vistos, lista = set(), []
        for e in ativas:
            if e[0] not in vistos:
                lista.append(e[:5] + (1,))           # 1 epoca
                vistos.add(e[0])
    else:
        lista = ativas

    log("=" * 60)
    log(f"Sessao iniciada | modo_teste={MODO_TESTE} | saltar_concluidos={SALTAR_CONCLUIDOS} "
        f"| chips={PASTA_CHIPS}")

    # Saltar experiencias ja concluidas (nao se aplica ao modo teste)
    if SALTAR_CONCLUIDOS and not MODO_TESTE:
        por_fazer = []
        for exp in lista:
            tipo, backbone = exp[0], exp[1]
            concluidas, incompletas = pastas_da_experiencia(tipo, backbone)
            for p in incompletas:
                log(f"   AVISO: pasta sem modelo final (treino interrompido?): {p}")
            if concluidas:
                log(f"   SALTADA {tipo} {backbone}: ja concluida em {os.path.basename(concluidas[-1])}")
            else:
                por_fazer.append(exp)
        lista = por_fazer

    if not lista:
        log("Nada para treinar: todas as experiencias ativas ja estao concluidas.")
        log("Sessao terminada")
        return

    for tipo, backbone, batches, args, lr, epocas in lista:
        log(f"   {tipo} {backbone} | batches={batches} | lr={lr} | epocas={epocas} | args='{args}'")

    for tipo, backbone, batches, args, lr, epocas in lista:
        for bs in batches:
            try:
                pasta, msgs, horas = treinar_com_tentativas(tipo, backbone, bs, args, lr, epocas)
                registar(tipo, backbone, bs, epocas, pasta, msgs, horas)
                break                                  # correu bem: passa a experiencia seguinte
            except Exception as e:
                erro = str(e) + "\n" + arcpy.GetMessages()
                log(f"ERRO {tipo} {backbone} com batch={bs}:\n{erro}\n{traceback.format_exc()}")
                txt = erro.lower()
                if any(s in txt for s in SINAIS_MEMORIA):
                    log(f"Provavel falta de memoria na GPU; a tentar batch menor para {tipo} {backbone}")
                    continue
                if any(s in txt for s in SINAIS_BLOQUEIO):
                    log("Ficheiro bloqueado em todas as tentativas; verificar o antivirus ou outros processos.")
                elif any(s in txt for s in SINAIS_PALAVRA_CHAVE):
                    log(f"Provavel palavra-chave invalida em '{tipo}' ou '{backbone}'.")
                break

    log("Sessao terminada")


if __name__ == "__main__":
    main()
