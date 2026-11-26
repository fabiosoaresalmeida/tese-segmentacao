# -*- coding: utf-8 -*-
r"""
9_cadeia_todas_rgb.py - corre SEM VIGILANCIA a experiencia "todas as classes so com RGB":
confirmar o ortomosaico RGB, exportar os chips RGB, treinar os 3 modelos, avaliar com TTA e
comparar com os modelos de 4 bandas.

1. VERIFICACOES no arranque (feitas logo, enquanto ainda esta ao computador):
   - todos os scripts desta pasta e os originais 53, 54 e 55 existem;
   - nenhum script tem MODO_TESTE = True;
   - existem o 126.tif, os rotulos e a mascara de treino do 52;
   - o processo ESPERAR_PID (se indicado) esta a correr.
   Se alguma falhar, a cadeia NAO arranca e diz porque.
2. ESPERA que o processo ESPERAR_PID termine (ex.: um treino em curso; ver o PID no
   nvidia-smi, linha "...arcgispro-py3\python.exe") e depois mais ESPERA_FINAL_S.
   ESPERAR_PID = None -> comeca logo.
3. CORRE os scripts, um de cada vez. Se um falhar, volta a tenta-lo (TENTATIVAS); cada
   passo salta o que ja esta feito (ortomosaico, chips, modelos treinados, classificacoes).
4. Mantem o Windows ACORDADO enquanto corre (o ecra pode desligar).

Tudo fica registado em log_cadeia_todas_rgb.txt (nesta pasta) e nos logs de cada script.

Correr numa janela do Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO):
    cd D:\Users\Cap_Almeida\Projeto\Python\4_TodasClasses\TodasClasses_RGB
    python 9_cadeia_todas_rgb.py
NAO fechar essa janela. Numa maquina virtual por ambiente remoto: DESLIGAR a ligacao
(fechar a janela do Ambiente de Trabalho Remoto), NUNCA "Terminar sessao".
"""
import os
import re
import sys
import time
import subprocess
from datetime import datetime

PASTA = os.path.dirname(os.path.abspath(__file__))

# =============================== CONFIGURACAO ===============================
SCRIPTS = [
    "1_preparar_orto_rgb.py",          # confirma (ou cria) o 126 RGB (minutos)
    "2_exportar_chips_rgb.py",         # chips RGB com o 53 (horas, CPU)
    "3_treinar_rgb.py",                # SAMLoRA, DeepLabV3, U-Net com o 54 (~20-32 h cada, GPU)
    "4_avaliar_rgb_TTA.py",            # 3 ortos de teste com TTA, com o 55 (~2-3 h)
    "5_comparar_4bandas_rgb.py",       # tabela 4 bandas vs RGB (segundos)
]
ESPERAR_PID = None          # PID de um processo a esperar (ex.: o treino em curso); None = nao esperar
INTERVALO_S = 300           # verificar de 5 em 5 minutos
ESPERA_FINAL_S = 180        # margem depois do fim desse processo
TENTATIVAS = 2              # cada script corre no maximo 2 vezes se falhar
ORIGINAIS = ["53_exportar_chips_todas_classes.py", "54_treinar_todas_classes.py",
             "55_avaliar_todas_classes_TTA.py"]
LOG = os.path.join(PASTA, "log_cadeia_todas_rgb.txt")
# ============================================================================


def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(linha, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def manter_acordado(ligar=True):
    """Impede o Windows de suspender/hibernar enquanto a cadeia corre."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        estado = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if ligar else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(estado)
        return True
    except Exception:
        return False


def processo_ativo(pid):
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                           timeout=60)
        return str(pid) in r.stdout
    except Exception:
        return False


def verificacoes():
    erros = []
    for s in SCRIPTS + ["comum_todas_rgb.py"]:
        caminho = os.path.join(PASTA, s)
        if not os.path.exists(caminho):
            erros.append(f"nao existe o script {s}")
            continue
        with open(caminho, encoding="utf-8", errors="ignore") as f:
            if re.search(r"^MODO_TESTE\s*=\s*True", f.read(), re.M):
                erros.append(f"{s} tem MODO_TESTE = True (por False)")
    sys.path.insert(0, PASTA)
    import comum_todas_rgb as C
    for o in ORIGINAIS:
        caminho = C.encontrar_original(o)
        if not caminho:
            erros.append(f"nao encontrei o script original {o} em {C.PASTAS_SCRIPTS}")
            continue
        with open(caminho, encoding="utf-8", errors="ignore") as f:
            if re.search(r"^MODO_TESTE\s*=\s*True", f.read(), re.M):
                erros.append(f"{o} tem MODO_TESTE = True (por False)")
    for f in (C.ORTO_126, C.ROTULOS_TREINO, C.MASCARA_TREINO):
        if not os.path.exists(f):
            erros.append(f"nao existe {f}")
    if ESPERAR_PID is not None and not processo_ativo(ESPERAR_PID):
        erros.append(f"o processo {ESPERAR_PID} nao esta a correr (confirmar o PID no nvidia-smi, "
                     f"ou ESPERAR_PID = None se o treino ja acabou)")
    return erros


def main():
    erros = verificacoes()
    if erros:
        print("A CADEIA NAO ARRANCOU:\n  - " + "\n  - ".join(erros))
        return
    acordado = manter_acordado(True)
    log("=" * 60)
    log(f"Cadeia todas as classes RGB | {len(SCRIPTS)} scripts: {SCRIPTS} | python: {sys.executable}")
    log(f"Verificacoes OK | Windows mantido acordado: {'sim' if acordado else 'NAO (verificar energia)'}")

    if ESPERAR_PID is not None:
        log(f"A espera que o processo {ESPERAR_PID} termine...")
        while processo_ativo(ESPERAR_PID):
            time.sleep(INTERVALO_S)
        log(f"Processo terminado. Primeiro script dentro de {ESPERA_FINAL_S} s.")
        time.sleep(ESPERA_FINAL_S)

    resumo = []
    for i, s in enumerate(SCRIPTS, 1):
        codigo, h = None, 0.0
        for tentativa in range(1, TENTATIVAS + 1):
            log(f"[{i}/{len(SCRIPTS)}] INICIO {s} (tentativa {tentativa}/{TENTATIVAS})")
            t0 = time.time()
            try:
                codigo = subprocess.run([sys.executable, os.path.join(PASTA, s)], cwd=PASTA).returncode
            except Exception as e:
                codigo = f"excecao: {e}"
            h = (time.time() - t0) / 3600
            log(f"[{i}/{len(SCRIPTS)}] {'OK' if codigo == 0 else f'ERRO ({codigo})'} {s} | {h:.2f} h")
            if codigo == 0:
                break
            if tentativa < TENTATIVAS:
                log("   nova tentativa dentro de 10 min")
                time.sleep(600)
        resumo.append(f"{s}: {'OK' if codigo == 0 else 'ERRO'} ({h:.1f} h na ultima tentativa)")
        # sem ortomosaico ou sem chips nao ha treino; se o treino falhar so num modelo, a
        # avaliacao corre na mesma com os modelos treinados (salta os que nao tem .emd)
        if codigo != 0 and s in ("1_preparar_orto_rgb.py", "2_exportar_chips_rgb.py"):
            log(f"{s} falhou: os passos seguintes dependem dele -> cadeia interrompida")
            break

    log("Cadeia terminada:\n  " + "\n  ".join(resumo))
    manter_acordado(False)


if __name__ == "__main__":
    main()
