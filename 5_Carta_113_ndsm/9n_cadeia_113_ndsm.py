# -*- coding: utf-8 -*-
r"""
9n_cadeia_113_ndsm.py - corre SEM VIGILANCIA a avaliacao da folha 113 com os modelos
com nDSM (5 bandas): construcoes (1n), rede viaria (2n), ocupacao do solo (3n) e
agregacoes (3c).
Copia do 9_cadeia_113.py; tambem verifica que o ortomosaico tem 5 bandas.

1. VERIFICACOES no arranque (feitas logo, enquanto ainda esta ao computador):
   - todos os scripts e o comum_113.py existem;
   - nenhum script tem MODO_TESTE = True;
   - o ortomosaico de 5 bandas e a cartografia da 113 existem;
   - o ortomosaico tem mesmo 5 bandas (senao os modelos seriam todos ignorados).
   Se alguma falhar, a cadeia NAO arranca e diz porque.
2. ESPERA que o treino em curso termine (processo ESPERAR_PID, ver no nvidia-smi a linha
   "...arcgispro-py3\python.exe") e depois mais ESPERA_FINAL_S, para a GPU libertar a memoria.
   ESPERAR_PID = None -> comeca logo.
3. CORRE os scripts, um de cada vez, no mesmo Python. Se um falhar, volta a tenta-lo
   (TENTATIVAS); as classificacoes ja feitas sao reaproveitadas, por isso a repeticao so
   refaz o que faltou. Depois passa ao seguinte.
4. Mantem o Windows ACORDADO enquanto corre (nao suspende nem hiberna; o ecra pode desligar).

Tudo fica registado em log_cadeia_113_ndsm.txt (nesta pasta) e nos logs de cada script.

Correr numa SEGUNDA janela do Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO):
    cd D:\Users\Cap_Almeida\Projeto\Python\5_Carta_113
    python 9n_cadeia_113_ndsm.py
NAO fechar essa janela. Numa maquina virtual por ambiente remoto: DESLIGAR a ligacao
(fechar a janela do Ambiente de Trabalho Remoto), NUNCA "Terminar sessao".
"""
import os
import re
import glob
import sys
import time
import subprocess
from datetime import datetime

PASTA = os.path.dirname(os.path.abspath(__file__))

# =============================== CONFIGURACAO ===============================
ORTO_5BANDAS = r"D:\Users\Cap_Almeida\Projeto\113\Ortomosaico\113_ndsm\113_nDSM_5bandas.tif"

# (script, argumentos)
SCRIPTS = [
    ("1n_avaliar_construcoes_113_ndsm.py", []),     # construcoes, 4 modelos
    ("2n_avaliar_rviaria_113_ndsm.py", []),         # rede viaria e ferrovia, 2 MTRE
    ("3n_avaliar_ocupacao_113_ndsm.py", []),        # ocupacao do solo, 2 modelos com TTA
    ("3c_agregacoes_ocupacao_113.py", [r"D:\DL\Avaliacao\113_ocupacao_nDSM_TTA"]),  # segundos
]
# ja corridos: 0_verificar_dominio_126_113.py e 3a_rotulos_ocupacao_113.py

ESPERAR_PID = None          # PID do treino em curso (nvidia-smi); None = nao esperar
INTERVALO_S = 300           # verificar de 5 em 5 minutos
ESPERA_FINAL_S = 180        # margem depois do fim do treino
TENTATIVAS = 2              # cada script corre no maximo 2 vezes se falhar

# ficheiros que tem de existir antes de arrancar
FICHEIROS_NECESSARIOS = [
    os.path.join(PASTA, "comum_113.py"),
    ORTO_5BANDAS,
    r"D:\Users\Cap_Almeida\Projeto\M888_SHP\113_2018_5\a_constr.shp",
    r"D:\Users\Cap_Almeida\Projeto\M888_SHP\113_2018_5\l_vias.shp",
    r"D:\Users\Cap_Almeida\Projeto\113\Verificacoes_dominio\radiometria_por_orto.csv",
    r"D:\Users\Cap_Almeida\Projeto\113\113_OcupacaoSolo\rotulos_ocupacao\rotulos_ocupacao.shp",
    r"D:\DL\Modelos\126_construcoes_ndsm",
    r"D:\DL\Modelos\126_ocupacao_nDSM\deeplabv3_r50",
    r"D:\DL\Modelos\126_ocupacao_nDSM\samlora",
    r"D:\DL\Modelos\126_RViaria_nDSM\mtre_rede_viaria_ndsm_resnet50_b8_lrauto_treino_20261126_1339",
    r"D:\DL\Modelos\126_RViaria_nDSM\mtre_ferrovia_ndsm_resnet50_b8_lrauto_treino_20261127_0634",
]
LOG = os.path.join(PASTA, "log_cadeia_113_ndsm.txt")
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
    for s, _ in SCRIPTS:
        caminho = os.path.join(PASTA, s)
        if not os.path.exists(caminho):
            erros.append(f"nao existe o script {s}")
            continue
        with open(caminho, encoding="utf-8", errors="ignore") as f:
            if re.search(r"^MODO_TESTE\s*=\s*True", f.read(), re.M):
                erros.append(f"{s} tem MODO_TESTE = True (por False)")
    for f in FICHEIROS_NECESSARIOS:
        if not os.path.exists(f):
            erros.append(f"nao existe {f}")
        elif "126_RViaria_nDSM" in f and not glob.glob(os.path.join(f, "*.emd")):
            erros.append(f"sem .emd em {f} (o treino acabou?)")
    if os.path.exists(ORTO_5BANDAS):
        # num processo a parte: o import do arcpy aqui estragaria o nome datetime
        print("a verificar o n.o de bandas do ortomosaico (carrega o arcpy)...", flush=True)
        r = subprocess.run([sys.executable, "-c",
                            "import sys, arcpy; print(arcpy.Describe(sys.argv[1]).bandCount)",
                            ORTO_5BANDAS], capture_output=True, text=True)
        nb = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
        if nb != "5":
            erros.append(f"o ortomosaico tem {nb} bandas (tem de ter 5: R, G, B, NIR, nDSM)")
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
    log(f"Cadeia 113 nDSM | {len(SCRIPTS)} scripts: {SCRIPTS} | python: {sys.executable}")
    log(f"Verificacoes OK | Windows mantido acordado: {'sim' if acordado else 'NAO (verificar energia)'}")

    if ESPERAR_PID is not None:
        log(f"A espera que o treino (PID {ESPERAR_PID}) termine...")
        while processo_ativo(ESPERAR_PID):
            time.sleep(INTERVALO_S)
        log(f"Treino terminado. Primeiro script dentro de {ESPERA_FINAL_S} s.")
        time.sleep(ESPERA_FINAL_S)

    resumo = []
    for i, (s, args) in enumerate(SCRIPTS, 1):
        for tentativa in range(1, TENTATIVAS + 1):
            log(f"[{i}/{len(SCRIPTS)}] INICIO {s} (tentativa {tentativa}/{TENTATIVAS})")
            t0 = time.time()
            try:
                codigo = subprocess.run([sys.executable, os.path.join(PASTA, s)] + args, cwd=PASTA).returncode
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

    log("Cadeia terminada:\n  " + "\n  ".join(resumo))
    manter_acordado(False)


if __name__ == "__main__":
    main()