# -*- coding: utf-8 -*-
r"""
99_cadeia_rgb.py - corre SEM VIGILANCIA a experiencia "ocupacao do solo so com RGB":
exportar e verificar os chips RGB, treinar a DeepLabV3 e o SAMLoRA, avaliar sem e com TTA
e comparar com as versoes de 4 bandas e do nDSM.

1. VERIFICACOES no arranque (feitas logo, enquanto ainda esta ao computador):
   - todos os scripts desta pasta existem e nenhum tem MODO_TESTE = True;
   - existem o ortomosaico RGB e os rotulos, a mascara e a zona de teste do 22;
   - o processo ESPERAR_PID (se indicado) esta a correr.
   Se alguma falhar, a cadeia NAO arranca e diz porque.
2. ESPERA que o processo ESPERAR_PID termine (ex.: um treino em curso; ver o PID no
   nvidia-smi, linha "...arcgispro-py3\python.exe") e depois mais ESPERA_FINAL_S.
   ESPERAR_PID = None -> comeca logo.
3. CORRE os scripts, um de cada vez. Se um falhar, volta a tenta-lo (TENTATIVAS). Os chips
   ja exportados (com relatorio) nao sao exportados de novo; os modelos treinados e as
   classificacoes feitas sao reutilizados pelos proprios scripts.
4. Mantem o Windows ACORDADO enquanto corre (o ecra pode desligar).

Tudo fica registado em log_cadeia_rgb.txt (nesta pasta) e nos logs de cada script.

Correr numa janela do Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO):
    cd D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\3_RGB
    python 99_cadeia_rgb.py
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
    "23_exportar_chips_ocupacao.py",   # chips RGB (~30 min a horas, CPU)
    "24_verificar_chips_sem_rotulo.py",# relatorio dos chips sem rotulo (minutos)
    "25_treinar_ocupacao.py",          # DeepLabV3 e SAMLoRA (~20-32 h cada, GPU)
    "30_avaliar_ocupacao.py",          # 3 ortos de teste sem TTA (~20 min)
    "30_avaliar_ocupacao_TTA.py",      # 3 ortos de teste com TTA (~1-1,5 h)
    "33_comparar_bandas.py",           # tabela RGB vs 4 bandas vs nDSM (segundos)
]
ORTO_RGB = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico_RGB\126.tif"
SEP = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\32_separacao_unet"
CHIPS = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\33_Training_RGB\chips_ocupacao_256_128"
ESPERAR_PID = None          # PID de um processo a esperar (ex.: o treino em curso); None = nao esperar
INTERVALO_S = 300           # verificar de 5 em 5 minutos
ESPERA_FINAL_S = 180        # margem depois do fim desse processo
TENTATIVAS = 2              # cada script corre no maximo 2 vezes se falhar
LOG = os.path.join(PASTA, "log_cadeia_rgb.txt")
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
    for s in SCRIPTS:
        caminho = os.path.join(PASTA, s)
        if not os.path.exists(caminho):
            erros.append(f"nao existe o script {s}")
            continue
        with open(caminho, encoding="utf-8", errors="ignore") as f:
            if re.search(r"^MODO_TESTE\s*=\s*True", f.read(), re.M):
                erros.append(f"{s} tem MODO_TESTE = True (por False)")
    for f in (ORTO_RGB, os.path.join(SEP, "rotulos_treino.shp"), os.path.join(SEP, "mascara_treino.shp"),
              os.path.join(SEP, "rotulos_teste.shp"), os.path.join(SEP, "zona_teste.shp")):
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
    log(f"Cadeia ocupacao do solo RGB | {len(SCRIPTS)} scripts: {SCRIPTS} | python: {sys.executable}")
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
        if s.startswith("23_") and os.path.exists(os.path.join(CHIPS, "exportacao_chips_relatorio.txt")):
            log(f"[{i}/{len(SCRIPTS)}] {s}: chips ja exportados em {CHIPS} -> saltado")
            resumo.append(f"{s}: saltado (chips ja exportados)")
            continue
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
        # sem chips nao ha treino; se o treino falhar so num modelo, a avaliacao corre na
        # mesma com os modelos treinados (salta os que nao tem .emd)
        if codigo != 0 and s.startswith("23_"):
            log(f"{s} falhou: os passos seguintes dependem dele -> cadeia interrompida")
            break

    log("Cadeia terminada:\n  " + "\n  ".join(resumo))
    manter_acordado(False)


if __name__ == "__main__":
    main()
