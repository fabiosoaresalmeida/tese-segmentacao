# -*- coding: utf-8 -*-
r"""
9_cadeia_rgb.py - corre SEM VIGILANCIA a experiencia "NIR medio vs treino so RGB"
(U-Net R152 das construcoes): preparar ortos, exportar chips RGB, treinar, avaliar.

1. VERIFICACOES no arranque (feitas logo, enquanto ainda esta ao computador):
   - todos os scripts, o comum_rgb.py e os scripts originais (07, 08, 09, 1_avaliar) existem;
   - nenhum script tem MODO_TESTE = True;
   - os ortomosaicos 126 e 235 e o modelo R152 de 4 bandas existem.
   Se alguma falhar, a cadeia NAO arranca e diz porque.
2. ESPERA que o treino em curso termine (processo ESPERAR_PID, ver no nvidia-smi a linha
   "...arcgispro-py3\python.exe") e depois mais ESPERA_FINAL_S, para a GPU libertar a memoria.
   ESPERAR_PID = None -> comeca logo.
3. CORRE os scripts, um de cada vez, no mesmo Python. Se um falhar, volta a tenta-lo
   (TENTATIVAS); as classificacoes ja feitas sao reaproveitadas, por isso a repeticao so
   refaz o que faltou. Depois passa ao seguinte.
4. Mantem o Windows ACORDADO enquanto corre (nao suspende nem hiberna; o ecra pode desligar).

Tudo fica registado em log_cadeia_rgb.txt (nesta pasta) e nos logs de cada script.

Correr numa SEGUNDA janela do Python Command Prompt do ArcGIS Pro (ArcGIS Pro FECHADO):
    cd D:\Users\Cap_Almeida\Projeto\Python\6_RGB_vs_NIR
    python 9_cadeia_rgb.py
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
    "1_preparar_ortos.py",             # 126 RGB, 126 NIR medio, 235 NIR medio (minutos)
    "2_exportar_chips_rgb.py",         # chips RGB com o 07 (horas, CPU)
    "3_treinar_rgb.py",                # U-Net R152 RGB com o 08 (~14 h, GPU)
    "4_avaliar_126.py",                # 126: NIR real vs NIR medio vs RGB, com o 09 (~2 h)
    "5_avaliar_235_A_nir_medio.py",    # 235, solucao A (~0,5-1 dia)
    "6_avaliar_235_B_rgb.py",          # 235, solucao B (~0,5-1 dia)
]
# cada passo verifica se ja esta feito (ortos, chips, modelo) e salta-o; as avaliacoes
# reaproveitam as classificacoes ja feitas

ESPERAR_PID = None          # PID de um processo a esperar (ex.: o treino em curso); None = nao esperar
INTERVALO_S = 300           # verificar de 5 em 5 minutos
ESPERA_FINAL_S = 180        # margem depois do fim desse processo
TENTATIVAS = 2              # cada script corre no maximo 2 vezes se falhar

# ficheiros que tem de existir antes de arrancar
FICHEIROS_NECESSARIOS = [
    os.path.join(PASTA, "comum_rgb.py"),
    os.path.join(PASTA, "avaliar_235_comum.py"),
    r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif",
    r"D:\Users\Cap_Almeida\Projeto\235\Ortomosaico\235.tif",
    r"D:\DL\Modelos\126_construcoes\unet_resnet152_b8_20261024_050812",
    r"D:\Users\Cap_Almeida\Projeto\235\Verificacoes_dominio\radiometria_por_orto.csv",
]
# scripts originais que tem de existir (procurados como no comum_rgb.py)
ORIGINAIS = ["07_exportar_chips.py", "08_treinar_modelos.py", "09_avaliar_modelos.py",
             "1_avaliar_construcoes_235.py"]
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
    for f in FICHEIROS_NECESSARIOS:
        if not os.path.exists(f):
            erros.append(f"nao existe {f}")
    sys.path.insert(0, PASTA)
    import comum_rgb
    for o in ORIGINAIS:
        if not any(os.path.exists(os.path.join(d, o)) for d in comum_rgb.PASTAS_SCRIPTS):
            erros.append(f"nao encontrei o script original {o} em {comum_rgb.PASTAS_SCRIPTS}")
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
    log(f"Cadeia RGB vs NIR medio | {len(SCRIPTS)} scripts: {SCRIPTS} | python: {sys.executable}")
    log(f"Verificacoes OK | Windows mantido acordado: {'sim' if acordado else 'NAO (verificar energia)'}")

    if ESPERAR_PID is not None:
        log(f"A espera que o treino (PID {ESPERAR_PID}) termine...")
        while processo_ativo(ESPERAR_PID):
            time.sleep(INTERVALO_S)
        log(f"Treino terminado. Primeiro script dentro de {ESPERA_FINAL_S} s.")
        time.sleep(ESPERA_FINAL_S)

    resumo = []
    for i, s in enumerate(SCRIPTS, 1):
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

    log("Cadeia terminada:\n  " + "\n  ".join(resumo))
    manter_acordado(False)


if __name__ == "__main__":
    main()
