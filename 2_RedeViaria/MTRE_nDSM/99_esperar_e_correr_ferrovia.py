# -*- coding: utf-8 -*-
r"""
MTRE_nDSM/99_esperar_e_correr_ferrovia.py

Espera que o treino do MTRE da rede viária com nDSM (43_Treino_RViaria_ndsm.py, já a
decorrer) termine e, a seguir, corre uma LISTA de scripts, um de cada vez, pela ordem
indicada: o treino do MTRE da ferrovia com nDSM (os chips da ferrovia já foram
exportados pelo 42). Assim a GPU nunca é usada por dois processos ao mesmo tempo.

O fim do treino é detetado pelos processos: ao arrancar, o script lê no nvidia-smi
os python.exe que estão a usar a GPU (ou usa os PID indicados em PIDS_TREINO) e
espera que TODOS terminem. Depois espera que a memória da GPU fique livre (abaixo de
MEMORIA_LIVRE_MB) e só então corre os scripts da lista. Se a memória não ficar livre
em ESPERA_MEMORIA_MAX_MIN minutos, não lança nada (o 43 cairia para lote 4 e o
treino deixaria de ser comparável com o de 4 bandas).

Antes de esperar, confirma que:
  - os scripts da lista existem;
  - o 43 que vai correr tem a experiência da ferrovia com ativo=True, a da rede viária
    com ativo=False e MODO_TESTE = False (a versão nova do 43, não a que está a treinar).

Cada script corre com o mesmo Python e na sua própria pasta. O resultado de cada um
(início, fim, duração e se correu bem) fica registado no ecrã e em LOG_CADEIA.

Correr numa SEGUNDA janela do Python Command Prompt, com o treino a decorrer na outra:
    python 99_esperar_e_correr_ferrovia.py
"""
import os
import re
import sys
import time
import subprocess
from datetime import datetime

# =============================== CONFIGURAÇÃO ===============================
PASTA = os.path.dirname(os.path.abspath(__file__))          # pasta MTRE_nDSM
PASTA_MODELOS = r"D:\DL\Modelos\126_RViaria_nDSM"     # onde fica o registo da cadeia

# PID dos processos a esperar. None = automático: os python.exe que estão a usar a GPU
# quando este script arranca (ver com nvidia-smi). Ex.: PIDS_TREINO = [6200]
PIDS_TREINO = None

# Scripts a correr depois do fim do treino, por esta ordem
SCRIPTS = [
    os.path.join(PASTA, "43_Treino_RViaria_ndsm.py"),    # treina o MTRE da ferrovia
]

# Se um script falhar: True = continua com o seguinte; False = para a cadeia.
CONTINUAR_SE_FALHAR = False

INTERVALO_S = 60           # verifica de minuto a minuto (só consulta a lista de processos)
MEMORIA_LIVRE_MB = 2000    # memória da GPU usada abaixo da qual a GPU é dada como livre
ESPERA_MEMORIA_MAX_MIN = 30
ESPERA_FINAL_S = 120       # margem final antes de lançar o primeiro script
LOG_CADEIA = os.path.join(PASTA_MODELOS, "log_cadeia_scripts.txt")
# ============================================================================


def registo(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(linha, flush=True)
    with open(LOG_CADEIA, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def correr(cmd):
    """Saída de um comando ('' se falhar)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="ignore", timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def pids_python_gpu():
    """PID dos python.exe (exceto este) que estão a usar a GPU, segundo o nvidia-smi."""
    pids = set()
    for linha in correr(["nvidia-smi", "--query-compute-apps=pid,process_name",
                         "--format=csv,noheader"]).splitlines():
        partes = [x.strip() for x in linha.split(",")]
        if len(partes) >= 2 and partes[0].isdigit() and "python" in partes[1].lower():
            pids.add(int(partes[0]))
    if not pids:   # em modo WDDM a consulta pode vir vazia: ler a tabela "Processes"
        for linha in correr(["nvidia-smi"]).splitlines():
            m = re.match(r"^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C|C\+G)\s+(\S+)", linha)
            if m and "python" in m.group(3).lower():
                pids.add(int(m.group(1)))
    pids.discard(os.getpid())
    return sorted(pids)


def vivo(pid):
    """True se o processo com este PID ainda existe."""
    saida = correr(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"])
    return f'"{pid}"' in saida


def memoria_usada_mb():
    """Memória usada na GPU 0 (MB), ou None se o nvidia-smi não responder."""
    saida = correr(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).strip()
    try:
        return int(saida.splitlines()[0])
    except (ValueError, IndexError):
        return None


def verificar_43():
    """Confirma que o 43 a correr é a versão com a ferrovia. Devolve a lista de problemas."""
    script = next((s for s in SCRIPTS if os.path.basename(s).startswith("43_")), None)
    if not script:
        return []
    with open(script, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    problemas = []
    # a última atribuição (a docstring do 43 também tem linhas "MODO_TESTE = ...")
    m = re.findall(r"^MODO_TESTE\s*=\s*(\w+)\s*(?:#.*)?$", txt, re.M)
    if not m or m[-1] != "False":
        problemas.append(f"MODO_TESTE não está False em {os.path.basename(script)}")

    def ativo(nome):
        bloco = re.search(r'"%s"\s*:\s*dict\((.*?)\n    \)' % re.escape(nome), txt, re.S)
        if not bloco:
            return None
        a = re.search(r"ativo\s*=\s*(\w+)", bloco.group(1))
        return a.group(1) if a else None

    if ativo("mtre_ferrovia_ndsm_resnet50") is None:
        problemas.append("o 43 não tem a experiência mtre_ferrovia_ndsm_resnet50 (versão antiga do script?)")
    elif ativo("mtre_ferrovia_ndsm_resnet50") != "True":
        problemas.append("a experiência mtre_ferrovia_ndsm_resnet50 não está com ativo=True")
    if ativo("mtre_rede_viaria_ndsm_resnet50") == "True":
        problemas.append("a experiência mtre_rede_viaria_ndsm_resnet50 está com ativo=True "
                         "(treinaria a rede viária outra vez)")
    return problemas


def main():
    # confirmar logo no início (para não esperar horas em vão)
    em_falta = [s for s in SCRIPTS if not os.path.exists(s)]
    if em_falta:
        print("ERRO: estes scripts não existem:\n  " + "\n  ".join(em_falta))
        return
    if not os.path.isdir(PASTA_MODELOS):
        print(f"ERRO: a pasta dos modelos não existe: {PASTA_MODELOS}")
        return
    problemas = verificar_43()
    if problemas:
        print("ERRO no 43:\n  " + "\n  ".join(problemas))
        return

    pids = PIDS_TREINO or pids_python_gpu()
    if not pids:
        print("ERRO: nenhum python.exe a usar a GPU no nvidia-smi. Indicar o PID do treino em PIDS_TREINO.")
        return
    vivos = [p for p in pids if vivo(p)]
    if not vivos:
        print(f"ERRO: os processos {pids} não existem (PID errado?).")
        return
    registo(f"À espera do fim dos processos python.exe {vivos} (treino em curso). "
            f"Depois corre(m) {len(SCRIPTS)} script(s): {', '.join(os.path.basename(s) for s in SCRIPTS)}.")
    n = 0
    while vivos:
        time.sleep(INTERVALO_S)
        vivos = [p for p in vivos if vivo(p)]
        n += 1
        if vivos and n % 30 == 0:      # de meia em meia hora, para não encher o ecrã
            print(f"[{datetime.now():%Y-%m-%d %H:%M}] treino ainda a decorrer (PID {vivos})", flush=True)
    registo("Processos do treino terminados. A confirmar que a memória da GPU ficou livre.")

    t0 = time.time()
    while True:
        mem = memoria_usada_mb()
        outros = pids_python_gpu()
        if mem is not None and mem < MEMORIA_LIVRE_MB and not outros:
            registo(f"GPU livre ({mem} MB usados). Primeiro script dentro de {ESPERA_FINAL_S} s.")
            break
        if time.time() - t0 > ESPERA_MEMORIA_MAX_MIN * 60:
            registo(f"GPU ainda ocupada ao fim de {ESPERA_MEMORIA_MAX_MIN} min ({mem} MB usados, "
                    f"python.exe na GPU: {outros or 'nenhum'}). Nada foi lançado.")
            return
        time.sleep(30)
    time.sleep(ESPERA_FINAL_S)

    for i, script in enumerate(SCRIPTS, 1):
        nome = os.path.basename(script)
        registo(f"[{i}/{len(SCRIPTS)}] INÍCIO {nome}")
        t0 = time.time()
        res = subprocess.run([sys.executable, script], cwd=os.path.dirname(script))
        horas = (time.time() - t0) / 3600
        if res.returncode == 0:
            registo(f"[{i}/{len(SCRIPTS)}] FIM {nome} | correu bem | {horas:.2f} h")
        else:
            registo(f"[{i}/{len(SCRIPTS)}] ERRO {nome} | código {res.returncode} | {horas:.2f} h")
            if not CONTINUAR_SE_FALHAR:
                registo("Cadeia interrompida (CONTINUAR_SE_FALHAR = False).")
                return
        time.sleep(60)      # pausa curta entre scripts, para a GPU libertar a memória

    registo("Todos os scripts concluídos.")


if __name__ == "__main__":
    main()
