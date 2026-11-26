# -*- coding: utf-8 -*-
r"""
MTRE_nDSM/99_esperar_e_correr_ferrovia.py

Espera que o treino do MTRE da rede viária com nDSM (43_Treino_RViaria_ndsm.py, já a
decorrer) termine e, a seguir, corre uma LISTA de scripts, um de cada vez, pela ordem
indicada: o treino do MTRE da ferrovia com nDSM (os chips da ferrovia já foram
exportados pelo 42). Assim a GPU nunca é usada por dois processos ao mesmo tempo.

O fim do treino é detetado pelo resumo_treinos.csv da pasta dos modelos: o 43 só
cria esse ficheiro, ou lhe acrescenta linhas, quando termina todas as experiências.
Quando o número de linhas aumenta (ou o ficheiro aparece), começam a correr os
scripts da lista.

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
RESUMO_TREINOS = r"D:\DL\Modelos\126_RViaria_nDSM\resumo_treinos.csv"

# Scripts a correr depois do fim do treino, por esta ordem
SCRIPTS = [
    os.path.join(PASTA, "43_Treino_RViaria_ndsm.py"),    # treina o MTRE da ferrovia
]

# Se um script falhar: True = continua com o seguinte; False = para a cadeia.
CONTINUAR_SE_FALHAR = False

INTERVALO_S = 600          # verifica de 10 em 10 minutos
ESPERA_FINAL_S = 120       # margem depois do fim do treino, para a GPU libertar a memória
LOG_CADEIA = os.path.join(os.path.dirname(RESUMO_TREINOS), "log_cadeia_scripts.txt")
# ============================================================================


def registo(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(linha, flush=True)
    with open(LOG_CADEIA, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def n_linhas():
    """Linhas (não vazias) do resumo_treinos.csv; 0 se ainda não existir."""
    if not os.path.exists(RESUMO_TREINOS):
        return 0
    with open(RESUMO_TREINOS, encoding="utf-8-sig", errors="ignore") as f:
        return sum(1 for linha in f if linha.strip())


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
    if not os.path.isdir(os.path.dirname(RESUMO_TREINOS)):
        print(f"ERRO: a pasta dos modelos não existe: {os.path.dirname(RESUMO_TREINOS)}")
        return
    problemas = verificar_43()
    if problemas:
        print("ERRO no 43:\n  " + "\n  ".join(problemas))
        return

    inicial = n_linhas()
    estado = f"{inicial} linhas" if os.path.exists(RESUMO_TREINOS) else "ainda não existe"
    registo(f"À espera do fim do treino da rede viária ({os.path.basename(RESUMO_TREINOS)}: {estado}). "
            f"Depois corre(m) {len(SCRIPTS)} script(s): {', '.join(os.path.basename(s) for s in SCRIPTS)}.")
    while n_linhas() <= inicial:
        time.sleep(INTERVALO_S)
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] treino ainda a decorrer", flush=True)
    registo(f"Treino terminado. Primeiro script dentro de {ESPERA_FINAL_S} s.")
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
