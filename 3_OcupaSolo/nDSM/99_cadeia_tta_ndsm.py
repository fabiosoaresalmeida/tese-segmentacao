# -*- coding: utf-8 -*-
r"""
99_cadeia_tta_ndsm.py

(Script de uma só utilização, já executado em 13-15/11/2026: fica para registo.
Depois disso a pasta de modelos do nDSM passou a chamar-se 126_ocupacao_nDSM.)

Corre, um de cada vez e por esta ordem:
  1. 30_avaliar_ocupacao.py  (pasta 1_Unet)  - avaliacao com test-time augmentation (GPU)
  2. 25_treinar_ocupacao.py (pasta 2_nDSM) - treino com os chips do nDSM (GPU)

Cada script corre com o mesmo Python e na sua propria pasta, so depois de o anterior
terminar, por isso a GPU nunca e usada por dois processos ao mesmo tempo.

ANTES de comecar, verifica (para nao descobrir um erro so passadas horas):
  - que os dois scripts existem;
  - no 30: que ha pelo menos um modelo ativo com test_time_augmentation True, e que a
    pasta de classificacoes desses modelos ainda nao existe (senao seriam reutilizadas,
    sem TTA);
  - no 25: que os argumentos nao tem espacos dentro das listas (ex.: [0,8], e nao [0, 8]),
    que a pasta dos chips existe e tem imagens e rotulos, e que a PASTA_MODELOS nao e a
    dos modelos ja treinados (senao seriam todos saltados);
  - que nao ha nenhum treino ou avaliacao a correr; se houver, espera que termine.

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro fechado):
    python "D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\2_nDSM\99_cadeia_tta_ndsm.py"
"""
import os
import re
import sys
import ast
import glob
import time
import subprocess
from datetime import datetime

# =============================== CONFIGURACAO ===============================
SCRIPT_AVALIACAO = r"D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\1_Unet\30_avaliar_ocupacao.py"
ARGS_AVALIACAO = []          # sem --cpu: a GPU esta livre (nada a treinar ao mesmo tempo)

SCRIPT_TREINO = r"D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\2_nDSM\25_treinar_ocupacao.py"

# Pastas de modelos ja treinados: o treino do nDSM NAO pode usar nenhuma delas
PASTAS_MODELOS_EXISTENTES = [
    r"D:\DL\Modelos\126_ocupacao_samlora",
    r"D:\DL\Modelos\126_ocupacao_2v",
]

CONTINUAR_SE_FALHAR = True   # se a avaliacao falhar, corre na mesma o treino
NOMES_PROCESSOS = ("25_treinar_fase1_ocupacao", "25_treinar_ocupacao", "30_avaliar_ocupacao",
                   "31_ensemble_ocupacao")
INTERVALO_S = 300
PAUSA_ENTRE_S = 120          # pausa entre os dois scripts, para a GPU libertar a memoria
LOG_CADEIA = r"D:\DL\Modelos\log_cadeia_ocupacao.txt"
# ============================================================================


def registo(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M}] {msg}"
    print(linha, flush=True)
    os.makedirs(os.path.dirname(LOG_CADEIA), exist_ok=True)
    with open(LOG_CADEIA, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def linhas_ativas(txt):
    """Linhas do script que nao estao comentadas."""
    return [l for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]


def valor_constante(txt, nome):
    """Valor de uma constante simples do tipo NOME = r"..." (ou None)."""
    m = re.search(rf'^{nome}\s*=\s*r?["\'](.+?)["\']', txt, re.M)
    return m.group(1) if m else None


def processos_ativos():
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
           "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    ativos = []
    for linha in out.splitlines():
        if "|" not in linha:
            continue
        pid, cmdline = linha.split("|", 1)
        if pid.strip() != str(os.getpid()) and any(n in cmdline for n in NOMES_PROCESSOS):
            ativos.append(f"{pid.strip()}: {cmdline.strip()[-80:]}")
    return ativos


# ------------------------------------------------------------------ verificacoes
def verificar_avaliacao():
    erros, avisos = [], []
    if not os.path.exists(SCRIPT_AVALIACAO):
        return [f"o script de avaliacao nao existe: {SCRIPT_AVALIACAO}"], avisos
    txt = open(SCRIPT_AVALIACAO, encoding="utf-8", errors="ignore").read()
    ativas = linhas_ativas(txt)

    # argumentos com TTA definidos e usados por algum modelo ativo
    nomes_args_tta = {m.group(1) for l in ativas
                      for m in [re.match(r'\s*(\w+)\s*=\s*".*test_time_augmentation\s+True', l)] if m}
    if not nomes_args_tta:
        erros.append("no 30 nao ha nenhuns argumentos com 'test_time_augmentation True'")
    modelos_tta = []
    for l in ativas:
        m = re.match(r'\s*\(\s*"([^"]+)"\s*,.*,\s*(\w+)\s*\)\s*,?\s*$', l)
        if m and m.group(2) in nomes_args_tta:
            modelos_tta.append(m.group(1))
    if nomes_args_tta and not modelos_tta:
        erros.append(f"no 30 nenhum modelo ativo da lista MODELOS usa {sorted(nomes_args_tta)}")

    # modelos da versao 1 (mapa {0: 9}) tem de prever o fundo, que e o sem representacao
    args_valores = {m.group(1): m.group(2) for l in ativas
                    for m in [re.match(r'\s*(\w+)\s*=\s*"(.*)"', l)] if m}
    for l in ativas:
        m = re.match(r'\s*\(\s*"([^"]+)"\s*,.*\{\s*0\s*:\s*9\s*\}\s*,\s*(\w+)\s*\)', l)
        if m and "predict_background True" not in args_valores.get(m.group(2), ""):
            erros.append(f"no 30 o modelo {m.group(1)} (mapa {{0: 9}}, versao 1) usa {m.group(2)} sem "
                         "'predict_background True': nunca preveria o sem representacao")

    # classificacoes ja existentes seriam reutilizadas (sem TTA)
    pasta_av = valor_constante(txt, "PASTA_AVALIACAO")
    if pasta_av:
        for rot in modelos_tta:
            pasta_cl = os.path.join(pasta_av, "classificacoes", rot)
            if glob.glob(os.path.join(pasta_cl, "class_*.tif")):
                erros.append(f"ja existem classificacoes em {pasta_cl}: seriam reutilizadas e o TTA "
                             "nao seria aplicado. Usa outro rotulo (ex.: ..._tta) ou apaga essa pasta.")
    if "FORCAR_CPU" in txt and "--cpu" in ARGS_AVALIACAO:
        avisos.append("a avaliacao vai correr em CPU (--cpu): com TTA pode demorar dias")
    return erros, avisos + [f"modelos com TTA: {modelos_tta}"]


def verificar_treino():
    erros, avisos = [], []
    if not os.path.exists(SCRIPT_TREINO):
        return [f"o script de treino nao existe: {SCRIPT_TREINO}"], avisos
    txt = open(SCRIPT_TREINO, encoding="utf-8", errors="ignore").read()

    # argumentos: formato "nome valor;nome valor", sem espacos dentro dos valores
    m = re.search(r'^ARGS_COMUNS\s*=\s*f?["\'](.+?)["\']', txt, re.M)
    if not m:
        avisos.append("nao encontrei ARGS_COMUNS no 25 (nao verificado)")
    else:
        args = m.group(1)
        ign = re.search(r"^IGNORE\s*=\s*(\d+)", txt, re.M)
        if ign:
            args = args.replace("{IGNORE}", ign.group(1))
        for par in args.split(";"):
            partes = par.strip().split(" ")
            if len(partes) != 2:
                erros.append(f"argumento mal formado no 25 (espaco dentro do valor?): '{par}'")
                continue
            try:
                ast.literal_eval(partes[1])
            except Exception:
                erros.append(f"valor invalido no argumento '{par}' do 25")

    # ORDEM: todos os modelos tem de existir (e nao estar comentados) em MODELOS
    ativas = linhas_ativas(txt)
    chaves = {m.group(1) for l in ativas for m in [re.match(r'\s*"(\w+)"\s*:\s*\(', l)] if m}
    ordem = None
    for l in ativas:
        m = re.match(r'\s*ORDEM\s*=\s*(\[.*\])', l)
        if m:
            try:
                ordem = ast.literal_eval(m.group(1))
            except Exception:
                pass
    if ordem is None:
        avisos.append("nao encontrei a ORDEM ativa no 25 (nao verificado)")
    else:
        falta = [n for n in ordem if n not in chaves]
        if falta:
            erros.append(f"no 25 a ORDEM {ordem} inclui modelos que nao estao ativos em MODELOS: "
                         f"{falta} (o treino parava logo com KeyError)")
        else:
            avisos.append(f"modelos a treinar: {ordem}")

    # chips
    chips = valor_constante(txt, "CHIPS")
    if not chips or not os.path.isdir(chips):
        erros.append(f"a pasta dos chips do 25 nao existe: {chips}")
    else:
        n_img = len(glob.glob(os.path.join(chips, "images", "*.tif")))
        n_lab = len(glob.glob(os.path.join(chips, "labels", "*.tif")))
        if n_img == 0 or n_img != n_lab:
            erros.append(f"chips do 25: {n_img} imagens e {n_lab} rotulos")
        else:
            avisos.append(f"chips: {n_img} imagens | {n_lab} rotulos ({chips})")

    # pasta de modelos
    pasta = valor_constante(txt, "PASTA_MODELOS")
    if not pasta:
        avisos.append("nao encontrei PASTA_MODELOS no 25 (nao verificado)")
    else:
        norm = lambda p: os.path.normcase(os.path.normpath(p))
        if norm(pasta) in {norm(p) for p in PASTAS_MODELOS_EXISTENTES}:
            erros.append(f"o 25 usa a PASTA_MODELOS {pasta}, onde ja estao modelos treinados: "
                         "seriam todos saltados. Usa uma pasta nova (ex.: ..._ndsm).")
        elif glob.glob(os.path.join(pasta, "*", "*.emd")):
            avisos.append(f"ja ha modelos com .emd em {pasta}: esses serao saltados")
        else:
            avisos.append(f"modelos novos em: {pasta}")
    return erros, avisos


# ------------------------------------------------------------------ principal
def correr(script, args):
    nome = os.path.basename(script)
    registo(f"INICIO {nome} {' '.join(args)}  ({os.path.dirname(script)})")
    t0 = time.time()
    res = subprocess.run([sys.executable, script] + args, cwd=os.path.dirname(script))
    horas = (time.time() - t0) / 3600
    if res.returncode == 0:
        registo(f"FIM {nome} | correu bem | {horas:.2f} h")
        return True
    registo(f"ERRO {nome} | codigo {res.returncode} | {horas:.2f} h")
    return False


def main():
    ok = True
    for titulo, funcao in (("AVALIACAO (TTA)", verificar_avaliacao), ("TREINO (nDSM)", verificar_treino)):
        erros, avisos = funcao()
        print(f"--- {titulo}")
        for a in avisos:
            print(f"    {a}")
        for e in erros:
            print(f"    ERRO: {e}")
        ok &= not erros
    if not ok:
        print("\nCorrige os erros acima e corre outra vez. Nada foi iniciado.")
        return

    ativos = processos_ativos()
    if ativos:
        registo(f"Ha processos a correr; a espera que terminem: {ativos}")
        while processos_ativos():
            time.sleep(INTERVALO_S)
        time.sleep(PAUSA_ENTRE_S)

    registo("CADEIA: avaliacao com TTA -> treino com nDSM")
    if not correr(SCRIPT_AVALIACAO, ARGS_AVALIACAO) and not CONTINUAR_SE_FALHAR:
        registo("Cadeia interrompida (CONTINUAR_SE_FALHAR = False).")
        return
    time.sleep(PAUSA_ENTRE_S)
    correr(SCRIPT_TREINO, [])
    registo("CADEIA concluida.")


if __name__ == "__main__":
    main()
