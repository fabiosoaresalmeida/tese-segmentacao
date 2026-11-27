# -*- coding: utf-8 -*-
"""
Ocupacao do solo na folha 113: metricas com agregacoes de classes, calculadas a partir
das matrizes de confusao que o 3b_avaliar_ocupacao_113.py ja gravou. NAO volta a
classificar, NAO precisa do arcpy e NAO altera os ficheiros do 3b. Demora segundos.

Agregacoes:
  completa       as 8 classes (controlo: tem de dar os mesmos numeros do 3b)
  vegetacao      Arvoredo denso + Arvoredo esparso + Mato denso -> Vegetacao
                 (o resto fica igual: mostra quanto custa a fronteira de densidade)
  colega         esquema da tabela 3.1 do colega, reduzido as nossas classes:
                   Cultivo    = Vinha + Pomar e vinha + Pomar
                   Vegetacao  = Arvoredo denso + Arvoredo esparso + Mato denso
                   Hidrografia = Agua
                   Sem_representacao (construcoes, vias, solo nu e o resto)
                 Diferencas de criterio: o colega poe Jardim ou horta em Vegetacao
                 (aqui e Ignorar) e tem Arrozal (aqui nao existe).

Saidas em <PASTA_AVALIACAO>\\agregacoes\\
  metricas_globais_agregacoes.csv   OA, kappa, mIoU, F1 macro... por modelo e agregacao
  metricas_classe_agregacoes.csv    precisao, recall, F1, IoU por classe
  matriz_<modelo>_<agregacao>.csv   matriz agregada (contagens e % por linha)
  destino_classes.txt               para onde vai cada classe da referencia (completa)
  resumo_agregacoes.txt             tabelas para ler
"""
import os
import sys
import csv
import glob

# ============================================================
PASTA_AVALIACAO = r"D:\DL\Avaliacao\113_ocupacao_TTA"
# Outra pasta sem alterar o script (ex.: os modelos com nDSM):
#   python 3c_agregacoes_ocupacao_113.py D:\DL\Avaliacao\113_ocupacao_nDSM_TTA
if len(sys.argv) > 1:
    PASTA_AVALIACAO = sys.argv[1]
# ============================================================

PASTA_MATRIZES = os.path.join(PASTA_AVALIACAO, "matrizes")
SAIDA = os.path.join(PASTA_AVALIACAO, "agregacoes")

VEG = ["Arvoredo_denso", "Arvoredo_esparso", "Mato_denso"]
CULT = ["Vinha", "Pomar_vinha", "Pomar"]
AGREGACOES = {
    "completa": {},
    "vegetacao": {c: "Vegetacao" for c in VEG},
    "colega": dict({c: "Vegetacao" for c in VEG}, **{c: "Cultivo" for c in CULT}, Agua="Hidrografia"),
}


def ler_matriz(caminho):
    """Le o bloco de CONTAGENS do csv do 3b. Devolve (classes, colunas, contagens)."""
    with open(caminho, encoding="utf-8-sig") as f:
        linhas = list(csv.reader(f, delimiter=";"))
    i = next(k for k, l in enumerate(linhas) if l and l[0].startswith("CONTAGENS"))
    cab = linhas[i + 1]
    colunas = cab[1:-2]                    # classes previstas + outro_* (sem total e recall)
    M, classes = {}, []
    for l in linhas[i + 2:]:
        if not l or not l[0]:
            break
        classes.append(l[0])
        M[l[0]] = {c: int(float(v or 0)) for c, v in zip(colunas, l[1:1 + len(colunas)])}
    return classes, colunas, M


def agregar(classes, colunas, M, mapa):
    f = lambda c: mapa.get(c, c)
    novas = []
    for c in classes:
        if f(c) not in novas:
            novas.append(f(c))
    cols = novas + [c for c in colunas if c not in classes]          # outro_* ficam no fim
    A = {r: {c: 0 for c in cols} for r in novas}
    for r in classes:
        for c in colunas:
            A[f(r)][f(c) if c in classes else c] += M[r][c]
    return novas, cols, A


def metricas(classes, cols, A):
    n = sum(sum(A[r].values()) for r in classes)
    pc = []
    for c in classes:
        tp = A[c][c]
        sup = sum(A[c].values())
        prev = sum(A[r][c] for r in classes)
        fp, fn = prev - tp, sup - tp
        p = tp / prev if prev else 0.0
        r_ = tp / sup if sup else 0.0
        pc.append(dict(classe=c, suporte=sup, previstos=prev, TP=tp, FP=fp, FN=fn, precision=p,
                       recall=r_, F1=2 * p * r_ / (p + r_) if p + r_ else 0.0,
                       IoU=tp / (tp + fp + fn) if tp + fp + fn else 0.0))
    cs = [m for m in pc if m["suporte"] > 0]
    oa = sum(m["TP"] for m in pc) / n if n else 0.0
    pe = sum(m["suporte"] * m["previstos"] for m in pc) / (n * n) if n else 0.0
    media = lambda k: sum(m[k] for m in cs) / len(cs) if cs else 0.0
    gl = dict(pixeis=n, exatidao_global=oa, kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
              mIoU=media("IoU"), fwIoU=sum(m["suporte"] / n * m["IoU"] for m in cs) if n else 0.0,
              F1_macro=media("F1"), F1_ponderado=sum(m["suporte"] / n * m["F1"] for m in cs) if n else 0.0)
    return pc, gl


def gravar_matriz(classes, cols, A, caminho):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["CONTAGENS (pixeis)"])
        w.writerow(["referencia \\ previsao"] + cols + ["total", "recall_%"])
        for r in classes:
            tot = sum(A[r].values())
            w.writerow([r] + [A[r][c] for c in cols] + [tot, round(100 * A[r][r] / tot, 2) if tot else ""])
        w.writerow([])
        w.writerow(["PERCENTAGEM DE CADA LINHA (para onde vai cada classe da referencia)"])
        w.writerow(["referencia \\ previsao"] + cols)
        for r in classes:
            tot = sum(A[r].values())
            w.writerow([r] + [round(100 * A[r][c] / tot, 2) if tot else "" for c in cols])


def main():
    os.makedirs(SAIDA, exist_ok=True)
    ficheiros = sorted(f for f in glob.glob(os.path.join(PASTA_MATRIZES, "*_completa_TOTAL*.csv"))
                       if "modo_teste" not in os.path.basename(f))
    if not ficheiros:
        raise FileNotFoundError(f"Nenhuma matriz *_completa_TOTAL.csv em {PASTA_MATRIZES}")

    l_glb, l_cls, txt, destino = [], [], [], []
    tabelas = {a: [] for a in AGREGACOES}
    for fich in ficheiros:
        modelo = os.path.basename(fich).split("_completa_TOTAL")[0]
        classes, colunas, M = ler_matriz(fich)
        print(f"{modelo}: {len(classes)} classes, {sum(sum(v.values()) for v in M.values()):,} pixeis")

        # para onde vai cada classe da referencia (agregacao completa)
        destino += ["", f"{modelo}"]
        for r in classes:
            tot = sum(M[r].values())
            if not tot:
                continue
            top = sorted(((v, c) for c, v in M[r].items() if v), reverse=True)[:4]
            destino.append(f"  {r:20s} -> " + ", ".join(f"{c} {100 * v / tot:.1f}%" for v, c in top))

        for agr, mapa in AGREGACOES.items():
            cl, cols, A = agregar(classes, colunas, M, mapa)
            pc, gl = metricas(cl, cols, A)
            gravar_matriz(cl, cols, A, os.path.join(SAIDA, f"matriz_{modelo}_{agr}.csv"))
            l_glb.append([modelo, agr] + [round(v, 4) if isinstance(v, float) else v for v in gl.values()])
            for m in pc:
                l_cls.append([modelo, agr, m["classe"], m["suporte"], m["previstos"], m["TP"], m["FP"], m["FN"]] +
                             [round(m[k], 4) for k in ("precision", "recall", "F1", "IoU")])
            tabelas[agr].append((modelo, gl, pc))

    with open(os.path.join(SAIDA, "metricas_globais_agregacoes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["modelo", "agregacao", "pixeis", "exatidao_global", "kappa", "mIoU", "fwIoU",
                    "F1_macro", "F1_ponderado"])
        w.writerows(l_glb)
    with open(os.path.join(SAIDA, "metricas_classe_agregacoes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["modelo", "agregacao", "classe", "suporte_px", "previstos_px", "TP", "FP", "FN",
                    "precision", "recall", "F1", "IoU"])
        w.writerows(l_cls)

    txt = ["OCUPACAO DO SOLO 113 - METRICAS COM AGREGACOES DE CLASSES",
           "(a partir das matrizes do 3b; 'completa' deve dar os mesmos numeros do 3b)", ""]
    for agr, linhas in tabelas.items():
        if not linhas:
            continue
        nomes = [m["classe"] for m in linhas[0][2]]
        cab = f"{'modelo':24s}{'OA':>7s}{'kappa':>7s}{'mIoU':>7s}{'F1mac':>7s} |" + "".join(f"{c[:11]:>12s}" for c in nomes)
        txt += [f"AGREGACAO: {agr}  (F1 por classe a direita)", cab, "-" * len(cab)]
        for modelo, gl, pc in linhas:
            txt.append(f"{modelo:24s}{gl['exatidao_global']:7.3f}{gl['kappa']:7.3f}{gl['mIoU']:7.3f}"
                       f"{gl['F1_macro']:7.3f} |" + "".join(f"{m['F1']:12.3f}" for m in pc))
        txt.append("")
    txt += ["PARA ONDE VAI CADA CLASSE DA REFERENCIA (agregacao completa, 4 destinos principais)"] + destino
    with open(os.path.join(SAIDA, "resumo_agregacoes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    with open(os.path.join(SAIDA, "destino_classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(destino) + "\n")
    print("\n".join(txt))
    print(f"\nResultados em {SAIDA}")


if __name__ == "__main__":
    main()
