# -*- coding: utf-8 -*-
r"""
5_comparar_4bandas_rgb.py - compara os modelos de TODAS AS CLASSES de 4 bandas (RGB + NIR,
avaliacao do 55) com os mesmos modelos treinados so com RGB (avaliacao do 4_avaliar_rgb_TTA.py).
Le so os CSV das duas avaliacoes (nao precisa do arcpy nem da GPU).

Para cada agrupamento (completa, generalizada, rede_viaria, ocupacao_sem_constr_vias, ...)
e para o TOTAL das 3 ortos de teste:
  - metricas globais (OA, kappa, mIoU, fwIoU, F1 macro) de 4 bandas, RGB e diferenca;
  - F1 por classe de 4 bandas, RGB e diferenca.

Saidas em D:\DL\Avaliacao\126_todas_classes_rgb_TTA:
  comparacao_4bandas_rgb_globais.csv, comparacao_4bandas_rgb_classes.csv,
  comparacao_4bandas_rgb.txt (tabela legivel)

Correr no Python Command Prompt do ArcGIS Pro (ou em qualquer Python 3):
    python 5_comparar_4bandas_rgb.py
"""
import os
import csv
from datetime import datetime

import comum_todas_rgb as C

METRICAS = [("exatidao_global", "OA"), ("kappa", "kappa"), ("mIoU", "mIoU"), ("fwIoU", "fwIoU"),
            ("F1_macro", "F1mac")]


def ler(pasta, nome):
    caminho = os.path.join(pasta, nome)
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"{caminho} (correr antes a avaliacao correspondente)")
    with open(caminho, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter=";") if r["orto"] == "TOTAL"]


def main():
    g4, gr = ler(C.AVAL_4B, "metricas_globais.csv"), ler(C.AVAL_RGB, "metricas_globais.csv")
    c4, cr = ler(C.AVAL_4B, "metricas_classe.csv"), ler(C.AVAL_RGB, "metricas_classe.csv")
    G4 = {(r["modelo"], r["agrupamento"]): r for r in g4}
    GR = {(r["modelo"], r["agrupamento"]): r for r in gr}
    F4 = {(r["modelo"], r["agrupamento"], r["nome"]): float(r["F1"]) for r in c4}
    FR = {(r["modelo"], r["agrupamento"], r["nome"]): float(r["F1"]) for r in cr}
    agrupamentos = list(dict.fromkeys(r["agrupamento"] for r in g4 + gr))
    pares = [(rot, rot + "_rgb") for rot in C.MODELOS.values()]

    lin_g, lin_c, txt = [], [], []
    txt.append("TODAS AS CLASSES: 4 BANDAS (RGB + NIR) vs SO RGB - total das 3 ortos de teste, com TTA")
    txt.append(f"Data: {datetime.now():%Y-%m-%d %H:%M} | 4 bandas: {C.AVAL_4B} | RGB: {C.AVAL_RGB}")
    txt.append("Diferenca = RGB - 4 bandas (em pontos; negativo = o RGB e pior)")
    for agr in agrupamentos:
        txt += ["", f"AGRUPAMENTO: {agr}"]
        cab = f"{'modelo':<18}" + "".join(f"{n:>8}{'RGB':>7}{'dif':>7}" for _, n in METRICAS)
        txt += [cab, "-" * len(cab)]
        nomes_cl = list(dict.fromkeys(k[2] for k in list(F4) + list(FR) if k[1] == agr))
        for r4, rr in pares:
            a, b = G4.get((r4, agr)), GR.get((rr, agr))
            if not a or not b:
                txt.append(f"{r4:<18}  (sem resultados {'de 4 bandas' if not a else 'RGB'})")
                continue
            linha = f"{r4:<18}"
            reg = {"modelo": r4, "agrupamento": agr}
            for k, n in METRICAS:
                v4, vr = float(a[k]), float(b[k])
                linha += f"{v4:>8.3f}{vr:>7.3f}{100 * (vr - v4):>+7.1f}"
                reg.update({f"{n}_4b": round(v4, 4), f"{n}_rgb": round(vr, 4), f"{n}_dif_pp": round(100 * (vr - v4), 2)})
            txt.append(linha)
            lin_g.append(reg)
            for cl in nomes_cl:
                v4, vr = F4.get((r4, agr, cl)), FR.get((rr, agr, cl))
                if v4 is None or vr is None:
                    continue
                lin_c.append({"modelo": r4, "agrupamento": agr, "classe": cl, "F1_4b": round(v4, 4),
                              "F1_rgb": round(vr, 4), "F1_dif_pp": round(100 * (vr - v4), 2)})
        # F1 por classe (uma linha por classe, colunas por modelo)
        if nomes_cl:
            txt += ["", f"  F1 por classe ({agr}): 4 bandas / RGB / dif"]
            cab = f"  {'classe':<24}" + "".join(f"{r4[:14]:>23}" for r4, _ in pares)
            txt += [cab]
            for cl in nomes_cl:
                s = f"  {cl[:23]:<24}"
                for r4, rr in pares:
                    v4, vr = F4.get((r4, agr, cl)), FR.get((rr, agr, cl))
                    s += (f"{v4:>9.3f}{vr:>7.3f}{100 * (vr - v4):>+7.1f}" if v4 is not None and vr is not None
                          else f"{'-':>23}")
                txt.append(s)

    os.makedirs(C.AVAL_RGB, exist_ok=True)
    for nome, linhas in (("comparacao_4bandas_rgb_globais.csv", lin_g), ("comparacao_4bandas_rgb_classes.csv", lin_c)):
        if linhas:
            with open(os.path.join(C.AVAL_RGB, nome), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
                w.writeheader()
                w.writerows(linhas)
    with open(os.path.join(C.AVAL_RGB, "comparacao_4bandas_rgb.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"\nGravado em {C.AVAL_RGB}")


if __name__ == "__main__":
    main()
