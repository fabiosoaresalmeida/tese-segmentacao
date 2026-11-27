# -*- coding: utf-8 -*-
r"""
13_comparar_fases.py - compara as tres fases das construcoes para os quatro modelos da
fase B, a partir dos CSV ja produzidos pelo 09 (nao usa o arcpy nem a GPU):

  A  ortomosaico com 4 bandas (R, G, B, NIR)       AVAL_A
  B  4 bandas + nDSM (5 bandas)                    AVAL_B
  C  so R, G, B (3 bandas)                         AVAL_RGB

Usa o TOTAL das 3 ortos de teste, a referencia REFERENCIA e os poligonos regularizados
(produto final); o F1 e o kappa da classificacao tambem sao mostrados.

Saidas em AVAL_RGB:
  comparacao_fases_ABC.txt   tabela para ler (e para a tese), com as diferencas C - A e B - C
  comparacao_fases_ABC.csv   os mesmos valores, para o Excel (separador ;)

Le as pastas do comum_RGB.py (em ..\3_3bandas_RGB).

Correr em qualquer Python (por exemplo, o do ArcGIS Pro), na pasta 1_Construcoes\4_Avaliacao_Figuras:
    python 13_comparar_fases.py
    python 13_comparar_fases.py fotogrametria      (outra referencia)
"""
import os
import sys
import csv
from datetime import datetime

# o comum_RGB.py esta na pasta da fase C (..\3_3bandas_RGB)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3_3bandas_RGB"))
import comum_RGB as C  # noqa: E402

REFERENCIA = sys.argv[1] if len(sys.argv) > 1 else "edicao"
FASES = [("A", "4 bandas (RGB+NIR)", C.AVAL_A),
         ("B", "4 bandas + nDSM", C.AVAL_B),
         ("C", "RGB", C.AVAL_RGB)]
NOMES = {"DEEPLAB": "DeepLabV3", "UNET": "U-Net", "SAMLORA": "SAMLoRA"}

# metricas: (chave, ficheiro, coluna, produto/etapa, legenda, casas decimais)
METRICAS = [
    ("F1px",  "pixel",  "F1",                 "classificacao", "F1 ao pixel (classificacao)", 3),
    ("kappa", "pixel",  "kappa",              "classificacao", "kappa (classificacao)", 3),
    ("F1pg",  "pixel",  "F1",                 "poligonos_reg", "F1 ao pixel (poligonos regularizados)", 3),
    ("IoUpx", "pixel",  "IoU",                "poligonos_reg", "IoU ao pixel (poligonos regularizados)", 3),
    ("P",     "objeto", "precision",          "regularizado",  "precisao por objeto", 3),
    ("R",     "objeto", "recall",             "regularizado",  "sensibilidade por objeto", 3),
    ("F1obj", "objeto", "F1",                 "regularizado",  "F1 por objeto", 3),
    ("cob",   "objeto", "cobertura50",        "regularizado",  "cobertura >= 50 %", 3),
    ("fus",   "objeto", "fusoes",             "regularizado",  "fusoes", 0),
    ("IoUob", "objeto", "IoU_medio",          "regularizado",  "IoU medio dos detetados", 3),
    ("desl",  "objeto", "desloc_centroide_m", "regularizado",  "deslocamento do centroide (m)", 2),
    ("dcont", "objeto", "dist_contorno_media_m", "regularizado", "distancia media entre contornos (m)", 2),
]


def ler(pasta, tipo):
    f = os.path.join(pasta, f"metricas_{tipo}.csv")
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh, delimiter=";")
                if r["referencia"] == REFERENCIA and r["orto"] == "TOTAL"]


def valor(linhas, modelo_pref, coluna, produto):
    """Valor do ultimo modelo (por nome, ou seja, o mais recente) que comeca por modelo_pref."""
    if not linhas:
        return None, None
    col_prod = "produto" if "produto" in linhas[0] else "etapa"
    c = sorted((r for r in linhas if r["modelo"].startswith(modelo_pref) and r[col_prod] == produto),
               key=lambda r: r["modelo"])
    if not c:
        return None, None
    return float(c[-1][coluna]), c[-1]["modelo"]


def fmt(v, casas):
    return "   -  " if v is None else f"{v:.{casas}f}"


def main():
    dados, usados = {}, {}
    for fase, _, pasta in FASES:
        tab = {"pixel": ler(pasta, "pixel"), "objeto": ler(pasta, "objeto")}
        if tab["pixel"] is None:
            print(f"AVISO: sem resultados da fase {fase} em {pasta}")
        for tipo, backbone, _ in C.MODELOS_FASE_B:
            pref = f"{tipo.lower()}_{backbone.lower()}_b"
            for chave, fich, col, prod, _, _ in METRICAS:
                v, nome = valor(tab[fich], pref, col, prod)
                dados[(pref, fase, chave)] = v
                if nome:
                    usados[(pref, fase)] = nome

    chaves = [m[0] for m in METRICAS]
    larg = [7 if k in ("fus",) else 6 for k in chaves]
    cab = f"{'modelo':22s} {'fase':4s} " + " ".join(f"{k:>{w}s}" for k, w in zip(chaves, larg))
    linhas = [f"COMPARACAO DAS FASES A (4 bandas), B (+ nDSM) e C (RGB) - edificios, folha 126",
              f"Data: {datetime.now():%Y-%m-%d %H:%M} | referencia: {REFERENCIA} | total das 3 ortos de teste",
              "", cab, "-" * len(cab)]
    csv_linhas = [["modelo", "fase", "pasta_modelo"] + chaves]
    for tipo, backbone, _ in C.MODELOS_FASE_B:
        pref = f"{tipo.lower()}_{backbone.lower()}_b"
        rot = f"{NOMES.get(tipo, tipo)} {backbone.replace('RESNET', 'ResNet-').replace('VIT_', 'ViT-')}"
        for fase, _, _ in FASES:
            vals = [dados[(pref, fase, k)] for k in chaves]
            linhas.append(f"{rot:22s} {fase:4s} " + " ".join(
                f"{fmt(v, m[5]):>{w}s}" for v, m, w in zip(vals, METRICAS, larg)))
            csv_linhas.append([rot, fase, usados.get((pref, fase), "")] +
                              ["" if v is None else v for v in vals])
        # diferencas: efeito do NIR (A - C) e do nDSM (B - A)
        for rotulo, f1, f2 in (("A - C (NIR)", "A", "C"), ("B - A (nDSM)", "B", "A")):
            difs = [None if dados[(pref, f1, k)] is None or dados[(pref, f2, k)] is None
                    else dados[(pref, f1, k)] - dados[(pref, f2, k)] for k in chaves]
            linhas.append(f"{'':22s} {'':4s} " + " ".join(
                f"{('   -  ' if d is None else f'{d:+.{m[5]}f}'):>{w}s}" for d, m, w in zip(difs, METRICAS, larg))
                + f"   {rotulo}")
        linhas.append("")
    linhas.append("Metricas:")
    linhas += [f"  {k:6s} {leg}" for k, _, _, _, leg, _ in METRICAS]
    linhas.append("A - C: ganho do NIR em relacao ao RGB; B - A: ganho do nDSM em relacao as 4 bandas.")
    linhas.append("Nas distancias (desl, dcont) e nas fusoes, valores negativos nas diferencas sao melhorias.")
    linhas.append("")
    linhas.append("Modelos usados:")
    for (pref, fase), nome in sorted(usados.items(), key=lambda t: (t[0][0], t[0][1])):
        linhas.append(f"  fase {fase}: {nome}")

    texto = "\n".join(linhas)
    print(texto)
    os.makedirs(C.AVAL_RGB, exist_ok=True)
    suf = "" if REFERENCIA == "edicao" else f"_{REFERENCIA}"
    with open(os.path.join(C.AVAL_RGB, f"comparacao_fases_ABC{suf}.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    with open(os.path.join(C.AVAL_RGB, f"comparacao_fases_ABC{suf}.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter=";").writerows(csv_linhas)
    print(f"\nGravado em {C.AVAL_RGB}")


if __name__ == "__main__":
    main()
