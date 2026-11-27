# -*- coding: utf-8 -*-
r"""
33_comparar_bandas.py  (qualquer Python 3; nao precisa do arcpy nem da GPU)

Compara os modelos de OCUPACAO DO SOLO treinados com tres combinacoes de bandas:
  RGB     3 bandas (R, G, B)                  pasta RGB    (esta experiencia)
  4B      4 bandas (R, G, B, NIR)             pasta UNET   (versao 2; SAMLoRA v1)
  nDSM    5 bandas (R, G, B, NIR, nDSM)       pasta nDSM
para os dois modelos treinados nas tres versoes (DeepLabV3 ResNet-50 e SAMLoRA), sem e com
TTA, no TOTAL das 3 ortos de teste. Le so os CSV das avaliacoes (30_avaliar_ocupacao*.py).

Saidas em PASTA_SAIDA:
  comparacao_bandas.txt           tabelas legiveis (metricas globais e F1 por classe)
  comparacao_bandas_globais.csv   uma linha por modelo x bandas x TTA x agrupamento
  comparacao_bandas_classes.csv   F1 por classe

Uma avaliacao que ainda nao exista (ex.: o TTA do RGB) aparece como "-" e e indicada no fim.

Correr:
    python 33_comparar_bandas.py
"""
import os
import csv
from datetime import datetime

AVAL = r"D:\DL\Avaliacao"
PASTA_SAIDA = os.path.join(AVAL, "126_ocupacao_RGB")

# (bandas, TTA) -> (pasta da avaliacao, {modelo: rotulo na avaliacao})
AVALIACOES = {
    ("RGB", False):  ("126_ocupacao_RGB",      {"DeepLabV3": "deeplabv3_r50_rgb",    "SAMLoRA": "samlora_rgb"}),
    ("4B", False):   ("126_ocupacao",          {"DeepLabV3": "deeplabv3_r50_v2",     "SAMLoRA": "samlora_v1"}),
    ("nDSM", False): ("126_ocupacao_nDSM",     {"DeepLabV3": "deeplabv3_r50_ndsm",   "SAMLoRA": "samlora_ndsm"}),
    ("RGB", True):   ("126_ocupacao_RGB_TTA",  {"DeepLabV3": "deeplabv3_r50_rgb_tta", "SAMLoRA": "samlora_rgb_tta"}),
    ("4B", True):    ("126_ocupacao_TTA",      {"DeepLabV3": "deeplabv3_r50_v2_tta", "SAMLoRA": "samlora_v1_tta"}),
    ("nDSM", True):  ("126_ocupacao_nDSM_TTA", {"DeepLabV3": "deeplabv3_r50_ndsm",   "SAMLoRA": "samlora_ndsm"}),
}
BANDAS = ["RGB", "4B", "nDSM"]
MODELOS = ["DeepLabV3", "SAMLoRA"]
AGRUPAMENTOS = ["completa", "generalizada"]
METRICAS = [("exatidao_global", "OA"), ("kappa", "kappa"), ("mIoU", "mIoU"), ("fwIoU", "fwIoU"),
            ("F1_macro", "F1mac")]
ORDEM_CLASSES = ["Vinha", "Pomar_vinha", "Vinha+Pomar_vinha", "Pomar", "Arvoredo_denso",
                 "Arvoredo_esparso", "Mato_denso", "Agua", "Sem_representacao"]


def ler(pasta, nome):
    caminho = os.path.join(AVAL, pasta, nome)
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f, delimiter=";") if r["orto"] == "TOTAL"]


def main():
    G, F, faltam = {}, {}, []
    for (b, tta), (pasta, rotulos) in AVALIACOES.items():
        gl, cl = ler(pasta, "metricas_globais.csv"), ler(pasta, "metricas_classe.csv")
        if gl is None:
            faltam.append(f"{b}{' TTA' if tta else ''}: {os.path.join(AVAL, pasta)}")
            continue
        for mod, rot in rotulos.items():
            for r in gl:
                if r["modelo"] == rot:
                    G[(mod, b, tta, r["agrupamento"])] = r
            for r in cl or []:
                if r["modelo"] == rot:
                    F[(mod, b, tta, r["agrupamento"], r["nome"])] = float(r["F1"])

    txt = ["OCUPACAO DO SOLO - COMPARACAO DAS BANDAS DE ENTRADA (total das 3 ortos de teste)",
           f"Data: {datetime.now():%Y-%m-%d %H:%M}",
           "RGB = 3 bandas | 4B = RGB + NIR (SAMLoRA v1) | nDSM = RGB + NIR + nDSM",
           "dif = diferenca face a 4 bandas, em pontos (negativo = pior do que 4 bandas)"]
    lin_g, lin_c = [], []
    for tta in (False, True):
        for agr in AGRUPAMENTOS:
            txt += ["", f"=== {'COM' if tta else 'SEM'} TTA | agrupamento {agr}"]
            cab = f"{'modelo':<11}{'bandas':<7}" + "".join(f"{n:>8}{'dif':>7}" for _, n in METRICAS)
            txt += [cab, "-" * len(cab)]
            for mod in MODELOS:
                ref = G.get((mod, "4B", tta, agr))
                for b in BANDAS:
                    r = G.get((mod, b, tta, agr))
                    if r is None:
                        txt.append(f"{mod:<11}{b:<7}  (sem avaliacao)")
                        continue
                    linha = f"{mod:<11}{b:<7}"
                    reg = {"modelo": mod, "bandas": b, "TTA": "sim" if tta else "nao", "agrupamento": agr}
                    for k, n in METRICAS:
                        v = float(r[k])
                        d = 100 * (v - float(ref[k])) if ref is not None and b != "4B" else None
                        linha += f"{v:>8.3f}" + (f"{d:>+7.1f}" if d is not None else f"{'':>7}")
                        reg[n] = round(v, 4)
                        reg[f"{n}_dif_4B_pp"] = round(d, 2) if d is not None else ""
                    txt.append(linha)
                    lin_g.append(reg)
            # F1 por classe
            nomes = [c for c in ORDEM_CLASSES if any(k[3] == agr and k[4] == c and k[2] == tta for k in F)]
            if nomes:
                txt += ["", f"  F1 por classe ({'com' if tta else 'sem'} TTA, {agr})"]
                cab = f"  {'classe':<20}" + "".join(f"{m[:9] + ' ' + b:>15}" for m in MODELOS for b in BANDAS)
                txt.append(cab)
                for c in nomes:
                    s = f"  {c[:19]:<20}"
                    for m in MODELOS:
                        for b in BANDAS:
                            v = F.get((m, b, tta, agr, c))
                            s += f"{v:>15.3f}" if v is not None else f"{'-':>15}"
                            if v is not None:
                                v4 = F.get((m, "4B", tta, agr, c))
                                lin_c.append({"modelo": m, "bandas": b, "TTA": "sim" if tta else "nao",
                                              "agrupamento": agr, "classe": c, "F1": round(v, 4),
                                              "F1_dif_4B_pp": round(100 * (v - v4), 2) if v4 is not None and b != "4B" else ""})
                    txt.append(s)
    if faltam:
        txt += ["", "Avaliacoes ainda nao feitas (aparecem como 'sem avaliacao'):"] + [f"  {x}" for x in faltam]

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    for nome, linhas in (("comparacao_bandas_globais.csv", lin_g), ("comparacao_bandas_classes.csv", lin_c)):
        if linhas:
            with open(os.path.join(PASTA_SAIDA, nome), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
                w.writeheader()
                w.writerows(linhas)
    with open(os.path.join(PASTA_SAIDA, "comparacao_bandas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"\nGravado em {PASTA_SAIDA}")


if __name__ == "__main__":
    main()
