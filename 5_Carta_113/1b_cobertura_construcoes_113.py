# -*- coding: utf-8 -*-
"""
Cobertura dos edificios de referencia da 113 com varios limiares (10 a 50 %), e metricas
por objeto com IoU 0,3 e 0,5.

Le os resultados que o 1_avaliar_construcoes_113.py deixou em poligonos.gdb (poligonos
regularizados de cada modelo e referencia limpa e projetada). NAO volta a classificar e
NAO altera a gdb: trabalha numa copia em memoria. Demora poucos minutos.

Cobertura = fracao da area de cada edificio de referencia coberta por poligonos previstos
(um ou varios). Um edificio conta como "coberto a 30 %" se pelo menos 30 % da sua area
estiver coberta. Nao penaliza fusoes nem falsos positivos: ler junto com a precisao.

Saidas em D:\\DL\\Avaliacao\\113_construcoes\\cobertura\\
  cobertura_limiares.csv     modelo, classe de area, limiar, n_ref, cobertos, taxa
  objetos_iou.csv            modelo, limiar IoU, detetados, falsos, omitidos, P, R, F1
  resumo_cobertura.txt       tabelas para ler
"""
import os
import csv

print("A carregar o arcpy...", flush=True)
import arcpy
import numpy as np

# ============================================================
PASTA_AVALIACAO = r"D:\DL\Avaliacao\113_construcoes"
LIMIARES_COB = [0.1, 0.2, 0.3, 0.4, 0.5]
LIMIARES_IOU = [0.3, 0.5]
CLASSES_AREA = [0, 25, 50, 100, 200, float("inf")]
ZONA = "ortomosaico"           # nome da zona usado no 1_avaliar (folha inteira)
# ============================================================

GDB = os.path.join(PASTA_AVALIACAO, "poligonos.gdb")
SAIDA = os.path.join(PASTA_AVALIACAO, "cobertura")
arcpy.env.overwriteOutput = True


def apagar(*itens):
    for it in itens:
        if it and arcpy.Exists(it):
            arcpy.management.Delete(it)


def referencia():
    """Mesma selecao do 1_avaliar: edificios inteiramente dentro da zona (teste) e os que
    atravessam o limite (margens, so para excluir previsoes)."""
    ref, zona = os.path.join(GDB, "referencia_projetada"), os.path.join(GDB, f"zona_{ZONA}")
    for fc in (ref, zona):
        if not arcpy.Exists(fc):
            raise FileNotFoundError(f"Nao existe: {fc}")
    teste, marg = "memory\\c_ref", "memory\\c_marg"
    apagar(teste, marg)
    lyr = arcpy.management.MakeFeatureLayer(ref, "lyr_cref")[0]
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", zona)
    arcpy.management.CopyFeatures(lyr, teste)
    arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", zona)
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", zona, selection_type="REMOVE_FROM_SELECTION")
    arcpy.management.CopyFeatures(lyr, marg)
    arcpy.management.Delete(lyr)
    arcpy.management.AddField(teste, "RID", "LONG")
    arcpy.management.CalculateField(teste, "RID", f"!{arcpy.Describe(teste).OIDFieldName}!", "PYTHON3")
    return teste, marg


def avaliar(reg_fc, ref, marg):
    p, inter = "memory\\c_pred", "memory\\c_inter"
    apagar(p, inter)
    arcpy.management.CopyFeatures(reg_fc, p)
    arcpy.management.AddField(p, "PID", "LONG")
    arcpy.management.CalculateField(p, "PID", f"!{arcpy.Describe(p).OIDFieldName}!", "PYTHON3")
    preds = {i: g.area for i, g in arcpy.da.SearchCursor(p, ["PID", "SHAPE@"]) if g is not None}
    refs = {i: g.area for i, g in arcpy.da.SearchCursor(ref, ["RID", "SHAPE@"]) if g is not None}
    pares = {}
    arcpy.analysis.Intersect([p, ref], inter, "NO_FID")
    for pid, rid, a in arcpy.da.SearchCursor(inter, ["PID", "RID", "SHAPE@AREA"]):
        pares[(pid, rid)] = pares.get((pid, rid), 0.0) + a

    # cobertura de cada edificio (todas as previsoes juntas; o Intersect ja separa por par)
    cob = {}
    for (pid, rid), a in pares.items():
        cob[rid] = cob.get(rid, 0.0) + a
    frac = {rid: min(cob.get(rid, 0.0) / ar, 1.0) if ar else 0.0 for rid, ar in refs.items()}

    linhas_cob = []
    for a0, a1 in list(zip(CLASSES_AREA[:-1], CLASSES_AREA[1:])) + [(None, None)]:
        ids = list(refs) if a0 is None else [r for r, ar in refs.items() if a0 <= ar < a1]
        rotulo = "TOTAL" if a0 is None else (f"{a0:g}-inf" if a1 == float("inf") else f"{a0:g}-{a1:g}")
        for lim in LIMIARES_COB:
            n_c = sum(1 for r in ids if frac[r] >= lim)
            linhas_cob.append([rotulo, lim, len(ids), n_c, round(n_c / len(ids), 4) if ids else 0.0])

    # previsoes nas margens (excluidas da contagem de falsos, como no 1_avaliar)
    nas_margens = set()
    if arcpy.Exists(marg) and int(arcpy.management.GetCount(marg)[0]):
        lyr = arcpy.management.MakeFeatureLayer(p, "lyr_cpm")[0]
        arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", marg)
        nas_margens = {pid for (pid,) in arcpy.da.SearchCursor(lyr, ["PID"])}
        arcpy.management.Delete(lyr)

    cand = sorted(((ai / (preds[pid] + refs[rid] - ai), pid, rid) for (pid, rid), ai in pares.items()),
                  reverse=True)
    linhas_iou = []
    for lim in LIMIARES_IOU:
        up, ur = set(), set()
        for iou, pid, rid in cand:
            if iou >= lim and pid not in up and rid not in ur:
                up.add(pid); ur.add(rid)
        tp = len(up)
        fp = len(preds) - tp - len(nas_margens - up)
        fn = len(refs) - tp
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        linhas_iou.append([lim, len(refs), len(preds), tp, fp, fn, round(P, 4), round(R, 4), round(F1, 4)])
    apagar(p, inter)
    return linhas_cob, linhas_iou


def main():
    os.makedirs(SAIDA, exist_ok=True)
    arcpy.env.workspace = GDB
    regs = sorted(fc for fc in (arcpy.ListFeatureClasses(f"*_{ZONA}_reg") or []))
    if not regs:
        raise RuntimeError(f"Nenhuma feature class *_{ZONA}_reg em {GDB}")
    print(f"{len(regs)} modelos: {regs}", flush=True)
    ref, marg = referencia()
    print(f"Referencia: {int(arcpy.management.GetCount(ref)[0])} edificios", flush=True)

    t_cob, t_iou = [], []
    for fc in regs:
        modelo = fc[: -len(f"_{ZONA}_reg")]
        print(f"  {modelo}...", flush=True)
        lc, li = avaliar(os.path.join(GDB, fc), ref, marg)
        t_cob += [[modelo] + l for l in lc]
        t_iou += [[modelo] + l for l in li]

    with open(os.path.join(SAIDA, "cobertura_limiares.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["modelo", "classe_area_m2", "limiar_cobertura", "n_ref", "cobertos", "taxa"])
        w.writerows(t_cob)
    with open(os.path.join(SAIDA, "objetos_iou.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["modelo", "limiar_iou", "n_ref", "n_pred", "detetados", "falsos", "omitidos",
                    "precision", "recall", "F1"])
        w.writerows(t_iou)

    modelos = sorted({l[0] for l in t_cob})
    txt = ["COBERTURA DOS EDIFICIOS DE REFERENCIA - FOLHA 113 (poligonos regularizados)", "",
           "Taxa de edificios cobertos, por limiar de cobertura (folha inteira):",
           f"{'modelo':40s}" + "".join(f"{int(l*100):>7d}%" for l in LIMIARES_COB)]
    for m in modelos:
        v = {l[2]: l[5] for l in t_cob if l[0] == m and l[1] == "TOTAL"}
        txt.append(f"{m:40s}" + "".join(f"{v[l]:>8.3f}" for l in LIMIARES_COB))
    for m in modelos:
        txt += ["", f"{m} - por classe de area (m2)",
                f"{'classe':>10s}{'n_ref':>8s}" + "".join(f"{int(l*100):>7d}%" for l in LIMIARES_COB)]
        for cl in [l[1] for l in t_cob if l[0] == m and l[2] == LIMIARES_COB[0]]:
            ls = [l for l in t_cob if l[0] == m and l[1] == cl]
            txt.append(f"{cl:>10s}{ls[0][3]:>8d}" + "".join(f"{l[5]:>8.3f}" for l in ls))
    txt += ["", "Metricas por objeto com IoU 0,3 e 0,5:",
            f"{'modelo':40s}{'IoU':>5s}{'P':>7s}{'R':>7s}{'F1':>7s}{'detet':>7s}{'falsos':>8s}"]
    for l in t_iou:
        txt.append(f"{l[0]:40s}{l[1]:>5.1f}{l[7]:>7.3f}{l[8]:>7.3f}{l[9]:>7.3f}{l[4]:>7d}{l[5]:>8d}")
    txt += ["", "Cobertura nao penaliza fusoes nem falsos positivos: ler junto com a precisao."]
    with open(os.path.join(SAIDA, "resumo_cobertura.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt), flush=True)
    print(f"\nResultados em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
