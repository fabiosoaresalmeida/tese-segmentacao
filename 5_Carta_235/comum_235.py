# -*- coding: utf-8 -*-
"""
comum_235.py - funcoes partilhadas pelos scripts de avaliacao na folha 235
(2_avaliar_rviaria_235.py e 3b_avaliar_ocupacao_235.py).

  - blocos da grelha GRELHA x GRELHA com a MESMA divisao do 0_verificar_dominio_126_235.py
    (e do 1_avaliar_construcoes_235.py): mesmos nomes (ex.: 235_bloco_L4C4) e mesmos limites;
  - leitura da radiometria de cada bloco (saidas do 0_verificar_dominio);
  - correlacao de Spearman;
  - projecao de uma feature class para o sistema do ortomosaico, com a transformacao
    usada na 126 (Lisbon_Lisbon_To_WGS_1984_2).

Tem de estar na mesma pasta dos scripts que o importam.
"""
import os
import csv

import numpy as np

GRELHA = 4            # igual ao GRELHA do 0_verificar_dominio
BLOCO_LEITURA = 4096  # igual ao BLOCO do 0_verificar_dominio


def blocos_pixels(largura, altura, prefixo, grelha=GRELHA, bloco_leitura=BLOCO_LEITURA):
    """Blocos em coordenadas de pixel (linha 0 = topo), com a divisao do 0_verificar.
    Devolve lista de dict(nome, r0, r1, c0, c1)."""
    W, H, G = largura, altura, grelha
    bloco = max(256, min(bloco_leitura, W // G, H // G))
    cols, rows = {}, {}
    for x in range(0, W, bloco):
        nc = min(x + bloco, W) - x
        gc = min(G - 1, int((x + nc / 2.0) / W * G))
        a, b = cols.get(gc, (x, x + nc))
        cols[gc] = (min(a, x), max(b, x + nc))
    for y in range(0, H, bloco):          # y conta a partir de baixo (como no 0_verificar)
        nr = min(y + bloco, H) - y
        gl = min(G - 1, int((1 - (y + nr / 2.0) / H) * G))
        r0, r1 = H - (y + nr), H - y      # linhas do array, a partir do topo
        a, b = rows.get(gl, (r0, r1))
        rows[gl] = (min(a, r0), max(b, r1))
    out = []
    for gl in sorted(rows):
        for gc in sorted(cols):
            (r0, r1), (c0, c1) = rows[gl], cols[gc]
            out.append(dict(nome=f"{prefixo}_bloco_L{gl + 1}C{gc + 1}", r0=r0, r1=r1, c0=c0, c1=c1))
    return out


def blocos_raster(raster_path, grelha=GRELHA, bloco_leitura=BLOCO_LEITURA):
    """Blocos do raster, com a extensao em coordenadas (xmin, ymin, xmax, ymax)."""
    import arcpy
    r = arcpy.Raster(raster_path)
    cw, ch = r.meanCellWidth, r.meanCellHeight
    xmin, ymax = r.extent.XMin, r.extent.YMax
    prefixo = os.path.splitext(os.path.basename(raster_path))[0]
    blocos = blocos_pixels(r.width, r.height, prefixo, grelha, bloco_leitura)
    for b in blocos:
        b["ext"] = (xmin + b["c0"] * cw, ymax - b["r1"] * ch, xmin + b["c1"] * cw, ymax - b["r0"] * ch)
    return blocos


def poligono_bloco(b, sr):
    import arcpy
    x0, y0, x1, y1 = b["ext"]
    return arcpy.Polygon(arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x0, y1), arcpy.Point(x1, y1),
                                      arcpy.Point(x1, y0), arcpy.Point(x0, y0)]), sr)


def criar_fc_blocos(blocos, sr, fc):
    """Feature class com um poligono por bloco (campo NOME)."""
    import arcpy
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)
    pasta, nome = os.path.split(fc)
    arcpy.management.CreateFeatureclass(pasta, nome, "POLYGON", spatial_reference=sr)
    arcpy.management.AddField(fc, "NOME", "TEXT", field_length=64)
    with arcpy.da.InsertCursor(fc, ["NOME", "SHAPE@"]) as ic:
        for b in blocos:
            ic.insertRow([b["nome"], poligono_bloco(b, sr)])
    return fc


def ler_radiometria(pasta, ref="126"):
    """{bloco: dict(media_R.., emd_R.., emd_medio, fora_126)} das saidas do 0_verificar."""
    if not pasta:
        return {}
    f_orto = os.path.join(pasta, "radiometria_por_orto.csv")
    f_dist = os.path.join(pasta, "radiometria_distancias.csv")
    if not (os.path.exists(f_orto) and os.path.exists(f_dist)):
        return {}
    num = lambda s: float(str(s).replace(",", "."))
    rad, max_int = {}, {}
    with open(f_orto, encoding="utf-8-sig") as f:
        for l in csv.DictReader(f, delimiter=";"):
            rad.setdefault(l["orto"], {})["media_" + l["banda"]] = num(l["media"])
    with open(f_dist, encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    for l in linhas:
        if l["tipo"].startswith("interna_") and l["b"].endswith(ref):
            max_int[l["banda"]] = max(max_int.get(l["banda"], 0.0), num(l["emd"]))
    for l in linhas:
        if l["tipo"].startswith("orto_"):
            d = rad.setdefault(l["a"], {})
            d["emd_" + l["banda"]] = num(l["emd"])
            if num(l["emd"]) > max_int.get(l["banda"], float("inf")):
                d["fora_126"] = "sim"
    for d in rad.values():
        d.setdefault("fora_126", "nao")
        emds = [v for k, v in d.items() if k.startswith("emd_")]
        d["emd_medio"] = float(np.mean(emds)) if emds else None
    return rad


def spearman(x, y):
    """Correlacao de Spearman simples, ou None (menos de 4 pares ou sem variacao)."""
    pares = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pares) < 4:
        return None
    xa, ya = np.array([p[0] for p in pares], float), np.array([p[1] for p in pares], float)
    rx, ry = xa.argsort().argsort().astype(float), ya.argsort().argsort().astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def projetar_se_preciso(fc, raster_ref, saida, transformacao, log=print):
    """fc no sistema do raster_ref. Se o sistema for diferente, projeta para 'saida' com a
    transformacao indicada (a da 126) ou, se nao for valida, a primeira da lista do ArcGIS."""
    import arcpy
    sr_fc = arcpy.Describe(fc).spatialReference
    sr_r = arcpy.Describe(raster_ref).spatialReference
    if sr_fc.name == sr_r.name:
        log(f"   {os.path.basename(fc)}: ja no sistema do ortomosaico ({sr_r.name})")
        return fc
    lista = arcpy.ListTransformations(sr_fc, sr_r, arcpy.Describe(fc).extent)
    t = transformacao if transformacao in lista else (lista[0] if lista else "")
    log(f"   {os.path.basename(fc)}: {sr_fc.name} -> {sr_r.name} | transformacao: {t or 'nenhuma'}"
        + ("" if t == transformacao else f" (AVISO: {transformacao} nao disponivel)"))
    if arcpy.Exists(saida):
        arcpy.management.Delete(saida)
    arcpy.management.Project(fc, saida, sr_r, t)
    return saida


def janela_teste(raster_path, lado_m, centro=None):
    """Extensao (xmin, ymin, xmax, ymax) de uma janela quadrada de lado_m metros, centrada no
    raster (ou em centro=(x, y)), alinhada a grelha do raster, para o modo teste rapido."""
    import arcpy
    r = arcpy.Raster(raster_path)
    e, cw = r.extent, r.meanCellWidth
    cx, cy = centro or ((e.XMin + e.XMax) / 2, (e.YMin + e.YMax) / 2)
    h = lado_m / 2
    snap = lambda v, o: o + round((v - o) / cw) * cw
    return (max(e.XMin, snap(cx - h, e.XMin)), max(e.YMin, snap(cy - h, e.YMin)),
            min(e.XMax, snap(cx + h, e.XMin)), min(e.YMax, snap(cy + h, e.YMin)))
