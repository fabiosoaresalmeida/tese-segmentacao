# -*- coding: utf-8 -*-
r"""
inventario_ortos_rede_CIGeoE.py - inventario de TODOS os rasters de uma arvore de pastas desorganizada,
para encontrar ortos com NIR (4 bandas) e pixel <= 0,5 m noutras regioes.

Percorre RAIZ e todas as subpastas (qualquer profundidade). Para cada raster (.tif, .tiff,
.img, .jp2, .ecw, .sid) regista: caminho completo, pastas, nome, bandas, pixel, tipo de
pixel, dimensoes, area, sistema de coordenadas, extensao, centro em coordenadas
geograficas (quando o sistema e conhecido), tamanho e data. Ficheiros que nao abrem (ex.:
'.prj.tif') ficam registados com o erro, sem parar o inventario.

Saidas em SAIDA:
  inventario_ortos.csv     uma linha por ficheiro (abrir no Excel e filtrar)
  resumo_por_pasta.csv     uma linha por pasta: n.o de ortos, bandas, pixeis, sistemas, area
  candidatos.txt           pastas com ortos de 4 bandas e pixel <= PIXEL_MAX
  footprints.gdb\ortos     um poligono por orto com sistema conhecido, em WGS84 Militares
                           IGeoE (abrir no ArcGIS Pro e ver onde ficam e se ha cartografia)

E RETOMAVEL: se for interrompido, ao voltar a correr salta os ficheiros ja inventariados.
Numa pasta de rede com milhares de ficheiros pode demorar horas: deixar a correr.

Correr no Python Command Prompt do ArcGIS Pro:
    python inventario_ortos.py
"""
import os
import csv
import time
import datetime as _dt   # alias: o 'import arcpy' volta a ligar o nome 'datetime' ao modulo
from collections import defaultdict

import arcpy

# =============================== CONFIGURACAO ===============================
RAIZ = r"\\igeoe.local\cpc\sfotog\00_Ortos"
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Inventario_ortos")   # junto ao script
EXTENSOES = {".tif", ".tiff", ".img", ".jp2", ".ecw", ".sid"}
IGNORAR_PASTAS = {"mdt", "mdt_wgs84_completo"}      # nomes de pastas a saltar (minusculas)
IGNORAR_SUFIXOS = (".prj.tif",)                      # ficheiros que nao sao rasters validos
PIXEL_MAX = 0.5                                      # candidatos: pixel <= isto (m)
SR_REFERENCIA = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"   # sistema dos footprints
#   (se nao existir neste PC, os footprints ficam em WGS84 geograficas)
# ============================================================================

CSV_INV = os.path.join(SAIDA, "inventario_ortos.csv")
CAMPOS = ["caminho", "pasta_1", "pasta_2", "pasta_pai", "ficheiro", "bandas", "pixel_x", "pixel_y",
          "tipo_pixel", "largura", "altura", "area_km2", "sr_nome", "sr_tipo", "sr_codigo", "unidade",
          "xmin", "ymin", "xmax", "ymax", "lon_centro", "lat_centro", "tamanho_mb", "modificado", "erro"]


def ler_feitos():
    if not os.path.exists(CSV_INV):
        return set()
    with open(CSV_INV, encoding="utf-8-sig") as f:
        return {l["caminho"] for l in csv.DictReader(f, delimiter=";")}


def listar():
    for d, subdirs, ficheiros in os.walk(RAIZ):
        subdirs[:] = [s for s in subdirs if s.lower() not in IGNORAR_PASTAS]
        for f in ficheiros:
            fl = f.lower()
            if os.path.splitext(fl)[1] in EXTENSOES and not fl.endswith(IGNORAR_SUFIXOS):
                yield os.path.join(d, f)


def descrever(caminho):
    rel = os.path.relpath(caminho, RAIZ)
    partes = rel.split(os.sep)
    linha = dict(caminho=caminho, pasta_1=partes[0] if len(partes) > 1 else "",
                 pasta_2=partes[1] if len(partes) > 2 else "",
                 pasta_pai=os.path.dirname(rel), ficheiro=partes[-1])
    try:
        st = os.stat(caminho)
        linha["tamanho_mb"] = round(st.st_size / 1e6, 1)
        linha["modificado"] = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
    except OSError:
        pass
    try:
        r = arcpy.Raster(caminho)
        sr, e = r.spatialReference, r.extent
        linha.update(bandas=r.bandCount, pixel_x=round(r.meanCellWidth, 4), pixel_y=round(r.meanCellHeight, 4),
                     tipo_pixel=r.pixelType, largura=r.width, altura=r.height,
                     sr_nome=sr.name if sr else "", sr_tipo=sr.type if sr else "",
                     sr_codigo=sr.factoryCode if sr else "",
                     unidade=(sr.linearUnitName if sr and sr.type == "Projected" else
                              sr.angularUnitName if sr and sr.type == "Geographic" else ""),
                     xmin=round(e.XMin, 2), ymin=round(e.YMin, 2), xmax=round(e.XMax, 2), ymax=round(e.YMax, 2))
        if sr and sr.type == "Projected" and sr.metersPerUnit:
            linha["area_km2"] = round(r.width * r.height * r.meanCellWidth * r.meanCellHeight
                                      * sr.metersPerUnit ** 2 / 1e6, 3)
        if sr and sr.type in ("Projected", "Geographic") and sr.name.lower() not in ("unknown", ""):
            c = arcpy.PointGeometry(arcpy.Point((e.XMin + e.XMax) / 2, (e.YMin + e.YMax) / 2), sr)
            g = c.projectAs(arcpy.SpatialReference(4326))
            linha["lon_centro"], linha["lat_centro"] = round(g.firstPoint.X, 5), round(g.firstPoint.Y, 5)
    except Exception as ex:
        linha["erro"] = str(ex).splitlines()[0][:200]
    return linha


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def resumos():
    with open(CSV_INV, encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    g = defaultdict(list)
    for l in linhas:
        g[l["pasta_pai"]].append(l)
    res, cand = [], []
    for pasta, ls in sorted(g.items()):
        ok = [l for l in ls if not l["erro"]]
        bandas = sorted({l["bandas"] for l in ok})
        pixeis = sorted({l["pixel_x"] for l in ok}, key=lambda v: num(v) or 0)
        srs = sorted({l["sr_nome"] for l in ok})
        area = sum(num(l["area_km2"]) or 0 for l in ok)
        anos = sorted({l["modificado"][:4] for l in ok if l["modificado"]})
        res.append([pasta, len(ls), len(ls) - len(ok), " ".join(bandas), " ".join(pixeis), " | ".join(srs),
                    round(area, 1), " ".join(anos)])
        bons = [l for l in ok if (num(l["bandas"]) or 0) >= 4 and (num(l["pixel_x"]) or 99) <= PIXEL_MAX + 1e-6]
        if bons:
            lon = [num(l["lon_centro"]) for l in bons if l["lon_centro"]]
            lat = [num(l["lat_centro"]) for l in bons if l["lat_centro"]]
            onde = (f"centro ~ lon {sum(lon) / len(lon):.3f}, lat {sum(lat) / len(lat):.3f}" if lon
                    else "sistema desconhecido (localizar no ArcGIS)")
            cand.append(f"{pasta}\n    {len(bons)} ortos com 4+ bandas e pixel <= {PIXEL_MAX} m | pixel "
                        f"{sorted({l['pixel_x'] for l in bons})} | {sorted({l['sr_nome'] for l in bons})} | "
                        f"{sum(num(l['area_km2']) or 0 for l in bons):.1f} km2 | {onde}")
    with open(os.path.join(SAIDA, "resumo_por_pasta.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["pasta", "n_ficheiros", "n_erros", "bandas", "pixeis", "sistemas", "area_km2", "anos_ficheiro"])
        w.writerows(res)
    with open(os.path.join(SAIDA, "candidatos.txt"), "w", encoding="utf-8") as f:
        f.write(f"PASTAS COM ORTOS DE 4 BANDAS E PIXEL <= {PIXEL_MAX} m ({len(cand)})\n\n" + "\n\n".join(cand) + "\n")
    print(f"\n{len(cand)} pastas candidatas -> {os.path.join(SAIDA, 'candidatos.txt')}")
    return linhas


def footprints(linhas):
    """Poligono da extensao de cada orto com sistema conhecido, no sistema da 126."""
    if os.path.exists(SR_REFERENCIA):
        sr_dest = arcpy.Describe(SR_REFERENCIA).spatialReference
    else:
        sr_dest = arcpy.SpatialReference(4326)
        print(f"AVISO: {SR_REFERENCIA} nao existe neste PC; footprints em WGS84 (lat/lon)")
    gdb = os.path.join(SAIDA, "footprints.gdb")
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(SAIDA, "footprints.gdb")
    fc = os.path.join(gdb, "ortos")
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)
    arcpy.management.CreateFeatureclass(gdb, "ortos", "POLYGON", spatial_reference=sr_dest)
    for nome, tipo, comp in (("pasta_pai", "TEXT", 255), ("ficheiro", "TEXT", 255), ("bandas", "SHORT", None),
                             ("pixel", "DOUBLE", None), ("sr_origem", "TEXT", 100), ("candidato", "SHORT", None)):
        arcpy.management.AddField(fc, nome, tipo, field_length=comp)
    srs, n = {}, 0
    with arcpy.da.InsertCursor(fc, ["SHAPE@", "pasta_pai", "ficheiro", "bandas", "pixel", "sr_origem",
                                    "candidato"]) as ic:
        for l in linhas:
            if l["erro"] or not l["lon_centro"]:
                continue
            try:
                if l["caminho"] not in srs:
                    srs[l["caminho"]] = arcpy.Describe(l["caminho"]).spatialReference
                sr = srs[l["caminho"]]
                x0, y0, x1, y1 = (float(l[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                pol = arcpy.Polygon(arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x0, y1), arcpy.Point(x1, y1),
                                                 arcpy.Point(x1, y0), arcpy.Point(x0, y0)]), sr)
                b, px = int(l["bandas"]), float(l["pixel_x"])
                ic.insertRow([pol.projectAs(sr_dest), l["pasta_pai"][:255], l["ficheiro"][:255], b, px,
                              l["sr_nome"][:100], int(b >= 4 and px <= PIXEL_MAX + 1e-6)])
                n += 1
            except Exception:
                continue
    print(f"{n} footprints -> {fc}")


def main():
    os.makedirs(SAIDA, exist_ok=True)
    feitos = ler_feitos()
    novo = not os.path.exists(CSV_INV)
    print(f"Raiz: {RAIZ} | ja inventariados: {len(feitos)}")
    t0, n = time.time(), 0
    with open(CSV_INV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, delimiter=";", extrasaction="ignore")
        if novo:
            w.writeheader()
        for caminho in listar():
            if caminho in feitos:
                continue
            w.writerow(descrever(caminho))
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"  {n} ficheiros ({(time.time() - t0) / 60:.1f} min) | ultimo: {caminho}", flush=True)
    print(f"{n} ficheiros novos inventariados em {(time.time() - t0) / 60:.1f} min -> {CSV_INV}")
    linhas = resumos()
    footprints(linhas)


if __name__ == "__main__":
    main()