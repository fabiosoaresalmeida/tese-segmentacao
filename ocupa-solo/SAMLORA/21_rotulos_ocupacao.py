# -*- coding: utf-8 -*-
"""
SHAPE de rótulos para o modelo de OCUPAÇÃO DO SOLO (vegetação + água), já sem
sobreposições, pronta para o Export Training Data (Classified Tiles, campo 'classe').

Classes (campo 'classe'), segundo as regras de edição:
  0  Sem representação (sobretudo solo nu) -> NÃO tem polígonos: tudo o que fica
     sem polígono sai como 0 nos chips e é aprendido como fundo
  1  A_Vinha
  2  A_Pomar_vinha
  3  A_Pomar
  4  A_Arvoredo_denso
  5  A_Arvoredo_esparso
  6  A_Mato_denso_ou_arbustos
  7  Água (A_Albufeira, A_Rio_de_duas_margens, A_Lagoa)
  8  Ignorar (construções + estradas com buffer) -> ignore_classes=[8] no treino

Vinha, Pomar e vinha e Pomar ficam separadas. Para avaliar a versão generalizada
(Pomar e vinha = Vinha), juntam-se as classes 1 e 2 na avaliação, sem novo treino.

Precedência nas sobreposições: Ignorar (estradas, construções) > Água > Vegetação
  - Vegetação x estradas/construções -> fica Ignorar
  - Água x estradas (pontes)          -> fica Ignorar (o tabuleiro não é água)
  - Água x vegetação                  -> fica Água
  - Construções x estradas            -> indiferente (ambas Ignorar, dissolvidas)

Correr com o python do ambiente arcgispro-py3.
"""
import os
import csv
from collections import defaultdict
import arcpy

# ============================ CONFIGURAÇÃO ============================
# Referência espacial: o ortomosaico (caminho completo do ficheiro)
MOSAIC = r"E:\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"

VEGETACAO = r"E:\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_vegetacao.shp"
HIDRO_POL = r"E:\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_hidro.shp"
CONSTR    = r"E:\Cap_Almeida\Projeto\M888_SHP\126_2018_4\a_constr.shp"
ESTRADAS  = r"E:\Cap_Almeida\Projeto\M888_SHP\126_2018_4\l_vias.shp"   # eixos
CLASS_FIELD = "source"

# Estradas: LARGURA TOTAL em metros por 'source' (as do treino das vias); buffer = largura/2
ESTRADAS_CLASS_FIELD = "source"
ESTRADAS_LARGURA = {
    "Auto_estrada": 11.0,
    "Estrada_larga": 7.0,
    "Estrada_estreita": 4.0,
    "Arruamento": 5.0,
    "Acesso_auto": 3.0,
    "Caminho_carreteiro": 2.5,
    "Caminho_a_pe_posto": 1.5,
    "Caminho_de_ferro_via_larga_unica": 3.5,
}
ESTRADAS_LARGURA_DEFEITO = 3.0

OUT_FOLDER = r"E:\Cap_Almeida\Projeto\126\rotulos_ocupacao"
GDB_NAME = "rotulos.gdb"
OUT_SHP = "rotulos_ocupacao.shp"

VEG_MAP = {
    "A_Vinha": 1,
    "A_Pomar_vinha": 2,
    "A_Pomar": 3,
    "A_Arvoredo_denso": 4,
    "A_Arvoredo_esparso": 5,
    "A_Mato_denso_ou_arbustos": 6,
}
AGUA_MAP = {"A_Albufeira": 7, "A_Rio_de_duas_margens": 7, "A_Lagoa": 7}
IGNORE = 8

NOMES = {1: "Vinha", 2: "Pomar_vinha", 3: "Pomar", 4: "Arvoredo_denso",
         5: "Arvoredo_esparso", 6: "Mato_denso", 7: "Agua", IGNORE: "Ignorar"}
# =====================================================================

arcpy.env.overwriteOutput = True


def log(msg):
    print(msg)


def find_field(fc, wanted):
    for f in arcpy.ListFields(fc):
        if f.name.lower() == wanted.lower():
            return f.name
    raise RuntimeError(f"Campo '{wanted}' não encontrado em {fc}")


def largura_de(larguras, c, defeito):
    """Procura a largura pelo valor de 'source' tal como está ou sem o prefixo L_/A_."""
    if c in larguras:
        return larguras[c]
    if c and str(c)[:2] in ("L_", "A_") and str(c)[2:] in larguras:
        return larguras[str(c)[2:]]
    return None if defeito is None else defeito


def prepare(gdb, name, path, sr):
    d = arcpy.Describe(path)
    out = os.path.join(gdb, name)
    if d.spatialReference.name == sr.name:
        arcpy.management.CopyFeatures(path, out)
    else:
        trans = arcpy.ListTransformations(d.spatialReference, sr, d.extent)
        t = trans[0] if trans else ""
        log(f"{name}: {d.spatialReference.name} -> {sr.name} | transformação: {t or 'nenhuma'}")
        arcpy.management.Project(path, out, sr, t)
    arcpy.management.RepairGeometry(out)
    return out


def set_class(fc, mapping=None, constant=None):
    """Campo 'classe'. Com mapping, apaga os elementos com source fora do mapa."""
    arcpy.management.AddField(fc, "classe", "SHORT")
    fields = ["classe"] + ([find_field(fc, CLASS_FIELD)] if mapping is not None else [])
    fora = defaultdict(int)
    with arcpy.da.UpdateCursor(fc, fields) as cur:
        for row in cur:
            if mapping is None:
                cur.updateRow([constant])
                continue
            v = mapping.get(row[1])
            if v is None:
                fora[row[1]] += 1
                cur.deleteRow()
            else:
                cur.updateRow([v, row[1]])
    for k, n in fora.items():
        log(f"  AVISO: '{k}' ({n} elementos) sem classe no mapeamento -> excluído")


def buffer_estradas(gdb, fc):
    """Buffer dos eixos com a LARGURA TOTAL de cada classe (distância = largura / 2)."""
    if arcpy.Describe(fc).shapeType != "Polyline":
        return fc
    src = find_field(fc, ESTRADAS_CLASS_FIELD)
    arcpy.management.AddField(fc, "buf_m", "DOUBLE")
    sem = set()
    with arcpy.da.UpdateCursor(fc, [src, "buf_m"]) as cur:
        for c, _ in cur:
            if largura_de(ESTRADAS_LARGURA, c, None) is None:
                sem.add(c)
            cur.updateRow([c, largura_de(ESTRADAS_LARGURA, c, ESTRADAS_LARGURA_DEFEITO) / 2.0])
    if sem:
        log(f"  AVISO: estradas sem largura definida (usa {ESTRADAS_LARGURA_DEFEITO} m): {sorted(sem)}")
    out = os.path.join(gdb, "estradas")
    arcpy.analysis.PairwiseBuffer(fc, out, "buf_m", "NONE")
    return out


def erase(in_fc, erase_fc, out_fc):
    """Pairwise Erase; se não estiver disponível na licença, tenta o Erase clássico."""
    try:
        arcpy.analysis.PairwiseErase(in_fc, erase_fc, out_fc)
    except Exception as e:
        log(f"  Pairwise Erase falhou ({e}); a tentar Erase")
        arcpy.analysis.Erase(in_fc, erase_fc, out_fc)
    return out_fc


def keep_fields(fc, keep=("classe",)):
    drop = [f.name for f in arcpy.ListFields(fc)
            if not f.required and f.name.lower() not in [k.lower() for k in keep]]
    if drop:
        arcpy.management.DeleteField(fc, drop)


def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    gdb = os.path.join(OUT_FOLDER, GDB_NAME)
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(OUT_FOLDER, GDB_NAME)

    if not arcpy.Exists(MOSAIC):
        raise RuntimeError(f"MOSAIC não encontrado: {MOSAIC}")
    sr = arcpy.Raster(MOSAIC).spatialReference
    arcpy.env.outputCoordinateSystem = sr
    log(f"Sistema de referência: {sr.name}")

    # --- preparar e classificar
    veg = prepare(gdb, "veg", VEGETACAO, sr)
    set_class(veg, VEG_MAP)
    agua = prepare(gdb, "agua", HIDRO_POL, sr)
    set_class(agua, AGUA_MAP)
    constr = prepare(gdb, "constr", CONSTR, sr)
    est = buffer_estradas(gdb, prepare(gdb, "estradas_eixos", ESTRADAS, sr))
    for fc in (constr, est):
        set_class(fc, constant=IGNORE)
        keep_fields(fc)
    for fc in (veg, agua):
        keep_fields(fc)

    # --- Ignorar: construções + estradas, dissolvido (sem sobreposições internas)
    ign_m = os.path.join(gdb, "ignorar_merge")
    arcpy.management.Merge([constr, est], ign_m)
    ign = os.path.join(gdb, "ignorar")
    arcpy.analysis.PairwiseDissolve(ign_m, ign, "classe", multi_part="SINGLE_PART")

    # --- precedência: Ignorar > Água > Vegetação
    log("A recortar por precedência...")
    agua_ok = erase(agua, ign, os.path.join(gdb, "agua_ok"))        # estradas/pontes ganham à água
    veg_tmp = erase(veg, ign, os.path.join(gdb, "veg_tmp"))         # estradas ganham à vegetação
    veg_ok = erase(veg_tmp, agua_ok, os.path.join(gdb, "veg_ok"))   # água ganha à vegetação

    # --- juntar
    final_gdb = os.path.join(gdb, "rotulos_ocupacao")
    arcpy.management.Merge([ign, agua_ok, veg_ok], final_gdb)
    arcpy.management.MultipartToSinglepart(final_gdb, final_gdb + "_sp")
    final_gdb += "_sp"
    keep_fields(final_gdb)
    arcpy.management.AddField(final_gdb, "nome", "TEXT", field_length=20)
    with arcpy.da.UpdateCursor(final_gdb, ["classe", "nome"]) as cur:
        for c, _ in cur:
            cur.updateRow([c, NOMES.get(c, "?")])

    # --- verificação: não pode haver sobreposições
    chk = os.path.join(gdb, "verif_sobreposicoes")
    arcpy.analysis.CountOverlappingFeatures(final_gdb, chk, 2)
    area_sob = sum(a or 0 for (a,) in arcpy.da.SearchCursor(chk, ["SHAPE@AREA"]))
    if area_sob > 1.0:
        log(f"AVISO: ainda há {area_sob:.1f} m² de sobreposições (ver {chk})")
    else:
        log("Sem sobreposições.")

    # --- exportar shapefile
    out_shp = os.path.join(OUT_FOLDER, OUT_SHP)
    if arcpy.Exists(out_shp):
        arcpy.management.Delete(out_shp)
    arcpy.conversion.ExportFeatures(final_gdb, out_shp)

    # --- resumo
    agg = defaultdict(lambda: [0, 0.0])
    for c, a in arcpy.da.SearchCursor(final_gdb, ["classe", "SHAPE@AREA"]):
        agg[c][0] += 1
        agg[c][1] += a or 0.0
    csv_path = os.path.join(OUT_FOLDER, "resumo_rotulos.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["classe", "nome", "n_poligonos", "area_ha"])
        log("\nclasse  nome               polígonos   área (ha)")
        for c in sorted(agg):
            n, a = agg[c]
            w.writerow([c, NOMES.get(c, "?"), n, round(a / 10000, 2)])
            log(f"{c:>6}  {NOMES.get(c, '?'):<18} {n:>9} {a / 10000:>11.1f}")
    log(f"\nShape: {out_shp}\nResumo: {csv_path}")


if __name__ == "__main__":
    main()
