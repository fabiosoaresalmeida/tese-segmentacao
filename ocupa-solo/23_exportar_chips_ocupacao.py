# -*- coding: utf-8 -*-
r"""
exportar_chips_ocupacao.py  (ArcGIS Pro 3.6, extensão Image Analyst)

Exporta os chips de treino do modelo de OCUPAÇÃO DO SOLO (Classified Tiles),
só dentro da máscara de treino (as 13 ortofotos de treino).

  - tile 256 x 256, stride 128 (50% de sobreposição)
  - ALL_TILES: exporta também os chips sem polígonos, porque a classe 0
    (sem representação / solo nu) é uma classe real que a rede tem de aprender
  - classe 8 = Ignorar (usar ignore_classes=[8] no treino)

No fim verifica os chips: n.º de imagens e de rótulos, píxeis por classe e
pesos de referência (median frequency balancing). Grava um relatório em txt.

Correr no Python Command Prompt (fechar o ArcGIS Pro ou remover do mapa as
camadas de entrada) ou na janela Python do ArcGIS Pro.
"""
import os
import glob
import json
import time
import datetime
import traceback
from collections import Counter
import arcpy

# =============================== PARÂMETROS ===============================
MOSAICO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"
ROTULOS_TREINO = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\22_separacao\rotulos_treino.shp"
MASCARA_TREINO = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\22_separacao\mascara_treino.shp"
CAMPO_CLASSE = "classe"

PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\23_Training\chips_ocupacao_256_128"
APAGAR_SAIDA_SE_EXISTIR = True   # True apaga a pasta de saída antes de exportar

TILE = 256
STRIDE = 128
FORMATO = "TIFF"
IGNORE = 8

NOMES = {0: "Sem_representacao", 1: "Vinha", 2: "Pomar_vinha", 3: "Pomar",
         4: "Arvoredo_denso", 5: "Arvoredo_esparso", 6: "Mato_denso", 7: "Agua",
         8: "Ignorar"}

VERIFICAR_CHIPS = True   # lê todos os rótulos exportados para contar os píxeis
# ==========================================================================

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False


class Relatorio:
    def __init__(self):
        self.linhas = []

    def __call__(self, txt=""):
        print(txt)
        self.linhas.append(str(txt))

    def gravar(self, caminho):
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas) + "\n")


def validar(rel):
    for nome, c in (("MOSAICO", MOSAICO), ("ROTULOS_TREINO", ROTULOS_TREINO),
                    ("MASCARA_TREINO", MASCARA_TREINO)):
        if not arcpy.Exists(c):
            raise FileNotFoundError(f"{nome} não existe: {c}")
    campos = [f.name.lower() for f in arcpy.ListFields(ROTULOS_TREINO)]
    if CAMPO_CLASSE.lower() not in campos:
        raise ValueError(f"O campo {CAMPO_CLASSE} não existe em {ROTULOS_TREINO}")

    sr_m = arcpy.Raster(MOSAICO).spatialReference
    sr_r = arcpy.Describe(ROTULOS_TREINO).spatialReference
    sr_k = arcpy.Describe(MASCARA_TREINO).spatialReference
    rel("SISTEMAS DE COORDENADAS")
    for n, sr in (("Mosaico", sr_m), ("Rótulos", sr_r), ("Máscara", sr_k)):
        rel(f"  {n:<8} {sr.name}")
    if not (sr_m.name == sr_r.name == sr_k.name):
        raise RuntimeError("O mosaico, os rótulos e a máscara não estão no mesmo sistema.")
    rel("  OK: mesmo sistema de coordenadas.")
    r = arcpy.Raster(MOSAICO)
    rel(f"  Mosaico: {r.width} x {r.height} píxeis | pixel {r.meanCellWidth} m | "
        f"{r.bandCount} bandas | {r.pixelType}")
    rel()

    classes = sorted({c for (c,) in arcpy.da.SearchCursor(ROTULOS_TREINO, [CAMPO_CLASSE])})
    rel(f"CLASSES NOS RÓTULOS DE TREINO: {classes}")
    desconhecidas = [c for c in classes if c not in NOMES]
    if desconhecidas:
        rel(f"  AVISO: classes sem nome definido: {desconhecidas}")
    rel()
    return r.meanCellWidth


def preparar_saida(rel):
    if os.path.isdir(PASTA_SAIDA) and os.listdir(PASTA_SAIDA):
        if APAGAR_SAIDA_SE_EXISTIR:
            arcpy.management.Delete(PASTA_SAIDA)
            rel(f"Pasta de saída apagada: {PASTA_SAIDA}")
        else:
            raise RuntimeError(f"A pasta de saída já tem ficheiros: {PASTA_SAIDA}. "
                               "Muda PASTA_SAIDA ou põe APAGAR_SAIDA_SE_EXISTIR = True.")


def ler_rotulo(caminho):
    """Lê um chip de rótulo como array (Pillow se existir, senão arcpy)."""
    try:
        from PIL import Image
        import numpy as np
        return np.array(Image.open(caminho))
    except Exception:
        return arcpy.RasterToNumPyArray(caminho)


def verificar_chips(rel, pixel):
    imgs = glob.glob(os.path.join(PASTA_SAIDA, "images", "*.tif"))
    labs = glob.glob(os.path.join(PASTA_SAIDA, "labels", "*.tif"))
    rel("CHIPS EXPORTADOS")
    rel(f"  imagens: {len(imgs)} | rótulos: {len(labs)}"
        f"  ({'OK' if len(imgs) == len(labs) else 'NÚMEROS DIFERENTES!'})")
    rel(f"  cada chip: {TILE} px = {TILE * pixel:.0f} m de lado; stride {STRIDE} px = {STRIDE * pixel:.0f} m")

    for js in ("esri_model_definition.emd", "esri_accumulated_stats.json"):
        rel(f"  {js}: {'existe' if os.path.exists(os.path.join(PASTA_SAIDA, js)) else 'NÃO existe'}")
    stats = os.path.join(PASTA_SAIDA, "esri_accumulated_stats.json")
    if os.path.exists(stats):
        try:
            with open(stats, encoding="utf-8") as f:
                d = json.load(f)
            cls = d.get("Classes") or d.get("classes")
            if cls:
                rel(f"  classes no metadata: {[c.get('ClassValue', c) for c in cls]}")
        except Exception as e:
            rel(f"  (não foi possível ler o json: {e})")
    rel()

    if not VERIFICAR_CHIPS or not labs:
        return
    print("A contar píxeis nos rótulos (pode demorar)...")
    total = Counter()
    chips_com = Counter()
    for i, lab in enumerate(labs, 1):
        a = ler_rotulo(lab)
        vals, cont = __import__("numpy").unique(a, return_counts=True)
        for v, c in zip(vals.tolist(), cont.tolist()):
            total[v] += c
            chips_com[v] += 1
        if i % 2000 == 0:
            print(f"  {i}/{len(labs)}")

    soma = sum(total.values())
    validos = sum(c for v, c in total.items() if v != IGNORE)
    rel("PÍXEIS POR CLASSE NOS CHIPS (conta as sobreposições do stride)")
    rel(f"  {'classe':>6}  {'nome':<20} {'% píxeis':>9} {'% sem Ignorar':>14} {'chips com a classe':>19}")
    freq = {}
    for v in sorted(total):
        p = 100 * total[v] / soma
        pv = 100 * total[v] / validos if v != IGNORE else float("nan")
        if v != IGNORE:
            freq[v] = total[v] / validos
        rel(f"  {v:>6}  {NOMES.get(v, '?'):<20} {p:>8.2f}% {pv:>13.2f}% "
            f"{chips_com[v]:>12} ({100 * chips_com[v] / len(labs):.1f}%)")
    rel()

    if freq:
        mediana = sorted(freq.values())[len(freq) // 2]
        rel("PESOS DE REFERÊNCIA (median frequency balancing, sem a classe Ignorar)")
        for v in sorted(freq):
            rel(f"  {v:>6}  {NOMES.get(v, '?'):<20} {mediana / freq[v]:>6.2f}")
        rel("  (o class_balancing=True do ArcGIS calcula os seus próprios pesos; estes são para comparação)")
        rel()


def main(rel):
    t0 = time.time()
    rel("=" * 78)
    rel("RELATÓRIO - EXPORTAÇÃO DE CHIPS - OCUPAÇÃO DO SOLO")
    rel("=" * 78)
    rel(f"Data/hora:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rel(f"ArcGIS Pro:  {arcpy.GetInstallInfo().get('Version', '?')}")
    rel()
    rel("PARÂMETROS")
    for n in ("MOSAICO", "ROTULOS_TREINO", "MASCARA_TREINO", "CAMPO_CLASSE",
              "PASTA_SAIDA", "TILE", "STRIDE", "FORMATO", "IGNORE"):
        rel(f"  {n:<15} = {globals()[n]}")
    rel()

    if arcpy.CheckExtension("ImageAnalyst") != "Available":
        raise RuntimeError("Extensão Image Analyst não disponível.")
    arcpy.CheckOutExtension("ImageAnalyst")

    pixel = validar(rel)
    preparar_saida(rel)

    rel("A exportar...")
    t1 = time.time()
    arcpy.ia.ExportTrainingDataForDeepLearning(
        in_raster=MOSAICO,
        out_folder=PASTA_SAIDA,
        in_class_data=ROTULOS_TREINO,
        image_chip_format=FORMATO,
        tile_size_x=TILE,
        tile_size_y=TILE,
        stride_x=STRIDE,
        stride_y=STRIDE,
        output_nofeature_tiles="ALL_TILES",
        metadata_format="Classified_Tiles",
        start_index=0,
        class_value_field=CAMPO_CLASSE,
        buffer_radius=0,
        in_mask_polygons=MASCARA_TREINO,
        rotation_angle=0,
        reference_system="MAP_SPACE",
    )
    rel(f"Exportação concluída em {(time.time() - t1) / 60:.1f} min")
    rel()

    verificar_chips(rel, pixel)

    rel("TREINO (Train Deep Learning Model)")
    rel(f"  Input Training Data = {PASTA_SAIDA}")
    rel(f"  Model arguments: ignore_classes=[{IGNORE}], class_balancing=True")
    rel("  Opcional: focal_loss=True; validação 10%")
    rel()
    rel(f"Tempo total: {(time.time() - t0) / 60:.1f} min")
    arcpy.CheckInExtension("ImageAnalyst")
    rel.gravar(os.path.join(PASTA_SAIDA, "exportacao_chips_relatorio.txt"))


if __name__ == "__main__":
    _rel = Relatorio()
    try:
        main(_rel)
    except Exception:
        _rel()
        _rel("ERRO - execução interrompida")
        _rel(traceback.format_exc())
        destino = PASTA_SAIDA if os.path.isdir(PASTA_SAIDA) else os.path.dirname(PASTA_SAIDA)
        _rel.gravar(os.path.join(destino, "exportacao_chips_relatorio_erro.txt"))
        raise