# -*- coding: utf-8 -*-
r"""
11_teste_ndsm_zero.py  (ArcGIS Pro 3.6)

Verifica se os modelos treinados com o nDSM (fase B) usam de facto essa banda.

Para cada modelo e cada orto de teste indicados:
  1. cria uma copia da orto de teste (5 bandas) com a banda do nDSM SUBSTITUIDA
     POR ZEROS (como se todo o terreno fosse plano, sem edificios);
  2. classifica essa copia com o modelo;
  3. compara com a classificacao feita com o nDSM real (a do 20_avaliar_modelos.py):
       - proporcao de pixeis cuja classe muda entre as duas classificacoes;
       - metricas ao pixel das duas classificacoes contra a referencia.

Interpretacao:
  - se quase nada mudar (por exemplo, menos de 1 % dos pixeis de edificio) e as
    metricas forem praticamente iguais, o modelo esta a IGNORAR o nDSM;
  - se a classificacao mudar muito e as metricas descerem, o modelo USA o nDSM.

Correr no Python Command Prompt do ArcGIS Pro (com a GPU livre):
    python "D:\Users\Cap_Almeida\Projeto\Python\23_teste_ndsm_zero.py"
"""

import os
import csv
import time
import glob
import numpy as np
import arcpy
from datetime import datetime

# ============================================================
# CONFIGURACAO
# ============================================================
PASTA_AVALIACAO = r"D:\DL\Avaliacao\126_original_ndsm"     # avaliacao da fase B (20_avaliar_modelos.py)
GRUPO           = "faseB_ndsm"
PASTA_MODELOS   = r"D:\DL\Modelos\126_original_ndsm"
REFERENCIA      = "edicao"

MODELOS = ["unet_resnet50"]      # inicio do nome da pasta do modelo (pode ter varios)
ORTOS   = ["126-I-3"]            # uma orto chega para o teste
BANDA_NDSM = 5

CELL_SIZE = 0.5
VALOR_EDIFICIO = 1
ARGS_INFERENCIA = "padding 64;batch_size 4;predict_background True;test_time_augmentation False"
# ============================================================

PASTA_TESTE = os.path.join(PASTA_AVALIACAO, "teste_ndsm_zero")
FICHEIRO_LOG = os.path.join(PASTA_TESTE, "log_teste_ndsm_zero.txt")


def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(FICHEIRO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def nome_seguro(txt):
    return txt.replace("-", "_").replace(" ", "_")


def ler_grelha(raster, grelha):
    g = arcpy.Raster(grelha)
    return arcpy.RasterToNumPyArray(raster, arcpy.Point(g.extent.XMin, g.extent.YMin),
                                    g.width, g.height, nodata_to_value=0)


def metricas(pred, ref, msk):
    tp = int(np.sum(pred & ref & msk)); fp = int(np.sum(pred & ~ref & msk))
    fn = int(np.sum(~pred & ref & msk)); tn = int(np.sum(~pred & ~ref & msk))
    n = tp + fp + fn + tn
    ua = tp / (tp + fp) if (tp + fp) else 0.0
    pa = tp / (tp + fn) if (tp + fn) else 0.0
    oa = (tp + tn) / n if n else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0
    return dict(UA=ua, PA=pa, F1=2 * ua * pa / (ua + pa) if (ua + pa) else 0.0,
                kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
                IoU=tp / (tp + fp + fn) if (tp + fp + fn) else 0.0)


def orto_sem_ndsm(orto_tif, saida):
    """Copia da orto com a banda do nDSM a zeros (as outras bandas ficam iguais)."""
    if os.path.exists(saida):
        return saida
    r = arcpy.Raster(orto_tif)
    if r.bandCount < BANDA_NDSM:
        raise RuntimeError(f"A orto {orto_tif} tem {r.bandCount} bandas; esperava pelo menos {BANDA_NDSM}")
    arr = arcpy.RasterToNumPyArray(orto_tif)              # (bandas, linhas, colunas)
    log(f"   nDSM original: min={arr[BANDA_NDSM - 1].min()} max={arr[BANDA_NDSM - 1].max()} "
        f"media={arr[BANDA_NDSM - 1].mean():.1f}")
    arr[BANDA_NDSM - 1] = 0
    ras = arcpy.NumPyArrayToRaster(arr, arcpy.Point(r.extent.XMin, r.extent.YMin),
                                   r.meanCellWidth, r.meanCellHeight)
    ras.save(saida)
    arcpy.management.DefineProjection(saida, r.spatialReference)
    return saida


def main():
    os.makedirs(PASTA_TESTE, exist_ok=True)
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.env.processorType = "CPU"
    arcpy.env.gpuId = 0
    arcpy.env.overwriteOutput = True

    log("=" * 60)
    log(f"Teste do nDSM a zeros | modelos={MODELOS} | ortos={ORTOS}")
    linhas = []
    for pref in MODELOS:
        dlpks = sorted(f for f in glob.glob(os.path.join(PASTA_MODELOS, "**", "*.dlpk"), recursive=True)
                       if os.path.basename(os.path.dirname(f)).lower().startswith(pref.lower()))
        if not dlpks:
            log(f"ERRO: nenhum modelo '{pref}*' em {PASTA_MODELOS}")
            continue
        dlpk = dlpks[-1]
        modelo = os.path.basename(os.path.dirname(dlpk))
        for o in ORTOS:
            so = nome_seguro(o)
            base = os.path.join(PASTA_AVALIACAO, "ortos_teste", GRUPO)
            orto_tif = os.path.join(base, f"orto_{so}.tif")
            msk = ler_grelha(os.path.join(base, f"mascara_{so}.tif"), orto_tif) == 1
            ref = (ler_grelha(os.path.join(base, f"ref_{REFERENCIA}_{so}.tif"), orto_tif) == 1) & msk
            cl_real = os.path.join(PASTA_AVALIACAO, "classificacoes", GRUPO, modelo, f"class_{so}.tif")
            if not os.path.exists(cl_real):
                log(f"ERRO: falta a classificacao com o nDSM real: {cl_real}")
                continue

            log(f"{modelo} | {o}")
            orto_zero = orto_sem_ndsm(orto_tif, os.path.join(PASTA_TESTE, f"orto_ndsm_zero_{so}.tif"))
            cl_zero = os.path.join(PASTA_TESTE, f"class_ndsm_zero_{modelo}_{so}.tif")
            if not os.path.exists(cl_zero):
                t0 = time.time()
                with arcpy.EnvManager(cellSize=CELL_SIZE, snapRaster=orto_tif):
                    arcpy.ia.ClassifyPixelsUsingDeepLearning(orto_zero, dlpk, ARGS_INFERENCIA).save(cl_zero)
                log(f"   classificacao com o nDSM a zeros em {(time.time() - t0) / 60:.1f} min")

            p_real = (ler_grelha(cl_real, orto_tif) == VALOR_EDIFICIO) & msk
            p_zero = (ler_grelha(cl_zero, orto_tif) == VALOR_EDIFICIO) & msk
            mudam = int(np.sum(p_real != p_zero))
            edif_real = int(p_real.sum())
            pct_zona = mudam / int(msk.sum()) * 100
            pct_edif = mudam / edif_real * 100 if edif_real else 0.0
            perdidos = int(np.sum(p_real & ~p_zero)); ganhos = int(np.sum(~p_real & p_zero))
            m_real, m_zero = metricas(p_real, ref, msk), metricas(p_zero, ref, msk)
            log(f"   pixeis que mudam de classe: {mudam} ({pct_zona:.2f} % da zona; "
                f"{pct_edif:.1f} % da area de edificio prevista com o nDSM real)")
            log(f"      edificio -> fundo: {perdidos} | fundo -> edificio: {ganhos}")
            for nome, m in (("nDSM real ", m_real), ("nDSM zeros", m_zero)):
                log(f"   {nome}: UA={m['UA']:.3f} PA={m['PA']:.3f} F1={m['F1']:.3f} "
                    f"kappa={m['kappa']:.3f} IoU={m['IoU']:.3f}")
            dif = m_real["F1"] - m_zero["F1"]
            if pct_edif < 1 and abs(dif) < 0.005:
                conclusao = "o modelo praticamente IGNORA o nDSM"
            elif dif > 0.02:
                conclusao = "o modelo USA o nDSM (sem ele, o F1 desce claramente)"
            else:
                conclusao = "o modelo usa pouco o nDSM"
            log(f"   CONCLUSAO: {conclusao} (diferenca de F1 = {dif:+.3f})")
            linhas.append([modelo, o, mudam, round(pct_zona, 3), round(pct_edif, 2), perdidos, ganhos] +
                          [round(m_real[k], 4) for k in ("UA", "PA", "F1", "kappa", "IoU")] +
                          [round(m_zero[k], 4) for k in ("UA", "PA", "F1", "kappa", "IoU")] + [conclusao])

    with open(os.path.join(PASTA_TESTE, "resultado_teste_ndsm_zero.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["modelo", "orto", "pixeis_que_mudam", "pct_zona", "pct_area_edificio",
                    "edificio_para_fundo", "fundo_para_edificio",
                    "UA_real", "PA_real", "F1_real", "kappa_real", "IoU_real",
                    "UA_zeros", "PA_zeros", "F1_zeros", "kappa_zeros", "IoU_zeros", "conclusao"])
        w.writerows(linhas)
    log("Teste terminado")


if __name__ == "__main__":
    main()