r"""
30_avaliar_ocupacao_TTA.py  (ArcGIS Pro 3.6)

Avaliacao dos modelos de OCUPACAO DO SOLO nas 3 ortos de teste
(126-I-3, 126-II-4, 126-IV-3), contra os rotulos de teste (rotulos_teste.shp).

Pensado para correr ENQUANTO HA UM TREINO NA GPU: antes de cada classificacao
consulta a memoria livre da GPU (nvidia-smi). So usa a GPU se tiver pelo menos
MEM_GPU_MIN_GB livres; caso contrario corre no CPU (mais lento, mas nao toca na
memoria da GPU e nao poe o treino em risco).

Para cada modelo x orto:
  1. prepara a orto de teste com uma margem de contexto (bordas exteriores
     espelhadas), a mascara da zona de teste e a referencia rasterizada na mesma
     grelha (0,5 m) - feito uma vez e reutilizado;
  2. classifica a orto (Classify Pixels Using Deep Learning), reutilizando
     classificacoes ja feitas;
  3. converte as previsoes para os codigos da referencia (MAPA_PREVISOES; ex.: o
     SAMLoRA da versao 1 preve o sem representacao como 0 -> 9);
  4. calcula a matriz de confusao, so dentro da zona de teste e sem os pixeis
     Ignorar (8) e sem poligono (0) da referencia;
  5. calcula as metricas:
     POR CLASSE: precision (exatidao do utilizador), recall (do produtor), F1, IoU,
                 suporte (pixeis de referencia) e pixeis previstos;
     GLOBAIS:    exatidao global (pixel accuracy), exatidao media (balanced accuracy),
                 kappa, mIoU, IoU ponderado pela frequencia (fwIoU), F1 macro e F1
                 ponderado, e % de previsoes fora das classes avaliadas.
As metricas sao calculadas por orto e no TOTAL das 3 ortos, para cada AGRUPAMENTO
de classes (ex.: "completa" e "generalizada", com Pomar e vinha junto com Vinha).

Saidas em PASTA_AVALIACAO:
  metricas_classe.csv, metricas_globais.csv, tabela_comparativa.txt,
  matrizes\<modelo>_<agrupamento>_<orto>.csv (contagens e % por linha),
  classificacoes\<modelo>\class_<orto>.tif, ortos_teste\..., log_avaliacao.txt

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro fechado):
    python "D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\30_avaliar_ocupacao.py"
Com a opcao --cpu, corre SEMPRE no CPU, mesmo que a GPU esteja livre (para correr ao
mesmo tempo que um treino, sem risco de ocupar a GPU entre dois modelos):
    python 30_avaliar_ocupacao.py --cpu
"""
print("script iniciado, a carregar o arcpy...", flush=True)

import arcpy

print("arcpy carregado", flush=True)
import os
import sys
import csv
import glob
import time
import subprocess
import traceback
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURACAO
# ============================================================
ORTOMOSAICO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126_ndsm\126_nDSM_5bandas_v2.tif"
SEP = r"D:\Users\Cap_Almeida\Projeto\126\126_OcupacaoSolo\32_separacao_unet"
REFERENCIA = os.path.join(SEP, "rotulos_teste.shp")     # versao 2: sem representacao = 9
CAMPO_CLASSE = "classe"
ZONA_TESTE = os.path.join(SEP, "zona_teste.shp")         # um poligono por orto
CAMPO_NOME = "NOME"
ORTOS_TESTE = ["126-I-3", "126-II-4", "126-IV-3"]

# Argumentos de inferencia
# v1: o fundo (0) e o sem representacao -> tem de ser previsto (predict_background True)
# v2: o sem representacao e a classe 9 e o 0 foi ignorado no treino -> nao prever o 0
ARGS_V1 = "padding 64;batch_size 4;predict_background True;test_time_augmentation False"
ARGS_V2 = "padding 64;batch_size 4;predict_background False;test_time_augmentation False"
ARGS_V1_TTA = "padding 64;batch_size 4;predict_background True;test_time_augmentation True"
ARGS_V2_TTA = "padding 64;batch_size 4;predict_background False;test_time_augmentation True"

# Modelos: (rotulo, pasta do modelo (ou .emd / .dlpk),
#           mapa previsao -> codigo da referencia, argumentos de inferencia)
# Numa pasta, o script usa o .emd (e nao o .dlpk): o .dlpk e extraido para uma pasta
# temporaria do ArcGIS e, se outro .dlpk com o MESMO NOME ja tiver sido usado, pode ser
# reutilizado o modelo errado. O .emd le os pesos diretamente da pasta do modelo.
MODELOS = [
    ("deeplabv3_r50_ndsm", r"D:\DL\Modelos\126_ocupacao_nDSM\deeplabv3_r50", {}, ARGS_V2_TTA),
    ("samlora_ndsm",       r"D:\DL\Modelos\126_ocupacao_nDSM\samlora", {}, ARGS_V2_TTA),
]
REFAZER = []
PASTA_AVALIACAO = r"D:\DL\Avaliacao\126_ocupacao_nDSM_TTA"

# Classes avaliadas (codigos da referencia) e respetivos nomes
CLASSES = {1: "Vinha", 2: "Pomar_vinha", 3: "Pomar", 4: "Arvoredo_denso",
           5: "Arvoredo_esparso", 6: "Mato_denso", 7: "Agua", 9: "Sem_representacao"}
# Pixeis da referencia excluidos da avaliacao: 8 = Ignorar, 0 = sem poligono (residual)
IGNORAR_REF = {0, 8}
SEM_REF = 255   # valor interno para os pixeis sem referencia rasterizada

# Agrupamentos de classes: nome -> {classe: classe_agrupada}
AGRUPAMENTOS = {
    "completa": {},
    "generalizada": {2: 1},          # Pomar e vinha passa a Vinha
}

# Classificacao
CELL_SIZE = 0.5
CONTEXTO_M = 256          # margem de contexto a volta de cada orto de teste
ESPELHAR_BORDAS = True    # preenche a margem fora do ortomosaico com o reflexo da imagem

# GPU: so e usada se tiver pelo menos esta memoria livre; caso contrario, CPU
MEM_GPU_MIN_GB = 12
GPU_ID = 0
FORCAR_CPU = "--cpu" in sys.argv   # opcao da linha de comandos

# Modo teste: True = so o primeiro modelo e um quadrado de LADO_TESTE_M no centro da
# primeira orto, para medir o tempo de classificacao antes de correr tudo
MODO_TESTE = False
LADO_TESTE_M = 500
# ============================================================

FICHEIRO_LOG = os.path.join(PASTA_AVALIACAO, "log_avaliacao.txt")
SINAIS_BLOQUEIO = ("winerror 32", "being used by another process")


# ------------------------------------------------------------------ utilitarios
def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(FICHEIRO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def nome_seguro(txt):
    return txt.replace("-", "_").replace(" ", "_").replace(".", "_")


def apagar(*itens):
    for it in itens:
        if it and arcpy.Exists(it):
            arcpy.management.Delete(it)


def contar(fc):
    return int(arcpy.management.GetCount(fc)[0])


def ler_grelha(raster, grelha, nodata):
    """Le o raster na grelha exata do raster 'grelha'; o NoData passa a 'nodata'."""
    g = arcpy.Raster(grelha)
    canto = arcpy.Point(g.extent.XMin, g.extent.YMin)
    return arcpy.RasterToNumPyArray(raster, canto, g.width, g.height, nodata_to_value=nodata)


def rasterizar(fc, campo, saida, grelha):
    """Rasteriza os poligonos com o valor do 'campo' na grelha exata do raster 'grelha'."""
    with arcpy.EnvManager(extent=grelha, snapRaster=grelha, cellSize=CELL_SIZE,
                          outputCoordinateSystem=grelha):
        arcpy.conversion.PolygonToRaster(fc, campo, saida, "CELL_CENTER", "NONE", CELL_SIZE)
    return saida


def encontrar_dlpk(caminho):
    """Ficheiro do modelo a usar: o .emd (preferido) ou o .dlpk."""
    if caminho.lower().endswith((".emd", ".dlpk")):
        return caminho if os.path.exists(caminho) else None
    for padrao in ("*.emd", "*.dlpk"):
        f = sorted(glob.glob(os.path.join(caminho, padrao)))
        if f:
            return f[0]
    return None


def bandas_do_modelo(modelo):
    """N.o de bandas com que o modelo foi treinado (ExtractBands do .emd), ou None."""
    emd = modelo if modelo.lower().endswith(".emd") else \
        next(iter(sorted(glob.glob(os.path.join(os.path.dirname(modelo), "*.emd")))), None)
    if not emd:
        return None
    try:
        import json
        with open(emd, encoding="utf-8") as f:
            eb = json.load(f).get("ExtractBands")
        return len(eb) if eb else None
    except Exception:
        return None


# ------------------------------------------------------------------ GPU ou CPU
def memoria_gpu_livre_gb():
    """Memoria livre da GPU em GB (nvidia-smi), ou None se nao for possivel saber."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        valores = [float(v) for v in out.strip().splitlines() if v.strip()]
        return valores[GPU_ID] / 1024 if len(valores) > GPU_ID else None
    except Exception:
        return None


def escolher_processador():
    livre = memoria_gpu_livre_gb()
    if FORCAR_CPU:
        return "CPU", livre
    if livre is not None and livre >= MEM_GPU_MIN_GB:
        return "GPU", livre
    return "CPU", livre


# ------------------------------------------------------------------ preparacao das ortos
def espelhar(entrada, saida, falta):
    d = arcpy.Raster(entrada)
    sr, cs = d.spatialReference, d.meanCellWidth
    arr = arcpy.RasterToNumPyArray(entrada)
    pad = ((falta["cima"], falta["baixo"]), (falta["esquerda"], falta["direita"]))
    if arr.ndim == 3:
        pad = ((0, 0),) + pad
    arr = np.pad(arr, pad, mode="reflect")
    canto = arcpy.Point(d.extent.XMin - falta["esquerda"] * cs, d.extent.YMin - falta["baixo"] * cs)
    ras = arcpy.NumPyArrayToRaster(arr, canto, cs, cs)
    ras.save(saida)
    arcpy.management.DefineProjection(saida, sr)


def zona_da_orto(o):
    zona = os.path.join("memory", f"z_{nome_seguro(o)}")
    apagar(zona)
    arcpy.analysis.Select(ZONA_TESTE, zona, f"{CAMPO_NOME} = '{o}'")
    if contar(zona) != 1:
        raise RuntimeError(f"Nao encontrei exatamente 1 poligono para {o} em {ZONA_TESTE}")
    if MODO_TESTE:
        # quadrado no centro da orto, para medir o tempo
        e = arcpy.Describe(zona).extent
        cx, cy, h = (e.XMin + e.XMax) / 2, (e.YMin + e.YMax) / 2, LADO_TESTE_M / 2
        sr = arcpy.Describe(zona).spatialReference
        apagar(zona)
        arcpy.management.CreateFeatureclass("memory", f"z_{nome_seguro(o)}", "POLYGON",
                                            spatial_reference=sr)
        pts = [(cx - h, cy - h), (cx - h, cy + h), (cx + h, cy + h), (cx + h, cy - h)]
        with arcpy.da.InsertCursor(zona, ["SHAPE@"]) as ic:
            ic.insertRow([arcpy.Polygon(arcpy.Array([arcpy.Point(x, y) for x, y in pts]), sr)])
    return zona


def preparar_ortos():
    """Para cada orto de teste: orto com contexto, mascara da zona e referencia ao pixel."""
    sub = "ortos_teste_modo_teste" if MODO_TESTE else "ortos_teste"
    pasta = os.path.join(PASTA_AVALIACAO, sub)
    os.makedirs(pasta, exist_ok=True)
    mos = arcpy.Describe(ORTOMOSAICO).extent
    preparadas = {}
    for o in ORTOS_TESTE[:1] if MODO_TESTE else ORTOS_TESTE:
        so = nome_seguro(o)
        orto_tif = os.path.join(pasta, f"orto_{so}.tif")
        msk_tif = os.path.join(pasta, f"mascara_{so}.tif")
        ref_tif = os.path.join(pasta, f"ref_{so}.tif")
        zona = zona_da_orto(o)

        if not os.path.exists(orto_tif):
            log(f"   a recortar {o} com {CONTEXTO_M} m de contexto")
            ctx = os.path.join("memory", "ctx")
            apagar(ctx)
            arcpy.analysis.Buffer(zona, ctx, f"{CONTEXTO_M} Meters")
            ext = arcpy.Describe(ctx).extent
            xmin, ymin = max(ext.XMin, mos.XMin), max(ext.YMin, mos.YMin)
            xmax, ymax = min(ext.XMax, mos.XMax), min(ext.YMax, mos.YMax)
            ret = f"{xmin} {ymin} {xmax} {ymax}"
            falta = {
                "esquerda": int(round((mos.XMin - ext.XMin) / CELL_SIZE)) if ext.XMin < mos.XMin else 0,
                "direita":  int(round((ext.XMax - mos.XMax) / CELL_SIZE)) if ext.XMax > mos.XMax else 0,
                "baixo":    int(round((mos.YMin - ext.YMin) / CELL_SIZE)) if ext.YMin < mos.YMin else 0,
                "cima":     int(round((ext.YMax - mos.YMax) / CELL_SIZE)) if ext.YMax > mos.YMax else 0,
            }
            if ESPELHAR_BORDAS and any(falta.values()):
                base = os.path.join(pasta, f"base_{so}.tif")
                arcpy.management.Clip(ORTOMOSAICO, ret, base, "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
                espelhar(base, orto_tif, falta)
                arcpy.management.Delete(base)
                log("      bordas espelhadas: " + ", ".join(f"{k} {v} px" for k, v in falta.items() if v))
            else:
                arcpy.management.Clip(ORTOMOSAICO, ret, orto_tif, "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
            apagar(ctx)

        if not os.path.exists(msk_tif):
            tmp = os.path.join("memory", "msk_tmp")
            apagar(tmp)
            arcpy.management.CopyFeatures(zona, tmp)
            arcpy.management.AddField(tmp, "um", "SHORT")
            arcpy.management.CalculateField(tmp, "um", "1", "PYTHON3")
            rasterizar(tmp, "um", msk_tif, orto_tif)
            apagar(tmp)
        if not os.path.exists(ref_tif):
            rasterizar(REFERENCIA, CAMPO_CLASSE, ref_tif, orto_tif)

        msk = ler_grelha(msk_tif, orto_tif, 0) == 1
        ref = ler_grelha(ref_tif, orto_tif, SEM_REF).astype(np.int64)
        valido = msk & ~np.isin(ref, list(IGNORAR_REF | {SEM_REF}))
        n_ign = int(np.sum(msk & np.isin(ref, list(IGNORAR_REF))))
        n_sem = int(np.sum(msk & (ref == SEM_REF)))
        log(f"   {o}: {int(valido.sum()):,} pixeis avaliados | excluidos: {n_ign:,} Ignorar/residuais, "
            f"{n_sem:,} sem referencia")
        preparadas[o] = dict(orto=orto_tif, ref=ref, valido=valido)
    return preparadas


# ------------------------------------------------------------------ classificacao
def classificar(orto_tif, dlpk, out_tif, args):
    proc, livre = escolher_processador()
    txt_livre = f"{livre:.1f} GB livres" if livre is not None else "memoria livre desconhecida"
    motivo = "forcado pela opcao --cpu" if FORCAR_CPU else f"minimo para usar a GPU: {MEM_GPU_MIN_GB} GB"
    log(f"      processador: {proc} (GPU com {txt_livre}; {motivo})")
    for tentativa in range(3):
        try:
            env = dict(cellSize=CELL_SIZE, snapRaster=orto_tif, processorType=proc)
            if proc == "GPU":
                env["gpuId"] = GPU_ID
            with arcpy.EnvManager(**env):
                res = arcpy.ia.ClassifyPixelsUsingDeepLearning(orto_tif, dlpk, args)
                res.save(out_tif)
            return proc
        except Exception as e:
            if any(s in str(e).lower() for s in SINAIS_BLOQUEIO) and tentativa < 2:
                log("      ficheiro bloqueado; nova tentativa dentro de 2 min")
                time.sleep(120)
                continue
            raise


# ------------------------------------------------------------------ metricas
def matriz_confusao(ref, pred, valido):
    """Matriz 256 x 256 (linhas = referencia, colunas = previsao)."""
    r = ref[valido].astype(np.int64)
    p = np.clip(pred[valido].astype(np.int64), 0, 255)
    return np.bincount(r * 256 + p, minlength=256 * 256).reshape(256, 256).astype(np.int64)


def agrupar(M, mapa):
    if not mapa:
        return M, sorted(CLASSES)
    M2 = np.zeros_like(M)
    f = lambda c: mapa.get(c, c)
    for i, j in zip(*np.nonzero(M)):
        M2[f(i), f(j)] += M[i, j]
    return M2, sorted({f(c) for c in CLASSES})


def nome_classe(c, mapa):
    juntas = [CLASSES[k] for k in CLASSES if mapa.get(k, k) == c]
    return "+".join(juntas) if juntas else str(c)


def calcular_metricas(M, labels):
    n = int(M[labels, :].sum())
    por_classe = []
    for c in labels:
        tp = int(M[c, c])
        sup = int(M[c, :].sum())
        prev = int(M[labels, c].sum())
        fp, fn = prev - tp, sup - tp
        prec = tp / prev if prev else 0.0
        rec = tp / sup if sup else 0.0
        por_classe.append(dict(classe=c, suporte=sup, previstos=prev, TP=tp, FP=fp, FN=fn,
                               precision=prec, recall=rec,
                               F1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                               IoU=tp / (tp + fp + fn) if (tp + fp + fn) else 0.0))
    com_sup = [m for m in por_classe if m["suporte"] > 0]
    oa = sum(m["TP"] for m in por_classe) / n if n else 0.0
    pe = sum(m["suporte"] * m["previstos"] for m in por_classe) / (n * n) if n else 0.0
    fora = n - sum(m["previstos"] for m in por_classe)
    glob_ = dict(
        pixeis=n,
        exatidao_global=oa,
        exatidao_media=float(np.mean([m["recall"] for m in com_sup])) if com_sup else 0.0,
        kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
        mIoU=float(np.mean([m["IoU"] for m in com_sup])) if com_sup else 0.0,
        fwIoU=sum(m["suporte"] / n * m["IoU"] for m in com_sup) if n else 0.0,
        F1_macro=float(np.mean([m["F1"] for m in com_sup])) if com_sup else 0.0,
        F1_ponderado=sum(m["suporte"] / n * m["F1"] for m in com_sup) if n else 0.0,
        pct_previsoes_fora=100 * fora / n if n else 0.0,
    )
    return por_classe, glob_


def gravar_matriz(M, labels, mapa, caminho):
    outros = [c for c in range(256) if c not in labels and M[labels, c].sum() > 0]
    cab = ["referencia \\ previsao"] + [nome_classe(c, mapa) for c in labels] + \
          [f"outro_{c}" for c in outros] + ["total", "recall_%"]
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["CONTAGENS (pixeis)"])
        w.writerow(cab)
        for r in labels:
            linha = [int(M[r, c]) for c in labels] + [int(M[r, c]) for c in outros]
            tot = int(M[r, :].sum())
            w.writerow([nome_classe(r, mapa)] + linha + [tot, round(100 * M[r, r] / tot, 2) if tot else ""])
        w.writerow([])
        w.writerow(["PERCENTAGEM DE CADA LINHA (para onde vai cada classe da referencia)"])
        w.writerow(cab[:-2])
        for r in labels:
            tot = M[r, :].sum()
            w.writerow([nome_classe(r, mapa)] +
                       [round(100 * M[r, c] / tot, 2) if tot else "" for c in labels + outros])


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(PASTA_AVALIACAO, exist_ok=True)
    os.makedirs(os.path.join(PASTA_AVALIACAO, "matrizes"), exist_ok=True)
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.CheckOutExtension("Spatial")

    log("=" * 60)
    log(f"Avaliacao iniciada | modo_teste={MODO_TESTE} | forcar_cpu={FORCAR_CPU} | "
        f"modelos={[m[0] for m in MODELOS]}")
    for nome, c in (("ORTOMOSAICO", ORTOMOSAICO), ("REFERENCIA", REFERENCIA), ("ZONA_TESTE", ZONA_TESTE)):
        if not arcpy.Exists(c):
            raise FileNotFoundError(f"{nome} nao existe: {c}")
    livre = memoria_gpu_livre_gb()
    log(f"GPU: {f'{livre:.1f} GB livres' if livre is not None else 'nao foi possivel consultar'} "
        f"-> {escolher_processador()[0]} (verificado antes de cada orto)")

    preparadas = preparar_ortos()
    l_cls, l_glb, comparativa = [], [], []

    for rotulo, caminho, mapa_pred, args in (MODELOS[:1] if MODO_TESTE else MODELOS):
        dlpk = encontrar_dlpk(caminho)
        if not dlpk:
            log(f"ERRO {rotulo}: nenhum .emd ou .dlpk em {caminho}")
            continue
        log(f"MODELO {rotulo} | {dlpk} | mapa das previsoes: {mapa_pred or 'nenhum'} | {args}")
        nb_modelo = bandas_do_modelo(dlpk)
        nb_orto = arcpy.Raster(next(iter(preparadas.values()))["orto"]).bandCount
        log(f"   bandas: modelo {nb_modelo if nb_modelo else '?'} | orto de teste {nb_orto}")
        if nb_modelo and nb_modelo != nb_orto:
            log(f"ERRO {rotulo}: o modelo foi treinado com {nb_modelo} bandas e a orto tem {nb_orto} "
                "(ORTOMOSAICO errado?) - modelo ignorado")
            continue
        pasta_cl = os.path.join(PASTA_AVALIACAO,
                                "classificacoes_modo_teste" if MODO_TESTE else "classificacoes", rotulo)
        if rotulo in REFAZER and os.path.isdir(pasta_cl):
            for f in glob.glob(os.path.join(pasta_cl, "class_*")):
                apagar(f) if f.lower().endswith(".tif") else None
            log(f"   classificacoes anteriores de {rotulo} apagadas (REFAZER)")
        os.makedirs(pasta_cl, exist_ok=True)
        M_total = np.zeros((256, 256), np.int64)

        for o, P in preparadas.items():
            so = nome_seguro(o)
            try:
                cl_tif = os.path.join(pasta_cl, f"class_{so}.tif")
                if not os.path.exists(cl_tif):
                    t0 = time.time()
                    proc = classificar(P["orto"], dlpk, cl_tif, args)
                    mins = (time.time() - t0) / 60
                    log(f"   {o}: classificacao em {mins:.1f} min ({proc})")
                    if MODO_TESTE:
                        # area realmente classificada (inclui a margem de contexto)
                        r = arcpy.Raster(P["orto"])
                        km2_teste = r.width * r.height * r.meanCellWidth * r.meanCellHeight / 1e6
                        # area a classificar nas ortos de teste completas, com o contexto
                        km2_total = 0.0
                        with arcpy.da.SearchCursor(ZONA_TESTE, [CAMPO_NOME, "SHAPE@"]) as sc:
                            for nome_o, g in sc:
                                if nome_o in ORTOS_TESTE:
                                    e = g.extent
                                    km2_total += ((e.XMax - e.XMin + 2 * CONTEXTO_M) *
                                                  (e.YMax - e.YMin + 2 * CONTEXTO_M)) / 1e6
                        log(f"   MODO TESTE: {km2_teste:.2f} km2 classificados (com contexto) em "
                            f"{mins:.1f} min -> {km2_total:.1f} km2 nas {len(ORTOS_TESTE)} ortos: "
                            f"~{mins / km2_teste * km2_total / 60:.1f} h no mesmo processador")
                else:
                    log(f"   {o}: classificacao ja existente, reutilizada")

                # NoData da classificacao -> 0 (o SAMLoRA v1 preve o sem representacao como 0)
                pred = ler_grelha(cl_tif, P["orto"], 0).astype(np.int64)
                if mapa_pred:
                    pred_m = pred.copy()
                    for de, para in mapa_pred.items():
                        pred_m[pred == de] = para
                    pred = pred_m
                M = matriz_confusao(P["ref"], pred, P["valido"])
                M_total += M

                for agr, mapa in AGRUPAMENTOS.items():
                    Ma, labels = agrupar(M, mapa)
                    pc, gl = calcular_metricas(Ma, labels)
                    for m in pc:
                        l_cls.append([rotulo, agr, o, m["classe"], nome_classe(m["classe"], mapa),
                                      m["suporte"], m["previstos"], m["TP"], m["FP"], m["FN"]] +
                                     [round(m[k], 4) for k in ("precision", "recall", "F1", "IoU")])
                    l_glb.append([rotulo, agr, o] + [round(v, 4) if isinstance(v, float) else v
                                                     for v in gl.values()])
                    gravar_matriz(Ma, labels, mapa, os.path.join(
                        PASTA_AVALIACAO, "matrizes", f"{rotulo}_{agr}_{so}.csv"))
                    log(f"   {o} | {agr}: OA={gl['exatidao_global']:.3f} kappa={gl['kappa']:.3f} "
                        f"mIoU={gl['mIoU']:.3f} F1macro={gl['F1_macro']:.3f}")
            except Exception as e:
                log(f"ERRO {rotulo} | {o}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")

        if M_total.sum() == 0:
            continue
        for agr, mapa in AGRUPAMENTOS.items():
            Ma, labels = agrupar(M_total, mapa)
            pc, gl = calcular_metricas(Ma, labels)
            for m in pc:
                l_cls.append([rotulo, agr, "TOTAL", m["classe"], nome_classe(m["classe"], mapa),
                              m["suporte"], m["previstos"], m["TP"], m["FP"], m["FN"]] +
                             [round(m[k], 4) for k in ("precision", "recall", "F1", "IoU")])
            l_glb.append([rotulo, agr, "TOTAL"] + [round(v, 4) if isinstance(v, float) else v
                                                   for v in gl.values()])
            gravar_matriz(Ma, labels, mapa, os.path.join(PASTA_AVALIACAO, "matrizes",
                                                         f"{rotulo}_{agr}_TOTAL.csv"))
            comparativa.append((rotulo, agr, labels, mapa, pc, gl))
            log(f"   TOTAL {rotulo} | {agr}: OA={gl['exatidao_global']:.3f} kappa={gl['kappa']:.3f} "
                f"mIoU={gl['mIoU']:.3f} fwIoU={gl['fwIoU']:.3f} F1macro={gl['F1_macro']:.3f}")

    # --- CSV
    sufixo = "_modo_teste" if MODO_TESTE else ""

    def gravar(nome, cab, linhas):
        with open(os.path.join(PASTA_AVALIACAO, nome.replace(".csv", f"{sufixo}.csv")), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas_classe.csv",
           ["modelo", "agrupamento", "orto", "classe", "nome", "suporte_px", "previstos_px", "TP", "FP",
            "FN", "precision", "recall", "F1", "IoU"], l_cls)
    gravar("metricas_globais.csv",
           ["modelo", "agrupamento", "orto", "pixeis", "exatidao_global", "exatidao_media", "kappa",
            "mIoU", "fwIoU", "F1_macro", "F1_ponderado", "pct_previsoes_fora"], l_glb)

    # --- tabela comparativa (total das ortos)
    linhas = ["AVALIACAO DOS MODELOS DE OCUPACAO DO SOLO - total das ortos de teste",
              f"Data: {datetime.now():%Y-%m-%d %H:%M} | referencia: {REFERENCIA}"]
    for agr, mapa in AGRUPAMENTOS.items():
        bloco = sorted((c for c in comparativa if c[1] == agr), key=lambda x: x[5]["mIoU"], reverse=True)
        if not bloco:
            continue
        labels = bloco[0][2]
        cab = (f"{'modelo':<22}{'OA':>7}{'kappa':>7}{'mIoU':>7}{'fwIoU':>7}{'F1mac':>7} | " +
               "".join(f"{nome_classe(c, mapa)[:9]:>10}" for c in labels))
        linhas += ["", f"AGRUPAMENTO: {agr}  (F1 por classe a direita)", cab, "-" * len(cab)]
        for rotulo, _, lb, _, pc, gl in bloco:
            f1 = {m["classe"]: m["F1"] for m in pc}
            linhas.append(f"{rotulo[:21]:<22}{gl['exatidao_global']:>7.3f}{gl['kappa']:>7.3f}"
                          f"{gl['mIoU']:>7.3f}{gl['fwIoU']:>7.3f}{gl['F1_macro']:>7.3f} | " +
                          "".join(f"{f1.get(c, 0):>10.3f}" for c in labels))
    linhas += ["", "OA: exatidao global. kappa: coeficiente kappa. mIoU: IoU medio das classes. "
               "fwIoU: IoU ponderado pela area de cada classe. F1mac: F1 medio das classes. "
               f"Excluidos da avaliacao: pixeis da referencia com {sorted(IGNORAR_REF)} e fora da zona de teste."]
    with open(os.path.join(PASTA_AVALIACAO, f"tabela_comparativa{sufixo}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    log("\n".join(linhas))
    log("Avaliacao terminada")


if __name__ == "__main__":
    main()