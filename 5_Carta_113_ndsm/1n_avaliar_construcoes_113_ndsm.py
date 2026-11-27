r"""
1n_avaliar_construcoes_113_ndsm.py  (ArcGIS Pro 3.6)

COPIA do 1_avaliar_construcoes_113.py so com a CONFIGURACAO alterada para os modelos
com nDSM (fase B, 5 bandas: R, G, B, NIR, nDSM): ortomosaico de 5 bandas da 113,
modelos de D:\DL\Modelos\126_construcoes_ndsm e pasta de avaliacao propria. O resto
(metricas, pos-processamento, blocos) e igual, para comparar com a fase A na 113.

Avaliacao dos modelos de edificios ja treinados numa AREA NOVA (outro ortomosaico),
para medir a capacidade de generalizacao. Usa as mesmas metricas e o mesmo
pos-processamento do 09_avaliar_modelos.py, para os resultados serem comparaveis.

Antes de comecar:
  1. PROCESSADOR: verifica a memoria livre da GPU (nvidia-smi). Se tiver menos de
     VRAM_MIN_GB livres, usa o CPU. A verificacao e repetida antes de cada
     classificacao: se a GPU ficar livre a meio, passa a usa-la.
     ATENCAO: no CPU a classificacao e muito mais lenta (cerca de 70 vezes).
  2. ORTOMOSAICO: o modelo nao depende do sistema de coordenadas, mas sim do tamanho
     do pixel no terreno (0,5 m, como no treino). O ortomosaico fica no seu sistema,
     sem reamostragem, se estiver num sistema PROJETADO, em METROS e com pixel de
     0,5 m. So e alterado (e o script informa) se:
       - estiver num sistema geografico ou noutra unidade: e projetado para
         SR_SE_PRECISO, com pixel de 0,5 m (bilinear);
       - tiver outro tamanho de pixel: e reamostrado para 0,5 m (bilinear).
  3. VETORES: se a shapefile dos edificios (e a zona de avaliacao, se existir) nao
     estiver no sistema do ortomosaico, e projetada para ele, e o script informa.
  4. BANDAS: so avalia os modelos treinados com o mesmo numero de bandas do
     ortomosaico (sem nDSM: os modelos da fase A, com 4 bandas).

Folha 113 (decisao a partir do 0_verificar_dominio_126_113.py):
  - ORTOS ORIGINAIS, SEM CALIBRACAO (como na 235; confirmar com o 0_verificar_dominio_126_113.py).
  - METRICAS POR BLOCO: alem do total, as metricas sao calculadas em cada bloco da
    mesma grelha GRELHA_BLOCOS x GRELHA_BLOCOS do 0_verificar (mesmos nomes, ex.
    113_bloco_L4C4), e juntadas a radiometria de cada bloco (media e EMD vs 126),
    para ver se os blocos mais escuros (fora da variacao da 126) tem piores resultados.
    A classificacao e feita uma so vez no ortomosaico inteiro.
  - LIMPEZA DA REFERENCIA: com LIMPAR_REFERENCIA = True, a a_constr da 113 e limpa
    aqui com a mesma regra do 03_Limpar_Constr_Projt.py (palavras em PALAVRAS_EXCLUIR
    no campo source). Os poligonos fora ou no limite do ortomosaico ficam fora da
    avaliacao pela selecao da area (COMPLETELY_WITHIN), como no 03. O log mostra as
    classes mantidas e excluidas.

Area avaliada:
  - ZONA_AVALIACAO = None: todo o ortomosaico (a area com dados), mais os blocos;
  - ou uma shapefile com um ou mais poligonos (campo CAMPO_NOME), avaliados em
    separado e no total; cada um e classificado com CONTEXTO_M de margem.
Os edificios inteiramente dentro da area sao a referencia por objeto; os que
atravessam o limite (margens) nao contam como erros.

A referencia e limpa como a da area de treino (03_Limpar_Constr_Projt.py).

Saidas em PASTA_AVALIACAO: metricas_pixel.csv, metricas_objeto.csv,
metricas_por_area.csv, metricas_por_bloco.csv, tabela_comparativa.txt, log_avaliacao.txt,
classificacoes\<modelo>\, poligonos.gdb e, se for preciso, os dados projetados.

Correr no Python Command Prompt do ArcGIS Pro (com o ArcGIS Pro FECHADO):
    python "D:\Users\Cap_Almeida\Projeto\Python\5_Carta_113\1_avaliar_construcoes_113.py"
"""
import os
import csv
import json
import glob
import time
import subprocess
import traceback
import unicodedata

print("A carregar o arcpy...", flush=True)
import arcpy
from datetime import datetime   # DEPOIS do arcpy: o arcpy volta a ligar o nome 'datetime' ao modulo
import numpy as np

# ============================================================
# CONFIGURACAO
# ============================================================
# --- area nova
ORTOMOSAICO_NOVO = r"D:\Users\Cap_Almeida\Projeto\113\Ortomosaico\113_ndsm\113_nDSM_5bandas.tif"   # orto original + nDSM (5 bandas)
REFERENCIA_NOVA  = r"D:\Users\Cap_Almeida\Projeto\M888_SHP\113_2018_5\a_constr.shp"
ZONA_AVALIACAO   = None        # None = todo o ortomosaico (+ blocos); ou shapefile com poligonos
CAMPO_NOME       = "NOME"      # campo com o nome de cada poligono da zona (se existir)
NOME_REFERENCIA  = "edicao"

# --- limpeza da referencia: a MESMA regra do 03_Limpar_Constr_Projt.py
#     (elimina os poligonos cujo "source" contenha uma das palavras, sem distinguir
#     maiusculas, minusculas e acentos; as ruinas sao mantidas)
#     False = a REFERENCIA_NOVA ja esta limpa
LIMPAR_REFERENCIA = True
CAMPO_CLASSE = "source"
PALAVRAS_EXCLUIR = ["campo", "limite", "piscina", "electrica"]

# --- metricas por bloco (so com ZONA_AVALIACAO = None)
#     GRELHA_BLOCOS e BLOCO_LEITURA iguais ao GRELHA e BLOCO do 0_verificar_dominio
GRELHA_BLOCOS = 4              # 0 = nao calcular por bloco
BLOCO_LEITURA = 4096
# pasta de saida do 0_verificar_dominio (None = sem radiometria na tabela dos blocos)
PASTA_VERIFICACAO = r"D:\Users\Cap_Almeida\Projeto\113\Verificacoes_dominio"

# --- sistema para onde projetar o ortomosaico, SO se ele nao estiver num sistema
#     projetado em metros (dataset de onde copiar o sistema; por exemplo o de treino)
SR_SE_PRECISO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"
TOLERANCIA_PIXEL = 0.01        # diferenca relativa tolerada no tamanho do pixel (1 %)

# --- modelos (fase B, 5 bandas com nDSM)
PASTA_MODELOS = r"D:\DL\Modelos\126_construcoes_ndsm"
# Inicio do nome das pastas dos modelos a avaliar. Lista vazia = todos os da pasta.
MODELOS = ["unet_resnet50", "unet_resnet152", "deeplab_resnet101", "samlora_vit_l"]

PASTA_AVALIACAO = r"D:\DL\Avaliacao\113_construcoes_ndsm"

# --- processador
VRAM_MIN_GB = 12               # GPU com menos memoria livre do que isto -> CPU
GPU_ID = 0

# --- classificacao (iguais ao 09_avaliar_modelos.py)
CELL_SIZE = 0.5
VALOR_EDIFICIO = 1
ARGS_INFERENCIA = "padding 64;batch_size 4;predict_background True;test_time_augmentation False"
CONTEXTO_M = 256               # margem a volta de cada poligono da zona (se houver zona)

# --- limpeza e regularizacao (m2)
AREA_MIN = 10
FURO_MAX = 20
METODO_REG = "RIGHT_ANGLES_AND_DIAGONALS"
TOLERANCIA, DENSIFICACAO, PRECISAO, PENALIZACAO_DIAGONAL = 1.0, 1.0, 0.25, 1.5

# --- avaliacao por objeto
LIMIAR_IOU = 0.5
CLASSES_AREA = [0, 25, 50, 100, 200, float("inf")]
PASSO_CONTORNO_M = 1.0

# Modo teste RAPIDO: True = so o primeiro modelo, numa janela de TESTE_LADO_M x TESTE_LADO_M
# (centro do ortomosaico, ou TESTE_CENTRO = (x, y) no sistema do ortomosaico), com
# CONTEXTO_M de margem. Corre o pipeline todo (classificacao, poligonos, metricas, bloco)
# em poucos minutos, so para confirmar que funciona. Os resultados nao contam.
MODO_TESTE = False
TESTE_LADO_M = 1500
TESTE_CENTRO = None
# ============================================================

FICHEIRO_LOG = os.path.join(PASTA_AVALIACAO, "log_avaliacao.txt")
GDB = os.path.join(PASTA_AVALIACAO, "poligonos.gdb")
SINAIS_BLOQUEIO = ("winerror 32", "being used by another process")
# minutos de classificacao por km2 (medidos na folha 126), para a estimativa de tempo
MIN_KM2_GPU = {"unet": 0.65, "deeplab": 0.30, "samlora": 0.55}
MIN_KM2_CPU = 45.0
MIN_KM2_METRICAS = 0.15


# ------------------------------------------------------------------ utilitarios
def log(msg):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(FICHEIRO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def normalizar(texto):
    """Minusculas, sem acentos nem espacos nas extremidades (como no 03)."""
    texto = unicodedata.normalize("NFKD", str(texto).strip().lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def nome_seguro(txt):
    return str(txt).replace("-", "_").replace(" ", "_").replace(".", "_")


def apagar(*itens):
    for it in itens:
        if it and arcpy.Exists(it):
            arcpy.management.Delete(it)


def contar(fc):
    return int(arcpy.management.GetCount(fc)[0])


def ler_grelha(raster, grelha):
    g = arcpy.Raster(grelha)
    return arcpy.RasterToNumPyArray(raster, arcpy.Point(g.extent.XMin, g.extent.YMin),
                                    g.width, g.height, nodata_to_value=0)


def rasterizar(fc, saida, grelha):
    tmp = os.path.join("memory", "rast_tmp")
    apagar(tmp)
    arcpy.management.CopyFeatures(fc, tmp)
    arcpy.management.AddField(tmp, "um", "SHORT")
    arcpy.management.CalculateField(tmp, "um", "1", "PYTHON3")
    with arcpy.EnvManager(extent=grelha, snapRaster=grelha, cellSize=CELL_SIZE):
        arcpy.conversion.PolygonToRaster(tmp, "um", saida, "CELL_CENTER", "NONE", CELL_SIZE)
    apagar(tmp)
    return saida


def bandas_do_modelo(dlpk):
    emds = glob.glob(os.path.join(os.path.dirname(dlpk), "*.emd"))
    if not emds:
        return None
    try:
        with open(emds[0], encoding="utf-8") as f:
            eb = json.load(f).get("ExtractBands")
        return len(eb) if eb else None
    except Exception:
        return None


# ------------------------------------------------------------------ processador
def memoria_gpu_livre_gb():
    """Memoria livre da GPU em GB (nvidia-smi), ou None se nao for possivel saber."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        valores = [float(v) for v in r.stdout.split()]
        return valores[GPU_ID] / 1024 if len(valores) > GPU_ID else None
    except Exception:
        return None


ESTADO = {"proc": None}


def escolher_processador():
    """GPU se tiver pelo menos VRAM_MIN_GB livres; caso contrario CPU. Regista as mudancas."""
    livre = memoria_gpu_livre_gb()
    if livre is not None and livre >= VRAM_MIN_GB:
        proc, motivo = "GPU", f"{livre:.1f} GB livres"
        arcpy.env.processorType = "GPU"
        arcpy.env.gpuId = GPU_ID
    else:
        proc = "CPU"
        motivo = "GPU nao encontrada (nvidia-smi)" if livre is None else \
                 f"so {livre:.1f} GB livres na GPU (minimo {VRAM_MIN_GB} GB)"
        arcpy.env.processorType = "CPU"
    if proc != ESTADO["proc"]:
        log(f"PROCESSADOR: {proc} ({motivo})" +
            ("  ATENCAO: no CPU a classificacao e muito mais lenta." if proc == "CPU" else ""))
        ESTADO["proc"] = proc
    return proc


# ------------------------------------------------------------------ sistemas de referencia
def mesmo_sr(a, b):
    """True se os dois sistemas forem o mesmo, mesmo com nomes diferentes."""
    if a.name == b.name:
        return True
    if a.factoryCode and a.factoryCode == b.factoryCode:
        return True
    try:
        if a.type != b.type or a.GCS.datumName != b.GCS.datumName:
            return False
        if a.type == "Projected":
            if a.projectionName != b.projectionName or a.linearUnitName != b.linearUnitName:
                return False
            for p in ("centralMeridian", "latitudeOfOrigin", "scaleFactor", "falseEasting", "falseNorthing"):
                if abs(getattr(a, p) - getattr(b, p)) > 1e-6:
                    return False
        return True
    except Exception:
        return False


def transformacao(sr_de, sr_para, extent):
    """Transformacao geografica entre datums diferentes ('' se o datum for o mesmo)."""
    try:
        if sr_de.GCS.datumName == sr_para.GCS.datumName:
            return ""
    except Exception:
        pass
    lst = arcpy.ListTransformations(sr_de, sr_para, extent)
    if lst:
        log(f"      transformacoes possiveis: {lst[:5]}; usada: {lst[0]}")
        return lst[0]
    log("      AVISO: nenhuma transformacao encontrada entre os datums; projecao sem transformacao")
    return ""


def em_metros(sr):
    try:
        return sr.type == "Projected" and abs(sr.metersPerUnit - 1.0) < 1e-9
    except Exception:
        return False


def preparar_ortomosaico():
    """Ortomosaico pronto a classificar: sistema projetado em metros e pixel de 0,5 m.
    So e alterado se for preciso; caso contrario e usado tal como esta."""
    r = arcpy.Raster(ORTOMOSAICO_NOVO)
    sr = r.spatialReference
    cx, cy = r.meanCellWidth, r.meanCellHeight
    log(f"Ortomosaico novo: {ORTOMOSAICO_NOVO}")
    log(f"   {r.bandCount} bandas | sistema: {sr.name} ({sr.type}) | pixel: {cx:.6g} x {cy:.6g} "
        f"{getattr(sr, 'linearUnitName', '') or getattr(sr, 'angularUnitName', '')}")

    if not em_metros(sr):
        destino = arcpy.Describe(SR_SE_PRECISO).spatialReference
        if not em_metros(destino):
            raise RuntimeError(f"O ortomosaico nao esta num sistema projetado em metros, e o sistema de "
                               f"SR_SE_PRECISO ({destino.name}) tambem nao. Indica outro em SR_SE_PRECISO.")
        saida = os.path.join(PASTA_AVALIACAO, "ortomosaico_projetado.tif")
        log(f"   AVISO: sistema nao projetado em metros. A projetar para {destino.name}, "
            f"pixel {CELL_SIZE} m (bilinear) -> {saida}")
        if not os.path.exists(saida):
            tr = transformacao(sr, destino, r.extent)
            t0 = time.time()
            arcpy.management.ProjectRaster(ORTOMOSAICO_NOVO, saida, destino, "BILINEAR",
                                           f"{CELL_SIZE} {CELL_SIZE}", tr)
            log(f"   ortomosaico projetado em {(time.time() - t0) / 60:.1f} min")
        else:
            log("   (ja existia uma versao projetada; e reutilizada)")
        return saida

    if abs(cx - CELL_SIZE) / CELL_SIZE > TOLERANCIA_PIXEL or abs(cy - CELL_SIZE) / CELL_SIZE > TOLERANCIA_PIXEL:
        saida = os.path.join(PASTA_AVALIACAO, "ortomosaico_reamostrado.tif")
        log(f"   AVISO: pixel de {cx:.4g} x {cy:.4g} m, diferente dos {CELL_SIZE} m do treino. "
            f"A reamostrar (bilinear) -> {saida}")
        if not os.path.exists(saida):
            t0 = time.time()
            arcpy.management.Resample(ORTOMOSAICO_NOVO, saida, f"{CELL_SIZE} {CELL_SIZE}", "BILINEAR")
            log(f"   ortomosaico reamostrado em {(time.time() - t0) / 60:.1f} min")
        else:
            log("   (ja existia uma versao reamostrada; e reutilizada)")
        return saida

    log(f"   OK: sistema projetado em metros e pixel de {CELL_SIZE} m; usado sem alteracoes")
    return ORTOMOSAICO_NOVO


def no_sistema_do_orto(fc, sr_orto, nome, saida):
    """Devolve fc no sistema do ortomosaico, projetando-a se for preciso."""
    sr = arcpy.Describe(fc).spatialReference
    if mesmo_sr(sr, sr_orto):
        log(f"   OK: {nome} no sistema do ortomosaico ({sr.name})")
        return fc
    log(f"   AVISO: {nome} esta em {sr.name}; a projetar para {sr_orto.name} -> {saida}")
    if not arcpy.Exists(saida):
        tr = transformacao(sr, sr_orto, arcpy.Describe(fc).extent)
        arcpy.management.Project(fc, saida, sr_orto, tr)
    return saida


# ------------------------------------------------------------------ zonas e referencia
def preparar_zonas(orto, sr_orto):
    """Lista de (nome, fc da zona). Sem zona: a area com dados do ortomosaico."""
    zonas = []
    if MODO_TESTE:
        e = arcpy.Describe(orto).extent
        cx, cy = TESTE_CENTRO or ((e.XMin + e.XMax) / 2, (e.YMin + e.YMax) / 2)
        h = TESTE_LADO_M / 2
        pol = arcpy.Polygon(arcpy.Array([arcpy.Point(cx - h, cy - h), arcpy.Point(cx - h, cy + h),
                                         arcpy.Point(cx + h, cy + h), arcpy.Point(cx + h, cy - h),
                                         arcpy.Point(cx - h, cy - h)]), sr_orto)
        fc = os.path.join(GDB, "zona_teste")
        arcpy.management.CopyFeatures([pol], fc)
        zonas.append(("teste", fc))
        log(f"MODO TESTE: janela de {TESTE_LADO_M:g} m centrada em ({cx:.0f}, {cy:.0f})")
    elif ZONA_AVALIACAO is None:
        fc = os.path.join(GDB, "zona_ortomosaico")
        if not arcpy.Exists(fc):
            try:
                tmp = os.path.join("memory", "dominio")
                apagar(tmp)
                arcpy.ddd.RasterDomain(orto, tmp, "POLYGON")
                arcpy.management.Dissolve(tmp, fc)
                apagar(tmp)
            except Exception:
                e = arcpy.Describe(orto).extent
                pol = arcpy.Polygon(arcpy.Array([e.lowerLeft, e.upperLeft, e.upperRight, e.lowerRight,
                                                 e.lowerLeft]), sr_orto)
                arcpy.management.CopyFeatures(pol, fc)
        zonas.append(("ortomosaico", fc))
    else:
        zf = no_sistema_do_orto(ZONA_AVALIACAO, sr_orto, "zona de avaliacao",
                                os.path.join(GDB, "zona_projetada"))
        campos = [f.name for f in arcpy.ListFields(zf)]
        campo = CAMPO_NOME if CAMPO_NOME in campos else arcpy.Describe(zf).OIDFieldName
        oid = arcpy.Describe(zf).OIDFieldName
        with arcpy.da.SearchCursor(zf, [oid, campo]) as sc:
            linhas = list(sc)
        for i, nome in linhas:
            fc = os.path.join("memory", f"zona_{i}")
            apagar(fc)
            arcpy.analysis.Select(zf, fc, f"{oid} = {i}")
            zonas.append((str(nome), fc))
    area = sum(sum(g.area for (g,) in arcpy.da.SearchCursor(fc, ["SHAPE@"])) for _, fc in zonas) / 1e6
    log(f"Area avaliada: {len(zonas)} poligono(s), {area:.1f} km2")
    return zonas, area


def preparar_referencia(ref, zonas):
    """Edificios inteiramente dentro da area (teste) e os que atravessam o limite (margens)."""
    uniao = os.path.join("memory", "zona_uniao")
    zm = os.path.join("memory", "zonas_m")
    apagar(uniao, zm)
    arcpy.management.Merge([fc for _, fc in zonas], zm)
    arcpy.management.Dissolve(zm, uniao)
    apagar(zm)
    teste, marg = os.path.join("memory", "ref_teste"), os.path.join("memory", "ref_marg")
    apagar(teste, marg)
    lyr = arcpy.management.MakeFeatureLayer(ref, "lyr_ref")[0]
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", uniao)
    arcpy.management.CopyFeatures(lyr, teste)
    arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", uniao)
    arcpy.management.SelectLayerByLocation(lyr, "COMPLETELY_WITHIN", uniao, selection_type="REMOVE_FROM_SELECTION")
    arcpy.management.CopyFeatures(lyr, marg)
    arcpy.management.Delete(lyr)
    log(f"Referencia: {contar(teste)} edificios dentro da area, {contar(marg)} nas margens")
    if contar(teste) == 0:
        raise RuntimeError("Nenhum edificio de referencia dentro da area: confirma a shapefile e o sistema.")
    return teste, marg


def preparar_zona(nome, zona, orto, ref_todos, ref_teste, ref_marg):
    pasta = os.path.join(PASTA_AVALIACAO, "zonas")
    os.makedirs(pasta, exist_ok=True)
    sz = nome_seguro(nome)
    if ZONA_AVALIACAO is None and not MODO_TESTE:
        orto_tif = orto                                   # o ortomosaico inteiro
    else:
        orto_tif = os.path.join(pasta, f"orto_{sz}.tif")
        if MODO_TESTE:
            apagar(orto_tif)
        if not os.path.exists(orto_tif):
            ctx = os.path.join("memory", "ctx")
            apagar(ctx)
            arcpy.analysis.Buffer(zona, ctx, f"{CONTEXTO_M} Meters")
            e, m = arcpy.Describe(ctx).extent, arcpy.Describe(orto).extent
            ret = f"{max(e.XMin, m.XMin)} {max(e.YMin, m.YMin)} {min(e.XMax, m.XMax)} {min(e.YMax, m.YMax)}"
            arcpy.management.Clip(orto, ret, orto_tif, "#", "#", "NONE", "NO_MAINTAIN_EXTENT")
            apagar(ctx)
    msk_tif = os.path.join(pasta, f"mascara_{sz}.tif")
    ref_tif = os.path.join(pasta, f"ref_{sz}.tif")
    if not os.path.exists(msk_tif):
        rasterizar(zona, msk_tif, orto_tif)
    if not os.path.exists(ref_tif):
        rasterizar(ref_todos, ref_tif, orto_tif)
    msk = ler_grelha(msk_tif, orto_tif) == 1
    ref = (ler_grelha(ref_tif, orto_tif) == 1) & msk
    ref_o, marg_o = os.path.join("memory", f"r_{sz}"), os.path.join("memory", f"g_{sz}")
    apagar(ref_o, marg_o)
    for origem, destino, rel in ((ref_teste, ref_o, "HAVE_THEIR_CENTER_IN"), (ref_marg, marg_o, "INTERSECT")):
        lyr = arcpy.management.MakeFeatureLayer(origem, "lyr_sel")[0]
        arcpy.management.SelectLayerByLocation(lyr, rel, zona)
        arcpy.management.CopyFeatures(lyr, destino)
        arcpy.management.Delete(lyr)
    log(f"   zona {nome}: {contar(ref_o)} edificios de referencia, {contar(marg_o)} nas margens")
    return dict(orto=orto_tif, zona=zona, msk=msk, ref=ref, ref_o=ref_o, marg_o=marg_o)


# ------------------------------------------------------------------ classificacao e poligonos
def classificar(orto_tif, dlpk, out_tif):
    for tentativa in range(3):
        escolher_processador()
        try:
            with arcpy.EnvManager(cellSize=CELL_SIZE, snapRaster=orto_tif):
                arcpy.ia.ClassifyPixelsUsingDeepLearning(orto_tif, dlpk, ARGS_INFERENCIA).save(out_tif)
            return
        except Exception as e:
            txt = str(e).lower()
            if any(s in txt for s in SINAIS_BLOQUEIO) and tentativa < 2:
                log("      ficheiro bloqueado; nova tentativa dentro de 2 min")
                time.sleep(120)
                continue
            if ("out of memory" in txt or "cuda" in txt) and ESTADO["proc"] == "GPU" and tentativa < 2:
                log("      falta de memoria na GPU; nova tentativa no CPU")
                arcpy.env.processorType = "CPU"
                ESTADO["proc"] = "CPU"
                with arcpy.EnvManager(cellSize=CELL_SIZE, snapRaster=orto_tif):
                    arcpy.ia.ClassifyPixelsUsingDeepLearning(orto_tif, dlpk, ARGS_INFERENCIA).save(out_tif)
                return
            raise


def criar_poligonos(class_tif, zona, nome_base):
    m = "memory"
    bruto, edif, recort, sem_fu = (os.path.join(m, x) for x in ("p_bruto", "p_edif", "p_rec", "p_sf"))
    limpo, reg = os.path.join(GDB, f"{nome_base}_limpo"), os.path.join(GDB, f"{nome_base}_reg")
    apagar(bruto, edif, recort, sem_fu, limpo, reg)
    arcpy.conversion.RasterToPolygon(class_tif, bruto, "NO_SIMPLIFY", "Value")
    arcpy.analysis.Select(bruto, edif, f"gridcode = {VALOR_EDIFICIO}")
    arcpy.analysis.Clip(edif, zona, recort)
    arcpy.management.EliminatePolygonPart(recort, sem_fu, "AREA", f"{FURO_MAX} SquareMeters", 0, "CONTAINED_ONLY")
    arcpy.management.CopyFeatures(sem_fu, limpo)
    with arcpy.da.UpdateCursor(limpo, ["SHAPE@AREA"]) as uc:
        for (a,) in uc:
            if a < AREA_MIN:
                uc.deleteRow()
    arcpy.ddd.RegularizeBuildingFootprint(limpo, reg, METODO_REG, TOLERANCIA, DENSIFICACAO,
                                          PRECISAO, PENALIZACAO_DIAGONAL)
    apagar(bruto, edif, recort, sem_fu)
    return limpo, reg


# ------------------------------------------------------------------ metricas (iguais ao 09)
def contagens(pred, ref, msk):
    tp = int(np.sum(pred & ref)); fp = int(np.sum(pred & ~ref & msk))
    fn = int(np.sum(~pred & ref)); tn = int(np.sum(~pred & ~ref & msk))
    return tp, fp, fn, tn


def metricas_pixel(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    ua = tp / (tp + fp) if (tp + fp) else 0.0
    pa = tp / (tp + fn) if (tp + fn) else 0.0
    ua_f = tn / (tn + fn) if (tn + fn) else 0.0
    pa_f = tn / (tn + fp) if (tn + fp) else 0.0
    oa = (tp + tn) / n if n else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0.0
    return dict(UA_edificio=ua, PA_edificio=pa, UA_fundo=ua_f, PA_fundo=pa_f,
                exatidao_global=oa, kappa=(oa - pe) / (1 - pe) if pe < 1 else 0.0,
                F1=2 * ua * pa / (ua + pa) if (ua + pa) else 0.0,
                IoU=tp / (tp + fp + fn) if (tp + fp + fn) else 0.0)


def dist_contornos(gp, gr):
    def media(a, b):
        la, lb = a.boundary(), b.boundary()
        n = max(int(la.length / PASSO_CONTORNO_M), 4)
        return sum(lb.distanceTo(la.positionAlongLine(i / n, True)) for i in range(n)) / n
    return (media(gp, gr) + media(gr, gp)) / 2


def avaliar_objetos(pred_fc, ref_fc, marg_fc):
    m = "memory"
    p, r, inter = (os.path.join(m, x) for x in ("e_pred", "e_ref", "e_inter"))
    apagar(p, r, inter)
    arcpy.management.CopyFeatures(pred_fc, p)
    arcpy.management.CopyFeatures(ref_fc, r)
    for fc, campo in ((p, "PID"), (r, "RID")):
        arcpy.management.AddField(fc, campo, "LONG")
        arcpy.management.CalculateField(fc, campo, f"!{arcpy.Describe(fc).OIDFieldName}!", "PYTHON3")
    preds = {i: g for i, g in arcpy.da.SearchCursor(p, ["PID", "SHAPE@"]) if g is not None}
    refs = {i: g for i, g in arcpy.da.SearchCursor(r, ["RID", "SHAPE@"]) if g is not None}
    pares = {}
    if preds and refs:
        arcpy.analysis.Intersect([p, r], inter, "NO_FID")
        for pid, rid, a in arcpy.da.SearchCursor(inter, ["PID", "RID", "SHAPE@AREA"]):
            pares[(pid, rid)] = pares.get((pid, rid), 0.0) + a
    cand = sorted(((ai / (preds[pid].area + refs[rid].area - ai), pid, rid)
                   for (pid, rid), ai in pares.items()), reverse=True)
    up, ur, ok = set(), set(), []
    for iou, pid, rid in cand:
        if iou >= LIMIAR_IOU and pid not in up and rid not in ur:
            up.add(pid); ur.add(rid); ok.append((iou, pid, rid))
    excluidas = set()
    if marg_fc and arcpy.Exists(marg_fc) and contar(marg_fc):
        lyr = arcpy.management.MakeFeatureLayer(p, "lyr_pm")[0]
        arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", marg_fc)
        excluidas = {pid for (pid,) in arcpy.da.SearchCursor(lyr, ["PID"])} - up
        arcpy.management.Delete(lyr)
    cob = {}
    for (pid, rid), ai in pares.items():
        cob[rid] = cob.get(rid, 0.0) + ai
    coberto = {rid for rid, g in refs.items() if cob.get(rid, 0.0) >= 0.5 * g.area}
    por_pred = {}
    for (pid, rid), ai in pares.items():
        if ai >= 0.5 * refs[rid].area:
            por_pred[pid] = por_pred.get(pid, 0) + 1
    ious, erros, desl, dcont = [], [], [], []
    for iou, pid, rid in ok:
        gp, gr = preds[pid], refs[rid]
        cp, cr = gp.trueCentroid, gr.trueCentroid
        ious.append(iou)
        erros.append((gp.area - gr.area) / gr.area * 100)
        desl.append(((cp.X - cr.X) ** 2 + (cp.Y - cr.Y) ** 2) ** 0.5)
        dcont.append(dist_contornos(gp, gr))
    detetados = {rid for _, _, rid in ok}
    por_classe = []
    for a0, a1 in zip(CLASSES_AREA[:-1], CLASSES_AREA[1:]):
        ids = [rid for rid, g in refs.items() if a0 <= g.area < a1]
        por_classe.append([a0, a1, len(ids), sum(1 for i in ids if i in detetados),
                           sum(1 for i in ids if i in coberto)])
    apagar(p, r, inter)
    return dict(n_ref=len(refs), n_pred=len(preds), tp=len(ok),
                fp=len(preds) - len(ok) - len(excluidas), fn=len(refs) - len(ok),
                excluidas=len(excluidas), cobertos=len(coberto),
                fusoes=sum(1 for v in por_pred.values() if v >= 2),
                ious=ious, erros=erros, desl=desl, dcont=dcont, por_classe=por_classe)


def juntar(lista):
    t = dict(n_ref=0, n_pred=0, tp=0, fp=0, fn=0, excluidas=0, cobertos=0, fusoes=0,
             ious=[], erros=[], desl=[], dcont=[], por_classe=None)
    for res in lista:
        for k in ("n_ref", "n_pred", "tp", "fp", "fn", "excluidas", "cobertos", "fusoes"):
            t[k] += res[k]
        for k in ("ious", "erros", "desl", "dcont"):
            t[k] += res[k]
        if t["por_classe"] is None:
            t["por_classe"] = [list(c) for c in res["por_classe"]]
        else:
            for c, n in zip(t["por_classe"], res["por_classe"]):
                c[2] += n[2]; c[3] += n[3]; c[4] += n[4]
    return t


def resumo_objetos(res):
    tp, fp, fn = res["tp"], res["fp"], res["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    med = lambda v: float(np.mean(v)) if v else 0.0
    return dict(precision=p, recall=r, F1=2 * p * r / (p + r) if (p + r) else 0.0,
                cobertura=res["cobertos"] / res["n_ref"] if res["n_ref"] else 0.0,
                IoU_medio=med(res["ious"]), erro_area_pct=med(res["erros"]),
                erro_area_abs_pct=med([abs(x) for x in res["erros"]]),
                desloc_centroide_m=med(res["desl"]),
                dist_contorno_media_m=med(res["dcont"]),
                dist_contorno_mediana_m=float(np.median(res["dcont"])) if res["dcont"] else 0.0)


# ------------------------------------------------------------------ limpeza da referencia
def limpar_referencia(ref):
    """Copia da referencia sem as classes a excluir (como no 03_Limpar_Constr_Projt.py)."""
    if not LIMPAR_REFERENCIA:
        return ref
    campos = [f.name for f in arcpy.ListFields(ref)]
    if CAMPO_CLASSE not in campos:
        raise RuntimeError(f"A referencia nao tem o campo {CAMPO_CLASSE}; desliga LIMPAR_REFERENCIA "
                           f"ou indica o campo certo.")
    contagem = {}
    for (c,) in arcpy.da.SearchCursor(ref, [CAMPO_CLASSE]):
        contagem[c] = contagem.get(c, 0) + 1
    palavras = [normalizar(p) for p in PALAVRAS_EXCLUIR]
    palavra = lambda c: next((p for p in palavras if p in normalizar(c)), None) if c is not None else None
    excl = lambda c: palavra(c) is not None
    log(f"Limpeza da referencia (regra do 03_Limpar_Constr_Projt.py: {PALAVRAS_EXCLUIR}):")
    for c in sorted(contagem, key=str):
        log(f"   {'EXCLUIDA (' + palavra(c) + ')' if excl(c) else 'mantida'}  {c}: {contagem[c]}")
    saida = os.path.join(GDB, "referencia_limpa")
    apagar(saida)
    lyr = arcpy.management.MakeFeatureLayer(ref, "lyr_limpa")[0]
    excluir = [c for c in contagem if excl(c)]
    if excluir:
        campo = arcpy.AddFieldDelimiters(ref, CAMPO_CLASSE)
        lista = ", ".join("'" + str(c).replace("'", "''") + "'" for c in excluir)
        arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", f"{campo} NOT IN ({lista})")
    arcpy.management.CopyFeatures(lyr, saida)
    arcpy.management.Delete(lyr)
    log(f"   referencia limpa: {contar(saida)} de {sum(contagem.values())} poligonos -> {saida}")
    return saida


# ------------------------------------------------------------------ blocos (grelha do 0_verificar)
def blocos_grelha(orto):
    """Blocos da grelha GRELHA_BLOCOS x GRELHA_BLOCOS, com a MESMA divisao do
    0_verificar_dominio_126_113.py (limites nos multiplos do bloco de leitura).
    Devolve lista de dict(nome, r0, r1, c0, c1, fc) - linhas/colunas do array (linha 0 = topo)."""
    r = arcpy.Raster(orto)
    W, H, G = r.width, r.height, GRELHA_BLOCOS
    cw, ch = r.meanCellWidth, r.meanCellHeight
    xmin, ymax = r.extent.XMin, r.extent.YMax
    bloco = max(256, min(BLOCO_LEITURA, W // G, H // G))
    cols, rows = {}, {}
    for x in range(0, W, bloco):
        nc = min(x + bloco, W) - x
        gc = min(G - 1, int((x + nc / 2.0) / W * G))
        a, b = cols.get(gc, (x, x + nc))
        cols[gc] = (min(a, x), max(b, x + nc))
    for y in range(0, H, bloco):           # y conta a partir de baixo (como no 0_verificar)
        nr = min(y + bloco, H) - y
        gl = min(G - 1, int((1 - (y + nr / 2.0) / H) * G))
        r0, r1 = H - (y + nr), H - y       # linhas do array, a partir do topo
        a, b = rows.get(gl, (r0, r1))
        rows[gl] = (min(a, r0), max(b, r1))
    prefixo = os.path.splitext(os.path.basename(ORTOMOSAICO_NOVO))[0]
    if os.path.normcase(orto) != os.path.normcase(ORTOMOSAICO_NOVO):
        log("   AVISO: o ortomosaico foi projetado/reamostrado; os blocos podem nao coincidir "
            "exatamente com os do 0_verificar_dominio")
    fc = os.path.join(GDB, "grelha_blocos")
    apagar(fc)
    arcpy.management.CreateFeatureclass(GDB, "grelha_blocos", "POLYGON",
                                        spatial_reference=r.spatialReference)
    arcpy.management.AddField(fc, "NOME", "TEXT", field_length=64)
    blocos = []
    with arcpy.da.InsertCursor(fc, ["NOME", "SHAPE@"]) as ic:
        for gl in sorted(rows):
            for gc in sorted(cols):
                (r0, r1), (c0, c1) = rows[gl], cols[gc]
                nome = f"{prefixo}_bloco_L{gl + 1}C{gc + 1}"
                x0, x1 = xmin + c0 * cw, xmin + c1 * cw
                y0, y1 = ymax - r1 * ch, ymax - r0 * ch
                pol = arcpy.Polygon(arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x0, y1),
                                                 arcpy.Point(x1, y1), arcpy.Point(x1, y0),
                                                 arcpy.Point(x0, y0)]), r.spatialReference)
                ic.insertRow([nome, pol])
                blocos.append(dict(nome=nome, r0=r0, r1=r1, c0=c0, c1=c1))
    for b in blocos:
        b["fc"] = os.path.join("memory", "blk_" + b["nome"].replace("-", "_"))
        apagar(b["fc"])
        arcpy.analysis.Select(fc, b["fc"], f"NOME = '{b['nome']}'")
    log(f"Blocos: {len(blocos)} ({G}x{G}, bloco de leitura {bloco} px) -> {fc}")
    return blocos


def ler_radiometria():
    """{bloco: dict(media_R.., emd_R.., fora_126)} a partir das saidas do 0_verificar."""
    if not PASTA_VERIFICACAO:
        return {}
    f_orto = os.path.join(PASTA_VERIFICACAO, "radiometria_por_orto.csv")
    f_dist = os.path.join(PASTA_VERIFICACAO, "radiometria_distancias.csv")
    if not (os.path.exists(f_orto) and os.path.exists(f_dist)):
        log(f"   AVISO: sem radiometria em {PASTA_VERIFICACAO}; tabela dos blocos sem radiometria")
        return {}
    num = lambda s: float(str(s).replace(",", "."))
    rad, max_int = {}, {}
    with open(f_orto, encoding="utf-8-sig") as f:
        for l in csv.DictReader(f, delimiter=";"):
            rad.setdefault(l["orto"], {})["media_" + l["banda"]] = num(l["media"])
    with open(f_dist, encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    for l in linhas:
        if l["tipo"].startswith("interna_") and l["b"].endswith("126"):
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


def metricas_blocos(modelo, blocos, P, pred_cl, reg, rad, linhas):
    """Metricas por pixel (classificacao) e por objeto (poligonos regularizados) em cada bloco.
    Edificios com centro no bloco contam; os que o atravessam com centro fora sao margens."""
    for b in blocos:
        sl = (slice(b["r0"], b["r1"]), slice(b["c0"], b["c1"]))
        c = contagens(pred_cl[sl], P["ref"][sl], P["msk"][sl])
        mp = metricas_pixel(*c)
        sub = {}
        for nome, origem, rel in (("p", reg, "HAVE_THEIR_CENTER_IN"), ("r", P["ref_o"], "HAVE_THEIR_CENTER_IN")):
            sub[nome] = os.path.join("memory", f"bk_{nome}")
            apagar(sub[nome])
            lyr = arcpy.management.MakeFeatureLayer(origem, "lyr_bk")[0]
            arcpy.management.SelectLayerByLocation(lyr, rel, b["fc"])
            arcpy.management.CopyFeatures(lyr, sub[nome])
            arcpy.management.Delete(lyr)
        # margens do bloco: margens da area + referencia que atravessa o bloco com centro fora
        mg1, mg2, marg = (os.path.join("memory", x) for x in ("bk_m1", "bk_m2", "bk_marg"))
        apagar(mg1, mg2, marg)
        lyr = arcpy.management.MakeFeatureLayer(P["ref_o"], "lyr_bk")[0]
        arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", b["fc"])
        arcpy.management.SelectLayerByLocation(lyr, "HAVE_THEIR_CENTER_IN", b["fc"],
                                               selection_type="REMOVE_FROM_SELECTION")
        arcpy.management.CopyFeatures(lyr, mg1)
        arcpy.management.Delete(lyr)
        lyr = arcpy.management.MakeFeatureLayer(P["marg_o"], "lyr_bk")[0]
        arcpy.management.SelectLayerByLocation(lyr, "INTERSECT", b["fc"])
        arcpy.management.CopyFeatures(lyr, mg2)
        arcpy.management.Delete(lyr)
        arcpy.management.Merge([mg1, mg2], marg)
        if contar(sub["r"]) or contar(sub["p"]):
            res = avaliar_objetos(sub["p"], sub["r"], marg)
            so_ = resumo_objetos(res)
        else:
            res, so_ = dict(n_ref=0, n_pred=0, tp=0, fp=0, fn=0), {}
        apagar(sub["p"], sub["r"], mg1, mg2, marg)
        rb = rad.get(b["nome"], {})
        linhas.append([NOME_REFERENCIA, modelo, b["nome"],
                       rb.get("media_R"), rb.get("media_G"), rb.get("media_B"),
                       rb.get("emd_R"), rb.get("emd_G"), rb.get("emd_B"), rb.get("emd_medio"),
                       rb.get("fora_126", ""),
                       *c, round(mp["F1"], 4), round(mp["IoU"], 4),
                       res["n_ref"], res["n_pred"], res["tp"], res["fp"], res["fn"],
                       round(so_.get("precision", 0.0), 4), round(so_.get("recall", 0.0), 4),
                       round(so_.get("F1", 0.0), 4), round(so_.get("desloc_centroide_m", 0.0), 3)])


def correlacao_spearman(x, y):
    """Correlacao de Spearman simples (sem empates corrigidos), ou None."""
    pares = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pares) < 4:
        return None
    xa, ya = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
    rx, ry = xa.argsort().argsort().astype(float), ya.argsort().argsort().astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ principal
def estimar_tempo(modelos, area_km2):
    proc = ESTADO["proc"]
    total = 0.0
    for mod in modelos:
        arq = next((k for k in MIN_KM2_GPU if mod.lower().startswith(k)), "unet")
        cl = (MIN_KM2_GPU[arq] if proc == "GPU" else MIN_KM2_CPU) * area_km2
        total += cl + MIN_KM2_METRICAS * area_km2
    log(f"Estimativa de tempo ({proc}, {len(modelos)} modelos, {area_km2:.1f} km2): "
        f"cerca de {total / 60:.1f} h (valores medidos na folha 126; so indicativo)")


def main():
    os.makedirs(PASTA_AVALIACAO, exist_ok=True)
    # Recomeca do zero tudo o que depende da referencia (limpeza, projecao, poligonos,
    # rasters de referencia e mascara). As classificacoes (classificacoes\) sao mantidas:
    # nao dependem da referencia e sao a parte mais demorada.
    if arcpy.Exists(GDB):
        try:
            arcpy.management.Delete(GDB)
        except Exception as e:
            raise RuntimeError(f"Nao foi possivel apagar {GDB} (ArcGIS Pro aberto com a gdb?): {e}")
    arcpy.management.CreateFileGDB(PASTA_AVALIACAO, "poligonos.gdb")
    for f in glob.glob(os.path.join(PASTA_AVALIACAO, "zonas", "ref_*.tif")) + \
             glob.glob(os.path.join(PASTA_AVALIACAO, "zonas", "mascara_*.tif")):
        arcpy.management.Delete(f)
    arcpy.env.overwriteOutput = True
    for ext in ("ImageAnalyst", "Spatial", "3D"):
        arcpy.CheckOutExtension(ext)

    log("=" * 60)
    log(f"Avaliacao numa area nova | modo_teste={MODO_TESTE}")
    for caminho in (ORTOMOSAICO_NOVO, REFERENCIA_NOVA, SR_SE_PRECISO):
        if not arcpy.Exists(caminho):
            raise FileNotFoundError(f"Nao existe: {caminho}")

    # 1. processador
    escolher_processador()

    # 2. ortomosaico: sistema projetado em metros e pixel de 0,5 m
    orto = preparar_ortomosaico()
    sr_orto = arcpy.Describe(orto).spatialReference
    n_bandas = arcpy.Describe(orto).bandCount

    # 3. referencia (e zona) no sistema do ortomosaico
    log("Sistemas de referencia dos vetores:")
    ref = no_sistema_do_orto(limpar_referencia(REFERENCIA_NOVA), sr_orto, "referencia",
                             os.path.join(GDB, "referencia_projetada"))
    zonas, area_km2 = preparar_zonas(orto, sr_orto)
    ref_teste, ref_marg = preparar_referencia(ref, zonas)

    # 4. modelos compativeis
    dlpks = sorted(glob.glob(os.path.join(PASTA_MODELOS, "**", "*.dlpk"), recursive=True))
    if MODELOS:
        dlpks = [f for f in dlpks if any(os.path.basename(os.path.dirname(f)).lower().startswith(p.lower())
                                         for p in MODELOS)]
    validos = []
    for f in dlpks:
        nb = bandas_do_modelo(f)
        mod = os.path.basename(os.path.dirname(f))
        if nb and nb != n_bandas:
            log(f"   IGNORADO {mod}: treinado com {nb} bandas, o ortomosaico tem {n_bandas}")
        else:
            validos.append(f)
    if MODO_TESTE:
        validos = validos[:1]
    log(f"{len(validos)} modelos a avaliar: {[os.path.basename(os.path.dirname(f)) for f in validos]}")
    if not validos:
        log("Nenhum modelo para avaliar.")
        return
    estimar_tempo([os.path.basename(os.path.dirname(f)) for f in validos], area_km2)

    preparadas = {nome: preparar_zona(nome, fc, orto, ref, ref_teste, ref_marg) for nome, fc in zonas}

    # blocos da grelha do 0_verificar (so para o ortomosaico inteiro)
    blocos, rad = [], {}
    if MODO_TESTE:
        # um "bloco" = a janela de teste, para correr tambem o codigo das metricas por bloco
        P = preparadas["teste"]
        h, w = P["msk"].shape
        blocos = [dict(nome="teste_janela", r0=0, r1=h, c0=0, c1=w, fc=P["zona"])]
    elif ZONA_AVALIACAO is None and GRELHA_BLOCOS:
        blocos = blocos_grelha(orto)
        rad = ler_radiometria()
        if rad:
            fora = sorted(n for n, d in rad.items() if d.get("fora_126") == "sim" and n.startswith(
                os.path.splitext(os.path.basename(ORTOMOSAICO_NOVO))[0]))
            log(f"   blocos fora da variacao radiometrica da 126: {fora}")

    l_pix, l_obj, l_area, l_bloco, comparativa = [], [], [], [], []
    for dlpk in validos:
        modelo = os.path.basename(os.path.dirname(dlpk))
        log(f"MODELO {modelo}")
        acc = dict(cl=np.zeros(4, np.int64), pg=np.zeros(4, np.int64), reg=[])
        for nome, P in preparadas.items():
            sz = nome_seguro(nome)
            try:
                pasta_cl = os.path.join(PASTA_AVALIACAO, "classificacoes", modelo)
                os.makedirs(pasta_cl, exist_ok=True)
                cl_tif = os.path.join(pasta_cl, f"class_{sz}.tif")
                if MODO_TESTE:
                    apagar(cl_tif)       # a janela pode ter mudado: refazer sempre
                t0 = time.time()
                if not os.path.exists(cl_tif):
                    classificar(P["orto"], dlpk, cl_tif)
                    log(f"   {nome}: classificacao em {(time.time() - t0) / 60:.1f} min ({ESTADO['proc']})")
                pred_cl = (ler_grelha(cl_tif, P["orto"]) == VALOR_EDIFICIO) & P["msk"]
                limpo, reg = criar_poligonos(cl_tif, P["zona"], nome_seguro(f"{modelo}_{nome}"))
                pg_tif = os.path.join(pasta_cl, f"poligonos_{sz}.tif")
                rasterizar(reg, pg_tif, P["orto"])
                pred_pg = (ler_grelha(pg_tif, P["orto"]) == 1) & P["msk"]
                for produto, pred, chave in (("classificacao", pred_cl, "cl"), ("poligonos_reg", pred_pg, "pg")):
                    c = contagens(pred, P["ref"], P["msk"])
                    acc[chave] += c
                    l_pix.append([NOME_REFERENCIA, modelo, nome, produto, *c] +
                                 [round(v, 4) for v in metricas_pixel(*c).values()])
                res = avaliar_objetos(reg, P["ref_o"], P["marg_o"])
                acc["reg"].append(res)
                so_ = resumo_objetos(res)
                l_obj.append([NOME_REFERENCIA, modelo, nome, res["n_ref"], res["n_pred"], res["tp"], res["fp"],
                              res["fn"], res["excluidas"], res["fusoes"]] + [round(v, 4) for v in so_.values()])
                for a0, a1, n, det, cb in res["por_classe"]:
                    l_area.append([NOME_REFERENCIA, modelo, nome, f"{a0:g}-{a1:g}", n, det,
                                   round(det / n, 4) if n else "", cb, round(cb / n, 4) if n else ""])
                log(f"   {nome}: pixel F1={metricas_pixel(*contagens(pred_cl, P['ref'], P['msk']))['F1']:.3f} | "
                    f"objeto F1={so_['F1']:.3f} | desl={so_['desloc_centroide_m']:.2f} m | "
                    f"concluida em {(time.time() - t0) / 60:.1f} min")
                if blocos:
                    t1 = time.time()
                    metricas_blocos(modelo, blocos, P, pred_cl, reg, rad, l_bloco)
                    log(f"   metricas por bloco em {(time.time() - t1) / 60:.1f} min")
            except Exception as e:
                log(f"ERRO {modelo} | {nome}: {e}\n{arcpy.GetMessages(2)}\n{traceback.format_exc()}")
        if not acc["reg"]:
            continue
        for produto, chave in (("classificacao", "cl"), ("poligonos_reg", "pg")):
            t = [int(x) for x in acc[chave]]
            l_pix.append([NOME_REFERENCIA, modelo, "TOTAL", produto, *t] +
                         [round(v, 4) for v in metricas_pixel(*t).values()])
        tot = juntar(acc["reg"])
        so_ = resumo_objetos(tot)
        l_obj.append([NOME_REFERENCIA, modelo, "TOTAL", tot["n_ref"], tot["n_pred"], tot["tp"], tot["fp"],
                      tot["fn"], tot["excluidas"], tot["fusoes"]] + [round(v, 4) for v in so_.values()])
        for a0, a1, n, det, cb in tot["por_classe"]:
            l_area.append([NOME_REFERENCIA, modelo, "TOTAL", f"{a0:g}-{a1:g}", n, det,
                           round(det / n, 4) if n else "", cb, round(cb / n, 4) if n else ""])
        mcl = metricas_pixel(*[int(x) for x in acc["cl"]])
        mpg = metricas_pixel(*[int(x) for x in acc["pg"]])
        comparativa.append((modelo, mcl, mpg, so_, tot))
        log(f"   TOTAL {modelo}: pixel F1={mcl['F1']:.3f} | objeto F1={so_['F1']:.3f} | "
            f"desl={so_['desloc_centroide_m']:.2f} m")

    def gravar(nome, cab, linhas):
        with open(os.path.join(PASTA_AVALIACAO, nome), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas_pixel.csv",
           ["referencia", "modelo", "zona", "produto", "TP", "FP", "FN", "TN", "UA_edificio", "PA_edificio",
            "UA_fundo", "PA_fundo", "exatidao_global", "kappa", "F1", "IoU"], l_pix)
    gravar("metricas_objeto.csv",
           ["referencia", "modelo", "zona", "n_ref", "n_pred", "detetados", "falsos", "omitidos",
            "previsoes_margens_excluidas", "fusoes", "precision", "recall", "F1", "cobertura50", "IoU_medio",
            "erro_area_pct", "erro_area_abs_pct", "desloc_centroide_m", "dist_contorno_media_m",
            "dist_contorno_mediana_m"], l_obj)
    gravar("metricas_por_area.csv",
           ["referencia", "modelo", "zona", "classe_area_m2", "n_ref", "detetados", "taxa_detecao",
            "cobertos50", "taxa_cobertura"], l_area)

    cab_bloco = ["referencia", "modelo", "bloco", "media_R", "media_G", "media_B",
                 "emd_R_vs_126", "emd_G_vs_126", "emd_B_vs_126", "emd_medio", "fora_variacao_126",
                 "TP", "FP", "FN", "TN", "F1_pixel", "IoU_pixel",
                 "n_ref", "n_pred", "detetados", "falsos", "omitidos",
                 "precision_obj", "recall_obj", "F1_obj", "desloc_centroide_m"]
    if l_bloco:
        gravar("metricas_por_bloco.csv", cab_bloco, l_bloco)

    cab = (f"{'modelo':<38}{'UA':>7}{'PA':>7}{'OA':>8}{'kappa':>7}{'F1px':>7}{'F1pg':>7} | "
           f"{'P':>6}{'R':>6}{'F1obj':>7}{'cob':>6}{'fus':>5}{'IoU':>6}{'erroA%':>8}{'desl':>6}{'dcont':>7}")
    linhas = ["AVALIACAO NUMA AREA NOVA - total da area",
              f"Data: {datetime.now():%Y-%m-%d %H:%M}",
              f"Ortomosaico: {orto}", f"Referencia: {ref}", "", cab, "-" * len(cab)]
    for modelo, mcl, mpg, so_, tot in sorted(comparativa, key=lambda x: x[3]["F1"], reverse=True):
        linhas.append(f"{modelo[:37]:<38}{mcl['UA_edificio']:>7.3f}{mcl['PA_edificio']:>7.3f}"
                      f"{mcl['exatidao_global']:>8.4f}{mcl['kappa']:>7.3f}{mcl['F1']:>7.3f}{mpg['F1']:>7.3f} | "
                      f"{so_['precision']:>6.3f}{so_['recall']:>6.3f}{so_['F1']:>7.3f}{so_['cobertura']:>6.3f}"
                      f"{tot['fusoes']:>5d}{so_['IoU_medio']:>6.3f}{so_['erro_area_pct']:>8.1f}"
                      f"{so_['desloc_centroide_m']:>6.2f}{so_['dist_contorno_media_m']:>7.2f}")
    linhas += ["", "Mesmas metricas do 09_avaliar_modelos.py (poligonos regularizados, IoU >= "
               f"{LIMIAR_IOU}).", "Ortomosaico original, sem calibracao radiometrica."]

    # radiometria vs desempenho por bloco
    if l_bloco and rad:
        i_emd, i_fora = cab_bloco.index("emd_medio"), cab_bloco.index("fora_variacao_126")
        i_f1p, i_f1o, i_nref = cab_bloco.index("F1_pixel"), cab_bloco.index("F1_obj"), cab_bloco.index("n_ref")
        linhas += ["", "RADIOMETRIA vs DESEMPENHO POR BLOCO (blocos com edificios de referencia)",
                   f"{'modelo':<38}{'rho_F1px':>9}{'rho_F1obj':>10}{'F1obj_dentro':>13}{'F1obj_fora':>11}"
                   f"{'n_dentro':>9}{'n_fora':>7}"]
        for modelo in [c[0] for c in comparativa]:
            ls = [l for l in l_bloco if l[1] == modelo and l[i_nref] > 0]
            emd = [l[i_emd] for l in ls]
            r_px = correlacao_spearman(emd, [l[i_f1p] for l in ls])
            r_ob = correlacao_spearman(emd, [l[i_f1o] for l in ls])
            dentro = [l[i_f1o] for l in ls if l[i_fora] == "nao"]
            fora = [l[i_f1o] for l in ls if l[i_fora] == "sim"]
            f = lambda v: f"{v:.3f}" if v is not None else "-"
            linhas.append(f"{modelo[:37]:<38}{f(r_px):>9}{f(r_ob):>10}"
                          f"{f(float(np.mean(dentro)) if dentro else None):>13}"
                          f"{f(float(np.mean(fora)) if fora else None):>11}{len(dentro):>9}{len(fora):>7}")
        linhas += ["rho = correlacao de Spearman entre o EMD medio do bloco (vs 126) e o F1;",
                   "negativa = blocos radiometricamente mais diferentes da 126 tem pior desempenho.",
                   "Com 16 blocos e indicativo; ver metricas_por_bloco.csv (e a paisagem de cada bloco)."]
    with open(os.path.join(PASTA_AVALIACAO, "tabela_comparativa.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    log("\n".join(linhas))
    log("Avaliacao terminada")


if __name__ == "__main__":
    main()