# -*- coding: utf-8 -*-
r"""
07_exportar_chips.py  (ArcGIS Pro 3.6)

Exporta os chips de treino (Export Training Data For Deep Learning) a partir do
ortomosaico e dos edifícios de TREINO na posição original da restituição,
limitados à máscara de treino (saídas do 06_separar_treino_teste.py).

Fase A: ortomosaico com 4 bandas (R, G, B, NIR).

Os parâmetros dos chips (256 px, passo de 128 px, 0,5 m) e as entradas são iguais
nas duas fases; só muda o ortomosaico.

Antes de exportar, verifica:
  - que as entradas existem;
  - que os edifícios, a máscara e o ortomosaico estão no mesmo sistema de coordenadas;
  - o campo de classe e os seus valores. Se não existir (ou se
    CRIAR_CLASSE_CONSTANTE = True), cria uma cópia temporária dos edifícios, em
    disco, com o campo "classe" = 1 (o ficheiro original não é alterado).

Depois de exportar:
  - conta os chips de imagem e de rótulo;
  - verifica que nenhum chip entra na zona de teste;
  - calcula a proporção de píxeis de cada classe (numa amostra de chips);
  - grava o relatório exportacao_relatorio.txt na pasta de saída.

Opcional: no fim, lança o script de treino (só no Python Command Prompt).

Pode correr na janela Python do ArcGIS Pro ou no Python Command Prompt:
    python "D:\Users\Cap_Almeida\Projeto\Python\07_exportar_chips.py"
"""

import os
import sys
import time
import shutil
import datetime
import subprocess
import traceback
import numpy as np
import arcpy

# =============================== CONFIGURAÇÃO ===============================
# --- entradas
ORTOMOSAICO = r"D:\Users\Cap_Almeida\Projeto\126\Ortomosaico\126.tif"
EDIFICIOS   = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf\a_constr_treino.shp"   # do 06_separar_treino_teste.py
MASCARA     = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf\mascara_treino.shp"    # um só polígono
ZONA_TESTE  = r"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\SHP_vf\zona_teste.shp"        # só para verificação

# --- classe
CAMPO_CLASSE = "code"            # campo com o valor da classe (1 = edifício)
CRIAR_CLASSE_CONSTANTE = False   # True: ignora CAMPO_CLASSE e usa classe = 1 em todos os edifícios

# --- chips (iguais nas duas fases, para os resultados serem comparáveis)
TILE   = 256                     # px  (256 px x 0,5 m = 128 m)
STRIDE = 128                     # px  (50 % de sobreposição)
CELL_SIZE = 0.5                  # m; None = resolução nativa do ortomosaico
FORMATO = "TIFF"
ROTACAO = 0                      # graus; 0 = sem aumento por rotação (o treino já faz aumento de dados)

# --- saída
PASTA_SAIDA = rf"D:\Users\Cap_Almeida\Projeto\126\126_Construcoes\Training\original_{TILE}_s{STRIDE}"
APAGAR_SE_EXISTIR = True        # True: apaga uma pasta de saída anterior com o mesmo nome

# --- verificações depois da exportação
VERIFICAR_ZONA_TESTE = True      # confirma que nenhum chip entra na zona de teste
PASSO_AMOSTRA = 10               # proporção de classes: lê 1 em cada N chips de rótulos

# --- processamento
FATOR_PARALELO = "50%"           # núcleos usados na exportação; None = predefinição

# --- treino a seguir (opcional; só corre no Python Command Prompt)
SCRIPT_TREINO = None             # por exemplo r"D:\Users\Cap_Almeida\Projeto\Python\08_treinar_modelos.py"
# ============================================================================

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False
if FATOR_PARALELO:
    arcpy.env.parallelProcessingFactor = FATOR_PARALELO


class Relatorio:
    def __init__(self):
        self.linhas = []

    def __call__(self, txt=""):
        print(txt, flush=True)
        self.linhas.append(str(txt))

    def gravar(self, caminho):
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas) + "\n")


def descrever_sr(sr):
    codigo = sr.factoryCode if sr.factoryCode else "sem código EPSG"
    datum = sr.GCS.datumName if sr.GCS else "?"
    return f"{sr.name} ({codigo}; datum {datum})"


def listar_tifs(pasta):
    if not os.path.isdir(pasta):
        return []
    return sorted(os.path.join(pasta, f) for f in os.listdir(pasta)
                  if f.lower().endswith((".tif", ".tiff")))


# ---------------------------------------------------------------- verificações
def validar_entradas(rel):
    for nome, caminho in (("ORTOMOSAICO", ORTOMOSAICO), ("EDIFICIOS", EDIFICIOS),
                          ("MASCARA", MASCARA)):
        if not arcpy.Exists(caminho):
            raise FileNotFoundError(f"{nome} não existe: {caminho}")
    if VERIFICAR_ZONA_TESTE and not arcpy.Exists(ZONA_TESTE):
        raise FileNotFoundError(f"ZONA_TESTE não existe: {ZONA_TESTE}")

    # sistemas de coordenadas: tudo tem de estar no sistema do ortomosaico
    sr_o = arcpy.Describe(ORTOMOSAICO).spatialReference
    rel("SISTEMAS DE COORDENADAS")
    rel(f"  Ortomosaico: {descrever_sr(sr_o)}")
    for nome, caminho in (("Edifícios", EDIFICIOS), ("Máscara", MASCARA)):
        sr = arcpy.Describe(caminho).spatialReference
        rel(f"  {nome + ':':<12} {descrever_sr(sr)}")
        if sr.name != sr_o.name:
            raise RuntimeError(f"{nome} noutro sistema de coordenadas. Projeta-o para o sistema "
                               "do ortomosaico (por exemplo com o 03_Limpar_Constr_Projt.py) antes de exportar.")
    rel("  OK: todos no sistema do ortomosaico.")
    rel()

    # pasta de saída
    if os.path.isdir(PASTA_SAIDA) and os.listdir(PASTA_SAIDA):
        if APAGAR_SE_EXISTIR:
            shutil.rmtree(PASTA_SAIDA)
            rel(f"Pasta de saída anterior apagada: {PASTA_SAIDA}")
        else:
            raise FileExistsError(f"A pasta de saída já tem conteúdo: {PASTA_SAIDA}. "
                                  "Muda PASTA_SAIDA ou põe APAGAR_SE_EXISTIR = True.")


def preparar_classes(rel):
    """Devolve (feature class a exportar, campo da classe).
    Se o campo não existir ou CRIAR_CLASSE_CONSTANTE = True, cria uma cópia em memória
    com o campo 'classe' = 1 (o ficheiro original não é alterado)."""
    campos = [f.name for f in arcpy.ListFields(EDIFICIOS)]
    n = int(arcpy.management.GetCount(EDIFICIOS)[0])
    rel("CLASSES")
    rel(f"  Edifícios de treino: {n}")
    if not CRIAR_CLASSE_CONSTANTE and CAMPO_CLASSE in campos:
        valores = {}
        for (v,) in arcpy.da.SearchCursor(EDIFICIOS, [CAMPO_CLASSE]):
            valores[v] = valores.get(v, 0) + 1
        rel(f"  Campo '{CAMPO_CLASSE}': " +
            ", ".join(f"{v} ({c})" for v, c in sorted(valores.items(), key=lambda x: str(x[0]))))
        if None in valores or 0 in valores:
            rel("  AVISO: há edifícios com classe vazia ou 0, que ficam como fundo nos rótulos.")
        if len([v for v in valores if v not in (None, 0)]) > 1:
            rel("  AVISO: há mais do que um valor de classe; o modelo será multiclasse.")
        rel()
        return EDIFICIOS, CAMPO_CLASSE

    motivo = "CRIAR_CLASSE_CONSTANTE = True" if CRIAR_CLASSE_CONSTANTE else \
             f"o campo '{CAMPO_CLASSE}' não existe"
    rel(f"  {motivo}: é criada uma cópia temporária com o campo 'classe' = 1")
    # A cópia tem de ficar em disco: a exportação corre em processos paralelos, que
    # não têm acesso ao espaço "memory" do processo principal (dá ERROR 999999).
    copia = os.path.join(arcpy.env.scratchGDB, "edificios_classe")
    rel(f"  Cópia temporária: {copia}")
    if arcpy.Exists(copia):
        arcpy.management.Delete(copia)
    arcpy.management.CopyFeatures(EDIFICIOS, copia)
    arcpy.management.AddField(copia, "classe", "SHORT")
    arcpy.management.CalculateField(copia, "classe", "1", "PYTHON3")
    rel()
    return copia, "classe"


def verificar_zona_teste(imagens):
    """Devolve os chips cuja extensão interseta a zona de teste."""
    sr = arcpy.Describe(ORTOMOSAICO).spatialReference
    with arcpy.da.SearchCursor(ZONA_TESTE, ["SHAPE@"], spatial_reference=sr) as sc:
        zonas = [g for (g,) in sc if g is not None]
    zona = zonas[0]
    for g in zonas[1:]:
        zona = zona.union(g)
    invasoras = []
    t = time.time()
    for i, caminho in enumerate(imagens, 1):
        e = arcpy.Describe(caminho).extent
        ret = arcpy.Polygon(arcpy.Array([
            arcpy.Point(e.XMin, e.YMin), arcpy.Point(e.XMin, e.YMax),
            arcpy.Point(e.XMax, e.YMax), arcpy.Point(e.XMax, e.YMin),
            arcpy.Point(e.XMin, e.YMin)]), sr)
        inter = ret.intersect(zona, 4)
        if inter is not None and inter.area > 0.01:
            invasoras.append((os.path.basename(caminho), inter.area))
        if i % 2000 == 0:
            print(f"      verificados {i}/{len(imagens)} ({time.time() - t:.0f} s)", flush=True)
    return invasoras


def estatistica_classes(rotulos):
    contagem = np.zeros(256, dtype=np.int64)
    amostra = rotulos[::max(1, PASSO_AMOSTRA)]
    for caminho in amostra:
        arr = arcpy.RasterToNumPyArray(caminho)
        if arr.ndim > 2:
            arr = arr[0]
        contagem += np.bincount(arr.astype(np.int64).ravel(), minlength=256)[:256]
    return contagem, len(amostra)


# ---------------------------------------------------------------- principal
def main(rel):
    t0 = time.time()
    rel("=" * 78)
    rel("RELATÓRIO - EXPORTAÇÃO DOS CHIPS (vetores na posição original)")
    rel("=" * 78)
    rel(f"Data/hora:   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    rel(f"ArcGIS Pro:  {arcpy.GetInstallInfo().get('Version', '?')}")
    rel()
    rel("PARÂMETROS")
    for nome in ("ORTOMOSAICO", "EDIFICIOS", "MASCARA", "ZONA_TESTE", "CAMPO_CLASSE",
                 "CRIAR_CLASSE_CONSTANTE", "TILE", "STRIDE", "CELL_SIZE", "FORMATO", "ROTACAO",
                 "PASTA_SAIDA"):
        rel(f"  {nome:<24} = {globals()[nome]}")
    rel()

    if arcpy.CheckExtension("ImageAnalyst") != "Available":
        raise RuntimeError("Extensão Image Analyst não disponível.")
    arcpy.CheckOutExtension("ImageAnalyst")

    validar_entradas(rel)
    rotulos_fc, campo = preparar_classes(rel)

    # --- exportação
    rel("EXPORTAÇÃO")
    print("[1/3] A exportar os chips (pode demorar)...", flush=True)
    t1 = time.time()
    with arcpy.EnvManager(cellSize=CELL_SIZE if CELL_SIZE else "MINOF"):
        arcpy.ia.ExportTrainingDataForDeepLearning(
            in_raster=ORTOMOSAICO,
            out_folder=PASTA_SAIDA,
            in_class_data=rotulos_fc,
            image_chip_format=FORMATO,
            tile_size_x=TILE, tile_size_y=TILE,
            stride_x=STRIDE, stride_y=STRIDE,
            output_nofeature_tiles="ONLY_TILES_WITH_FEATURES",
            metadata_format="Classified_Tiles",
            start_index=0,
            class_value_field=campo,
            in_mask_polygons=MASCARA,
            rotation_angle=ROTACAO,
            reference_system="MAP_SPACE",
        )
    t_exp = time.time() - t1
    imagens = listar_tifs(os.path.join(PASTA_SAIDA, "images"))
    rotulos = listar_tifs(os.path.join(PASTA_SAIDA, "labels"))
    rel(f"  Chips de imagem:   {len(imagens)}")
    rel(f"  Chips de rótulos:  {len(rotulos)}")
    rel(f"  Tempo:             {t_exp / 60:.1f} min")
    if len(imagens) != len(rotulos):
        rel("  AVISO: número de imagens e de rótulos diferente.")
    rel()

    # --- zona de teste
    rel("VERIFICAÇÃO - CHIPS NA ZONA DE TESTE")
    if VERIFICAR_ZONA_TESTE and imagens:
        print("[2/3] A verificar se algum chip entra na zona de teste...", flush=True)
        invasoras = verificar_zona_teste(imagens)
        if invasoras:
            rel(f"  ATENÇÃO: {len(invasoras)} chips entram na zona de teste; "
                "o teste deixa de ser independente.")
            for nome, a in invasoras[:50]:
                rel(f"    {nome}  ({a:.1f} m2 dentro da zona de teste)")
        else:
            rel(f"  OK: nenhum dos {len(imagens)} chips entra na zona de teste.")
    else:
        rel("  Não executada.")
    rel()

    # --- proporção de classes
    rel("PROPORÇÃO DE PÍXEIS POR CLASSE (amostra)")
    if rotulos:
        print("[3/3] A calcular a proporção de classes...", flush=True)
        contagem, n_am = estatistica_classes(rotulos)
        total = contagem.sum() or 1
        rel(f"  Amostra: {n_am} de {len(rotulos)} chips")
        for v in np.nonzero(contagem)[0]:
            nome = "fundo" if v == 0 else f"classe {v}"
            rel(f"  {v:>3} {nome:<10} {contagem[v]:>14,d} px  {contagem[v] / total:7.2%}")
    rel()
    rel("PRÓXIMO PASSO")
    rel(f"  Treino com PASTA_CHIPS = {PASTA_SAIDA}")
    rel()
    rel(f"Tempo total: {(time.time() - t0) / 60:.1f} min")
    rel.gravar(os.path.join(PASTA_SAIDA, "exportacao_relatorio.txt"))

    # --- treino a seguir (opcional)
    if SCRIPT_TREINO:
        no_prompt = os.path.basename(sys.executable).lower().startswith("python")
        if not no_prompt:
            print("SCRIPT_TREINO indicado, mas o script está a correr dentro do ArcGIS Pro: "
                  "o treino não é lançado. Corre-o no Python Command Prompt.")
        else:
            print(f"A lançar o treino: {SCRIPT_TREINO}", flush=True)
            subprocess.run([sys.executable, SCRIPT_TREINO], check=True)
            print("Treino concluído.")


if __name__ == "__main__":
    _rel = Relatorio()
    try:
        main(_rel)
    except Exception:
        _rel()
        _rel("ERRO - execução interrompida")
        _rel(traceback.format_exc())
        destino = PASTA_SAIDA if os.path.isdir(PASTA_SAIDA) else os.path.dirname(PASTA_SAIDA)
        _rel.gravar(os.path.join(destino, "exportacao_relatorio_ERRO.txt"))
        raise
