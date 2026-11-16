# -*- coding: utf-8 -*-
"""
Áreas de treino e teste da rede viária.
Usa folhas_ortos.shp (criado pelo 40a) e as ortofotos de teste escolhidas.
Executar na janela Python ou num Notebook do ArcGIS Pro 3.6.
"""
import os
import arcpy

arcpy.env.overwriteOutput = True

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
PASTA_SAIDA = r"D:\Users\Cap_Almeida\Projeto\126\126_RedeViaria\SHP"
FOLHAS = os.path.join(PASTA_SAIDA, "folhas_ortos.shp")
TESTE = ["126-I-2", "126-III-4", "126-IV-1"]   # substituir pela escolha feita no 39


# ---------------------------------------------------------------------------
lista = ", ".join(f"'{t}'" for t in TESTE)

teste = os.path.join(PASTA_SAIDA, "area_teste.shp")
arcpy.analysis.Select(FOLHAS, teste, f"FOLHA IN ({lista})")
n = int(arcpy.management.GetCount(teste)[0])
if n != len(TESTE):
    raise SystemExit(f"Encontradas {n} de {len(TESTE)} ortos de teste: verificar os nomes")

# Treino = restantes ortos, sem qualquer sobreposição com o teste
# (as extensões de ortos vizinhas podem sobrepor-se nas margens)
tmp = "memory/treino"
arcpy.analysis.Select(FOLHAS, tmp, f"FOLHA NOT IN ({lista})")
treino = os.path.join(PASTA_SAIDA, "area_treino.shp")
arcpy.analysis.Erase(tmp, teste, treino)
arcpy.management.Delete("memory")

for nome, fc in (("Treino", treino), ("Teste", teste)):
    folhas, km2 = [], 0.0
    with arcpy.da.SearchCursor(fc, ["FOLHA", "SHAPE@AREA"]) as cur:
        for f, a in cur:
            folhas.append(f)
            km2 += a / 1e6
    print(f"{nome}: {len(folhas)} ortos, {km2:.1f} km² -> {fc}")
    print(f"  {', '.join(sorted(folhas))}")