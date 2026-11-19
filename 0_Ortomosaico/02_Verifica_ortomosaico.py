# -*- coding: utf-8 -*-
"""
Verificação do ortomosaico da folha 126 (série M888).

Parte 1 - Confirma as propriedades do mosaico: dimensões, número de bandas,
          resolução, extensão e sistema de referência (igual ao dos tiles).
Parte 2 - Calcula, sobre o mosaico COMPLETO, as estatísticas da banda 4
          (valores distintos, média, correlação com R, G e B) e a
          distribuição do NDVI (percentis 5/50/95 e fração > 0,3).

O mosaico é processado em blocos para não esgotar a memória.
Píxeis com as 4 bandas a 0 são tratados como NoData e excluídos.
"""
import numpy as np
import arcpy

# ------------------------------------------------------------------
# Configuração
# ------------------------------------------------------------------
MOSAICO = r"D:\Users\Cap Almeida\Projeto\126\Ortomosaico\126.tif"
TILE_REF = r"D:\Users\Cap Almeida\Projeto\M888_Ortos\126_Ortofotos\126-I-1.tif"

# valores esperados
LARGURA_ESP, ALTURA_ESP = 32000, 20000
BANDAS_ESP = 4
CELULA_ESP = 0.5
EXTENSAO_ESP = (216000.0, 460000.0, 232000.0, 470000.0)  # XMin, YMin, XMax, YMax

BLOCO = 2000         # lado de cada bloco, em píxeis
LIMIAR_NDVI = 0.3
N_BINS = 2000        # histograma do NDVI entre -1 e 1 (passo 0,001)


def estado(cond):
    return "OK" if cond else "FALHA"


# ------------------------------------------------------------------
# Parte 1 - Propriedades do mosaico
# ------------------------------------------------------------------
r = arcpy.Raster(MOSAICO)
e = r.extent
extensao = (e.XMin, e.YMin, e.XMax, e.YMax)
sr_mos = r.spatialReference
sr_ref = arcpy.Describe(TILE_REF).spatialReference

print("=" * 60)
print("PARTE 1 - PROPRIEDADES DO MOSAICO")
print("=" * 60)
print(f"Dimensões: {r.width} x {r.height} píxeis "
      f"[{estado(r.width == LARGURA_ESP and r.height == ALTURA_ESP)}]")
print(f"Bandas: {r.bandCount} [{estado(r.bandCount == BANDAS_ESP)}]")
print(f"Tipo de píxel: {r.pixelType}")
print(f"Resolução: {r.meanCellWidth} x {r.meanCellHeight} m "
      f"[{estado(r.meanCellWidth == CELULA_ESP and r.meanCellHeight == CELULA_ESP)}]")
print(f"Extensão: {extensao} "
      f"[{estado(all(abs(a - b) < 1e-6 for a, b in zip(extensao, EXTENSAO_ESP)))}]")
print(f"Sistema de referência: {sr_mos.name} "
      f"[{estado(sr_mos.exportToString() == sr_ref.exportToString())}]")
print(f"Área: {(e.XMax - e.XMin) / 1000:.1f} x {(e.YMax - e.YMin) / 1000:.1f} km "
      f"= {(e.XMax - e.XMin) * (e.YMax - e.YMin) / 1e6:.1f} km²")

# ------------------------------------------------------------------
# Parte 2 - Banda 4 e NDVI sobre o mosaico completo
# ------------------------------------------------------------------
print()
print("=" * 60)
print("PARTE 2 - BANDA 4 E NDVI (mosaico completo, por blocos)")
print("=" * 60)

cell = r.meanCellWidth
n_linhas, n_colunas = r.height, r.width

# acumuladores
contagem_b4 = np.zeros(256, dtype=np.int64)         # histograma da banda 4
n_validos = 0
n_nodata = 0
soma = np.zeros(4, dtype=np.float64)                # soma de cada banda
soma_q = np.zeros(4, dtype=np.float64)              # soma dos quadrados
soma_cruz = np.zeros(3, dtype=np.float64)           # soma de b4 * (R, G, B)
hist_ndvi = np.zeros(N_BINS, dtype=np.int64)
bins = np.linspace(-1.0, 1.0, N_BINS + 1)
n_ndvi = 0
n_acima = 0

total_blocos = ((n_linhas + BLOCO - 1) // BLOCO) * ((n_colunas + BLOCO - 1) // BLOCO)
bloco_atual = 0

for lin0 in range(0, n_linhas, BLOCO):
    nl = min(BLOCO, n_linhas - lin0)
    for col0 in range(0, n_colunas, BLOCO):
        nc = min(BLOCO, n_colunas - col0)
        bloco_atual += 1

        # canto inferior esquerdo do bloco (as linhas contam a partir do topo)
        x = e.XMin + col0 * cell
        y = e.YMax - (lin0 + nl) * cell
        arr = arcpy.RasterToNumPyArray(MOSAICO, arcpy.Point(x, y), nc, nl,
                                       nodata_to_value=0)  # (bandas, linhas, colunas)

        valido = np.any(arr != 0, axis=0)
        n_nodata += int((~valido).sum())
        if not valido.any():
            continue

        b = arr[:, valido].astype(np.float64)          # (4, n)
        n_validos += b.shape[1]

        contagem_b4 += np.bincount(arr[3][valido], minlength=256)
        soma += b.sum(axis=1)
        soma_q += (b * b).sum(axis=1)
        soma_cruz += (b[:3] * b[3]).sum(axis=1)

        red, nir = b[0], b[3]
        den = nir + red
        ok_den = den > 0
        ndvi = (nir[ok_den] - red[ok_den]) / den[ok_den]
        hist_ndvi += np.histogram(ndvi, bins=bins)[0]
        n_ndvi += ndvi.size
        n_acima += int((ndvi > LIMIAR_NDVI).sum())

    print(f"  processados {bloco_atual}/{total_blocos} blocos")

# ---- estatísticas da banda 4
media = soma / n_validos
var = soma_q / n_validos - media ** 2
cov = soma_cruz / n_validos - media[:3] * media[3]
corr = cov / np.sqrt(var[:3] * var[3])


# ---- percentis do NDVI a partir do histograma
def percentil_hist(hist, bordas, p):
    acum = np.cumsum(hist)
    i = int(np.searchsorted(acum, p / 100.0 * acum[-1]))
    return (bordas[i] + bordas[i + 1]) / 2


p5, p50, p95 = (percentil_hist(hist_ndvi, bins, p) for p in (5, 50, 95))

print()
print(f"Píxeis válidos: {n_validos:,}   NoData (4 bandas a 0): {n_nodata:,}")
print()
print("Banda 4")
print(f"  Valores distintos: {int((contagem_b4 > 0).sum())}")
print(f"  Média: {media[3]:.1f}")
print(f"  Correlação com R: {corr[0]:.3f}")
print(f"  Correlação com G: {corr[1]:.3f}")
print(f"  Correlação com B: {corr[2]:.3f}")
print()
print("NDVI (banda 1 = vermelho, banda 4 = NIR)")
print(f"  Percentis 5 / 50 / 95: {p5:.2f} / {p50:.2f} / {p95:.2f}")
print(f"  Fração de píxeis com NDVI > {LIMIAR_NDVI}: {n_acima / n_ndvi * 100:.1f} %")
print(f"  (píxeis com R + NIR = 0 excluídos: {n_validos - n_ndvi:,})")