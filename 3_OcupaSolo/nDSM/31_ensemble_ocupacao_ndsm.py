r"""
31_ensemble_ocupacao_ndsm.py  (ArcGIS Pro 3.6)

Versao com os modelos nDSM: le as classificacoes de varias pastas de avaliacao
(PASTAS_CLASSIFICACOES), corre varios ENSAIOS seguidos (conjuntos de modelos) e grava
cada um numa subpasta de PASTA_SAIDA, sem tocar nos resultados do ensemble anterior.

ENSEMBLE dos modelos de ocupacao do solo ja avaliados pelo 30_avaliar_ocupacao.py.
Nao classifica nada: usa as classificacoes que ja existem em
PASTA_AVALIACAO\classificacoes\<modelo>\class_<orto>.tif e as ortos, mascaras e
referencias preparadas em PASTA_AVALIACAO\ortos_teste\.

Metodos (para cada pixel):
  1. maioria     - fica a classe com mais votos (1 voto por modelo); nos empates
                   ganha a classe do modelo com maior PRIORIDADE.
  2. ponderado   - cada modelo vota com o seu F1 de VALIDACAO nessa classe (os pesos
                   vem da validacao, e nao do teste, para nao enviesar a avaliacao).
  3. especialista - parte do MODELO_BASE; onde um modelo ESPECIALISTA preve uma das
                   classes dificeis (ex.: Pomar, Arvoredo esparso) e pelo menos
                   MIN_CONCORDANCIA modelos (incluindo ele) concordam, fica essa classe.
  4. maioria_especialista - parte do resultado da maioria e aplica a mesma regra do
                   especialista nas classes dificeis.

Concordancia entre modelos: para cada pixel, o n.o de modelos que preve a classe mais
votada (4 = todos concordam ... 1 = todos diferentes). Calcula a % de pixeis e a
exatidao em cada nivel (concordancia_ensemble.csv) e grava um raster de concordancia
por orto (classificacoes\concordancia\conc_<orto>.tif), para usar como camada de
confianca (rever na edicao so as zonas de baixa concordancia).

Para comparar, avalia tambem cada modelo individual com exatamente o mesmo codigo.
As metricas sao as do 30_avaliar_ocupacao.py (importadas dele): por classe (precision,
recall, F1, IoU) e globais (OA, kappa, mIoU, fwIoU, F1 macro...), por orto e no TOTAL,
nos agrupamentos "completa" e "generalizada".

Saidas em PASTA_SAIDA\<ensaio>\ (nao apaga os resultados do 30 nem do 31 anterior):
  metricas_classe_ensemble.csv, metricas_globais_ensemble.csv,
  tabela_comparativa_ensemble.txt, matrizes\ens_<metodo>_<agrupamento>_<orto>.csv,
  classificacoes\ens_<metodo>\class_<orto>.tif (para ver no ArcGIS),
  log_ensemble.txt

Correr no Python Command Prompt do ArcGIS Pro (pode correr com um treino na GPU:
so usa o CPU e a memoria RAM):
    python "D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\2_nDSM\31_ensemble_ocupacao_ndsm.py"
"""
print("script iniciado, a carregar o arcpy...", flush=True)

import os
import csv
import time
import importlib.util
import datetime as _dt     # nome proprio: nao colide com o 'datetime' de outros modulos

import numpy as np
import arcpy

# ============================================================
# CONFIGURACAO
# ============================================================
# Script de avaliacao (de onde vem as funcoes das metricas, as classes e os agrupamentos)
SCRIPT_30 = r"D:\Users\Cap_Almeida\Projeto\Python\3_OcupacaoSolo\1_Unet\30_avaliar_ocupacao.py"
PASTA_AVALIACAO = r"D:\DL\Avaliacao\126_ocupacao"
ORTOS_TESTE = ["126-I-3", "126-II-4", "126-IV-3"]

# Modelos a combinar, por ORDEM DE PRIORIDADE (desempate): o primeiro e o melhor global
# Pasta onde ficam os resultados destes ensaios (uma subpasta por ensaio)
PASTA_SAIDA = r"D:\DL\Avaliacao\126_ocupacao_ensemble_nDSM"

# Onde estao as classificacoes de cada modelo (por omissao: PASTA_AVALIACAO\classificacoes)
PASTAS_CLASSIFICACOES = {
    "samlora_ndsm":       r"D:\DL\Avaliacao\126_ocupacao_nDSM\classificacoes",
    "deeplabv3_r50_ndsm": r"D:\DL\Avaliacao\126_ocupacao_nDSM\classificacoes",
    # com TTA: (pasta das classificacoes, nome da subpasta do modelo)
    "samlora_ndsm_tta":       (r"D:\DL\Avaliacao\126_ocupacao_nDSM_TTA\classificacoes", "samlora_ndsm"),
    "deeplabv3_r50_ndsm_tta": (r"D:\DL\Avaliacao\126_ocupacao_nDSM_TTA\classificacoes", "deeplabv3_r50_ndsm"),
    "samlora_v1_tta":         r"D:\DL\Avaliacao\126_ocupacao_TTA\classificacoes",
    "deeplabv3_r50_v2_tta":   r"D:\DL\Avaliacao\126_ocupacao_TTA\classificacoes",
    "unet_r50_v2_tta":        r"D:\DL\Avaliacao\126_ocupacao_TTA\classificacoes",
    "pspnet_r50_v2_tta":      r"D:\DL\Avaliacao\126_ocupacao_TTA\classificacoes",
}

# Ensaios: nome -> modelos por ORDEM DE PRIORIDADE (o 1.o e tambem o MODELO_BASE do especialista)
ENSAIOS = {
    "4_sam_ndsm":   ["samlora_ndsm", "deeplabv3_r50_v2", "unet_r50_v2", "pspnet_r50_v2"],
    "5_com_v1":     ["samlora_ndsm", "samlora_v1", "deeplabv3_r50_v2", "unet_r50_v2", "pspnet_r50_v2"],
    "5_dl_ndsm":    ["samlora_ndsm", "deeplabv3_r50_v2", "deeplabv3_r50_ndsm", "unet_r50_v2", "pspnet_r50_v2"],
    # os mesmos ensaios com as classificacoes TTA
    "4_sam_ndsm_TTA": ["samlora_ndsm_tta", "deeplabv3_r50_v2_tta", "unet_r50_v2_tta", "pspnet_r50_v2_tta"],
    "5_com_v1_TTA":   ["samlora_ndsm_tta", "samlora_v1_tta", "deeplabv3_r50_v2_tta", "unet_r50_v2_tta",
                       "pspnet_r50_v2_tta"],
    "5_dl_ndsm_TTA":  ["samlora_ndsm_tta", "deeplabv3_r50_v2_tta", "deeplabv3_r50_ndsm_tta", "unet_r50_v2_tta",
                       "pspnet_r50_v2_tta"],
}
MODELOS = list(next(iter(ENSAIOS.values())))   # definido em cada ensaio

# Conversao das previsoes para os codigos da referencia (como no 30)
MAPAS = {"samlora_v1": {0: 9}, "samlora_v1_tta": {0: 9}}

# F1 de VALIDACAO por modelo e classe (do model_metrics.html / relatorio de treino).
# No SAMLoRA v1, a coluna "NoData" (classe 0) corresponde ao sem representacao (9).
F1_VALIDACAO = {
    "samlora_v1":       {1: 0.8547, 2: 0.6714, 3: 0.4863, 4: 0.8057, 5: 0.5493,
                         6: 0.7019, 7: 0.9515, 9: 0.7304},
    "unet_r50_v2":      {1: 0.8219, 2: 0.5820, 3: 0.4378, 4: 0.7834, 5: 0.4061,
                         6: 0.6245, 7: 0.8819, 9: 0.5996},
    "deeplabv3_r50_v2": {1: 0.8616, 2: 0.7298, 3: 0.5641, 4: 0.8290, 5: 0.5734,
                         6: 0.7494, 7: 0.8799, 9: 0.6896},
    "pspnet_r50_v2":    {1: 0.7950, 2: 0.5524, 3: 0.2588, 4: 0.7515, 5: 0.3897,
                         6: 0.5786, 7: 0.7877, 9: 0.5318},
}
# Valores que o ensemble nunca escolhe: 0 (NoData/fundo) e 8 (Ignorar). O SAMLoRA preve o 8
# em 1-2 % dos pixeis avaliados; sem esta regra, o 8 ganhava alguns empates.
CLASSES_SEM_VOTO = {0, 8}
PESO_OUTRAS = 0.01   # peso de um voto numa classe sem F1 (ex.: 8 Ignorar)

# Metodo especialista
MODELO_BASE = "samlora_ndsm"           # passa a ser o 1.o modelo de cada ensaio
ESPECIALISTAS = {                     # classe -> modelos especialistas nessa classe
    3: ["unet_r50_v2", "pspnet_r50_v2"],   # Pomar
    5: ["unet_r50_v2", "pspnet_r50_v2"],   # Arvoredo esparso
}
MIN_CONCORDANCIA = 2                  # n.o minimo de modelos a prever a classe

METODOS = ["maioria", "especialista", "maioria_especialista"]   # "ponderado" exige F1 de validacao de todos
# Resultados para os quais se calcula a exatidao por nivel de concordancia
AVALIAR_CONCORDANCIA = ["ens_maioria", "ens_maioria_especialista", "samlora_ndsm"]
GRAVAR_RASTERS = True                 # grava as classificacoes do ensemble
# ============================================================

FICHEIRO_LOG = os.path.join(PASTA_SAIDA, "log_ensemble.txt")
N_VALORES = 10                        # codigos 0..9


def log(msg):
    linha = f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(FICHEIRO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def carregar_30():
    spec = importlib.util.spec_from_file_location("aval30", SCRIPT_30)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)        # so define funcoes e configuracao (main nao corre)
    return A


def nome_seguro(txt):
    return txt.replace("-", "_").replace(" ", "_").replace(".", "_")


# ------------------------------------------------------------------ previsoes
def ler_previsao(A, modelo, so, orto):
    base = PASTAS_CLASSIFICACOES.get(modelo, os.path.join(PASTA_AVALIACAO, "classificacoes"))
    base, sub = base if isinstance(base, tuple) else (base, modelo)
    tif = os.path.join(base, sub, f"class_{so}.tif")
    if not os.path.exists(tif):
        raise FileNotFoundError(f"Classificacao em falta: {tif} (correr antes o 30)")
    p = A.ler_grelha(tif, orto, 0).astype(np.int16)    # NoData -> 0
    mapa = MAPAS.get(modelo, {})
    if mapa:
        q = p.copy()
        for de, para in mapa.items():
            q[p == de] = para
        p = q
    return p


def votar(preds, peso):
    """preds: {modelo: array}; peso(modelo, classe) -> valor do voto.
    As classes de CLASSES_SEM_VOTO (0 e 8) nao recebem votos: o ensemble nunca as escolhe.
    Num pixel sem nenhum voto valido, fica a previsao do 1.o modelo da lista."""
    forma = next(iter(preds.values())).shape
    votos = np.zeros((N_VALORES,) + forma, np.float32)
    for m, p in preds.items():
        for c in range(N_VALORES):
            if c in CLASSES_SEM_VOTO:
                continue
            w = peso(m, c)
            if w:
                votos[c][p == c] += w
    out = votos.argmax(axis=0).astype(np.int16)
    sem_votos = votos.max(axis=0) == 0
    out[sem_votos] = preds[MODELOS[0]][sem_votos]
    return out


def ens_maioria(preds):
    n = len(MODELOS)
    rank = {m: i for i, m in enumerate(MODELOS)}
    # 1 voto por modelo + um desempate minusculo pela prioridade do modelo
    return votar(preds, lambda m, c: 1.0 + 1e-3 * (n - rank[m]))


def ens_ponderado(preds):
    n = len(MODELOS)
    rank = {m: i for i, m in enumerate(MODELOS)}
    return votar(preds, lambda m, c: F1_VALIDACAO[m].get(c, PESO_OUTRAS) + 1e-4 * (n - rank[m]))


def aplicar_especialistas(out, preds):
    for c, esp in ESPECIALISTAS.items():
        n_votos = np.zeros(out.shape, np.int8)
        for p in preds.values():
            n_votos += (p == c)
        algum_esp = np.zeros(out.shape, bool)
        for m in esp:
            algum_esp |= preds[m] == c
        out[(n_votos >= MIN_CONCORDANCIA) & algum_esp] = c
    return out


def ens_especialista(preds):
    out = preds[MODELO_BASE].copy()
    # onde o modelo base preve uma classe sem voto (ex.: 8 no SAMLoRA), fica a maioria
    sem_voto = np.isin(out, list(CLASSES_SEM_VOTO))
    if sem_voto.any():
        out[sem_voto] = ens_maioria(preds)[sem_voto]
    return aplicar_especialistas(out, preds)


def ens_maioria_especialista(preds):
    return aplicar_especialistas(ens_maioria(preds), preds)


def nivel_concordancia(preds):
    """N.o de modelos que preve a classe mais votada em cada pixel."""
    forma = next(iter(preds.values())).shape
    nivel = np.zeros(forma, np.int8)
    for c in range(N_VALORES):
        n = np.zeros(forma, np.int8)
        for p in preds.values():
            n += (p == c)
        np.maximum(nivel, n, out=nivel)
    return nivel


FUNCOES = {"maioria": ens_maioria, "ponderado": ens_ponderado, "especialista": ens_especialista,
           "maioria_especialista": ens_maioria_especialista}


def gravar_raster(arr, orto, saida):
    r = arcpy.Raster(orto)
    ras = arcpy.NumPyArrayToRaster(arr.astype(np.uint8), arcpy.Point(r.extent.XMin, r.extent.YMin),
                                   r.meanCellWidth, r.meanCellHeight)
    ras.save(saida)
    arcpy.management.DefineProjection(saida, r.spatialReference)


# ------------------------------------------------------------------ principal
def main():
    os.makedirs(os.path.join(PASTA_SAIDA, "matrizes"), exist_ok=True)
    arcpy.env.overwriteOutput = True
    A = carregar_30()
    t_ini = time.time()

    log("=" * 60)
    log(f"Ensemble iniciado | modelos (prioridade): {MODELOS} | metodos: {METODOS}")
    log(f"Especialista: base {MODELO_BASE}, classes {ESPECIALISTAS}, "
        f"concordancia minima {MIN_CONCORDANCIA}")
    for m in MODELOS:
        if "ponderado" in METODOS and m not in F1_VALIDACAO:
            raise ValueError(f"Falta o F1 de validacao do modelo {m}")

    nomes = MODELOS + [f"ens_{x}" for x in METODOS]
    M_total = {n: np.zeros((256, 256), np.int64) for n in nomes}
    l_cls, l_glb = [], []
    niveis = list(range(len(MODELOS), 0, -1))
    # conc[(orto, nivel)] = [n_pixeis, {nome: n_certos}]
    conc = {}

    def registar(nome, orto, M):
        for agr, mapa in A.AGRUPAMENTOS.items():
            Ma, labels = A.agrupar(M, mapa)
            pc, gl = A.calcular_metricas(Ma, labels)
            for m in pc:
                l_cls.append([nome, agr, orto, m["classe"], A.nome_classe(m["classe"], mapa),
                              m["suporte"], m["previstos"], m["TP"], m["FP"], m["FN"]] +
                             [round(m[k], 4) for k in ("precision", "recall", "F1", "IoU")])
            l_glb.append([nome, agr, orto] + [round(v, 4) if isinstance(v, float) else v
                                              for v in gl.values()])
            if nome.startswith("ens_"):
                A.gravar_matriz(Ma, labels, mapa, os.path.join(
                    PASTA_SAIDA, "matrizes", f"{nome}_{agr}_{nome_seguro(orto)}.csv"))
            yield agr, labels, mapa, pc, gl

    for o in ORTOS_TESTE:
        t0 = time.time()
        so = nome_seguro(o)
        pasta_o = os.path.join(PASTA_AVALIACAO, "ortos_teste")
        orto = os.path.join(pasta_o, f"orto_{so}.tif")
        msk_tif = os.path.join(pasta_o, f"mascara_{so}.tif")
        ref_tif = os.path.join(pasta_o, f"ref_{so}.tif")
        for f in (orto, msk_tif, ref_tif):
            if not os.path.exists(f):
                raise FileNotFoundError(f"Em falta: {f} (correr antes o 30)")

        msk = A.ler_grelha(msk_tif, orto, 0) == 1
        ref = A.ler_grelha(ref_tif, orto, A.SEM_REF).astype(np.int64)
        valido = msk & ~np.isin(ref, list(A.IGNORAR_REF | {A.SEM_REF}))
        preds = {m: ler_previsao(A, m, so, orto) for m in MODELOS}

        resultados = dict(preds)
        for metodo in METODOS:
            resultados[f"ens_{metodo}"] = FUNCOES[metodo](preds)
            if GRAVAR_RASTERS:
                pasta_cl = os.path.join(PASTA_SAIDA, "classificacoes", f"ens_{metodo}")
                os.makedirs(pasta_cl, exist_ok=True)
                gravar_raster(resultados[f"ens_{metodo}"], orto, os.path.join(pasta_cl, f"class_{so}.tif"))

        for nome in nomes:
            M = A.matriz_confusao(ref, resultados[nome], valido)
            M_total[nome] += M
            for agr, _, _, _, gl in registar(nome, o, M):
                if agr == "completa":
                    log(f"   {o} | {nome:<20} OA={gl['exatidao_global']:.3f} kappa={gl['kappa']:.3f} "
                        f"mIoU={gl['mIoU']:.3f} F1macro={gl['F1_macro']:.3f}")
        # concordancia entre modelos e exatidao em cada nivel
        nivel = nivel_concordancia(preds)
        if GRAVAR_RASTERS:
            pasta_cc = os.path.join(PASTA_SAIDA, "classificacoes", "concordancia")
            os.makedirs(pasta_cc, exist_ok=True)
            gravar_raster(nivel, orto, os.path.join(pasta_cc, f"conc_{so}.tif"))
        n_val = int(valido.sum())
        partes = []
        for k in niveis:
            sel = valido & (nivel == k)
            n_k = int(sel.sum())
            certos = {nome: int(np.sum(resultados[nome][sel] == ref[sel])) for nome in AVALIAR_CONCORDANCIA}
            for chave in ((o, k), ("TOTAL", k)):
                reg = conc.setdefault(chave, [0, {nome: 0 for nome in AVALIAR_CONCORDANCIA}])
                reg[0] += n_k
                for nome, v in certos.items():
                    reg[1][nome] += v
            oa = certos["ens_maioria"] / n_k if n_k else 0.0
            partes.append(f"{k} modelos: {100 * n_k / n_val:.1f}% dos pixeis (OA maioria {oa:.3f})")
        log(f"   {o}: concordancia -> " + " | ".join(partes) + f" | {time.time() - t0:.0f} s")
        del preds, resultados, nivel

    # --- totais
    comparativa = []
    for nome in nomes:
        for agr, labels, mapa, pc, gl in registar(nome, "TOTAL", M_total[nome]):
            comparativa.append((nome, agr, labels, mapa, pc, gl))
            if agr == "completa":
                log(f"   TOTAL {nome:<20} OA={gl['exatidao_global']:.3f} kappa={gl['kappa']:.3f} "
                    f"mIoU={gl['mIoU']:.3f} fwIoU={gl['fwIoU']:.3f} F1macro={gl['F1_macro']:.3f}")

    # --- CSV
    def gravar(nome, cab, linhas):
        with open(os.path.join(PASTA_SAIDA, nome), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cab)
            w.writerows(linhas)

    gravar("metricas_classe_ensemble.csv",
           ["modelo", "agrupamento", "orto", "classe", "nome", "suporte_px", "previstos_px", "TP", "FP",
            "FN", "precision", "recall", "F1", "IoU"], l_cls)
    gravar("metricas_globais_ensemble.csv",
           ["modelo", "agrupamento", "orto", "pixeis", "exatidao_global", "exatidao_media", "kappa",
            "mIoU", "fwIoU", "F1_macro", "F1_ponderado", "pct_previsoes_fora"], l_glb)

    # --- concordancia: CSV e resumo
    tot_orto = {}
    for (o, k), (n_k, _) in conc.items():
        tot_orto[o] = tot_orto.get(o, 0) + n_k
    l_conc = []
    for o in ORTOS_TESTE + ["TOTAL"]:
        acum_n, acum_c = 0, {nome: 0 for nome in AVALIAR_CONCORDANCIA}
        for k in niveis:
            n_k, certos = conc.get((o, k), [0, {}])
            acum_n += n_k
            for nome in AVALIAR_CONCORDANCIA:
                acum_c[nome] += certos.get(nome, 0)
            l_conc.append([o, k, n_k, round(100 * n_k / tot_orto[o], 2) if tot_orto.get(o) else 0,
                           round(100 * acum_n / tot_orto[o], 2) if tot_orto.get(o) else 0] +
                          [round(certos.get(nome, 0) / n_k, 4) if n_k else "" for nome in AVALIAR_CONCORDANCIA] +
                          [round(acum_c[nome] / acum_n, 4) if acum_n else "" for nome in AVALIAR_CONCORDANCIA])
    gravar("concordancia_ensemble.csv",
           ["orto", "n_modelos_concordam", "pixeis", "pct_pixeis", "pct_acumulada"] +
           [f"OA_{nome}" for nome in AVALIAR_CONCORDANCIA] +
           [f"OA_acumulada_{nome}" for nome in AVALIAR_CONCORDANCIA], l_conc)

    linhas_conc = ["", "CONCORDANCIA ENTRE MODELOS (total das ortos) - exatidao global em cada nivel",
                   f"{'concordam':<12}{'% pixeis':>10}{'% acum.':>9} | " +
                   "".join(f"{nome[:20]:>22}" for nome in AVALIAR_CONCORDANCIA)]
    for linha in l_conc:
        if linha[0] == "TOTAL":
            linhas_conc.append(f"{str(linha[1]) + ' de ' + str(len(MODELOS)):<12}{linha[3]:>10.1f}{linha[4]:>9.1f} | " +
                               "".join(f"{v:>22.3f}" if v != "" else f"{'-':>22}"
                                       for v in linha[5:5 + len(AVALIAR_CONCORDANCIA)]))
    linhas_conc.append("% acum.: pixeis com pelo menos esse n.o de modelos de acordo. "
                       "Ver concordancia_ensemble.csv para os valores por orto e a exatidao acumulada.")

    # --- tabela comparativa
    linhas = ["ENSEMBLE DOS MODELOS DE OCUPACAO DO SOLO - total das ortos de teste",
              f"Data: {_dt.datetime.now():%Y-%m-%d %H:%M} | modelos (prioridade): {MODELOS}",
              f"especialista: base {MODELO_BASE}, classes {list(ESPECIALISTAS)}, "
              f"concordancia minima {MIN_CONCORDANCIA}"]
    for agr, mapa in A.AGRUPAMENTOS.items():
        bloco = sorted((c for c in comparativa if c[1] == agr), key=lambda x: x[5]["mIoU"], reverse=True)
        labels = bloco[0][2]
        cab = (f"{'modelo':<22}{'OA':>7}{'kappa':>7}{'mIoU':>7}{'fwIoU':>7}{'F1mac':>7} | " +
               "".join(f"{A.nome_classe(c, mapa)[:9]:>10}" for c in labels))
        linhas += ["", f"AGRUPAMENTO: {agr}  (F1 por classe a direita)", cab, "-" * len(cab)]
        for nome, _, lb, _, pc, gl in bloco:
            f1 = {m["classe"]: m["F1"] for m in pc}
            linhas.append(f"{nome[:21]:<22}{gl['exatidao_global']:>7.3f}{gl['kappa']:>7.3f}"
                          f"{gl['mIoU']:>7.3f}{gl['fwIoU']:>7.3f}{gl['F1_macro']:>7.3f} | " +
                          "".join(f"{f1.get(c, 0):>10.3f}" for c in labels))
    linhas += ["", "ens_maioria: 1 voto por modelo (desempate pela prioridade; 0 e 8 nao recebem votos). "
               "ens_ponderado: voto = F1 de validacao do modelo nessa classe. "
               "ens_especialista: modelo base, substituido nas classes dificeis quando um "
               "especialista as preve e ha concordancia minima. "
               "ens_maioria_especialista: maioria + a mesma regra nas classes dificeis."]
    linhas += linhas_conc
    with open(os.path.join(PASTA_SAIDA, "tabela_comparativa_ensemble.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas) + "\n")
    log("\n".join(linhas))
    log(f"Ensemble terminado em {(time.time() - t_ini) / 60:.1f} min")


if __name__ == "__main__":
    PASTA_RAIZ = PASTA_SAIDA
    ESPEC_BASE = {c: list(v) for c, v in ESPECIALISTAS.items()}
    for _nome, _modelos in ENSAIOS.items():
        MODELOS = list(_modelos)
        # especialistas: usa a versao _tta quando o ensaio e com TTA
        ESPECIALISTAS = {c: [m + "_tta" if m + "_tta" in MODELOS else m for m in v]
                         for c, v in ESPEC_BASE.items()}
        _falta = [m for v in ESPECIALISTAS.values() for m in v if m not in MODELOS]
        if _falta:
            raise ValueError(f"Ensaio {_nome}: especialistas fora da lista de modelos: {_falta}")
        MODELO_BASE = MODELOS[0]
        if MODELO_BASE not in AVALIAR_CONCORDANCIA:
            AVALIAR_CONCORDANCIA = AVALIAR_CONCORDANCIA[:2] + [MODELO_BASE]
        PASTA_SAIDA = os.path.join(PASTA_RAIZ, _nome)
        os.makedirs(PASTA_SAIDA, exist_ok=True)
        FICHEIRO_LOG = os.path.join(PASTA_SAIDA, "log_ensemble.txt")
        print(f"\n######## ENSAIO {_nome}: {MODELOS}", flush=True)
        main()
