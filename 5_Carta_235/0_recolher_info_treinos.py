# -*- coding: utf-8 -*-
"""
0_recolher_info_treinos.py

Junta num zip a informacao dos treinos (sem os modelos em si), para preparar a
avaliacao na folha 235 com as mesmas condicoes da 126:
  - .emd (classes e valores, bandas, tamanho dos chips, tipo de modelo, backbone)
  - .json, .csv, .txt, .log, .xlsx (resumos, relatorios, logs, epocas)
  - os ficheiros _registo
NAO copia .pth, .dlpk, .pt, imagens nem rasters (sao grandes e nao sao precisos).

Corre em qualquer Python (nao precisa de arcpy):
    python 0_recolher_info_treinos.py
"""
import os
import zipfile

PASTA_MODELOS = r"D:\DL\Modelos"
PASTAS = [
    "126_construcoes",
    "126_RViaria",
    "126_RViaria_Resnet50",
    "126_ocupacao_samlora",
    "126_ocupacao_2v",
]
IGNORAR_PASTAS = ["lixo"]                     # ex.: Lixo_0_0NoData, Treinos_lixo
EXTENSOES = {".emd", ".json", ".csv", ".txt", ".log", ".xlsx", ".xls", ".py", ".yaml", ".yml"}
MAX_MB = 20                                   # ignora ficheiros maiores do que isto

SAIDA = os.path.join(PASTA_MODELOS, "info_treinos_126.zip")


def main():
    n, ignorados = 0, []
    with zipfile.ZipFile(SAIDA, "w", zipfile.ZIP_DEFLATED) as z:
        for pasta in PASTAS:
            raiz = os.path.join(PASTA_MODELOS, pasta)
            if not os.path.isdir(raiz):
                print(f"AVISO: nao existe {raiz}")
                continue
            for d, subdirs, ficheiros in os.walk(raiz):
                subdirs[:] = [s for s in subdirs if not any(i in s.lower() for i in IGNORAR_PASTAS)]
                for f in ficheiros:
                    caminho = os.path.join(d, f)
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in EXTENSOES:
                        continue
                    if os.path.getsize(caminho) > MAX_MB * 1024 * 1024:
                        ignorados.append(caminho)
                        continue
                    z.write(caminho, os.path.relpath(caminho, PASTA_MODELOS))
                    n += 1
        # lista de todos os ficheiros (incluindo os nao copiados), para ver a estrutura
        lista = []
        for pasta in PASTAS:
            for d, _, ficheiros in os.walk(os.path.join(PASTA_MODELOS, pasta)):
                for f in ficheiros:
                    c = os.path.join(d, f)
                    lista.append(f"{os.path.relpath(c, PASTA_MODELOS)}\t{os.path.getsize(c) / 1e6:.1f} MB")
        z.writestr("_lista_ficheiros.txt", "\n".join(lista))
    print(f"{n} ficheiros copiados -> {SAIDA} ({os.path.getsize(SAIDA) / 1e6:.1f} MB)")
    for c in ignorados:
        print(f"   ignorado (> {MAX_MB} MB): {c}")


if __name__ == "__main__":
    main()