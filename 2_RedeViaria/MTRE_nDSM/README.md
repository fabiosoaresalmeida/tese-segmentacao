# MTRE com nDSM – rede viária

Treino e avaliação do Multi-Task Road Extractor (MTRE) com o ortomosaico de 5 bandas
(R, G, B, NIR e nDSM normalizado), para comparar com o MTRE de 4 bandas
(`mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_1137`).

A única diferença em relação ao MTRE de 4 bandas é a banda do nDSM. Tudo o resto é igual:

- etiquetas (`etiquetas_rede_viaria_binario.shp`) e área de treino;
- chips de 512 × 512 com passo de 256;
- ResNet-50 com LinkNet e os mesmos argumentos do MTRE;
- lote 8, até 50 épocas, paragem pela perda de validação e taxa de aprendizagem automática;
- ortos de teste e métricas.

## Ordem de execução

| Passo | Script | O que faz | Saída |
|---|---|---|---|
| 1 | `MTRE_nDSM/42_Export_RViaria_ndsm.py` | Verifica o ortomosaico de 5 bandas contra o `126.tif` e exporta os chips | `Training\rede_viaria_chip512_binario_ndsm` |
| 2 | `MTRE_nDSM/43_Treino_RViaria_ndsm.py` com `MODO_TESTE = True` | 1 época: confirma que o MTRE aceita 5 bandas e mede o tempo por época | modelo de teste |
| 3 | `MTRE_nDSM/43_Treino_RViaria_ndsm.py` com `MODO_TESTE = False` | Treino real | `D:\DL\Modelos\126_RViaria_nDSM\mtre_rede_viaria_ndsm_resnet50_b8_lrauto_treino_*` |
| 4 | `44_avaliar_RViaria.py` | Avalia o MTRE de 4 bandas e o MTRE com nDSM | CSV em `D:\DL\Avaliacao\126_RViaria` |

## Avaliação

No 44, para os modelos com nDSM, a entrada da inferência é um recorte do ortomosaico de 5 bandas com a extensão e a grelha exatas de cada ortofoto de teste. O recorte é criado uma vez em `D:\DL\Avaliacao\126_RViaria\ortos_5bandas`. O resto do processo é igual: espelhamento das margens, três versões de vetorização e as mesmas métricas.

`AVALIAR` escolhe os modelos da corrida. A inferência do MTRE de 4 bandas já feita é reutilizada.
