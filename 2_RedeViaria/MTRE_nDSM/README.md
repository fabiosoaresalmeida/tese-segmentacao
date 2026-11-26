# MTRE com nDSM – rede viária e ferrovia

Treino e avaliação do Multi-Task Road Extractor (MTRE) com o ortomosaico de 5 bandas
(R, G, B, NIR e nDSM normalizado), para comparar com os MTRE de 4 bandas:

| Modelo com nDSM | Modelo de 4 bandas equivalente |
|---|---|
| `mtre_rede_viaria_ndsm_resnet50_…` | `mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_1137` |
| `mtre_ferrovia_ndsm_resnet50_…` | `mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209` |

A única diferença em relação aos MTRE de 4 bandas é a banda do nDSM. Tudo o resto é igual:

- etiquetas (`etiquetas_rede_viaria_binario.shp`, `etiquetas_ferrovia_binario.shp`) e área de treino;
- chips de 512 × 512 com passo de 256;
- ResNet-50 com LinkNet e os mesmos argumentos do MTRE;
- lote 8, paragem pela perda de validação e taxa de aprendizagem automática;
- até 50 épocas na rede viária e até 100 na ferrovia (poucos chips);
- ortos de teste e métricas.

## Ordem de execução

| Passo | Script | O que faz | Saída |
|---|---|---|---|
| 1 | `MTRE_nDSM/42_Export_RViaria_ndsm.py` | Verifica o ortomosaico de 5 bandas contra o `126.tif` e exporta os chips dos dois conjuntos (os já exportados são saltados) | `Training\rede_viaria_chip512_binario_ndsm`, `Training\ferrovia_chip512_binario_ndsm` |
| 2 | `MTRE_nDSM/43_Treino_RViaria_ndsm.py` com `MODO_TESTE = True` | 1 época: confirma que o MTRE aceita 5 bandas e mede o tempo por época | modelo de teste |
| 3 | `MTRE_nDSM/43_Treino_RViaria_ndsm.py` com `MODO_TESTE = False` | Treino real dos modelos com `ativo=True` | `D:\DL\Modelos\126_RViaria_nDSM\mtre_*_ndsm_resnet50_b8_lrauto_treino_*` |
| 4 | `44_avaliar_RViaria.py` | Avalia os MTRE de 4 bandas e com nDSM (rede viária e ferrovia) | CSV em `D:\DL\Avaliacao\126_RViaria` |

O teste da rede viária (26/11/2026) confirmou que o MTRE aceita as 5 bandas; a ferrovia pode ir diretamente para o treino real.

## Avaliação

No 44, para os modelos com nDSM, a entrada da inferência é um recorte do ortomosaico de 5 bandas com a extensão e a grelha exatas de cada ortofoto de teste. O recorte é criado uma vez em `D:\DL\Avaliacao\126_RViaria\ortos_5bandas` e serve aos dois modelos. O resto do processo é igual: espelhamento das margens, três versões de vetorização e as mesmas métricas.

`AVALIAR` escolhe os modelos da corrida. As inferências dos MTRE de 4 bandas já feitas são reutilizadas, e um modelo com nDSM ainda não treinado é saltado com um aviso.
