# MTRE só com RGB – rede viária e ferrovia

Treino e avaliação do Multi-Task Road Extractor (MTRE) com o ortomosaico RGB de 3 bandas
(`D:\Users\Cap_Almeida\Projeto\126\Ortomosaico_RGB\126.tif`, bandas 1-3 do `126.tif`, criado pelo
`6_RGB_vs_NIR/1_preparar_ortos.py`), para comparar com os MTRE de 4 bandas (RGB + NIR):

| Modelo RGB | Modelo de 4 bandas equivalente |
|---|---|
| `mtre_rede_viaria_rgb_resnet50_…` | `mtre_rede_viaria_resnet50_b8_lrauto_treino_20261016_1137` |
| `mtre_ferrovia_rgb_resnet50_…` | `mtre_ferrovia_resnet50_b8_lrauto_treino_20261017_0209` |

A única diferença em relação aos MTRE de 4 bandas é a falta do NIR. Tudo o resto é igual:

- etiquetas (`etiquetas_rede_viaria_binario.shp`, `etiquetas_ferrovia_binario.shp`) e área de treino;
- chips de 512 × 512 com passo de 256;
- ResNet-50 com LinkNet e os mesmos argumentos do MTRE;
- lote 8, paragem pela perda de validação e taxa de aprendizagem automática;
- até 50 épocas na rede viária e até 100 na ferrovia (poucos chips);
- ortos de teste e métricas.

Os scripts confirmam esta equivalência a partir do `parametros.json` do registo de treino de cada MTRE de 4 bandas (pasta `<modelo>_registo`): o **42** compara o número de chips no fim da exportação e o **43** compara o número de chips e os parâmetros antes do treino.

## Ordem de execução

| Passo | Script | O que faz | Saída |
|---|---|---|---|
| 1 | `MTRE_RGB/42_Export_RViaria_rgb.py` | Verifica o ortomosaico RGB contra o `126.tif` (bandas, célula, sistema, grelha e bandas 1-3 iguais numa amostra) e exporta os chips dos dois conjuntos | `Training\rede_viaria_chip512_binario_rgb`, `Training\ferrovia_chip512_binario_rgb` |
| 2 | `MTRE_RGB/43_Treino_RViaria_rgb.py` | Treino dos dois modelos, um a seguir ao outro (`MODO_TESTE = True` para 1 época de teste) | `D:\DL\Modelos\126_RViaria_RGB\mtre_*_rgb_resnet50_b8_lrauto_treino_*` |
| 3 | `44_avaliar_RViaria.py` | Avalia os MTRE de 4 bandas, com nDSM e RGB e compara-os | CSV em `D:\DL\Avaliacao\126_RViaria`, incluindo `comparacao_<corrida>.csv` |

A exportação corre no processador e pode ser feita com outro treino a decorrer; o treino só deve começar com a GPU livre.

## Avaliação

No 44, para os modelos RGB, a entrada da inferência é um recorte do ortomosaico RGB com a extensão e a grelha exatas de cada ortofoto de teste, criado uma vez em `D:\DL\Avaliacao\126_RViaria\ortos_mosaicos`. O resto do processo é igual. No fim, o `comparacao_<corrida>.csv` inclui os pares 4 bandas vs RGB.
