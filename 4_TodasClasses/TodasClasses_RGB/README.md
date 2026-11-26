# Todas as classes só com RGB

Treino e avaliação do modelo com **todas as classes** (vegetação, água, sem representação, construções, vias e ferrovia) usando só o ortomosaico **RGB de 3 bandas**. O objetivo é comparar com o mesmo modelo de 4 bandas (RGB + NIR) da pasta de cima, `4_TodasClasses`.

O ortomosaico RGB é `D:\Users\Cap_Almeida\Projeto\126\Ortomosaico_RGB\126.tif` (bandas 1-3 do `126.tif`). É o mesmo usado em `6_RGB_vs_NIR` e em `2_RedeViaria/MTRE_RGB`.

## Equivalência com o modelo de 4 bandas

Os scripts desta pasta não alteram os originais. Importam o **53**, o **54** e o **55** da pasta de cima (`4_TodasClasses`), mudam só a configuração (ortomosaico, pastas e nomes dos modelos) e correm o `main()` de cada um. A única diferença é a falta do NIR. Tudo o resto é igual:

- rótulos de treino e de teste, e máscara das 13 ortos de treino (do 52);
- chips de 256 × 256 com passo de 128 (`ALL_TILES`, *Classified Tiles*);
- modelos SAMLoRA (ViT-B), DeepLabV3 e U-Net (ResNet-50);
- lote 8 (4 e 2 se faltar memória), até 50 épocas com paragem pela perda de validação, taxa de aprendizagem de 2 × 10⁻⁴, 10 % de validação, `FREEZE_MODEL`, `class_balancing True;ignore_classes [0]`;
- ortos de teste 126-I-3, 126-II-4 e 126-IV-3, com contexto de 256 m e TTA;
- métricas e agrupamentos do 55 (completa, generalizada, rede viária, ocupação sem construções e vias…).

Com 3 bandas, os pesos pré-treinados do RGB são usados tal como estão. No modelo de 4 bandas, a primeira camada é reinicializada e os pesos do NIR começam aleatórios.

## Ordem de execução

| Passo | Script | O que faz | Tempo | Saída |
|---|---|---|---|---|
| 1 | `1_preparar_orto_rgb.py` | Confirma o 126 RGB contra o `126.tif` (3 bandas de 8 bits, pixel, sistema, grelha e bandas 1-3 iguais numa amostra); só o cria se faltar | minutos | `Ortomosaico_RGB\126.tif` |
| 2 | `2_exportar_chips_rgb.py` | Exporta os chips com o 53; confirma que têm 3 bandas e o mesmo número de chips da versão de 4 bandas | horas (CPU) | `126_TodasClasses\53_Training\chips_todas_rgb_256_128` |
| 3 | `3_treinar_rgb.py` | Treina os 3 modelos com o 54 (salta os já treinados) | ~20–32 h por modelo (GPU) | `D:\DL\Modelos\126_todas_classes_rgb\<modelo>` |
| 4 | `4_avaliar_rgb_TTA.py` | Avalia com TTA com o 55 (salta os modelos sem `.emd`) | ~2–3 h | `D:\DL\Avaliacao\126_todas_classes_rgb_TTA` |
| 5 | `5_comparar_4bandas_rgb.py` | Compara 4 bandas e RGB por agrupamento: métricas globais e F1 por classe | segundos | `comparacao_4bandas_rgb.txt` e dois CSV na pasta da avaliação RGB |
| — | `9_cadeia_todas_rgb.py` | Corre os passos 1 a 5 sem vigilância, com verificações no arranque, nova tentativa em caso de erro e o Windows mantido acordado | — | `log_cadeia_todas_rgb.txt` |

A exportação dos chips corre no CPU e pode ser feita com outro treino a decorrer. O treino só deve começar com a GPU livre: na cadeia, indicar o PID desse treino em `ESPERAR_PID`.

## Configuração

Os caminhos estão todos no `comum_todas_rgb.py`. Esta pasta fica dentro de `4_TodasClasses` (`Python\4_TodasClasses\TodasClasses_RGB`). Os scripts originais 53, 54 e 55 são procurados na pasta de cima, depois em `Python\4_TodasClasses`, nesta pasta e em `Python`.

O 5 compara com a avaliação de 4 bandas em `D:\DL\Avaliacao\126_todas_classes_TTA`, que tem de existir (55 já corrido).
