# Ocupação do solo só com RGB

Treino e avaliação dos modelos de ocupação do solo com o ortomosaico **RGB de 3 bandas** (`D:\Users\Cap_Almeida\Projeto\126\Ortomosaico_RGB\126.tif`, as bandas 1-3 do `126.tif`). O objetivo é medir o contributo do infravermelho próximo (NIR).

Os scripts são cópias dos da pasta `nDSM/`, com as mesmas alterações de configuração. Só mudam a imagem, as pastas e os nomes dos modelos:

| | 4 bandas (`UNET/`) | 5 bandas (`nDSM/`) | 3 bandas (`RGB/`) |
|---|---|---|---|
| Imagem | R, G, B, NIR | R, G, B, NIR, nDSM | R, G, B |
| Chips | `33_Training_unet\...` | `33_Training_nDSM\...` | `33_Training_RGB\chips_ocupacao_256_128` |
| Modelos | `D:\DL\Modelos\126_ocupacao_2v` (+ SAMLoRA v1) | `D:\DL\Modelos\126_ocupacao_nDSM` | `D:\DL\Modelos\126_ocupacao_RGB` |
| Avaliação | `126_ocupacao`, `126_ocupacao_TTA` | `126_ocupacao_nDSM`, `126_ocupacao_nDSM_TTA` | `126_ocupacao_RGB`, `126_ocupacao_RGB_TTA` |

Tudo o resto é igual:
- os rótulos, a máscara de treino e as ortos de teste do 22 (`32_separacao_unet`);
- chips de 256 × 256 com passo de 128;
- DeepLabV3 (ResNet-50) e SAMLoRA (ViT-B);
- lote 8, até 50 épocas com paragem antecipada, taxa de aprendizagem de 2 × 10⁻⁴, `FREEZE_MODEL`, `class_balancing True;ignore_classes [0,8]`;
- avaliação com contexto de 256 m, sem e com TTA.

Com 3 bandas, os pesos pré-treinados da primeira camada são usados tal como estão. Nas versões de 4 e 5 bandas, essa camada é reinicializada e os pesos das bandas extra começam aleatórios.

## Ordem de execução

| Passo | Script | Tempo |
|---|---|---|
| 23 | `23_exportar_chips_ocupacao.py` – chips RGB; confirma que o mosaico tem 3 bandas | ~30 min (CPU) |
| 24 | `24_verificar_chips_sem_rotulo.py` – relatório dos chips sem rótulo | minutos |
| 25 | `25_treinar_ocupacao.py` – DeepLabV3 e SAMLoRA (salta os já treinados) | ~20–32 h cada (GPU) |
| 30 | `30_avaliar_ocupacao.py` – 3 ortos de teste, sem TTA | ~20 min |
| 30 TTA | `30_avaliar_ocupacao_TTA.py` – 3 ortos de teste, com TTA | ~1–1,5 h |
| 33 | `33_comparar_bandas.py` – tabela RGB × 4 bandas × nDSM, sem e com TTA | segundos |
| — | `99_cadeia_rgb.py` – corre os passos todos sem vigilância (salta os chips já exportados, repete um passo que falhe, mantém o Windows acordado) | — |

O 33 lê os CSV das seis avaliações (3 combinações de bandas × sem/com TTA) em `D:\DL\Avaliacao`. As avaliações que ainda não existam aparecem como "sem avaliação".
