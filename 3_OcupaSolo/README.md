# Ocupação do solo – folha 126 (Peso da Régua)

Scripts e resultados da segmentação semântica da ocupação do solo (vegetação e água) por deep learning no ArcGIS Pro 3.6, para a produção da carta topográfica 1:25 000.

Foram comparadas quatro arquiteturas (U-Net, DeepLabV3, PSPNet e SAMLoRA), a inferência com e sem *test-time augmentation* (TTA), a adição de uma banda de alturas (nDSM), a remoção do infravermelho próximo (só RGB) e vários métodos de *ensemble*:

| Experiência | Imagem de entrada | Modelos |
|---|---|---|
| **4 bandas** | Ortofotografia R, G, B, NIR (0,5 m) | U-Net, DeepLabV3 e PSPNet (ResNet-50); SAMLoRA (ViT-B) |
| **4 bandas + TTA** | Ortofotografia R, G, B, NIR | os mesmos 4 modelos |
| **5 bandas (nDSM)** | Ortofotografia + nDSM (MDS − MDT da DGT, 2025) | DeepLabV3 (ResNet-50) e SAMLoRA (ViT-B) |
| **5 bandas + TTA** | Ortofotografia + nDSM | os mesmos 2 modelos |
| **3 bandas (RGB)** | Ortofotografia R, G, B (sem o NIR) | DeepLabV3 (ResNet-50) e SAMLoRA (ViT-B) — em curso |
| **3 bandas + TTA** | Ortofotografia R, G, B | os mesmos 2 modelos — em curso |
| **Ensemble** | classificações dos modelos de 4 bandas | maioria, ponderado, especialista, maioria + especialista |

## Classes

Os rótulos vêm da edição cartográfica da série M888 (shapefiles de 2018); a ortofotografia é de 2013.

| Código | Classe | Origem (campo `source`) | Notas |
|---|---|---|---|
| 1 | Vinha | `A_Vinha` | |
| 2 | Pomar e vinha | `A_Pomar_vinha` | junta-se à Vinha no agrupamento "generalizada" |
| 3 | Pomar | `A_Pomar` | |
| 4 | Arvoredo denso | `A_Arvoredo_denso` | |
| 5 | Arvoredo esparso | `A_Arvoredo_esparso` | |
| 6 | Mato denso ou arbustos | `A_Mato_denso_ou_arbustos` | |
| 7 | Água | `A_Albufeira`, `A_Rio_de_duas_margens`, `A_Lagoa` | |
| 8 | Ignorar | construções (`a_constr`) e estradas (`l_vias` com buffer) | não entra na *loss* nem nas métricas |
| 9 | Sem representação | área da folha sem nenhum outro rótulo (sobretudo solo nu) | polígonos explícitos (versão 2) |
| 0 | NoData / sem polígono | – | ignorado no treino (versão 2) e na avaliação |

As vias são convertidas em polígonos com um buffer de meia largura: auto-estrada 11 m, estrada larga 7 m, estrada estreita 4 m, arruamento 5 m, acesso auto 3 m, caminho carreteiro 2,5 m, caminho de pé posto 1,5 m, caminho de ferro 3,5 m (outras: 3 m). Nas sobreposições, a precedência é **Ignorar > Água > Vegetação** (as pontes ficam como Ignorar).

Dois agrupamentos são avaliados sem novo treino:
- **completa** – as 8 classes avaliadas (1–7 e 9);
- **generalizada** – Pomar e vinha (2) junto com a Vinha (1).

## Versões dos rótulos

| Versão | Pasta | Sem representação | `ignore_classes` | Modelos em uso |
|---|---|---|---|---|
| v1 | `SAMLORA/` | 0 (fundo, sem polígonos) | `[8]` | SAMLoRA v1 |
| v2 – 4 bandas | `UNET/` | 9 (polígonos explícitos) | `[0,8]` | U-Net, DeepLabV3 e PSPNet v2 |
| v2 – 5 bandas | `nDSM/` | 9 | `[0,8]` | DeepLabV3 e SAMLoRA nDSM |
| v2 – 3 bandas | `RGB/` | 9 | `[0,8]` | DeepLabV3 e SAMLoRA RGB |

Na v1, as redes com codificador ResNet do ArcGIS tratam o valor 0 como NoData e nunca aprendiam o sem representação; o SAMLoRA aprendia-o. A v2 passou o sem representação para a classe 9 e as redes ResNet foram treinadas de novo. Para o SAMLoRA as duas versões são equivalentes (a mesma área, com outro código), por isso foi mantido o SAMLoRA v1, avaliado com a conversão das previsões 0 → 9.

A pasta `UNET/` contém os scripts de **todos** os modelos v2 de 4 bandas (não só da U-Net) e o ensemble.

## Estrutura

```
3_OcupaSolo/
├── SAMLORA/                                 versão 1 (histórica)
│   ├── 21_rotulos_ocupacao.py
│   ├── 22_separar_treino_teste_ocupacao.py
│   ├── 23_exportar_chips_ocupacao.py
│   ├── 24_verificar_chips_sem_rotulo.py
│   └── 25_treinar_fase1_ocupacao.py
├── UNET/                                    versão 2, 4 bandas
│   ├── 21_rotulos_ocupacao.py               rótulos sem sobreposições e classe 9
│   ├── 22_separar_treino_teste_ocupacao.py  zona de teste, máscara de treino, rótulos recortados
│   ├── 23_exportar_chips_ocupacao.py        chips de treino (Classified Tiles)
│   ├── 24_verificar_chips_sem_rotulo.py     diagnóstico de chips sem rótulo
│   ├── 25_treinar_fase1_ocupacao.py         treino U-Net, DeepLabV3, PSPNet
│   ├── 30_avaliar_ocupacao.py               classificação e métricas nas ortos de teste
│   └── 31_ensemble_ocupacao.py              ensembles e concordância entre modelos
├── nDSM/                                    versão 2, 5 bandas
│   ├── 23_exportar_chips_ocupacao.py        chips com orto + nDSM
│   ├── 24_verificar_chips_sem_rotulo.py
│   ├── 25_treinar_ocupacao.py               treino DeepLabV3 e SAMLoRA
│   ├── 30_avaliar_ocupacao.py               avaliação sem TTA
│   ├── 30_avaliar_ocupacao_TTA.py           avaliação com TTA
│   ├── 31_ensemble_ocupacao_ndsm.py         ensembles com os modelos nDSM
│   └── 99_cadeia_tta_ndsm.py                cadeia avaliação TTA → treino nDSM (já executada)
├── RGB/                                     versão 2, 3 bandas (sem o NIR)
│   ├── 23_exportar_chips_ocupacao.py        chips com o ortomosaico RGB
│   ├── 24_verificar_chips_sem_rotulo.py
│   ├── 25_treinar_ocupacao.py               treino DeepLabV3 e SAMLoRA
│   ├── 30_avaliar_ocupacao.py               avaliação sem TTA
│   ├── 30_avaliar_ocupacao_TTA.py           avaliação com TTA
│   ├── 33_comparar_bandas.py                comparação RGB × 4 bandas × nDSM
│   └── 99_cadeia_rgb.py                     cadeia chips → treino → avaliação → comparação
└── Resultados/
    ├── 4Bandas/                             avaliação e ensemble, 4 bandas
    ├── 4Bandas_TTA/                         avaliação com TTA, 4 bandas
    ├── 5Bandas_nDSM/                        avaliação, 5 bandas
    └── 5Bandas_nDSM_TTA/                    avaliação com TTA, 5 bandas
```

As versões com nDSM e só com RGB usam os scripts 21 e 22 de `UNET/` (os mesmos rótulos, a mesma máscara e as mesmas ortos de teste); só muda a imagem.

## Cadeia de processamento

| N.º | Script | Entradas | Saídas |
|---|---|---|---|
| 21 | `21_rotulos_ocupacao.py` | `a_vegetacao.shp`, `a_hidro.shp`, `a_constr.shp`, `l_vias.shp`, ortomosaico | `rotulos_ocupacao.shp` (campo `classe`), `resumo_rotulos.csv` |
| 22 | `22_separar_treino_teste_ocupacao.py` | ortofotos individuais, `rotulos_ocupacao.shp` | `ortos_grelha.shp`, `zona_teste.shp`, `mascara_treino.shp`, `rotulos_treino.shp`, `rotulos_teste.shp`, relatório |
| 23 | `23_exportar_chips_ocupacao.py` | ortomosaico (4 ou 5 bandas), `rotulos_treino.shp`, `mascara_treino.shp` | chips 256 × 256 px (passo 128), *Classified Tiles*, relatório com píxeis por classe |
| 24 | `24_verificar_chips_sem_rotulo.py` | pasta dos chips, `rotulos_treino.shp` | `chips_sem_rotulo.shp` (diagnóstico) |
| 25 | `25_treinar_*.py` | chips | um modelo por pasta (`.emd`, `.pth`, `.dlpk`, `model_metrics.html`), relatório e resumo |
| 30 | `30_avaliar_ocupacao*.py` | modelos, ortomosaico, `rotulos_teste.shp`, `zona_teste.shp` | classificações, `metricas_classe.csv`, `metricas_globais.csv`, `tabela_comparativa.txt`, matrizes de confusão, `log_avaliacao.txt` |
| 31 | `31_ensemble_ocupacao*.py` | classificações e ortos de teste preparadas pelo 30 | classificações dos ensembles, rasters de concordância, `*_ensemble.csv`, `tabela_comparativa_ensemble.txt`, `concordancia_ensemble.csv` |

### Descrição dos passos

1. **Rótulos (21)** – projeta as shapefiles para o sistema do ortomosaico, atribui a classe pelo campo `source`, converte os eixos das vias em polígonos com a meia largura de cada tipo e junta-os às construções na classe Ignorar. Aplica a precedência Ignorar > Água > Vegetação e cria os polígonos de sem representação (9) como a área da folha sem nenhum outro rótulo. No fim verifica que não há sobreposições.
2. **Treino/teste (22)** – cria a grelha das 16 ortofotos e reserva três para teste: **126-I-3, 126-II-4 e 126-IV-3** (diferentes das usadas nos edifícios e na rede viária). As restantes 13 formam a máscara de treino. Os polígonos que atravessam o limite são **recortados**, para que nenhum píxel da zona de teste entre no treino.
3. **Chips (23)** – *Export Training Data For Deep Learning* com *Classified Tiles*, 256 × 256 px a 0,5 m (128 m), passo de 128 px (50 % de sobreposição), `ALL_TILES`, só dentro da máscara de treino: **30 892 chips**, iguais nas versões de 4 e 5 bandas. O relatório conta os píxeis por classe e calcula pesos de referência (*median frequency balancing*).
4. **Verificação (24)** – encontra chips com imagem e sem rótulo e verifica se têm polígonos por baixo (na v2 não deve haver nenhum, porque a classe 9 cobre toda a área).
5. **Treino (25)** – modelos treinados em sequência, com os mesmos parâmetros (ver tabela abaixo); se faltar memória na GPU, repete com um lote menor; salta os modelos já treinados.
6. **Avaliação (30)** – recorta cada ortofoto de teste com 256 m de contexto (bordas exteriores da folha espelhadas), rasteriza a máscara e a referência na mesma grelha, classifica com *Classify Pixels Using Deep Learning* (`padding 64`, `batch_size 4`) e calcula, por ortofoto e no total:
   - **por classe**: precisão (exatidão do utilizador), sensibilidade (exatidão do produtor), F1, IoU, suporte e píxeis previstos;
   - **globais**: exatidão global (OA), exatidão média, kappa, mIoU, IoU ponderado pela área (fwIoU), F1 macro e ponderado, e % de previsões fora das classes avaliadas.

   Os píxeis da referência com 0 ou 8 são excluídos. O modelo é lido pelo `.emd` (e não pelo `.dlpk`, porque o ArcGIS pode reutilizar um `.dlpk` já extraído com o mesmo nome). Cada classificação fica com um `.json` (modelo, data do modelo, argumentos, ortomosaico) e é refeita se algum destes mudar.
7. **Ensemble (31)** – combina as classificações já feitas, sem nova inferência:
   - **maioria** – um voto por modelo, desempate pela prioridade (o melhor modelo primeiro);
   - **ponderado** – cada modelo vota com o seu F1 de **validação** nessa classe;
   - **especialista** – parte do melhor modelo e, nas classes difíceis (Pomar e Arvoredo esparso), aceita a previsão da U-Net ou da PSPNet quando pelo menos 2 modelos concordam;
   - **maioria + especialista** – maioria com a mesma regra nas classes difíceis.

   As classes 0 e 8 não recebem votos. Grava também um raster de **concordância** (n.º de modelos que preveem a classe vencedora), para usar como camada de confiança na edição.
8. **Cadeia (99)** – script de uma só utilização que correu a avaliação com TTA e, a seguir, o treino com nDSM, verificando antes a configuração dos dois scripts.

## Parâmetros de treino

| Parâmetro | Valor |
|---|---|
| Chips | 256 × 256 px, passo 128, 30 892 chips |
| Validação | 10 % dos chips, escolhidos ao acaso |
| Épocas | máximo 50, com paragem antecipada (`STOP_TRAINING`, monitor `VALID_LOSS`) |
| Taxa de aprendizagem | 2 × 10⁻⁴ (fixa) |
| Lote | 8 (v1: 16 na DeepLabV3 e na PSPNet, 8 na U-Net e no SAMLoRA) |
| Codificador | pré-treinado e congelado (`FREEZE_MODEL`); a 1.ª camada é treinável para as bandas extra (NIR, nDSM) |
| Inicialização das bandas extra | por defeito (aleatória) |
| Argumentos | `class_balancing True;ignore_classes [0,8]` (v1: `[8]`) |
| Codificadores | ResNet-50 (U-Net, DeepLabV3, PSPNet); ViT-B (SAMLoRA) |

A lista em `ignore_classes` tem de ser escrita **sem espaços** (`[0,8]`): a ferramenta separa os argumentos por espaços e `[0, 8]` não é lido corretamente.

## Distribuição das classes no teste

Total das 3 ortofotos de teste (113,6 milhões de píxeis avaliados, 28,4 km²), sem Ignorar.

| Classe | Área (ha) | % | 126-I-3 | 126-II-4 | 126-IV-3 |
|---|---|---|---|---|---|
| Vinha | 890,3 | 31,3 | 61,6 % | 30,0 % | 8,4 % |
| Pomar e vinha | 312,8 | 11,0 | 12,1 % | 86,7 % | 1,2 % |
| Pomar | 28,5 | 1,0 | 23,7 % | 76,3 % | 0,0 % |
| Arvoredo denso | 470,8 | 16,6 | 25,0 % | 11,9 % | 63,0 % |
| Arvoredo esparso | 70,3 | 2,5 | 9,4 % | 0,0 % | 90,6 % |
| Mato denso | 279,2 | 9,8 | 2,0 % | 44,0 % | 53,9 % |
| Água | 109,7 | 3,9 | 40,6 % | 58,0 % | 1,4 % |
| Sem representação | 678,4 | 23,9 | 24,2 % | 21,9 % | 53,9 % |

As colunas das ortofotos indicam onde está a área de cada classe. O Pomar e o Arvoredo esparso são raros e estão quase todos numa só ortofoto, o que torna o seu F1 pouco estável.

## Resultados principais

Total das três ortofotos de teste, agrupamento "completa".

### Modelos individuais

| Modelo | Bandas | TTA | OA | Kappa | mIoU | fwIoU | F1 macro |
|---|---|---|---|---|---|---|---|
| SAMLoRA ViT-B (v1) | 4 | não | 0,775 | 0,715 | 0,538 | 0,650 | 0,656 |
| | 4 | sim | 0,780 | 0,721 | 0,541 | 0,655 | 0,657 |
| | **5** | não | 0,786 | 0,727 | 0,542 | 0,658 | 0,658 |
| | **5** | **sim** | **0,789** | **0,731** | **0,544** | **0,661** | 0,660 |
| DeepLabV3 ResNet-50 | 4 | não | 0,766 | 0,706 | 0,524 | 0,633 | 0,648 |
| | 4 | sim | 0,772 | 0,715 | 0,532 | 0,641 | 0,655 |
| | 5 | não | 0,759 | 0,699 | 0,515 | 0,632 | 0,636 |
| | 5 | sim | 0,766 | 0,707 | 0,521 | 0,640 | 0,642 |
| U-Net ResNet-50 | 4 | não | 0,740 | 0,679 | 0,518 | 0,611 | 0,652 |
| | 4 | sim | 0,751 | 0,693 | 0,532 | 0,624 | **0,666** |
| PSPNet ResNet-50 | 4 | não | 0,723 | 0,660 | 0,499 | 0,586 | 0,640 |
| | 4 | sim | 0,728 | 0,667 | 0,505 | 0,593 | 0,645 |

F1 por classe (sem TTA):

| Modelo | Vinha | Pomar e vinha | Pomar | Arv. denso | Arv. esparso | Mato denso | Água | Sem repr. |
|---|---|---|---|---|---|---|---|---|
| SAMLoRA v1 | 0,880 | 0,665 | 0,185 | 0,780 | 0,299 | 0,711 | 0,966 | 0,762 |
| SAMLoRA nDSM | 0,879 | 0,651 | 0,239 | 0,795 | 0,233 | 0,717 | 0,969 | 0,782 |
| DeepLabV3 v2 | 0,863 | 0,673 | 0,225 | 0,787 | 0,263 | 0,703 | 0,933 | 0,738 |
| DeepLabV3 nDSM | 0,871 | 0,653 | 0,151 | 0,794 | 0,286 | 0,662 | 0,931 | 0,742 |
| U-Net v2 | 0,862 | 0,628 | 0,320 | 0,762 | 0,334 | 0,660 | 0,934 | 0,713 |
| PSPNet v2 | 0,854 | 0,639 | 0,308 | 0,768 | 0,381 | 0,639 | 0,882 | 0,645 |

### Ensemble (4 modelos de 4 bandas, sem TTA)

| Método | OA | Kappa | mIoU | F1 macro | F1 Pomar | F1 Arv. esparso |
|---|---|---|---|---|---|---|
| **Maioria + especialista** | 0,792 | 0,739 | **0,573** | **0,702** | 0,397 | 0,396 |
| Especialista | 0,788 | 0,731 | 0,570 | 0,698 | 0,395 | 0,394 |
| Maioria | **0,795** | **0,743** | 0,563 | 0,687 | 0,291 | 0,369 |
| Ponderado | 0,793 | 0,738 | 0,553 | 0,680 | 0,291 | 0,351 |

Concordância entre os 4 modelos (exatidão da maioria em cada nível):

| Modelos de acordo | % dos píxeis | OA |
|---|---|---|
| 4 de 4 | 61,8 | 0,913 |
| 3 de 4 | 22,5 | 0,681 |
| 2 de 4 | 15,0 | 0,502 |
| 1 de 4 | 0,7 | 0,352 |

> Valores de 17/11/2026, com a regra que impede votos nas classes 0 e 8. Face à versão anterior, só mudou o especialista (OA 0,770 → 0,788, mIoU 0,562 → 0,570), que deixou de manter as previsões de Ignorar do SAMLoRA; a maioria e a maioria + especialista ficaram praticamente iguais.

### Conclusões

- O **SAMLoRA** foi o melhor modelo individual em OA e mIoU; a **U-Net** teve o melhor F1 macro com TTA, por ser a melhor nas classes raras (Pomar, Arvoredo esparso).
- O **ensemble maioria + especialista** foi o melhor resultado global: +3,5 pontos de mIoU e +4,6 de F1 macro face ao melhor modelo individual sem TTA, sobretudo no Pomar e no Arvoredo esparso.
- A **concordância** é uma boa medida de confiança: onde os 4 modelos concordam (62 % da área) a exatidão é de 91 %; as zonas com 2 ou menos modelos de acordo (16 %) concentram a revisão na edição.
- O **TTA** melhorou as métricas globais de todos os modelos (+0,2 a +1,4 pontos de mIoU), sem alterar a ordem entre arquiteturas, com 3 a 4 vezes o tempo de inferência. Por classe nem sempre melhora (o Pomar desceu nos dois SAMLoRA).
- O **nDSM** teve um efeito pequeno e dependente da arquitetura: ligeiramente positivo no SAMLoRA (+1,1 pontos de OA, +0,4 de mIoU), ligeiramente negativo no DeepLabV3 (−0,9 de mIoU). Não ajudou onde a altura seria mais útil (Arvoredo esparso, Mato × Arvoredo), o que é coerente com os 12 anos entre a ortofotografia (2013) e o nDSM (2025).
- A **generalização** de Pomar e vinha com a Vinha aumenta a OA em 3–5 pontos (F1 Vinha + Pomar e vinha entre 0,90 e 0,92), por serem as classes mais confundidas entre si.

## Notas para a interpretação

- **SAMLoRA e a classe Ignorar**: o SAMLoRA prevê a classe 8 em 1,3–2,3 % dos píxeis avaliados, apesar de `ignore_classes`; as redes ResNet nunca o fazem. Esses píxeis contam como erro (coluna `pct_previsoes_fora`). No ensemble, o 8 não recebe votos.
- **Métricas por ortofoto**: o mIoU, a exatidão média e o F1 macro de cada ortofoto só incluem as classes com referência nessa ortofoto (ex.: não há Pomar na 126-IV-3); no total entram todas.
- **Validação otimista**: os chips de validação são escolhidos ao acaso e sobrepõem-se aos de treino (passo de 50 %). As métricas de validação (`model_metrics.html`) servem para a paragem antecipada e para os pesos do ensemble ponderado; a avaliação é a do teste.
- **Diferenças pequenas**: com 3 ortofotos de teste, diferenças abaixo de ~1 ponto de mIoU (TTA, nDSM) devem ser lidas como tendências.
- **Datas**: rótulos de 2018 sobre ortofotografia de 2013; pode haver diferenças pontuais entre a imagem e a referência.

## Ficheiros de resultados

| Ficheiro | Conteúdo |
|---|---|
| `tabela_comparativa.txt` | Resumo por modelo (total das ortofotos), agrupamentos completa e generalizada, F1 por classe |
| `metricas_globais.csv` | Métricas globais por modelo, agrupamento e ortofoto |
| `metricas_classe.csv` | Precisão, sensibilidade, F1, IoU, suporte, TP, FP e FN por classe |
| `log_avaliacao.txt` | Registo completo da avaliação (modelo usado, bandas, processador, tempos) |
| `tabela_comparativa_ensemble.txt` | Resumo dos ensembles e da concordância |
| `metricas_globais_ensemble.csv`, `metricas_classe_ensemble.csv` | Métricas dos ensembles e dos modelos individuais, com o mesmo código |
| `concordancia_ensemble.csv` | % de píxeis e exatidão por nível de concordância, por ortofoto e acumulada |
| `log_ensemble.txt` | Registo do ensemble |

Nas pastas `*_TTA`, os modelos de 4 bandas têm o sufixo `_tta`; os de 5 bandas mantêm o nome (é a pasta que indica a TTA).

## Requisitos

- ArcGIS Pro 3.6 com Image Analyst e Spatial Analyst.
- Bibliotecas de deep learning do ArcGIS Pro (PyTorch, fastai, `arcgis.learn`).
- GPU NVIDIA (usada uma RTX A4000 de 16 GB). O script 30 consulta a memória livre da GPU (`nvidia-smi`) e passa ao CPU se não houver `MEM_GPU_MIN_GB` livres; a opção `--cpu` força o CPU.

## Utilização

Os caminhos e parâmetros estão na secção `CONFIGURACAO` (ou `PARÂMETROS`) no início de cada script. Correr no *Python Command Prompt* do ArcGIS Pro, na pasta do script, com o ArcGIS Pro fechado nos scripts que usam a GPU:

```
python 30_avaliar_ocupacao.py
python 30_avaliar_ocupacao.py --cpu
```

- **Tempos de referência** (RTX A4000): treino de 21 a 32 h por modelo; classificação de uma ortofoto de teste em 2,5–4 min sem TTA e 7,5–15 min com TTA; ensemble em ~2 min (só CPU).
- O script 30 tem um `MODO_TESTE` (um quadrado de 500 m) para medir o tempo antes de correr tudo.
- No 30, manter na lista `MODELOS` todos os modelos a incluir na tabela: as classificações já feitas são reutilizadas.

## Dados

Os dados (ortofotografia de 2013, shapefiles da edição cartográfica de 2018 e MDS/MDT da DGT de 2025) não estão incluídos no repositório.
