# Rede viária — extração com deep learning (folha 126, Sertã)

Scripts, resultados e figuras da extração da rede viária e da ferrovia a partir das
ortofotos da folha 126 (Sertã), com redes neuronais convolucionais no ArcGIS Pro 3.6.
São comparados dois modelos: o **Multi-Task Road Extractor** (MTRE) e o **U-Net**,
ambos com backbone ResNet-50.

## Estrutura

```
2_RedeViaria/
├── 40a_Sondagem_RViaria.py        preparação dos dados
├── 40b_Area_Treino_Teste.py
├── 41_Classificar_RViaria.py
├── 41b_Binario_RViaria.py
├── MTRE/
│   ├── 42_Export_RViaria.py       exportação e treino do Multi-Task Road Extractor
│   └── 43_Treino_RViaria.py
├── MTRE_nDSM/                     MTRE com o ortomosaico de 5 bandas (R, G, B, NIR, nDSM)
│   ├── 42_Export_RViaria_ndsm.py  ver MTRE_nDSM/README.md
│   ├── 43_Treino_RViaria_ndsm.py
│   ├── 99_esperar_e_correr_ferrovia.py
│   └── README.md
├── MTRE_RGB/                      MTRE só com o ortomosaico RGB (3 bandas)
│   ├── 42_Export_RViaria_rgb.py   ver MTRE_RGB/README.md
│   ├── 43_Treino_RViaria_rgb.py
│   └── README.md
├── unet/
│   ├── 42_Export_RViaria_unet.py  exportação e treino do U-Net
│   ├── 43_Treino_RViaria.py         (1.ª tentativa, LR automático: não aprendeu)
│   └── 43_Treino_RViaria_LR_fixo.py (versão usada: LR do LR Finder limitado)
├── 44_avaliar_RViaria.py          avaliação de todos os modelos
├── 45_figuras_RViaria.py          figuras de comparação visual
├── 46_uniao_MTRE_Unet_RViaria.py  avaliação da união MTRE + U-Net
├── Resultados/                    CSV e parâmetros das corridas oficiais
└── figuras/                       figuras geradas pelo 45
```

## Ordem de execução

| Passo | Script | O que faz | Cria |
|---|---|---|---|
| 1 | `40a_Sondagem_RViaria.py` | Projeta os eixos (Datum Lisboa → sistema das ortos, `Lisbon_Lisbon_To_WGS_1984_2`), mede os km por classe em cada orto e ordena as combinações de 3 ortos de teste pela diversidade | `l_vias_IGeoE.shp`, `folhas_ortos.shp`, `sondagem_vias.csv` |
| 2 | `40b_Area_Treino_Teste.py` | Separa as áreas de treino e de teste | `area_treino.shp`, `area_teste.shp` |
| 3 | `41_Classificar_RViaria.py` | Etiquetas de 3 classes: buffer dos eixos com a largura de cada tipo de via, recortado à área de treino | `etiquetas_vias.shp` |
| 4 | `41b_Binario_RViaria.py` | Etiquetas binárias (rede viária; ferrovia) | `etiquetas_rede_viaria_binario.shp`, `etiquetas_ferrovia_binario.shp` |
| 5 | `MTRE/42_Export_RViaria.py` | Chips 512 × 512 (stride 256) para o MTRE | `rede_viaria_chip512_binario`, `ferrovia_chip512_binario` |
| 6 | `MTRE/43_Treino_RViaria.py` | Treino dos dois MTRE | modelos `mtre_rede_viaria_…`, `mtre_ferrovia_…` |
| 7 | `unet/42_Export_RViaria_unet.py` | Chips 256 × 256 (stride 128) para o U-Net (só CPU) | `rede_viaria_chip256_unet_binario` |
| 8 | `unet/43_Treino_RViaria_LR_fixo.py` | Treino do U-Net binário | modelo `unet_resnet50_rede_binario_…` |
| 9 | `44_avaliar_RViaria.py` | Inferência nas ortos de teste, vetorização em 3 versões e métricas | `Resultados/*_<corrida>.csv` |
| 10 | `45_figuras_RViaria.py` | Figuras A, B, B2 e C | `figuras/*.png` |
| 11 | `46_uniao_MTRE_Unet_RViaria.py` | União das máscaras do MTRE e do U-Net, avaliada como no 44 | `Resultados/uniao_*_<corrida>.csv` |

Os passos 5–6 e 7–8 são independentes. Os caminhos (`D:\...`) estão na secção
CONFIGURAÇÃO no início de cada script.

## Dados e classes

- **Imagem:** 16 ortofotos (0,5 m) da folha 126, em WGS84 militar IGeoE.
- **Referência:** eixos da restituição IGeoE (`l_vias.shp`, campo `source`), em Datum Lisboa.
- **Ortos de teste:** 126-I-2, 126-III-4 e 126-IV-1. A combinação foi escolhida no 40a
  (2.ª de 560 combinações válidas). As restantes 13 ortos são de treino.
- **Classes de treino:** 1 Via pavimentada (autoestrada, estradas, arruamentos),
  2 Via não pavimentada (acessos, caminhos carreteiros e de pé posto), 3 Ferrovia.
- **Largura das etiquetas:** largura típica de cada tipo de via menos 2 × 0,75 m (desvio
  do eixo), com o mínimo de 1 m. Na autoestrada há um eixo por sentido, com 11 m cada.
- **Modelos:** o MTRE só aceita duas classes. Por isso há dois MTRE (rede viária;
  ferrovia) e um U-Net binário (rede viária). O U-Net multiclasse não aprendeu em
  3 tentativas e foi abandonado.

## Avaliação (44 e 46)

Para evitar o ruído nas margens, a inferência é feita sobre a orto com 32 m espelhados
à volta e depois recortada à extensão original. O padding da inferência é 1/4 do chip
de cada modelo. Cada máscara prevista é vetorizada em três versões:

| Versão | Processo |
|---|---|
| `bruto` | máscara → Thin → Raster to Polyline → remoção dos troços < 10 m |
| `pos` | limpeza do raster (fase A) → Thin → Raster to Polyline → fase B |
| `centerline` | fase A → polígonos → Simplify Polygon → Polygon to Centerline → Trim Line → fase B |

- **Fase A (raster):**
  - média focal 3 × 3;
  - abertura de 1 píxel;
  - remoção das manchas com menos de 50 m²;
  - preenchimento dos buracos até 8 m².
- **Fase B (vetor):**
  - remoção das travessas entre faixas paralelas (≤ 20 m, ângulo ≥ 30°);
  - Smooth Line PAEK com 12 m;
  - Extend Line até 20 m;
  - remoção dos troços com menos de 10 m.

Métricas, calculadas por orto e no TOTAL (soma das contagens e dos comprimentos):

- **por píxel:** matriz de confusão, OA, Kappa, F1, IoU e mIoU, contra o buffer dos eixos
  de referência (`matriz_confusao_*.csv`);
- **por píxel com tolerância:** precisão e recall relaxados (`pixel_tolerancia_*.csv`);
- **sobre os eixos (Wiedemann):** completeness, correctness e quality, com tolerâncias de
  2, 3 e 5 m (`metricas_eixos_*.csv`);
- **completeness por classe de origem** (`completeness_classes_*.csv`).

A versão `centerline` precisa da licença do Polygon to Centerline. O script pede-a com
`CheckOutExtension`.

## Resultados

Corridas oficiais:

- `Resultados/*_20261103_101943.*`: 44, com os três modelos;
- `Resultados/uniao_*_20261103_150702.*`: 46, com o MTRE, o U-Net e a união.

Na corrida da união, o MTRE e o U-Net têm exatamente os mesmos valores da corrida do 44.

Eixos, TOTAL das 3 ortos de teste, tolerância de 3 m:

| Modelo | Versão | Completeness | Correctness | Quality |
|---|---|---|---|---|
| MTRE rede viária | bruto | 0,617 | 0,829 | 0,545 |
| MTRE rede viária | pos | 0,611 | 0,835 | 0,540 |
| MTRE rede viária | centerline | 0,598 | 0,828 | 0,528 |
| U-Net rede viária | bruto | 0,209 | 0,936 | 0,199 |
| U-Net rede viária | pos | 0,198 | 0,933 | 0,187 |
| U-Net rede viária | centerline | 0,197 | 0,929 | 0,187 |
| União MTRE + U-Net | bruto | 0,618 | 0,829 | 0,546 |
| União MTRE + U-Net | pos | 0,613 | 0,834 | 0,541 |
| União MTRE + U-Net | centerline | 0,594 | 0,832 | 0,526 |
| MTRE ferrovia | bruto | 0,564 | 0,077 | 0,068 |
| MTRE ferrovia | pos | 0,559 | 0,069 | 0,061 |
| MTRE ferrovia | centerline | 0,572 | 0,152 | 0,137 |

## Figuras (45)

As figuras são feitas com a corrida `20261103_101943` e os eixos da versão `bruto`, com
uma tolerância de 3 m. Há cinco recortes: zona urbana, zona mista, zona dispersa,
autoestrada e ferrovia.

- `fig_A_<recorte>`: do modelo ao resultado final. Mostra a referência, a máscara prevista,
  os erros da área e os eixos extraídos (correto / falso positivo / omitido).
- `fig_B_rede_<recorte>`: a mesma comparação numa área duas vezes maior.
- `fig_B2_classes_<recorte>`: igual, sem os falsos positivos e com as vias omitidas
  coloridas pela classe.
- `fig_C_etapas_<modelo>_<recorte>`: os eixos bruto, pos e centerline sobre a referência.
  A ciano está o que cada versão removeu em relação ao bruto.

Cores: amarelo = referência, magenta = previsão, verde = correto, vermelho = falso
positivo, azul = omitido.

## Requisitos

- ArcGIS Pro 3.6 com Image Analyst e Spatial Analyst. A versão centerline precisa também
  da licença do Polygon to Centerline (Topographic Production).
- Ambiente `arcgispro-py3` com as bibliotecas de deep learning (testado com Python 3.13.7,
  PyTorch 2.5.1, CUDA 12.6 e ArcGIS API for Python 2.4.2).
- GPU usada: NVIDIA RTX A4000.
- Os scripts de treino e de avaliação devem ser corridos na linha de comandos
  (`arcgispro-py3`).
