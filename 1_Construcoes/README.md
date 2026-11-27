# Extração de edifícios – folha 126 (Peso da Régua)

Scripts, resultados e figuras da extração automática de edifícios por segmentação semântica (deep learning) no ArcGIS Pro 3.6, para a produção da carta topográfica 1:25 000.

Foram comparadas três arquiteturas (U-Net, DeepLabV3 e SAMLoRA) com três combinações de bandas:

| Fase | Pasta | Imagem de entrada | Modelos |
|---|---|---|---|
| **A** | `1_4bandas_NIR` | Ortofotografia com 4 bandas (R, G, B, NIR), 0,5 m | 9 (U-Net ResNet-34/50/101/152, DeepLabV3 ResNet-34/50/101, SAMLoRA ViT-B/L) |
| **B** | `2_5bandas_nDSM` | Ortofotografia + nDSM LiDAR (5 bandas) | 4 (U-Net ResNet-50 e 152, DeepLabV3 ResNet-101, SAMLoRA ViT-L) |
| **C** | `3_3bandas_RGB` | Só R, G, B (3 bandas) | os 4 da fase B (em curso) |

Em todas as fases, os vetores de edifícios são usados na **posição original** da restituição (ao nível do telhado), e a avaliação é feita contra essa posição. Os chips (16 508, 256 × 256 px, passo 128) são os mesmos nas três fases; só mudam as bandas.

## Estrutura

```
1_Construcoes/
├── 0_Preparacao/                    comum às três fases
│   ├── 03_Limpar_Constr_Projt.py    limpeza e projeção dos edifícios (+ campo classe = 1)
│   └── 06_separar_treino_teste.py   separação treino/teste, máscara e zona de teste
├── 1_4bandas_NIR/                   fase A
│   ├── 07_exportar_chips.py         chips de treino (4 bandas)
│   ├── 08_treinar_modelos.py        treino
│   └── resultados/126-4bandas/      avaliação da fase A
├── 2_5bandas_nDSM/                  fase B
│   ├── 07_exportar_chips_nDSM.py    chips de treino (5 bandas)
│   ├── 08_treinar_modelos_nDSM.py   treino (com SALTAR_CONCLUIDOS)
│   ├── 11_teste_ndsm_zero.py        teste de utilização da banda nDSM
│   └── resultados/126-5bandas/      avaliação da fase B
├── 3_3bandas_RGB/                   fase C
│   ├── comum_RGB.py                 configuração da fase C (pastas, épocas, paragem, modelos)
│   ├── 07_exportar_chips_RGB.py     ortomosaico RGB e chips (usa o 07 da fase A)
│   ├── 08_treinar_modelos_RGB.py    treino dos 4 modelos (usa o 08 da fase B)
│   └── 09_avaliar_modelos_RGB.py    avaliação (usa o 09)
└── 4_Avaliacao_Figuras/             comum às três fases
    ├── 09_avaliar_modelos.py        avaliação na zona de teste (píxel e objeto), por grupos
    ├── 10_figuras.py                mapas de erros e comparação fase A × fase B
    ├── 11_figuras_2.py              figuras da tese com arcpy (2, 3, 4, 5, 7, 13)
    ├── 12_figuras_sem_arcpy.py      figuras da tese sem arcpy (1, 6, 8, 9, 12)
    ├── 13_comparar_fases.py         tabela A × B × C (lê os CSV do 09)
    └── figuras/                     figuras geradas pelo 10_figuras.py
```

Os scripts da fase C não copiam código: carregam o 07, o 08 e o 09 originais (procurados nas subpastas de `1_Construcoes`), mudam só a configuração e chamam o `main()` de cada um. Assim os chips, o treino e as métricas são exatamente os das fases A e B.

## Cadeia de processamento

| N.º | Script | Entradas | Saídas |
|---|---|---|---|
| 03 | `03_Limpar_Constr_Projt.py` | `a_constr.shp`, ortomosaico | `a_constr_Limpa.shp` |
| 06 | `06_separar_treino_teste.py` | `a_constr_Limpa.shp`, `ortos_grelha.shp` | `zona_teste.shp`, `mascara_treino.shp`, `a_constr_treino.shp`, `a_constr_teste.shp`, `a_constr_margens_teste.shp` |
| 07 | `07_exportar_chips[_nDSM\|_RGB].py` | ortomosaico, `a_constr_treino.shp`, `mascara_treino.shp` | chips 256 × 256 px (passo 128), *Classified Tiles* |
| 08 | `08_treinar_modelos[_nDSM\|_RGB].py` | chips | modelos (`.dlpk`), `log_treinos.txt`, `resumo_treinos.csv`, `epocas_treinos.csv` |
| 09 | `09_avaliar_modelos[_RGB].py` | modelos, ortomosaico, `a_constr_Limpa.shp`, `zona_teste.shp` | `metricas_*.csv`, `tabela_comparativa.txt`, classificações, polígonos regularizados |
| 10–12 | `10_figuras.py`, `11_figuras_2.py`, `12_figuras_sem_arcpy.py` | pastas de avaliação, logs de treino, vetores | figuras PNG |
| 11 | `11_teste_ndsm_zero.py` | um modelo da fase B e a sua avaliação | comparação com o nDSM substituído por zeros |
| 13 | `13_comparar_fases.py` | `metricas_*.csv` das três fases | `comparacao_fases_ABC.txt` / `.csv` |

### Descrição dos passos

1. **Limpeza (03)** – projeta os edifícios para o sistema do ortomosaico (se necessário), elimina os polígonos fora do ortomosaico e os que não são edifícios visíveis (campos de jogos, limites, piscinas, infraestruturas elétricas). As ruínas são mantidas.
2. **Treino/teste (06)** – reserva três ortofotos para teste (126-I-3 urbana, 126-II-3 mista, 126-IV-1 dispersa) e usa as restantes 13 no treino. Os edifícios que atravessam o limite (margens) ficam no treino; a máscara impede que os chips entrem na zona de teste.
3. **Chips (07)** – 256 × 256 px a 0,5 m, passo de 128 px, só chips com edifícios; verifica que nenhum chip interseta a zona de teste.
4. **Treino (08)** – parâmetros iguais para todos os modelos: lote 8, 10 % de validação, taxa de aprendizagem fixa de 2 × 10⁻⁴, codificador pré-treinado e congelado, `class_balancing`. Fases A e B: 20 épocas sem paragem antecipada. Fase C: até 50 épocas com paragem antecipada (`EPOCAS` e `PARAGEM` no `comum_RGB.py`); em todos os casos é gravado o modelo da melhor época. Os scripts das fases B e C saltam os modelos já concluídos (`SALTAR_CONCLUIDOS`).
5. **Avaliação (09)** – classifica cada ortofoto de teste com 256 m de contexto (bordas exteriores espelhadas), vectoriza, limpa (buracos < 20 m², manchas < 10 m²) e regulariza (*Regularize Building Footprint*). Calcula:
   - **ao píxel**: matriz de confusão, UA, PA, OA, kappa, F1 e IoU, para a classificação e para os polígonos regularizados;
   - **por objeto**: correspondência 1:1 com IoU ≥ 0,5, precisão, sensibilidade, F1, cobertura (≥ 50 %), fusões, IoU médio, erro de área, deslocamento do centroide, distância entre contornos e deteção por classe de área.

   A avaliação é feita contra duas referências: a edição cartográfica e a restituição fotogramétrica.
6. **Figuras (10, 11, 12)** – `10_figuras.py`: por arquitetura, melhores modelos e comparação fase A × fase B (mapas de erros: verde = correto, vermelho = falso positivo, azul = omitido). `11_figuras_2.py` (com arcpy) e `12_figuras_sem_arcpy.py` (só matplotlib): figuras da tese (fluxo de trabalho, desvio dos telhados, nDSM, divisão treino/teste, chips, perda de validação, pós-processamento, critério de deteção, métricas, deteção por área, erros típicos). `python 11_figuras_2.py 4` refaz só a figura 4.
7. **Teste do nDSM (11)** – classifica com a banda do nDSM a zeros para verificar se o modelo usa efetivamente a altura.
8. **Comparação das fases (13)** – junta os resultados das três fases numa tabela, com as diferenças A − C (efeito do NIR) e B − A (efeito do nDSM).

## Resultados principais

Total das três ortofotos de teste, referência da edição cartográfica (3415 edifícios), polígonos regularizados.

| Modelo | Fase | F1 píxel | Kappa | F1 objeto | Cobertura | IoU médio | Deslocamento (m) |
|---|---|---|---|---|---|---|---|
| U-Net ResNet-50 | A | 0,775 | 0,770 | 0,574 | 78,2 % | 0,668 | 1,89 |
| | **B** | **0,830** | **0,826** | **0,670** | 82,5 % | **0,715** | **1,49** |
| U-Net ResNet-152 | A | 0,779 | 0,774 | 0,580 | 79,9 % | 0,669 | 1,85 |
| | B | 0,828 | 0,824 | 0,669 | 81,6 % | 0,713 | 1,50 |
| DeepLabV3 ResNet-101 | A | 0,761 | 0,755 | 0,539 | 73,9 % | 0,657 | 1,98 |
| | B | 0,779 | 0,775 | 0,593 | 69,5 % | 0,680 | 1,65 |
| SAMLoRA ViT-L | A | 0,753 | 0,747 | 0,505 | 82,4 % | 0,663 | 1,98 |
| | B | 0,758 | 0,752 | 0,586 | 83,1 % | 0,695 | 1,70 |

- A **U-Net** foi a melhor arquitetura nas duas fases; o codificador teve pouca influência.
- O **nDSM** aumentou o F1 por objeto entre 5,4 e 9,6 pontos e reduziu o deslocamento dos centroides em cerca de 0,35 m em todos os modelos.
- O desvio residual (~1,5 m) resulta de a ortofotografia ter sido retificada com o MDT (telhados deslocados até ~2 m).
- As referências da edição e da fotogrametria dão resultados praticamente iguais (diferença máxima de 0,003 no F1 por objeto).

As tabelas completas estão em `1_4bandas_NIR/resultados/126-4bandas/tabela_comparativa.txt` e `2_5bandas_nDSM/resultados/126-5bandas/tabela_comparativa.txt`. Os resultados da fase C serão acrescentados em `3_3bandas_RGB/resultados/`.

### Ficheiros de resultados

| Ficheiro | Conteúdo |
|---|---|
| `tabela_comparativa.txt` | Resumo por modelo e referência (total das ortofotos) |
| `metricas_pixel.csv` | Matriz de confusão e métricas ao píxel, por modelo, ortofoto e produto |
| `metricas_objeto.csv` | Métricas por objeto (polígonos limpos e regularizados) |
| `metricas_por_area.csv` | Deteção e cobertura por classe de área (0–25, 25–50, 50–100, 100–200, > 200 m²) |
| `log_avaliacao.txt` | Registo completo da avaliação |

## Requisitos

- ArcGIS Pro 3.6 com Image Analyst, Spatial Analyst e 3D Analyst.
- Bibliotecas de deep learning do ArcGIS Pro (PyTorch, fastai, `arcgis.learn`).
- GPU NVIDIA (usada uma RTX A4000 de 16 GB). Antes dos treinos: *CUDA – Sysmem Fallback Policy* = "Prefer No Sysmem Fallback" e "Modo de Edição Rápida" desativado na janela de comandos.

## Utilização

Os caminhos e parâmetros estão na secção `CONFIGURACAO` (ou `PARÂMETROS`) no início de cada script (na fase C, no `comum_RGB.py`). Correr no *Python Command Prompt* do ArcGIS Pro, dentro da subpasta do script, com o ArcGIS Pro fechado nos scripts que usam a GPU:

```
cd D:\Users\Cap_Almeida\Projeto\Python\1_Construcoes\3_3bandas_RGB
python 08_treinar_modelos_RGB.py teste     (verificação: 1 época de cada tipo)
python 08_treinar_modelos_RGB.py
```

A pasta `3_3bandas_RGB` tem de estar ao lado das outras subpastas, porque os seus scripts carregam o 07, o 08 e o 09 das fases A e B.

Os scripts de treino e avaliação das fases A e B têm um `MODO_TESTE` para uma execução rápida de verificação.

## Dados

Os dados (ortofotografia, vetores e MDS/MDT LiDAR da DGT, 2024) não estão incluídos no repositório.
