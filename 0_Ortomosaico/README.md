# Ortomosaico – folha 126 (Peso da Régua)

Scripts de construção, verificação e análise do ortomosaico da folha 126 da Carta Militar de Portugal 1:25 000 (série M888), no ArcGIS Pro 3.6. O ortomosaico é a imagem de entrada de todos os modelos de segmentação (edifícios, rede viária e ocupação do solo), e a análise das ortofotos define as áreas de teste.

## Estrutura

```
ortomosaico/
├── 01_Cria_ortomosaico.py      fusão das 16 ortofotos num único raster
├── 02_Verifica_ortomosaico.py  verificação do mosaico e identificação da 4.ª banda
└── 05_analisar_ortos.py        estatísticas por ortofoto e sugestão das áreas de teste
```

A numeração segue a cadeia de processamento do repositório: o script 03 está em `construcoes/` e os scripts 06 e seguintes usam os resultados destes.

## Ordem de execução

| N.º | Script | Entradas | Saídas |
|---|---|---|---|
| 01 | `01_Cria_ortomosaico.py` | 16 ortofotos (`126-I-1.tif` … `126-IV-4.tif`) | `126.tif` |
| 02 | `02_Verifica_ortomosaico.py` | `126.tif`, uma ortofoto de referência | relatório na consola |
| 05 | `05_analisar_ortos.py` | ortofotos, camada de edifícios filtrada | `ortos_grelha.shp`, `ortos_grelha_relatorio.txt`, `ortos_grelha_estatisticas.csv` |

### Descrição dos passos

1. **Construção do mosaico (01)** – lista as ortofotos da pasta e imprime, para cada uma, o número de bandas, o tipo de píxel, o sistema de referência, a resolução e a extensão. Funde-as com *Mosaic To New Raster*:
   - sistema de referência herdado das próprias ortofotos (sem reprojeção);
   - primeira ortofoto como *snap raster* e célula de 0,5 m (resolução nativa);
   - 4 bandas, 8 bits sem sinal;
   - método `LAST`: como as ortofotos são contíguas e sem sobreposição, cada píxel provém de uma única ortofoto e os valores originais são preservados.
2. **Verificação (02)** – compara as propriedades do mosaico com os valores esperados (dimensões, bandas, resolução, extensão e sistema de referência igual ao das ortofotos) e marca cada uma como `OK` ou `FALHA`. Percorre depois o mosaico completo em blocos de 2000 × 2000 píxeis e calcula as estatísticas da 4.ª banda (valores distintos, média, correlação com R, G e B) e a distribuição do NDVI, para confirmar que se trata de infravermelho próximo.
3. **Análise das ortofotos (05)** – atribui cada edifício à ortofoto que contém o seu centróide e calcula, por ortofoto, o número de edifícios, a área edificada, a densidade, a área mediana e a percentagem de edifícios pequenos. Divide as ortofotos em três estratos de densidade (disperso, misto, denso) e sugere uma ortofoto de cada estrato para teste, de modo que em conjunto tenham entre 10 % e 20 % dos edifícios (alvo: 15 %).

   O critério da margem da carta é escolhido na configuração (`MODO_MARGEM`):

   | Modo | Comportamento | Quando usar |
   |---|---|---|
   | `penalizacao` | Cada ortofoto interior soma `PESO_MARGEM` (0,1) à pontuação | Favorecer áreas de teste na periferia, com menos vizinhas de treino |
   | `desempate` | A margem só decide entre combinações com a mesma pontuação | Usar apenas a proporção de edifícios e a representatividade |

## Características dos dados

| Propriedade | Valor |
|---|---|
| Ortofotos | 16, em grelha de 4 × 4 (quadrantes I–IV, subdivisões 1–4) |
| Dimensão de cada ortofoto | 4000 × 2500 m (8000 × 5000 píxeis) |
| Bandas | 4 (vermelho, verde, azul, infravermelho próximo), 8 bits sem sinal |
| Resolução | 0,5 m |
| Sistema de referência | WGS84, Transversa de Mercator, ponto central na Melriça (39° 40′ 05,7″ N; 8° 07′ 59,2″ W), fator de escala 1, falsa origem 200 000 m E / 300 000 m N (coordenadas militares IGeoE) |
| Extensão | X 216 000–232 000 m; Y 460 000–470 000 m |
| Mosaico | 32 000 × 20 000 píxeis, 16 × 10 km (160 km²) |

## Resultados da verificação (02)

| Indicador | Valor |
|---|---|
| Píxeis válidos | 639 980 931 |
| Píxeis sem informação (4 bandas a 0) | 19 069 (0,003 %) |
| Banda 4 – valores distintos / média | 256 / 119,5 |
| Banda 4 – correlação com R / G / B | 0,629 / 0,707 / 0,637 |
| NDVI – percentis 5 / 50 / 95 | −0,10 / 0,22 / 0,56 |
| NDVI – píxeis acima de 0,3 | 36,7 % |

A distribuição contínua exclui um canal alfa, as correlações moderadas excluem uma banda duplicada e o NDVI confirma que a 4.ª banda é infravermelho próximo. A banda foi mantida como entrada dos modelos.

## Requisitos

- ArcGIS Pro 3.6 (ArcPy, NumPy).
- Script 05: nenhuma extensão adicional.

## Utilização

Os caminhos e parâmetros estão no início de cada script (secção `PARÂMETROS` ou `Configuração`). Correr na janela Python do ArcGIS Pro ou no *Python Command Prompt* do ArcGIS Pro:

```
python 01_Cria_ortomosaico.py
```

O script 02 lê o mosaico por blocos, para não carregar em memória os cerca de 2,5 GB da imagem completa.

## Dados

As ortofotografias e os vetores não estão incluídos no repositório.
