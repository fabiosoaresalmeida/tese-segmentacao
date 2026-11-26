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

Os scripts confirmam esta equivalência a partir do `parametros.json` do registo de treino de cada MTRE de 4 bandas (pasta `<modelo>_registo`):

- o **42** compara, no fim da exportação, o número de chips com o do modelo de 4 bandas;
- o **43** compara, antes do treino, o número de chips e os parâmetros (épocas, lote, chip, argumentos, validação e monitor), avisa se houver diferenças e grava o modelo equivalente no `parametros.json` do novo treino.

## Ordem de execução

| Passo | Script | O que faz | Saída |
|---|---|---|---|
| 1 | `MTRE_nDSM/42_Export_RViaria_ndsm.py` | Verifica o ortomosaico de 5 bandas contra o `126.tif` e exporta os chips dos dois conjuntos (os já exportados são saltados) | `Training\rede_viaria_chip512_binario_ndsm`, `Training\ferrovia_chip512_binario_ndsm` |
| 2 | `MTRE_nDSM/43_Treino_RViaria_ndsm.py` com `MODO_TESTE = True` | 1 época: confirma que o MTRE aceita 5 bandas e mede o tempo por época | modelo de teste |
| 3 | `MTRE_nDSM/43_Treino_RViaria_ndsm.py` com `MODO_TESTE = False` | Treino real dos modelos com `ativo=True` | `D:\DL\Modelos\126_RViaria_nDSM\mtre_*_ndsm_resnet50_b8_lrauto_treino_*` |
| 4 | `44_avaliar_RViaria.py` | Avalia os MTRE de 4 bandas e com nDSM (rede viária e ferrovia) e compara-os | CSV em `D:\DL\Avaliacao\126_RViaria`, incluindo `comparacao_ndsm_<corrida>.csv` |

O teste da rede viária (26/11/2026) confirmou que o MTRE aceita as 5 bandas; a ferrovia pode ir diretamente para o treino real.

Para a ferrovia começar sozinha quando o treino da rede viária terminar, correr `MTRE_nDSM/99_esperar_e_correr_ferrovia.py` numa segunda janela do Python Command Prompt. Deteta o fim do treino pelo `resumo_treinos.csv` (que o 43 só cria ou atualiza no fim) e corre a seguir o 43, que treina a ferrovia; os chips da ferrovia já têm de estar exportados pelo 42. Antes de esperar, confirma que o 43 tem a ferrovia com `ativo=True`, a rede viária com `ativo=False` e `MODO_TESTE = False`.

## Avaliação

No 44, para os modelos com nDSM, a entrada da inferência é um recorte do ortomosaico de 5 bandas com a extensão e a grelha exatas de cada ortofoto de teste. O recorte é criado uma vez em `D:\DL\Avaliacao\126_RViaria\ortos_5bandas` e serve aos dois modelos. O resto do processo é igual: espelhamento das margens, três versões de vetorização e as mesmas métricas.

No fim, para cada par de `COMPARAR_NDSM` (modelo de 4 bandas, modelo com nDSM) avaliado na corrida, o 44 grava `comparacao_ndsm_<corrida>.csv` com a completeness, a correctness e a quality dos dois modelos (TOTAL das 3 ortos) e a diferença nDSM − 4 bandas, por versão de vetorização e tolerância. Os valores com 3 m de tolerância aparecem também no ecrã.

`AVALIAR` escolhe os modelos da corrida. As inferências dos MTRE de 4 bandas já feitas são reutilizadas, e um modelo com nDSM ainda não treinado é saltado com um aviso.
