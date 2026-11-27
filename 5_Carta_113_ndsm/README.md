# 5_Carta_113_ndsm

Avaliação na folha 113 dos modelos treinados na 126 **com nDSM** (5 bandas: R, G, B, NIR, nDSM),
com o ortomosaico `113_nDSM_5bandas.tif`. São cópias dos scripts de `5_Carta_113` em que só muda a
configuração (ortomosaico, modelos e pasta de avaliação), para comparar com a avaliação de 4 bandas.

| Script | Base | Modelos | Resultados |
|---|---|---|---|
| `1n_avaliar_construcoes_113_ndsm.py` | `1_avaliar_construcoes_113.py` | `D:\DL\Modelos\126_construcoes_ndsm` | — |
| `2n_avaliar_rviaria_113_ndsm.py` | `2_avaliar_rviaria_113.py` | MTRE rede viária e ferrovia de `D:\DL\Modelos\126_RViaria_nDSM` (sem U-Net) | `D:\DL\Avaliacao\113_RViaria_nDSM` |
| `3n_avaliar_ocupacao_113_ndsm.py` | `3b_avaliar_ocupacao_113.py` | DeepLabV3 e SAMLoRA de `D:\DL\Modelos\126_ocupacao_nDSM`, com TTA | `D:\DL\Avaliacao\113_ocupacao_nDSM_TTA` |
| `3c_agregacoes_ocupacao_113.py` | — | agregações das matrizes do 3n | — |
| `9n_cadeia_113_ndsm.py` | `9_cadeia_113.py` | corre 1n, 2n, 3n e 3c sem vigilância | `log_cadeia_113_ndsm.txt` |

Notas:
- Os scripts precisam do `comum_113.py` (em `5_Carta_113`): no PC correm na mesma pasta que ele.
- O 2n usa o `medir()` corrigido do `44_avaliar_RViaria.py` (recorte robusto das linhas extraídas),
  com que foram avaliados os MTRE com nDSM na 126; o `2_avaliar_rviaria_113.py` ainda tem a versão antiga.
- O 9n verifica, antes de arrancar, que o ortomosaico tem 5 bandas e que as pastas dos modelos MTRE têm o `.emd`.
