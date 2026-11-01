# 04 · Associação de objetos, duplicatas, reidentificação limitada e oclusão

| Problema | O que o sistema faz | O que **não** resolve |
|---|---|---|
| **Detecção duplicada** na sobreposição de tiles | Descarta caixas coladas à borda interna do tile + NMS (IoU e IoS) | Duas caixas legítimas muito próximas (animais encostados) podem ser fundidas |
| **Mesmo animal, dois IDs** (fragmentação) | 2ª etapa do ByteTrack; `track_buffer`; **costura limitada de tracklets** (`tracking/stitch.py`): une fim de uma trilha ao início de outra se o intervalo é curto e posição extrapolada/tamanho são compatíveis | Reaparecimento após oclusão longa ou fora do quadro: **não há reidentificação por aparência** (bovinos são visualmente parecidos; foi uma decisão consciente) |
| **Troca de ID** (dois animais se cruzam) | Kalman + IoU reduz, não elimina | Animais muito próximos e parecidos continuam sendo o pior caso |
| **Oclusão** (árvore, sombra) | A confiança baixa é aproveitada na 2ª etapa; trilhas perdidas sobrevivem ao buffer | Animais ocultos durante todo o vídeo simplesmente não existem para o sistema |
| **Jitter na linha** | Zona morta (histerese) + *cooldown* por ID e linha | Linha mal posicionada continua gerando contagem errada |

A costura é **opcional** e **sempre sinalizada** no resultado: ela pode unir animais diferentes ou deixar de unir o mesmo animal. Os IDs unidos aparecem como `stitching.ids_unidos`.
