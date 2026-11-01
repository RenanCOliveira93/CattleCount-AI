# 03 · Rastreamento (ByteTrack)

Implementação própria (`cattlecount/tracking/`) do ByteTrack (Zhang et al., ECCV 2022):

1. **Filtro de Kalman** (estado: centro, razão de aspecto, altura e velocidades) prevê onde cada trilha estará no próximo quadro.
2. **Associação em duas etapas** (Húngaro sobre custo 1 − IoU): (a) detecções de **confiança alta** × trilhas ativas e perdidas; (b) detecções de **confiança baixa** × trilhas que sobraram. A 2ª etapa recupera animais parcialmente ocultos, cuja confiança cai na oclusão, em vez de descartá-los.
3. **Trilhas novas** só nascem de detecção com score ≥ `new_track_thresh`; trilhas sem associação ficam *perdidas* por `track_buffer` quadros e podem ser recuperadas.
4. **Compensação de movimento de câmera** (opcional): movimento global estimado por fluxo óptico Lucas‑Kanade + RANSAC (afim parcial) e aplicado às previsões do Kalman. Ajuda quando a câmera (drone) se move; **não** é necessária com câmera fixa.

**Por que próprio?** Evita a dependência `boxmot` (conflito de versões) e mantém o rastreador fora da licença AGPL. Foi **verificado contra o `BYTETracker` do Ultralytics** nas mesmas detecções (ver [12_resultados](12_resultados.md)).

## Persistência e reuso
As trilhas de um vídeo são salvas em `tracks.csv`. Linha, região, zona morta e costura podem ser reconfiguradas e **recontadas sem nova detecção**.
