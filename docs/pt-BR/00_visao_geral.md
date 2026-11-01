# 00 · Visão geral

O **CattleCount AI** detecta bovinos em imagens aéreas e conta animais em vídeo. Ele responde a **três perguntas diferentes**, e a regra central do projeto é nunca confundi-las:

| Grandeza | Pergunta | Como é obtida | O que NÃO é |
|---|---|---|---|
| **Visíveis agora** (contagem instantânea) | Quantos animais aparecem neste quadro/imagem? | Nº de detecções com score ≥ limiar escolhido | Não inclui animais ocultos (árvores, sombra, oclusão) nem fora do enquadramento |
| **Passagens** (contagem acumulada) | Quantas vezes animais cruzaram a linha / entraram ou saíram da região? | Eventos sobre trilhas ByteTrack (cruzamento com zona morta, direção, debounce) | Não é nº de animais: o mesmo animal pode cruzar várias vezes, e uma troca de ID pode gerar passagem a mais ou a menos |
| **Referência de rebanho** | Qual é o tamanho do rebanho? | **Não calculada como número único.** Só *faixas de referência*: piso observado simultâneo e IDs únicos filtrados/brutos | Nunca é população. IDs únicos inflam com trocas/fragmentação e deflacionam com animais nunca vistos |

## Dois modos

1. **Imagem** (`POST /infer/image`): conta **visíveis**. Imagens grandes (> 1280 px) são analisadas por fatias de 1024 px na resolução nativa (ver [01_deteccao](01_deteccao.md)).
2. **Vídeo** (`POST /videos`, assíncrono): detecção por quadro → ByteTrack → trilhas persistidas → contagem de **passagens** por linha/região. As trilhas ficam salvas: o usuário pode **mudar a linha e recontar sem refazer a detecção** (`POST /analyses/{id}/count`); cada recontagem é registrada para auditoria.

## O que foi e o que não foi validado

Ver [10_limitacoes_operacionais](10_limitacoes_operacionais.md) e [12_resultados](12_resultados.md). Em resumo: o detector foi avaliado em fazenda **inédita** (e o desempenho cai muito); o rastreador foi validado em vídeo terrestre real e em vídeo aéreo **sintético**; **não existe** verdade de rastreamento em vídeo aéreo real neste projeto.
