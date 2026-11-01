# 10 · Limitações operacionais

## Do modelo
- Treinado só no ICAERUS (3 fazendas europeias, 60–100 m, nadir). **Não validado** para outras raças, relevos, altitudes, ângulos oblíquos, ou iluminação diferente. Em fazenda inédita o modelo **superconta** (ver 12_resultados).
- Animais **escuros** e **ocultos** (árvore, sombra) são perdidos com frequência. Animais encostados podem ser fundidos.
- Escala: o modelo foi treinado com animais de ~46 px (mediana). Altitudes muito diferentes mudam a escala e quebram o modelo.

## Do rastreamento e da contagem
- **Não há verdade de rastreamento em vídeo aéreo real** neste projeto. O rastreador foi validado em vídeo terrestre real (Mendeley) e em vídeo aéreo **sintético**.
- Com detector fraco, as trilhas fragmentam e a contagem de passagens fica abaixo da verdade; com troca de ID pode ficar acima.
- **Câmera em movimento**: linha/região são do quadro, não do terreno (ver 05_contagem).
- Sem reidentificação por aparência: um animal que sai e volta é outro ID (a costura limitada ajuda só em intervalos curtos).
- **IDs únicos ≠ população.** O sistema não estima tamanho de rebanho.

## Do sistema
- **Autenticação**: apenas chave opcional (`API_KEY`); sem usuários, sem papéis. Não exponha à internet sem proxy autenticado.
- **Escala**: 1 worker de vídeo (fila em memória: jobs em andamento se perdem se o processo cair). Não há reprocessamento automático.
- **Armazenamento**: vídeos e trilhas ficam em volume local, sem política de retenção.
- Vídeo anotado só é reencodado para H.264 com o ffmpeg do pacote `imageio-ffmpeg`; sem ele, o MP4 pode não tocar em alguns navegadores.
- **Docker Desktop não foi testado**; a pilha foi executada com **Podman 6.1.4** (ver 11_como_rodar).
- O frontend não tem testes de acessibilidade além de rótulos ARIA básicos.

## Licenças
- Pesos e código **Ultralytics: AGPL‑3.0**. Uso comercial/serviço em rede exige avaliar AGPL ou licença comercial.
- Dados: ICAERUS e Mendeley **CC BY 4.0** (atribuição obrigatória: `sample_data/ATTRIBUTION.md`).
- Código do projeto (ByteTrack próprio, contagem, API): sem componente AGPL no caminho de inferência (ONNX Runtime).

## Uso responsável
Estimativas de modelo não substituem contagem física para decisões de manejo, comerciais, sanitárias ou regulatórias.
