# 05 · Contagem: linha, região e as três grandezas

## Linha virtual
Para cada trilha, calcula‑se a distância assinada do ponto âncora (centro ou base da caixa) à linha. Há **cruzamento** quando o sinal muda, ignorando pontos dentro da **zona morta** (`deadband_px`; mantém o lado anterior, evitando contagens repetidas por jitter). O ponto de cruzamento é interpolado e precisa cair **dentro do segmento** (cruzar o prolongamento da linha não conta). `cooldown_frames` impede reconte do mesmo ID logo após um evento. **Direção**: `in` = do lado negativo para o positivo da distância assinada `(x−ax)·dy − (y−ay)·dx` (para linha vertical de cima para baixo na imagem, esquerda→direita).

## Região (ROI)
Evento `enter`/`exit` quando o ponto âncora muda de dentro/fora por `min_frames` quadros consecutivos (debounce). Também calcula a **ocupação** por quadro.

## Auditoria
Cada evento tem: nº, tipo, alvo, direção, ID da trilha, quadro, tempo, posição, **comprimento da trilha**, score médio, **flags** (ex.: `trilha_curta`) e o **digest** da configuração. Cada recontagem fica registrada (`count_runs`) com sua configuração e opções (`GET /analyses/{id}/audit`).

## Câmera em movimento
Linha e região são referenciadas ao **quadro**, não ao terreno. Com câmera em movimento, "cruzar a linha" significa "o animal passou pela posição X do quadro", o que também ocorre se **a câmera** se move. Para contagem de passagens, prefira câmera fixa ou voo planejado e interprete com cuidado.

## As três grandezas
Ver [00_visao_geral](00_visao_geral.md). `herd_bounds` devolve apenas: `piso_observado_simultaneo` (maior contagem instantânea suavizada por mediana móvel), `ids_unicos_filtrados` (IDs com ≥ 5 observações), `ids_unicos_brutos` e um aviso. **Não existe campo "população" na API**, e há um teste automatizado garantindo isso.
