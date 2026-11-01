# 07 · Erros de contagem: de onde vêm

| Fonte | Efeito na contagem de **visíveis** | Efeito nas **passagens** | Como aparece nos números |
|---|---|---|---|
| Falso positivo (galho seco, sombra, rocha clara) | + (superconta) | + (trilhas espúrias cruzando) | viés positivo, precisão baixa |
| Animal não detectado (cor escura, oclusão, borda) | − | − | viés negativo, revocação baixa |
| Duplicata (tiles, animais encostados) | + | + | dependente de NMS/IoS |
| Troca de ID | 0 | ± | IDSW; passagem a mais ou a menos |
| Fragmentação de trilha | 0 | + (mesmo animal conta duas vezes) ou − (trilha curta filtrada) | Frag, IDs únicos > GT |
| Linha mal posicionada / zona morta | 0 | ± | passagem perdida ou duplicada |
| Câmera móvel | 0 | + (cruzamento causado pela câmera) | ver [05_contagem](05_contagem.md) |
| Animais ocultos ou fora do quadro | − (não são vistos) | − | **nunca** se corrige com mais software |

**Contagem global ≠ contagem localizada.** Um erro de contagem pequeno pode esconder falsos positivos e falsos negativos que se cancelam; por isso reportam‑se também precisão/revocação.
