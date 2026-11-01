# 08 · Mudança de domínio (dataset shift)

**Resultado central do projeto:** o detector que funciona na fazenda de validação (Mauron) **falha em fazenda inédita** (Jalogny). Números exatos em [12_resultados](12_resultados.md); aqui, o que isso significa.

## O que mudou entre treino e teste
| Fator | Treino/val (Mauron, Derval) | Teste (Jalogny) |
|---|---|---|
| Drone/câmera | DJI Mavic 3T (4000×3000) | DJI Mavic 3E (5280×3956) |
| Paisagem | pastagem aberta | pastagem **com mata, galhos secos, sombras e cursos d'água** |
| Cor/iluminação | uma estação | várias datas (jun–set) |
| Animais | mistura de pelagens do treino | pelagens e tamanhos aparentes diferentes |

## O que se observou (de fato)
1. **Falsos positivos em mata.** Inspeção visual de uma imagem de teste: os animais reais (à direita) foram detectados, mas dezenas de caixas surgiram em galhos secos e troncos claros. Nas imagens de teste **sem nenhum animal**, o modelo previu animais em todas as medidas de limiar usadas.
2. **Superestimativa generalizada**, em todos os voos de teste (não é um voo isolado).
3. **Revocação baixa** a IoU 0,5 também: parte dos animais reais não é detectada ou tem caixa frouxa.
4. Treinar mais épocas melhorou val e contagem de teste, **mas não resolveu**: o problema é de domínio, não de ajuste fino.

## O que isto NÃO diz
- Não prova que a *causa* seja a mata (não foi feito experimento de ablação por cenário), só que é o padrão visível.
- Não generaliza para outras fazendas: o teste é **uma** fazenda (5 voos).

## Estratégia de avaliação por condição (altitude, iluminação, densidade, oclusão)
O projeto implementa o **mecanismo** (estratos por voo, altitude, densidade e fazenda, IC por bootstrap de voos), mas os **dados disponíveis limitam as conclusões**:
- *Altitude*: só há 60 m e 100 m, e o único voo de teste a 60 m está confundido com o próprio voo.
- *Iluminação*: os voos "ensolarado/nublado" do resumo existem, mas não há voos nublados suficientes entre os confiáveis; **não foi avaliada**.
- *Densidade*: avaliada (0 | 1‑5 | 6‑15 | ≥ 16).
- *Oclusão*: não há anotação de oclusão; **não foi avaliada**.

## O que fazer na prática
1. **Calibrar na fazenda**: o experimento de adaptação (seção 4 de 12_resultados) ajustou o modelo com **2 voos (35 imagens)** de Jalogny e, nos 3 voos retidos, o erro de contagem caiu de ~11 para ~1,7 animais/imagem. **Custo medido:** o mesmo modelo piorou e passou a subcontar na fazenda original (Mauron). Conclusão prática: calibrar **por fazenda** funciona muito melhor que um modelo único, mas exige imagens anotadas da fazenda-alvo e re-treino/modelo separado.
2. **Validar antes de confiar**: conte manualmente algumas imagens da fazenda e compare (o relatório do sistema guarda a configuração para repetir).
3. Tratar a contagem como **estimativa com viés conhecido**, nunca como censo.
