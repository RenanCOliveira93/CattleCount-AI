# 02 · Dados: descoberta, validação e divisão

## Critérios de validação (antes de treinar)
Licença · tipo de anotação · diversidade de cenários · resolução · tamanho dos objetos · condições de aquisição · **consistência das anotações com o que o autor declara**.

## Dataset principal — ICAERUS "aerial cows" (Zenodo)
- **Registro**: Zenodo 11048412 · licença **CC BY 4.0** · 16,6 GB · fotos de drone (DJI Mavic 3E/3T), voo nadir a **60 e 100 m**, três fazendas (Mauron, Derval, Jalogny) + outras sem rótulo; caixas no formato YOLO.
- **Problemas encontrados na versão baixada (e como foram tratados):**
  - O resumo oficial declara 4 941 caixas, mas os arquivos de rótulo contêm **2 666**; 306 imagens estão **sem arquivo de rótulo**; 13 voos divergem do resumo. **Arquivo de rótulo ausente ≠ ausência de gado.**
  - Regra adotada: um voo é **"confiável"** só se o nº de caixas nos rótulos = nº do resumo oficial (±1). Voos não confiáveis foram **excluídos** da avaliação e do treino (não foram usados como "negativos").
- **Atenção à unidade de amostragem**: imagens do mesmo voo são quase duplicadas entre si. A divisão é **por voo** (e por fazenda no teste), nunca por imagem.
- **Divisão decidida antes de qualquer resultado de modelo:**
  - **Treino**: voos DJI_202306051814_022, 202306051839_025, 202306071703_029, 202306071745_034, 202306071745_035 (Mauron) e DJI_202308091442_012 (Derval).
  - **Validação**: DJI_202306071624_028 (Mauron).
  - **Teste (fazenda inédita)**: voos confiáveis de **Jalogny** (outro local, outro drone/câmera, cor e vegetação diferentes).
- **Subconjunto baixado**: o servidor entrega ~0,2–0,6 MB/s, então **não foi baixado o dataset inteiro**: foram baixadas 199 imagens (2,26 GB) escolhidas por plano reproduzível (`training/zenodo_plan.py`, semente 0): imagens com gado primeiro e até 4 imagens sem gado por voo. Resultados valem para esse subconjunto.

## Dataset de rastreamento — Mendeley Data dk54zg67dd (vídeos terrestres)
- Vídeos de celular, **nível do solo**, CC BY 4.0, com **MOT de verdade-terreno para um único clipe** (≈ 16 s, 479 quadros; o arquivo de GT tem **9 identidades**, enquanto a descrição do autor cita 8 animais). O próprio autor descreve que o GT partiu de YOLOv8‑COCO + DeepSORT e foi corrigido, então **favorece detectores tipo COCO**.
- Uso: apenas validar o rastreador em movimento real de bovinos. **Não é cenário aéreo** e não valida o detector aéreo.

## O que não foi usado
- **Roboflow Universe**: a spec cita como fonte de descoberta. Nenhum dataset de lá foi usado: licença e qualidade de anotação variam por projeto e exigiriam a mesma validação individual, que não foi feita aqui.
- **Não existe verdade de rastreamento em vídeo aéreo real** neste projeto.

## Citação e atribuição
ICAERUS aerial cows — Zenodo, doi:10.5281/zenodo.11048412 (CC BY 4.0). Mendeley Data dk54zg67dd (CC BY 4.0). Arquivos de exemplo e atribuição: `sample_data/ATTRIBUTION.md`.
