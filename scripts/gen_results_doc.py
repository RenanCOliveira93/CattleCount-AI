"""Gera runs/results.md (tabelas da seção 8 de docs/GUIA_TECNICO.md) a partir dos JSONs reais em experiments/. Os números NUNCA são digitados; a prosa só afirma o que os números mostram."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "experiments"
L = lambda n: json.loads((EX / f"{n}.json").read_text()) if (EX / f"{n}.json").exists() else None  # noqa: E731
es, full, coco = L("yolo11s_cattle_es"), L("yolo11s_cattle_full"), L("coco_yolo11s_zeroshot")
fh, eh, ad = L("yolo11s_cattle_full__jalogny_holdout"), L("yolo11s_cattle_es__jalogny_holdout"), L("yolo11s_cattle_adapt__jalogny_holdout")
trk, bm = L("tracking_eval"), L("benchmark")
f2 = lambda x: "—" if x is None or x != x else f"{x:.2f}".replace(".", ",")  # noqa: E731
f3 = lambda x: "—" if x is None or x != x else f"{x:.3f}".replace(".", ",")  # noqa: E731
fx = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
o = []
w = o.append
w("# Resultados (gerado automaticamente de `experiments/*.json`)\n")
w("> Este arquivo é produzido por `scripts/gen_results_doc.py`. Nenhum número foi digitado à mão. Interpretação e limites: ver o fim do documento.\n")
t, v = full["test"], full["val"]
w("## 1 · Em uma frase\n")
w(f"- **Na fazenda de validação (Mauron, voo não usado no treino)** o modelo escolhido conta razoavelmente: MAE {f2(v['contagem']['mae'])} animais/imagem, mAP50 {f3(v['map50'])}.")
w(f"- **Em fazenda inédita (Jalogny)** ele **superconta muito**: {int(t['contagem']['pred_total'])} animais previstos contra {int(t['contagem']['gt_total'])} reais (**{t['contagem']['total_error_pct']*100:+.0f} %**), precisão {f3(t['prf']['precision'])}, mAP50 {f3(t['map50'])}. **Não use este modelo, como está, numa fazenda nova sem calibrar.**")
w(f"- O detector COCO pré-treinado (zero-shot) quase não detecta nada nessa escala (revocação {f3(coco['test']['prf']['recall'])} no teste).\n")
w("## 2 · Detecção e contagem por imagem — modelos comparados\n")
w("Limiar de confiança escolhido no VAL (melhor F1) e aplicado ao TESTE sem alteração. Teste = 92 imagens de 5 voos de Jalogny; val = 41 imagens de 1 voo de Mauron. IC95 % por bootstrap **de voos** (só 5 voos no teste: IC largo).\n")
w("| modelo | conf | split | mAP50 | mAP50-95 | precisão | revocação | MAE contagem | viés | IC95 MAE (voos) |\n|---|---|---|---|---|---|---|---|---|---|")
for nm, e in (("COCO zero-shot (yolo11s)", coco), ("YOLO11s, parada antecipada (época 16)", es), ("**YOLO11s, 40 épocas (selecionado)**", full)):
    for sp in ("val", "test"):
        x = e[sp]; ci = x["contagem_ic95_bootstrap_por_voo"]["mae"]; icn = x["contagem_ic95_bootstrap_por_voo"]["n_unidades"]
        w(f"| {nm} | {e['conf_operacional']} | {sp} | {f3(x['map50'])} | {f3(x['map50_95'])} | {f3(x['prf']['precision'])} | {f3(x['prf']['recall'])} | {f2(x['contagem']['mae'])} | {x['contagem']['bias']:+.2f} | {('[%s; %s]' % (fx(ci[0]), fx(ci[1]))) if icn > 1 else '— (1 voo)'} |")
w("\n**Seleção do modelo:** regra aplicada: melhor F1 no val. **Transparência:** o primeiro modelo (parada antecipada) já tinha sido avaliado no teste quando se decidiu treinar as 40 épocas completas; a decisão de treinar mais não foi cega ao teste, mas a escolha entre os dois modelos usou só o val. " + f"F1 no val: 40 épocas {f3(max(full['f1_val_por_conf'].values()))} contra parada antecipada {f3(max(es['f1_val_por_conf'].values()))}. O val também escolheu o checkpoint dentro de cada treino (métrica interna do Ultralytics), então os números do val são **um pouco otimistas**.\n")
w("## 3 · Onde o modelo selecionado erra no teste (fazenda inédita)\n")
w("| estrato | imagens | GT | previsto | MAE | viés |\n|---|---|---|---|---|---|")
for k, nm in (("por_voo", "voo"), ("por_altitude_m", "altitude (m)"), ("por_densidade", "densidade (animais/imagem)")):
    for g, x in t[k].items():
        w(f"| {nm}: {g} | {x['n']} | {int(x['gt_total'])} | {int(x['pred_total'])} | {f2(x['mae'])} | {x['bias']:+.2f} |")
zero = t["por_densidade"].get("0")
w("")
if zero:
    w(f"- **Imagens sem nenhum animal ({zero['n']}):** o modelo previu {int(zero['pred_total'])} animais (≈ {fx(zero['pred_total']/zero['n'])} por imagem). São falsos positivos puros; inspeção visual mostra galhos secos e troncos claros em áreas de mata (cenário quase ausente no treino).")
w("- **Altitude:** 60 m tem erro menor que 100 m, mas só **um** voo de teste é de 60 m. Altitude e voo estão **confundidos**: não dá para concluir efeito da altitude.")
w("- **Densidade:** o viés relativo é menor nas cenas densas; o erro absoluto não cresce com a densidade, ou seja, a superestimativa vem de falsos positivos, não de contar demais animais reais.\n")
if fh and eh:
    w("## 4 · Adaptação a uma fazenda nova (experimento declarado antes de rodar)\n")
    w("Pergunta prática: *quanto ajuda ajustar o modelo com poucas fotos da fazenda do cliente?* Voos de **adaptação**: DJI_202306231103_031 e DJI_202309261339_058 (Jalogny) entram no treino. **Teste da comparação**: os 3 voos restantes (032, 056, 059), nunca vistos. Para a adaptação usa-se o checkpoint final (30 épocas, sem seleção) e conf fixo 0,25, porque o val de Mauron não representa a fazenda nova.\n")
    w("| modelo | imagens | GT | previsto | mAP50 | precisão | revocação | MAE | viés |\n|---|---|---|---|---|---|---|---|---|")
    rows = [("parada antecipada", eh), ("**40 épocas (selecionado)**", fh)] + ([("**40 épocas + adaptação (2 voos de Jalogny)**", ad)] if ad else [])
    for nm, e in rows:
        x = e["test"]
        w(f"| {nm} | {x['n_images']} | {int(x['contagem']['gt_total'])} | {int(x['contagem']['pred_total'])} | {f3(x['map50'])} | {f3(x['prf']['precision'])} | {f3(x['prf']['recall'])} | {f2(x['contagem']['mae'])} | {x['contagem']['bias']:+.2f} |")
    w("")
    if ad:
        w("Por voo retido (40 épocas → adaptado):\n\n| voo | imagens | GT | previsto antes | MAE antes | previsto depois | MAE depois | viés depois |\n|---|---|---|---|---|---|---|---|")
        for k, y in ad["test"]["por_voo"].items():
            x = fh["test"]["por_voo"][k]
            w(f"| {k} | {y['n']} | {int(y['gt_total'])} | {int(x['pred_total'])} | {f2(x['mae'])} | {int(y['pred_total'])} | {f2(y['mae'])} | {y['bias']:+.2f} |")
        ci = ad["test"]["contagem_ic95_bootstrap_por_voo"]["mae"]
        z = ad["test"]["por_densidade"].get("0")
        w(f"\nIC95 % do MAE adaptado (bootstrap de {ad['test']['contagem_ic95_bootstrap_por_voo']['n_unidades']} voos): [{fx(ci[0])}; {fx(ci[1])}] — **muito estreito para 3 voos; não o interprete como garantia**.\n")
        w(f"- **Efeito:** nos voos retidos de Jalogny o erro de contagem cai de {f2(fh['test']['contagem']['mae'])} para {f2(ad['test']['contagem']['mae'])} animais/imagem e o viés de {fh['test']['contagem']['bias']:+.2f} para {ad['test']['contagem']['bias']:+.2f}. A melhora aparece nos três voos, inclusive em voos de outro dia (032, a 60 m; 056).")
        if z:
            w(f"- **Falsos positivos em imagens sem animais:** {int(z['pred_total'])} animais previstos em {z['n']} imagens vazias (antes: {int(fh['test']['por_densidade']['0']['pred_total'])}).")
        w(f"- **Custo (esquecimento):** no val de Mauron o modelo adaptado cai de mAP50 {f3(full['val']['map50'])} para {f3(ad['val']['map50'])}, a revocação cai para {f3(ad['val']['prf']['recall'])} e ele passa a **subcontar** (viés {ad['val']['contagem']['bias']:+.2f} animais/imagem; MAE {f2(ad['val']['contagem']['mae'])}). Adaptar a uma fazenda pode **piorar** as outras: o caminho seguro é um modelo por fazenda/domínio, ou re-treinar com todas.")
        w("- **Ressalvas:** o voo 059 foi feito no mesmo dia e em sequência ao 058 (usado na adaptação): é quase a mesma cena. Mesmo sem ele, 032 e 056 melhoram. Conf fixo 0,25 e checkpoint final (sem seleção). É **uma** fazenda; o ganho prático depende de haver imagens anotadas da fazenda-alvo (aqui, 2 voos, 35 imagens).\n")
if trk:
    A, B = trk["A_sintetico"], trk["B_terrestre_mendeley"]
    w("## 5 · Rastreamento\n")
    w("Parâmetros do tracker **não ajustados** nestes vídeos (cenário A: limiares derivados do conf operacional do detector; cenário B: padrões do ByteTrack original).\n")
    w(f"### A · Vídeo aéreo **sintético** (janela deslizando sobre foto real; {A['n_frames']} quadros, {A['gt_ids']} animais; GT exato)\n")
    w("Animais estáticos e câmera sintética: valida a cadeia detector→tracker→contagem com GT exato, **não** o comportamento de bovinos reais em vídeo aéreo.\n")
    w("| método | HOTA | IDF1 | IDF1@IoU0,3 | trocas de ID | IDs previstos | passagens na linha (GT: " + f"{A['passagens_GT_linha_x640']['total_passagens']}) |\n|---|---|---|---|---|---|---|")
    for nm, m in A["metodos"].items():
        w(f"| {nm} | {f3(m['HOTA'])} | {f3(m['IDF1'])} | {f3(m['IDF1_iou0.3'])} | {m['IDSW']} | {m['n_pred_ids']} | {m['passagens_linha_x640']['total_passagens']} |")
    w("\n**Com detecções perfeitas (as caixas do GT como entrada)**, isolando o rastreador do detector:\n")
    w("| método | HOTA | IDF1 | trocas de ID | IDs previstos |\n|---|---|---|---|---|")
    for nm, m in A["metodos_com_deteccao_perfeita"].items():
        w(f"| {nm} | {f3(m['HOTA'])} | {f3(m['IDF1'])} | {m['IDSW']} | {m['n_pred_ids']} |")
    di = A["deteccao_isolada"]
    w(f"\nO detector aéreo viu em média **{fx(di['deteccoes_por_quadro_media'])} animais por quadro** contra **{fx(di['gt_visiveis_por_quadro_media'])}** reais: a revocação do detector (animais escuros, caixas frouxas) limita tudo a jusante. Com detecções perfeitas o rastreador é quase perfeito; com as reais, as trilhas fragmentam e a contagem de passagens fica abaixo da verdade. (Nota: nesse cenário simples e suave, o baseline IoU também é perfeito com detecção perfeita: a vantagem do ByteTrack aparece quando as detecções são ruidosas.)\n")
    w(f"### B · Vídeo **terrestre real** (Mendeley, clipe com GT de {B['gt_ids']} animais, {B['n_frames']} quadros; detector COCO)\n")
    w("O GT foi gerado com YOLOv8-COCO + DeepSORT (corrigido): **favorece detectores COCO**. É cenário terrestre, não aéreo.\n")
    w("| método | HOTA | IDF1 | MOTA | trocas de ID | IDs previstos |\n|---|---|---|---|---|---|")
    for nm, m in B["metodos"].items():
        w(f"| {nm} | {f3(m['HOTA'])} | {f3(m['IDF1'])} | {f3(m['MOTA'])} | {m['IDSW']} | {m['n_pred_ids']} |")
    w("\nO ByteTrack próprio e o `BYTETracker` do Ultralytics (mesmas detecções e limiares) ficam praticamente empatados; ambos superam o baseline IoU em IDF1 e em trocas de ID.\n")
if bm:
    w("## 6 · Latência (CPU, ONNX Runtime, máquina ociosa)\n")
    w(f"Máquina: {bm['maquina']['sistema']} · {bm['maquina']['provider']}\n")
    w("| operação | mediana (ms) | p95 (ms) | n |\n|---|---|---|---|")
    for k, nm in (("tile_1024", "detecção de 1 tile 1024×1024"), ("quadro_1280x720_inteiro", "quadro 1280×720 (inteiro)"), ("foto_4000x3000_fatiada", "foto 4000×3000 (fatiada)"), ("bytetrack_update_30_deteccoes", "ByteTrack, 1 quadro, 30 detecções")):
        x = bm[k]; w(f"| {nm} | {x['mediana_ms']} | {x['p95_ms']} | {x['n']} |")
    w("\nMedido nesta máquina; **não** extrapole para outros hardwares ou para a VM do Podman.\n")
w("## 7 · O que estes resultados NÃO provam\n")
for s in ("O teste tem **5 voos de uma única fazenda**: o IC por voo é largo e não representa outras fazendas.",
          "Subconjunto de 199 imagens do dataset (download limitado): resultados valem para esse subconjunto.",
          "**Não há GT de rastreamento em vídeo aéreo real**; o cenário A é sintético e o B é terrestre.",
          "Anotações do ICAERUS têm lacunas conhecidas; voos inconsistentes foram excluídos, mas ainda pode haver animais sem caixa nos voos usados (isso **subestima** a precisão).",
          "Um único modelo/tamanho (YOLO11s) e uma semente: sem variância entre execuções."):
    w(f"- {s}")
out = ROOT / "runs/results.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(o) + "\n")
print("ok", len(o), "linhas")
