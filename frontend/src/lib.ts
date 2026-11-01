export type Pt = [number, number];
export interface LineCfg { a: Pt; b: Pt; name: string }
export interface RegionCfg { polygon: Pt[]; name: string }

/** Converte posição do mouse (px de tela) para pixels da imagem original, considerando o redimensionamento do <canvas>/<img>. */
export function toImageCoords(clientX: number, clientY: number, rect: { left: number; top: number; width: number; height: number }, imgW: number, imgH: number): Pt {
  const x = ((clientX - rect.left) / rect.width) * imgW;
  const y = ((clientY - rect.top) / rect.height) * imgH;
  return [Math.min(Math.max(x, 0), imgW), Math.min(Math.max(y, 0), imgH)];
}

export function buildCountConfig(line: LineCfg | null, region: RegionCfg | null) {
  return {
    lines: line ? [{ a: line.a, b: line.b, name: line.name }] : [],
    regions: region && region.polygon.length >= 3 ? [{ polygon: region.polygon, name: region.name }] : [],
  };
}

export function sparkPath(values: number[], w: number, h: number, pad = 4): string {
  if (!values.length) return "";
  const max = Math.max(...values, 1);
  const dx = values.length > 1 ? (w - 2 * pad) / (values.length - 1) : 0;
  return values.map((v, i) => `${i ? "L" : "M"}${(pad + i * dx).toFixed(1)},${(h - pad - (v / max) * (h - 2 * pad)).toFixed(1)}`).join(" ");
}

export function fmtTime(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "—";
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

export function trailPoints(frames: Record<string, number[][]>, id: number, upTo: number, maxLen = 40): Pt[] {
  const out: Pt[] = [];
  for (let k = Math.max(1, upTo - maxLen); k <= upTo; k++) {
    const r = (frames[String(k)] ?? []).find((b) => b[4] === id);
    if (r) out.push([(r[0] + r[2]) / 2, (r[1] + r[3]) / 2]);
  }
  return out;
}

export const COPY = {
  visiveis: "Visíveis agora (instantânea): animais detectados no quadro. Não inclui ocultos nem fora do enquadramento.",
  passagens: "Passagens (acumulada): eventos de cruzamento da linha ou entrada/saída da região. Um animal que cruza duas vezes conta duas vezes.",
  rebanho: "Referência de rebanho: NÃO é o tamanho do rebanho. IDs únicos inflam com trocas de ID e deflacionam com animais nunca vistos.",
};
