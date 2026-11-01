import { describe, expect, it } from "vitest";
import { buildCountConfig, fmtTime, sparkPath, toImageCoords, trailPoints } from "./lib";

describe("lib", () => {
  it("converte coordenadas de tela para imagem e limita às bordas", () => {
    const rect = { left: 10, top: 20, width: 400, height: 300 };
    expect(toImageCoords(210, 170, rect, 4000, 3000)).toEqual([2000, 1500]);
    expect(toImageCoords(0, 0, rect, 4000, 3000)).toEqual([0, 0]);
    expect(toImageCoords(9999, 9999, rect, 4000, 3000)).toEqual([4000, 3000]);
  });
  it("monta a configuração de contagem; região com <3 pontos é ignorada", () => {
    const l = { a: [0, 0] as [number, number], b: [0, 10] as [number, number], name: "L" };
    expect(buildCountConfig(l, { polygon: [[0, 0], [1, 1]], name: "R" })).toEqual({ lines: [{ a: [0, 0], b: [0, 10], name: "L" }], regions: [] });
    expect(buildCountConfig(null, { polygon: [[0, 0], [1, 0], [1, 1]], name: "R" }).regions).toHaveLength(1);
  });
  it("sparkline e tempo", () => {
    expect(sparkPath([], 100, 20)).toBe("");
    expect(sparkPath([0, 5, 10], 100, 24).startsWith("M4.0,20.0 L50.0")).toBe(true);
    expect(fmtTime(75.25)).toBe("1:15.3");
    expect(fmtTime(null)).toBe("—");
  });
  it("rastro de um ID", () => {
    const f = { "1": [[0, 0, 10, 10, 7, 0.9]], "2": [[10, 0, 20, 10, 7, 0.9]], "3": [] };
    expect(trailPoints(f, 7, 3)).toEqual([[5, 5], [15, 5]]);
  });
});
