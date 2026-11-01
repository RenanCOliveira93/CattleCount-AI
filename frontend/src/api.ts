const B = "/api";
async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let d = r.statusText;
    try { const b = await r.json(); d = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* corpo não-JSON */ }
    throw new Error(`${r.status}: ${d}`);
  }
  return r.json() as Promise<T>;
}
const key = () => { try { return localStorage.getItem("cc_api_key") ?? ""; } catch { return ""; } };
const H = (): Record<string, string> => (key() ? { "x-api-key": key() } : {});

export interface Det { xyxy: number[]; score: number }
export interface ImageResult { analysis_id: number; visible_count: number; conf: number; image: { width: number; height: number }; warnings: string[]; detections: Det[];
  annotated_jpeg_base64?: string; timings_ms: { infer_ms: number }; interpretation: string; advisory: string; model_id: string }
export interface Herd { piso_observado_simultaneo: number; ids_unicos_filtrados: number; ids_unicos_brutos: number; aviso: string }
export interface CountEvent { event_id: number; type: string; line: string | null; region: string | null; direction: string; track_id: number; frame: number; t_s: number | null; x: number; y: number; track_len: number; mean_score: number; flags: string[] }
export interface CountResult { run_id: number; config_digest: string; events: CountEvent[]; totals: Record<string, Record<string, number>>; herd_bounds: Herd;
  instantaneous: { frames: number[]; count: number[]; count_smoothed: number[] }; stitching: { ids_unidos: number; aviso: string } | null }
export interface Analysis { id: number; kind: string; filename: string; status: string; progress: number; error: string | null; params: Record<string, unknown>; ingest: { width: number; height: number; warnings: string[]; meta: Record<string, number> };
  result: Record<string, unknown>; model_id: string; sha256: string; runs: { run_id: number; digest: string; totals: Record<string, Record<string, number>>; herd_bounds: Herd; options: Record<string, unknown> }[] }
export interface TracksResp { fps_efetivo: number; frames: Record<string, number[][]> }

export const api = {
  health: () => fetch(`${B}/health`).then(j<{ status: string; model_loaded: boolean; model_id: string; auth_required: boolean }>),
  limits: () => fetch(`${B}/limits`).then(j<{ advisory: string; limites: string[] }>),
  evaluations: () => fetch(`${B}/evaluations`).then(j<Record<string, unknown>>),
  models: () => fetch(`${B}/models`).then(j<Record<string, unknown>[]>),
  inferImage: (f: File, conf: number) => { const d = new FormData(); d.append("file", f); d.append("conf", String(conf)); return fetch(`${B}/infer/image`, { method: "POST", body: d, headers: H() }).then(j<ImageResult>); },
  uploadVideo: (f: File, o: { conf: number; stride: number; compensate: boolean }) => { const d = new FormData(); d.append("file", f); d.append("conf", String(o.conf)); d.append("stride", String(o.stride)); d.append("compensate_motion", String(o.compensate)); return fetch(`${B}/videos`, { method: "POST", body: d, headers: H() }).then(j<{ analysis_id: number }>); },
  analysis: (id: number) => fetch(`${B}/analyses/${id}`).then(j<Analysis>),
  list: () => fetch(`${B}/analyses`).then(j<{ id: number; kind: string; filename: string; status: string; progress: number; created_at: string; n_runs: number; visible_count: number | null }[]>),
  tracks: (id: number) => fetch(`${B}/analyses/${id}/tracks`).then(j<TracksResp>),
  count: (id: number, config: unknown, stitch: boolean) => fetch(`${B}/analyses/${id}/count`, { method: "POST", headers: { "content-type": "application/json", ...H() }, body: JSON.stringify({ config, stitch }) }).then(j<CountResult>),
  audit: (id: number) => fetch(`${B}/analyses/${id}/audit`).then(j<Record<string, unknown>>),
  report: (id: number) => fetch(`${B}/analyses/${id}/report`).then(j<Record<string, unknown>>),
  del: (id: number) => fetch(`${B}/analyses/${id}`, { method: "DELETE", headers: H() }).then(j<unknown>),
  preview: (id: number, k: number) => `${B}/analyses/${id}/preview.jpg?k=${k}`,
  eventsCsv: (id: number, run: number) => `${B}/analyses/${id}/runs/${run}/events.csv`,
  seriesCsv: (id: number, run: number) => `${B}/analyses/${id}/runs/${run}/series.csv`,
  tracksCsv: (id: number) => `${B}/analyses/${id}/tracks.csv`,
  annotated: (id: number, run: number) => `${B}/analyses/${id}/runs/${run}/annotated.mp4`,
  reportHtml: (id: number, run?: number) => `${B}/analyses/${id}/report.html${run ? `?run_id=${run}` : ""}`,
};
