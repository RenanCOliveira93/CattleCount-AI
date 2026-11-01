import { useState } from "react";
import { api, type ImageResult } from "./api";
import { COPY } from "./lib";

export default function ImageTab() {
  const [file, setFile] = useState<File | null>(null);
  const [conf, setConf] = useState(0.25);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState<ImageResult | null>(null);

  async function run() {
    if (!file) return;
    setBusy(true); setErr(""); setRes(null);
    try { setRes(await api.inferImage(file, conf)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  return (
    <div>
      <p className="muted">Modo 1 — contagem de animais <b>visíveis</b> em uma imagem aérea. Imagens grandes são analisadas por fatias de 1024 px na resolução nativa.</p>
      <div className="bar">
        <input aria-label="arquivo de imagem" type="file" accept="image/jpeg,image/png" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <label>confiança mínima <input type="range" min={0.05} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(+e.target.value)} /> {conf.toFixed(2)}</label>
        <button onClick={run} disabled={!file || busy}>{busy ? "Analisando…" : "Contar"}</button>
      </div>
      {err && <p role="alert" className="error">{err}</p>}
      {res && (
        <div data-testid="image-result">
          <div className="grid3">
            <section className="card"><h3>Visíveis nesta imagem</h3><p className="big" data-testid="visible">{res.visible_count}</p><p className="hint">{COPY.visiveis}</p>
              <p className="muted">{res.image.width}×{res.image.height} · {res.timings_ms.infer_ms} ms · modelo {res.model_id}</p></section>
            <section className="card warn"><h3>Avisos de entrada</h3>{res.warnings.length ? <ul>{res.warnings.map((w) => <li key={w}>{w}</li>)}</ul> : <p className="muted">nenhum</p>}</section>
          </div>
          {res.annotated_jpeg_base64 && <img className="full" alt="imagem anotada" src={`data:image/jpeg;base64,${res.annotated_jpeg_base64}`} />}
          <p className="hint">{res.advisory}</p>
        </div>
      )}
    </div>
  );
}
