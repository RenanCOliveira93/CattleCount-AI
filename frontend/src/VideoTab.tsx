import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Analysis, type CountResult, type TracksResp } from "./api";
import { EventsTable, ThreeCounts } from "./Counts";
import { buildCountConfig, toImageCoords, trailPoints, type LineCfg, type Pt, type RegionCfg } from "./lib";

type Mode = "ver" | "linha" | "regiao";

export default function VideoTab({ openId }: { openId: number | null }) {
  const [file, setFile] = useState<File | null>(null);
  const [conf, setConf] = useState(0.25);
  const [stride, setStride] = useState(1);
  const [compensate, setCompensate] = useState(false);
  const [stitch, setStitch] = useState(false);
  const [aid, setAid] = useState<number | null>(openId);
  const [an, setAn] = useState<Analysis | null>(null);
  const [tracks, setTracks] = useState<TracksResp | null>(null);
  const [k, setK] = useState(1);
  const [mode, setMode] = useState<Mode>("ver");
  const [line, setLine] = useState<LineCfg | null>(null);
  const [region, setRegion] = useState<RegionCfg | null>(null);
  const [pending, setPending] = useState<Pt[]>([]);
  const [res, setRes] = useState<CountResult | null>(null);
  const [showVideo, setShowVideo] = useState(false);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const img = useRef<HTMLImageElement | null>(null);

  useEffect(() => { if (openId != null) { setAid(openId); setRes(null); setTracks(null); setShowVideo(false); } }, [openId]);

  // progresso por sondagem (a API também expõe /ws para clientes que preferirem push)
  useEffect(() => {
    if (aid == null) return;
    let alive = true;
    const poll = async () => { try { const a = await api.analysis(aid); if (alive) setAn(a); return a; } catch (e) { if (alive) setErr((e as Error).message); return null; } };
    poll();
    const t = setInterval(async () => { const a = await poll(); if (a && (a.status === "done" || a.status === "error")) clearInterval(t); }, 1000);
    return () => { alive = false; clearInterval(t); };
  }, [aid]);

  useEffect(() => {
    if (an?.status === "done" && !tracks) api.tracks(an.id).then((t) => { setTracks(t); setK(1); }).catch((e) => setErr(e.message));
  }, [an?.status, an?.id, tracks]);

  const srcFrame = useCallback((kk: number) => Math.max(0, (kk - 1) * Number(an?.params.stride ?? 1)), [an]);

  const draw = useCallback(() => {
    const c = canvas.current, im = img.current;
    if (!c || !im || !im.complete || !an) return;
    c.width = an.ingest.width; c.height = an.ingest.height;
    const g = c.getContext("2d")!;
    g.drawImage(im, 0, 0, c.width, c.height);
    const lw = Math.max(2, c.width / 400);
    g.lineWidth = lw; g.font = `${Math.max(12, c.width / 60)}px sans-serif`;
    for (const b of tracks?.frames[String(k)] ?? []) {
      g.strokeStyle = `hsl(${(b[4] * 67) % 360},90%,55%)`; g.fillStyle = g.strokeStyle;
      g.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); g.fillText(`#${b[4]}`, b[0], Math.max(12, b[1] - 3));
      const tp = tracks ? trailPoints(tracks.frames, b[4], k) : [];
      if (tp.length > 1) { g.beginPath(); tp.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.stroke(); }
    }
    g.strokeStyle = "#ffd400"; g.fillStyle = "#ffd400";
    if (line) { g.beginPath(); g.moveTo(...line.a); g.lineTo(...line.b); g.stroke(); }
    const poly = region?.polygon ?? [];
    const pts = mode === "regiao" ? pending : poly;
    if (pts.length) { g.strokeStyle = "#ff9d00"; g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); if (mode !== "regiao") g.closePath(); g.stroke(); }
    if (mode === "linha") pending.forEach((p) => { g.beginPath(); g.arc(p[0], p[1], lw * 2, 0, 7); g.fill(); });
  }, [an, tracks, k, line, region, pending, mode]);

  useEffect(() => {
    if (!an || an.status !== "done") return;
    const im = new Image();
    im.onload = () => { img.current = im; draw(); };
    const t = setTimeout(() => { im.src = api.preview(an.id, srcFrame(k)); }, 80);
    return () => clearTimeout(t);
  }, [an, k, srcFrame]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(draw, [draw]);

  function onCanvas(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!an || mode === "ver") return;
    const p = toImageCoords(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect(), an.ingest.width, an.ingest.height);
    if (mode === "linha") {
      const n = [...pending, p];
      if (n.length === 2) { setLine({ a: n[0], b: n[1], name: "linha1" }); setPending([]); setMode("ver"); } else setPending(n);
    } else setPending([...pending, p]);
  }
  function closeRegion() { if (pending.length >= 3) { setRegion({ polygon: pending, name: "regiao1" }); } setPending([]); setMode("ver"); }

  async function upload() {
    if (!file) return;
    setBusy(true); setErr(""); setRes(null); setTracks(null); setLine(null); setRegion(null);
    try { const r = await api.uploadVideo(file, { conf, stride, compensate }); setAid(r.analysis_id); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }
  async function count() {
    if (!an) return;
    setBusy(true); setErr("");
    try { setRes(await api.count(an.id, buildCountConfig(line, region), stitch)); setShowVideo(false); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  }

  const done = an?.status === "done";
  return (
    <div>
      <p className="muted">Modo 2 — contagem de <b>passagens</b> por linha ou região em vídeo, com rastreamento ByteTrack. Trace a linha sobre o quadro e conte; a recontagem não repete a detecção.</p>
      <div className="bar">
        <input aria-label="arquivo de vídeo" type="file" accept="video/mp4,video/quicktime,.avi,.mkv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <label>confiança <input type="number" min={0.05} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(+e.target.value)} /></label>
        <label>stride <input type="number" min={1} max={30} value={stride} onChange={(e) => setStride(+e.target.value)} /></label>
        <label><input type="checkbox" checked={compensate} onChange={(e) => setCompensate(e.target.checked)} /> compensar movimento da câmera</label>
        <button onClick={upload} disabled={!file || busy}>Enviar e processar</button>
      </div>
      {err && <p role="alert" className="error">{err}</p>}
      {an && (
        <div className="card" data-testid="job">
          <b>#{an.id} {an.filename}</b> — <span data-testid="status">{an.status}</span>
          {an.status === "running" || an.status === "queued" ? <progress max={1} value={an.progress} aria-label="progresso" /> : null}
          {an.error && <p className="error">{an.error}</p>}
          {an.ingest?.warnings?.length ? <ul className="muted">{an.ingest.warnings.map((w) => <li key={w}>{w}</li>)}</ul> : null}
        </div>
      )}
      {done && tracks && (
        <div>
          <div className="bar">
            <button className={mode === "linha" ? "on" : ""} onClick={() => { setMode("linha"); setPending([]); }}>Traçar linha (2 cliques)</button>
            <button className={mode === "regiao" ? "on" : ""} onClick={() => { setMode("regiao"); setPending([]); }}>Desenhar região</button>
            {mode === "regiao" && <button onClick={closeRegion}>Fechar região</button>}
            <button onClick={() => { setLine(null); setRegion(null); setPending([]); setMode("ver"); }}>Limpar</button>
            <label><input type="checkbox" checked={stitch} onChange={(e) => setStitch(e.target.checked)} /> costura limitada de trilhas</label>
            <button className="primary" onClick={count} disabled={busy || (!line && !region)}>Contar passagens</button>
          </div>
          <canvas ref={canvas} className="full" data-testid="canvas" onClick={onCanvas} aria-label="quadro do vídeo com trilhas e linha de contagem" style={{ cursor: mode === "ver" ? "default" : "crosshair" }} />
          <input type="range" aria-label="quadro" min={1} max={Math.max(1, Object.keys(tracks.frames).length)} value={k} onChange={(e) => setK(+e.target.value)} className="full" />
          <p className="muted">quadro analisado {k}/{Object.keys(tracks.frames).length} · {(tracks.frames[String(k)] ?? []).length} trilhas ativas</p>
        </div>
      )}
      {res && an && (
        <div data-testid="count-result">
          <ThreeCounts r={res} />
          {res.stitching && <p className="hint">Costura aplicada: {res.stitching.ids_unidos} ID(s) unidos. {res.stitching.aviso}</p>}
          <div className="bar">
            <a href={api.eventsCsv(an.id, res.run_id)} download>Eventos (CSV)</a>
            <a href={api.seriesCsv(an.id, res.run_id)} download>Série visível (CSV)</a>
            <a href={api.tracksCsv(an.id)} download>Trilhas (CSV)</a>
            <a href={api.reportHtml(an.id, res.run_id)} target="_blank" rel="noreferrer">Relatório</a>
            <button onClick={() => setShowVideo(true)}>Gerar vídeo anotado</button>
          </div>
          {showVideo && <video className="full" controls src={api.annotated(an.id, res.run_id)} data-testid="annotated" />}
          <h3>Eventos (auditoria)</h3>
          <EventsTable events={res.events} onSeek={(f) => setK(Math.max(1, f))} />
        </div>
      )}
    </div>
  );
}
