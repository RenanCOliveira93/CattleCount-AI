import { useEffect, useState } from "react";
import { api } from "./api";

export function History({ onOpen }: { onOpen: (id: number) => void }) {
  const [rows, setRows] = useState<Awaited<ReturnType<typeof api.list>>>([]);
  const [aud, setAud] = useState<Record<string, unknown> | null>(null);
  const load = () => api.list().then(setRows).catch(() => setRows([]));
  useEffect(() => { load(); }, []);
  return (
    <div>
      <table>
        <thead><tr><th>#</th><th>tipo</th><th>arquivo</th><th>status</th><th>visíveis</th><th>recontagens</th><th /></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td><td>{r.kind}</td><td>{r.filename}</td><td>{r.status}</td><td>{r.visible_count ?? "—"}</td><td>{r.n_runs}</td>
              <td>
                {r.kind === "video" && <button onClick={() => onOpen(r.id)}>abrir</button>}
                <button onClick={() => api.audit(r.id).then(setAud)}>auditoria</button>
                <button onClick={() => api.del(r.id).then(load)}>excluir</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && <p className="muted">Nenhuma análise ainda.</p>}
      {aud && <pre className="card">{JSON.stringify(aud, null, 1)}</pre>}
    </div>
  );
}

export function About() {
  const [lim, setLim] = useState<{ advisory: string; limites: string[] } | null>(null);
  const [ev, setEv] = useState<Record<string, unknown>>({});
  useEffect(() => { api.limits().then(setLim).catch(() => null); api.evaluations().then(setEv).catch(() => setEv({})); }, []);
  return (
    <div>
      <section className="card warn"><h3>Limites e interpretação</h3><p>{lim?.advisory}</p><ul>{lim?.limites.map((l) => <li key={l}>{l}</li>)}</ul></section>
      <h3>Experimentos registrados</h3>
      {Object.keys(ev).length ? <pre className="card">{JSON.stringify(ev, null, 1)}</pre> : <p className="muted">Nenhum experimento registrado em experiments/registry.json.</p>}
    </div>
  );
}
