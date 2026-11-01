import { COPY, fmtTime, sparkPath } from "./lib";
import type { CountEvent, CountResult, Herd } from "./api";

export function ThreeCounts({ r }: { r: Pick<CountResult, "totals" | "herd_bounds" | "instantaneous"> }) {
  const sm = r.instantaneous?.count_smoothed ?? [];
  const peak = sm.length ? Math.max(...sm) : 0;
  return (
    <div className="grid3" data-testid="three-counts">
      <section className="card" aria-label="Visíveis agora">
        <h3>Visíveis agora</h3>
        <p className="big" data-testid="peak">{peak}</p>
        <p className="muted">pico simultâneo (mediana móvel)</p>
        <svg viewBox="0 0 240 48" className="spark" role="img" aria-label="série de animais visíveis"><path d={sparkPath(sm, 240, 48)} fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
        <p className="hint">{COPY.visiveis}</p>
      </section>
      <section className="card" aria-label="Passagens">
        <h3>Passagens (acumulado)</h3>
        {Object.keys(r.totals).length === 0 && <p className="muted">Configure uma linha ou região para contar passagens.</p>}
        {Object.entries(r.totals).map(([name, t]) => (
          <div key={name} className="row" data-testid={`tot-${name}`}>
            <b>{name}</b>
            {"in" in t ? <span>entrada {t.in} · saída {t.out} · líquido {t.net}</span> : <span>entrou {t.enter} · saiu {t.exit} · ocupação máx. {t.ocupacao_max}</span>}
          </div>
        ))}
        <p className="hint">{COPY.passagens}</p>
      </section>
      <HerdCard h={r.herd_bounds} />
    </div>
  );
}

export function HerdCard({ h }: { h: Herd }) {
  return (
    <section className="card warn" aria-label="Referência de rebanho">
      <h3>Referência de rebanho <span className="tag">não é população</span></h3>
      <div className="row"><span>piso observado simultâneo</span><b data-testid="piso">{h.piso_observado_simultaneo}</b></div>
      <div className="row"><span>IDs únicos (≥5 quadros)</span><b>{h.ids_unicos_filtrados}</b></div>
      <div className="row"><span>IDs únicos (brutos)</span><b>{h.ids_unicos_brutos}</b></div>
      <p className="hint">{COPY.rebanho}</p>
    </section>
  );
}

export function EventsTable({ events, onSeek }: { events: CountEvent[]; onSeek?: (frame: number) => void }) {
  if (!events.length) return <p className="muted">Nenhum evento.</p>;
  return (
    <div className="scroll">
      <table>
        <thead><tr><th>#</th><th>tipo</th><th>alvo</th><th>direção</th><th>ID</th><th>tempo</th><th>trilha</th><th>score</th><th>alertas</th></tr></thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.event_id} onClick={() => onSeek?.(e.frame)} className={e.flags.length ? "flag" : ""}>
              <td>{e.event_id}</td><td>{e.type === "line_cross" ? "linha" : "região"}</td><td>{e.line ?? e.region}</td><td>{e.direction}</td><td>#{e.track_id}</td>
              <td>{fmtTime(e.t_s)}</td><td>{e.track_len}</td><td>{e.mean_score.toFixed(2)}</td><td>{e.flags.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
