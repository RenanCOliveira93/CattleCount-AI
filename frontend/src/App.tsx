import { useEffect, useState } from "react";
import { api } from "./api";
import ImageTab from "./ImageTab";
import VideoTab from "./VideoTab";
import { About, History } from "./Views";

const TABS = ["Imagem", "Vídeo", "Histórico", "Sobre e limites"] as const;

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Imagem");
  const [openId, setOpenId] = useState<number | null>(null);
  const [health, setHealth] = useState<{ model_loaded: boolean; model_id: string } | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => { api.health().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true)); }, []);
  return (
    <main>
      <header>
        <h1>CattleCount AI</h1>
        <span className={`pill ${down || health?.model_loaded === false ? "bad" : "ok"}`} data-testid="health">
          {down ? "API indisponível" : health?.model_loaded ? `modelo ${health.model_id}` : "modelo ausente"}
        </span>
      </header>
      <nav role="tablist">{TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}</nav>
      <div hidden={tab !== "Imagem"}><ImageTab /></div>
      <div hidden={tab !== "Vídeo"}><VideoTab openId={openId} /></div>
      {tab === "Histórico" && <History onOpen={(id) => { setOpenId(id); setTab("Vídeo"); }} />}
      {tab === "Sobre e limites" && <About />}
      <footer className="muted">Estimativas de um modelo de visão computacional. Não substituem contagem física para decisões de manejo, comercial ou regulatórias.</footer>
    </main>
  );
}
