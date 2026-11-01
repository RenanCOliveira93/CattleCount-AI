# 09 · Arquitetura e API

```
 navegador ──► nginx (frontend estático + proxy /api) ──► FastAPI ──► PostgreSQL (análises, recontagens, eventos)
                                                          │
                                                          ├─ ONNX Runtime (detector, 1024 px, fatias)
                                                          ├─ ByteTrack próprio + contagem (cattlecount/)
                                                          └─ volume de jobs (vídeo, tracks.csv, vídeo anotado)
```
- **Assíncrono**: `POST /videos` valida e devolve `202` + `analysis_id`; um *worker* (1 thread) processa; progresso por `GET /analyses/{id}` (polling) e por **WebSocket** `/ws`.
- **Recontagem**: `POST /analyses/{id}/count` (linha/região/zona morta/costura) sobre trilhas persistidas, sem nova detecção.
- **Exportações**: `/analyses/{id}/runs/{run}/events.csv`, `series.csv`, `/analyses/{id}/tracks.csv`, `annotated.mp4` (H.264, sob demanda), `report` (JSON), `report.html`, `audit`.
- **Validação de entrada** (`cattlecount/ingest.py`): formato, corrupção, limites de tamanho; avisos explícitos (altitude ausente, FPS baixo, resolução baixa). Nada é "consertado" em silêncio.
- **Autenticação**: opcional por chave (`API_KEY` → cabeçalho `X-API-Key` nas rotas que gravam). **Não há usuários nem papéis.** Em rede não confiável, coloque atrás de um proxy autenticado.
- OpenAPI interativo em `/docs` do backend.
