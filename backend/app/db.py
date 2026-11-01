from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def now():
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    """Uma entrada processada (imagem ou vídeo). `result` guarda o que foi medido; nada é preenchido por suposição."""
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(10))
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(12), default="queued")   # queued|running|done|error
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    ingest: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    runs: Mapped[list["CountRun"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class CountRun(Base):
    """Cada (re)contagem com uma configuração de linha/região. Mantém histórico para auditoria."""
    __tablename__ = "count_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    config: Mapped[dict] = mapped_column(JSON)
    digest: Mapped[str] = mapped_column(String(16))
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    totals: Mapped[dict] = mapped_column(JSON)
    herd_bounds: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    analysis: Mapped[Analysis] = relationship(back_populates="runs")
    events: Mapped[list["EventRow"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class EventRow(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("count_runs.id", ondelete="CASCADE"), index=True)
    event_no: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(16))
    target: Mapped[str] = mapped_column(String(64))
    direction: Mapped[str] = mapped_column(String(8))
    track_id: Mapped[int] = mapped_column(Integer)
    frame: Mapped[int] = mapped_column(Integer)
    t_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    track_len: Mapped[int] = mapped_column(Integer)
    mean_score: Mapped[float] = mapped_column(Float)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    run: Mapped[CountRun] = relationship(back_populates="events")


def make_engine(url: str):
    if url.startswith("sqlite:///") and ":memory:" not in url:
        from pathlib import Path
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kw)


def make_session_factory(engine):
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
