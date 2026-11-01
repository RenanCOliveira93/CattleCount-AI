import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("CATTLECOUNT_ROOT", Path(__file__).resolve().parents[2]))


@dataclass
class Settings:
    root: Path = ROOT
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", f"sqlite:///{ROOT}/data/app.db"))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", ROOT / "data/jobs")))
    model_onnx: Path = field(default_factory=lambda: Path(os.environ.get("MODEL_ONNX", ROOT / "models/onnx/cattle_yolo11s_1024.onnx")))
    detect_mode: str = field(default_factory=lambda: os.environ.get("DETECT_MODE", "sliced"))   # sliced = fatias 1024 nativas; full = quadro inteiro
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "512")))
    api_key: str | None = field(default_factory=lambda: os.environ.get("API_KEY") or None)
    cors_origins: list[str] = field(default_factory=lambda: os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(","))
    conf: float = field(default_factory=lambda: float(os.environ.get("DEFAULT_CONF", "0.25")))
