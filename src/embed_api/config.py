from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(StrEnum):
    ONNX = "onnx"
    TORCH = "torch"


DEFAULT_MODEL_PATHS = {Backend.ONNX: "models/e5-int8", Backend.TORCH: "models/e5"}


class Settings(BaseSettings):
    """Runtime configuration, read from `EMBED_*` environment variables."""

    model_config = SettingsConfigDict(env_prefix="EMBED_")

    backend: Backend = Field(
        Backend.ONNX,
        description="onnx: the int8 model from scripts/export_onnx.py (default). torch: the "
        "fp32 model from scripts/download_model.py via sentence-transformers (the reference).",
    )
    model_path: str | None = Field(
        None, description="Model directory. Defaults to models/e5-int8 (onnx) or models/e5 (torch)."
    )
    num_threads: int | None = Field(
        None,
        ge=1,
        description="Inference threads. Set it to the container's CPU limit: the runtimes "
        "otherwise size their pools from the host's cores and get throttled.",
    )
    queue_timeout_seconds: float = Field(
        30, gt=0, description="Longest a request waits for inference before 503 overloaded."
    )
    max_total_tokens: int = Field(
        8192, ge=512, description="Token budget per request. Bounds CPU time and memory."
    )
    max_body_bytes: int = Field(1_000_000, ge=1024, description="Largest accepted request body.")
    log_level: str = "INFO"

    @property
    def model_dir(self) -> Path:
        return Path(self.model_path or DEFAULT_MODEL_PATHS[self.backend])
