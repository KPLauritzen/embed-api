from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-large"
DEFAULT_MODEL_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
DEFAULT_ONNX_PATH = "models/e5-int8"


class Backend(StrEnum):
    ONNX = "onnx"
    TORCH = "torch"


class Settings(BaseSettings):
    """Runtime configuration, read from `EMBED_*` environment variables."""

    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")

    backend: Backend = Field(
        Backend.ONNX,
        description="onnx: int8 model exported by scripts/export_onnx.py (default; ~2x faster, "
        "no torch needed). torch: fp32 via sentence-transformers (install the `torch` extra).",
    )
    model_id: str = Field(
        DEFAULT_MODEL_ID,
        description="Hugging Face model id. Reported in responses; downloaded from the Hub "
        "if model_path is unset.",
    )
    model_revision: str = Field(
        DEFAULT_MODEL_REVISION, description="Pinned Hugging Face commit for reproducible output."
    )
    model_path: str | None = Field(
        None,
        description="Local model directory. For onnx it defaults to models/e5-int8; for torch, "
        "unset means downloading model_id from the Hub.",
    )
    device: str | None = Field(
        None, description="torch backend only: cpu, cuda or mps. Auto-selected when unset."
    )
    num_threads: int | None = Field(
        None,
        ge=1,
        description="Inference threads. Set it to the container's CPU limit: torch and "
        "ONNX Runtime otherwise size their pools from the host's cores and get throttled.",
    )
    encode_batch_size: int = Field(16, ge=1, description="Inputs per forward pass.")
    max_concurrent_batches: int = Field(
        1, ge=1, description="Requests encoding at once. 1 lets one batch use every thread."
    )
    max_total_tokens: int = Field(
        8192,
        ge=512,
        description="Token budget per request (after truncation). Bounds CPU time and memory.",
    )
    max_body_bytes: int = Field(1_000_000, ge=1024, description="Largest accepted request body.")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
