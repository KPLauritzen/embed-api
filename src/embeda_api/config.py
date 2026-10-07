from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-large"
DEFAULT_MODEL_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"


class Settings(BaseSettings):
    """Runtime configuration, read from `EMBEDA_*` environment variables."""

    model_config = SettingsConfigDict(env_prefix="EMBEDA_", env_file=".env", extra="ignore")

    model_id: str = Field(
        DEFAULT_MODEL_ID,
        description="Hugging Face model id. Reported in responses; downloaded from the Hub "
        "if model_path is unset.",
    )
    model_revision: str = Field(
        DEFAULT_MODEL_REVISION, description="Pinned Hugging Face commit for reproducible output."
    )
    model_path: str | None = Field(
        None, description="Load the model from this local directory instead of the Hub."
    )
    device: str | None = Field(None, description="cpu, cuda or mps. Auto-selected when unset.")
    num_threads: int | None = Field(
        None,
        ge=1,
        description="Torch intra-op threads. Set it to the container's CPU limit: torch "
        "otherwise sizes its pool from the host's cores and gets throttled.",
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
