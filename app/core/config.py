"""Central application configuration.

Why this module exists
----------------------
Every tunable value (model names, paths, limits, URLs) lives here and nowhere
else. Other modules call ``get_settings()`` instead of reading ``os.environ``
directly, which gives us:

* one validated source of truth (typos in ``.env`` fail fast at startup,
  not 20 minutes later in the middle of a chat request),
* easy overriding in tests (``get_settings.cache_clear()`` + monkeypatch),
* a single place to document what each setting means.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

# Project root = <root>/app/core/config.py -> parents[2].
# We anchor relative paths to this directory instead of the process's current
# working directory. Otherwise `uvicorn` started from /app and `streamlit`
# started from /frontend would silently use two different database files.
BASE_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed, validated application settings loaded from env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        # Env var names are case-insensitive so DATABASE_URL == database_url.
        case_sensitive=False,
        # Ignore unknown variables: a shared .env may hold values for other
        # tools (e.g. Docker), and we must not crash because of them.
        extra="ignore",
    )

    # --- Application ---------------------------------------------------------
    app_name: str = "Local AI Business Advisor"
    app_version: str = "0.1.0"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # --- API server ----------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://127.0.0.1:8000"
    # Stored as a plain comma-separated string on purpose: pydantic-settings
    # expects JSON for List[...] env vars, which is error-prone for humans
    # editing a .env file. The parsed list is exposed via a property below.
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # --- Database ------------------------------------------------------------
    database_url: str = "sqlite:///./data/advisor.db"
    database_echo: bool = False

    # --- Ollama (local LLM) --------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Local inference (especially on CPU) can take minutes; a generous timeout
    # avoids spurious failures while still preventing infinite hangs.
    ollama_request_timeout: int = Field(default=120, ge=1)

    # --- Vector store --------------------------------------------------------
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection_name: str = Field(default="business_documents", min_length=3)

    # --- Uploads -------------------------------------------------------------
    upload_dir: Path = Path("./data/uploads")
    max_upload_size_mb: int = Field(default=25, ge=1, le=1024)
    allowed_upload_extensions: str = ".pdf,.csv,.xlsx,.txt"

    # --- Chat behaviour ------------------------------------------------------
    chat_history_window: int = Field(default=10, ge=0, le=100)
    max_message_length: int = Field(default=8000, ge=1)

    # --- RAG: document processing -------------------------------------------
    pdf_max_pages: int = Field(default=500, ge=1)
    # Harakat (tashkeel) rarely appear in business documents and make the same
    # word look different to the embedding model, so we strip them by default.
    strip_arabic_diacritics: bool = True
    rag_chunk_size: int = Field(default=800, ge=100, le=8000)
    rag_chunk_overlap: int = Field(default=150, ge=0)
    rag_min_chunk_chars: int = Field(default=30, ge=1)
    rag_embedding_batch_size: int = Field(default=32, ge=1, le=512)

    # --- RAG: retrieval ------------------------------------------------------
    rag_top_k: int = Field(default=4, ge=1, le=20)
    # Candidates fetched per query variant before fusion/threshold filtering.
    rag_fetch_k: int = Field(default=8, ge=1, le=50)
    # Minimum cosine similarity (0-1). Tune on real documents; see retriever.py.
    rag_score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    rag_max_context_chars: int = Field(default=6000, ge=500)
    rag_query_max_chars: int = Field(default=2000, ge=50)
    rag_multi_query_enabled: bool = True
    rag_num_paraphrases: int = Field(default=2, ge=0, le=5)
    rag_translate_arabic_queries: bool = True

    # --- Conversation memory -------------------------------------------------
    # Recent messages kept verbatim in the prompt (estimated tokens, see memory.py).
    memory_max_buffer_tokens: int = Field(default=1500, ge=200, le=16000)
    memory_summary_max_words: int = Field(default=200, ge=50, le=1000)
    # How many previous sessions' digests are loaded as long-term memory (0 = off).
    memory_cross_session_limit: int = Field(default=3, ge=0, le=10)
    memory_long_term_max_chars: int = Field(default=3000, ge=200)

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("database_url")
    @classmethod
    def _normalise_sqlite_url(cls, value: str) -> str:
        """Ensure a SQLite URL and anchor relative file paths to BASE_DIR.

        We only support SQLite in this architecture (per the tech stack), so
        rejecting other dialects early gives a clear error instead of an
        obscure driver-not-found failure later.
        """
        try:
            url = make_url(value)
        except ArgumentError as exc:
            raise ValueError(f"DATABASE_URL is not a valid URL: {value!r}") from exc

        if url.get_backend_name() != "sqlite":
            raise ValueError(
                "Only SQLite is supported. Expected DATABASE_URL like "
                "'sqlite:///./data/advisor.db'."
            )

        # ':memory:' / empty database means an in-memory DB (used by tests).
        if url.database in (None, "", ":memory:"):
            return value

        db_path = Path(url.database)
        if not db_path.is_absolute():
            db_path = (BASE_DIR / db_path).resolve()
        # as_posix() keeps the URL valid on Windows too (forward slashes).
        return str(url.set(database=db_path.as_posix()))

    @field_validator("chroma_persist_dir", "upload_dir")
    @classmethod
    def _resolve_relative_path(cls, value: Path) -> Path:
        """Resolve relative directories against the project root (see BASE_DIR)."""
        return value if value.is_absolute() else (BASE_DIR / value).resolve()

    @field_validator("allowed_upload_extensions")
    @classmethod
    def _validate_extensions(cls, value: str) -> str:
        """Normalise extensions to lowercase with a leading dot.

        Users often write ``pdf`` or ``PDF`` in a .env file. Normalising here
        means the upload service can do a simple ``suffix.lower() in allowed``
        check without worrying about formatting.
        """
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("ALLOWED_UPLOAD_EXTENSIONS must contain at least one entry.")
        normalised = [item if item.startswith(".") else f".{item}" for item in items]
        return ",".join(normalised)

    @model_validator(mode="after")
    def _forbid_debug_in_production(self) -> "Settings":
        """Fail fast if DEBUG is left on in production.

        Debug mode can expose stack traces and verbose SQL (which may contain
        confidential business data) to clients, so we refuse to boot rather
        than rely on someone noticing a log line.
        """
        if self.environment == "production" and self.debug:
            raise ValueError("DEBUG must be false when ENVIRONMENT=production.")
        return self

    @model_validator(mode="after")
    def _validate_chunk_geometry(self) -> "Settings":
        """Reject chunk settings that would make the splitter loop or stall.

        An overlap >= chunk size means every new chunk would start before the
        previous one ended, so progress through the text is impossible.
        """
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")
        return self

    # ------------------------------------------------------------------ #
    # Derived values (computed, never read from the environment)
    # ------------------------------------------------------------------ #
    @property
    def cors_origins_list(self) -> List[str]:
        """CORS origins as a clean list, ignoring blanks and stray spaces."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Allowed upload extensions as a list (already normalised)."""
        return self.allowed_upload_extensions.split(",")

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload limit in bytes, which is what request handlers compare against."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def sqlite_file_path(self) -> Path | None:
        """Filesystem path of the SQLite file, or None for in-memory databases."""
        database = make_url(self.database_url).database
        if database in (None, "", ":memory:"):
            return None
        return Path(database)

    # ------------------------------------------------------------------ #
    # Side-effecting helpers
    # ------------------------------------------------------------------ #
    def ensure_directories(self) -> None:
        """Create the data directories the app writes to.

        This is deliberately an explicit method called at application startup
        rather than something done at import time. Import-time side effects
        make tests slow and surprising (merely importing config would create
        folders on disk).

        Raises:
            RuntimeError: If a directory cannot be created (e.g. permissions),
                with a message that names the offending path.
        """
        targets = [self.chroma_persist_dir, self.upload_dir]
        if self.sqlite_file_path is not None:
            targets.append(self.sqlite_file_path.parent)

        for directory in targets:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot create required directory '{directory}': {exc}"
                ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    ``lru_cache`` ensures ``.env`` is parsed and validated only once. Tests can
    reset it with ``get_settings.cache_clear()``. It is also usable as a
    FastAPI dependency: ``settings: Settings = Depends(get_settings)``.
    """
    return Settings()
