"""ChromaDB vector store backed by Ollama embeddings.

WHY (architecture decisions)
----------------------------
1. **Persistent, embedded database.** ``chromadb.PersistentClient`` stores data
   in a local folder: no server to run, which fits the offline requirement.
   Telemetry is disabled so nothing leaves the machine.

2. **Cosine distance.** Chroma defaults to L2 distance, whose values are hard
   to interpret. With cosine, ``similarity = 1 - distance`` is a 0-1 score we
   can threshold ("is this chunk relevant enough?") in the retriever.

3. **Model-specific collection name.** Vectors from different embedding models
   are incompatible (different dimensions and geometry). Putting the model name
   in the collection name means switching ``OLLAMA_EMBEDDING_MODEL`` creates a
   fresh collection instead of silently mixing or crashing on mismatched sizes.

4. **nomic-embed-text task prefixes.** The model was trained with
   ``search_document:`` / ``search_query:`` prefixes; omitting them measurably
   hurts retrieval. ``PrefixedEmbeddings`` adds them transparently so no caller
   can forget.

5. **Errors are translated and writes are all-or-nothing.** Ollama being down
   is the most common failure, so embedding errors become ``EmbeddingError``
   with an actionable hint. If indexing fails midway, the partial document is
   removed so search never returns half a file.

6. **Dependency injection.** The embeddings and Chroma client can be injected,
   so tests run against an in-memory client with fake embeddings and never need
   Ollama.

Language note: ``nomic-embed-text`` is trained mainly on English. Arabic works
but with weaker recall; this is why the retriever also searches with an English
translation of the question. If Arabic quality matters most, a multilingual
Ollama model such as ``bge-m3`` is a drop-in replacement via
``OLLAMA_EMBEDDING_MODEL`` (documents must then be re-indexed, which the
model-specific collection name handles automatically).
"""

import hashlib
import logging
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingError, RAGError, VectorStoreError

logger = logging.getLogger(__name__)

_MAX_COLLECTION_NAME = 63  # Chroma's documented upper bound.


# =============================================================================
# Data models
# =============================================================================
class RetrievedChunk(BaseModel):
    """One search hit: the text, its similarity (0-1) and its metadata."""

    model_config = ConfigDict(frozen=True)

    text: str
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity; higher is better.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> Optional[int]:
        page = self.metadata.get("page")
        return int(page) if page is not None else None


class IndexedDocument(BaseModel):
    """Summary of one indexed document, for the UI's document list."""

    doc_id: str
    source: str
    chunk_count: int


# =============================================================================
# Embeddings
# =============================================================================
def _task_prefixes(model_name: str) -> tuple[str, str]:
    """Return ``(document_prefix, query_prefix)`` required by the model family."""
    name = model_name.lower()
    if "nomic-embed" in name:
        return "search_document: ", "search_query: "
    if re.search(r"(?:^|[/_-])e5(?:[/_-]|$)", name):
        return "passage: ", "query: "
    return "", ""


class PrefixedEmbeddings(Embeddings):
    """Wraps an embeddings backend to add task prefixes and translate errors."""

    def __init__(
        self, base: Embeddings, *, doc_prefix: str, query_prefix: str, hint: str
    ) -> None:
        self._base = base
        self._doc_prefix = doc_prefix
        self._query_prefix = query_prefix
        self._hint = hint

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._base.embed_documents([self._doc_prefix + t for t in texts])
        except Exception as exc:
            raise EmbeddingError(f"{self._hint} Cause: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._base.embed_query(self._query_prefix + text)
        except Exception as exc:
            raise EmbeddingError(f"{self._hint} Cause: {exc}") from exc


def build_embeddings(settings: Optional[Settings] = None) -> Embeddings:
    """Create the Ollama embeddings client described by the settings."""
    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {
        "model": cfg.ollama_embedding_model,
        "base_url": cfg.ollama_base_url,
    }
    try:
        base: Embeddings = OllamaEmbeddings(
            **kwargs, client_kwargs={"timeout": cfg.ollama_request_timeout}
        )
    except (TypeError, ValueError):
        # Older langchain-ollama releases do not accept client_kwargs.
        base = OllamaEmbeddings(**kwargs)

    doc_prefix, query_prefix = _task_prefixes(cfg.ollama_embedding_model)
    hint = (
        f"Embedding request failed. Check that Ollama is running at {cfg.ollama_base_url} "
        f"and the model is installed (`ollama pull {cfg.ollama_embedding_model}`)."
    )
    return PrefixedEmbeddings(base, doc_prefix=doc_prefix, query_prefix=query_prefix, hint=hint)


def build_collection_name(base_name: str, model_name: str) -> str:
    """Collection name unique per embedding model, valid for ChromaDB.

    Chroma allows 3-63 chars of ``[a-zA-Z0-9._-]`` starting/ending with an
    alphanumeric. Long names are truncated and given a hash suffix so two
    different long model names can never collide.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{base_name}__{model_name}").strip("_-")
    if len(slug) > _MAX_COLLECTION_NAME:
        digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[: _MAX_COLLECTION_NAME - 9].rstrip('_-')}_{digest}"
    return slug


# =============================================================================
# Vector store service
# =============================================================================
def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only types Chroma accepts (str/int/float/bool); drop ``None``."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


class VectorStoreService:
    """Index, search and delete document chunks in ChromaDB."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embeddings: Optional[Embeddings] = None,
        client: Optional[ClientAPI] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings or build_embeddings(self._settings)
        self.collection_name = build_collection_name(
            self._settings.chroma_collection_name, self._settings.ollama_embedding_model
        )

        try:
            if client is None:
                self._settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(self._settings.chroma_persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            self._client: ClientAPI = client
            self._store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not initialise ChromaDB: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: Sequence[Document], *, replace_existing: bool = True) -> int:
        """Embed and store chunks; return how many were stored.

        Every chunk needs ``doc_id`` and ``chunk_id`` metadata (produced by
        ``chunking.split_document``). With ``replace_existing`` the previous
        version of each document is deleted first, so re-uploading an edited
        file never leaves stale chunks behind.

        Raises:
            VectorStoreError: Invalid input or storage failure.
            EmbeddingError: Ollama unreachable / model missing.
        """
        if not chunks:
            raise VectorStoreError("No chunks to index.")

        prepared: list[Document] = []
        for chunk in chunks:
            meta = chunk.metadata
            if not meta.get("doc_id") or not meta.get("chunk_id"):
                raise VectorStoreError("Every chunk needs 'doc_id' and 'chunk_id' metadata.")
            if not chunk.page_content.strip():
                continue
            prepared.append(
                Document(page_content=chunk.page_content, metadata=_sanitize_metadata(meta))
            )
        if not prepared:
            raise VectorStoreError("All chunks were empty.")

        doc_ids = sorted({str(c.metadata["doc_id"]) for c in prepared})
        batch_size = self._settings.rag_embedding_batch_size

        try:
            if replace_existing:
                for doc_id in doc_ids:
                    self.delete_document(doc_id)
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                self._store.add_documents(
                    documents=batch, ids=[str(d.metadata["chunk_id"]) for d in batch]
                )
        except Exception as exc:
            # All-or-nothing: remove whatever was written before the failure.
            logger.error("Indexing failed; rolling back documents %s.", doc_ids)
            for doc_id in doc_ids:
                try:
                    self.delete_document(doc_id)
                except RAGError:
                    logger.exception("Rollback of document %s failed.", doc_id)
            if isinstance(exc, RAGError):
                raise
            raise VectorStoreError(f"Failed to index chunks: {exc}") from exc

        logger.info("Indexed %d chunks for %d document(s).", len(prepared), len(doc_ids))
        return len(prepared)

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks of a document; return how many were deleted."""
        try:
            collection = self._client.get_collection(self.collection_name)
            ids = collection.get(where={"doc_id": doc_id}, include=[])["ids"]
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{doc_id}': {exc}") from exc

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def similarity_search(
        self, query: str, *, k: int = 4, doc_ids: Optional[Sequence[str]] = None
    ) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks most similar to ``query`` (best first).

        Args:
            doc_ids: Restrict the search to these documents (e.g. the ones the
                user attached to the current chat). ``None`` searches all.
        """
        if not query.strip():
            raise VectorStoreError("Search query must not be empty.")
        if k < 1:
            raise VectorStoreError("k must be at least 1.")
        if self.count() == 0:
            return []

        where: Optional[dict[str, Any]] = None
        if doc_ids:
            unique = sorted(set(doc_ids))
            where = {"doc_id": unique[0]} if len(unique) == 1 else {"doc_id": {"$in": unique}}

        try:
            results = self._store.similarity_search_with_score(query, k=k, filter=where)
        except RAGError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        hits: list[RetrievedChunk] = []
        for document, distance in results:
            # Cosine distance = 1 - similarity. Clamp to [0, 1] for float noise.
            similarity = min(1.0, max(0.0, 1.0 - float(distance)))
            hits.append(
                RetrievedChunk(
                    text=document.page_content, score=similarity, metadata=dict(document.metadata)
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def count(self) -> int:
        """Total number of stored chunks."""
        try:
            return self._client.get_collection(self.collection_name).count()
        except Exception as exc:
            raise VectorStoreError(f"Could not count chunks: {exc}") from exc

    def list_documents(self) -> list[IndexedDocument]:
        """Summarise indexed documents (one entry per ``doc_id``)."""
        try:
            data = self._client.get_collection(self.collection_name).get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Could not list documents: {exc}") from exc

        counts: Counter[str] = Counter()
        sources: dict[str, str] = {}
        for meta in data.get("metadatas") or []:
            doc_id = str(meta.get("doc_id", ""))
            if not doc_id:
                continue
            counts[doc_id] += 1
            sources.setdefault(doc_id, str(meta.get("source", "unknown")))
        return [
            IndexedDocument(doc_id=doc_id, source=sources[doc_id], chunk_count=n)
            for doc_id, n in sorted(counts.items(), key=lambda item: sources[item[0]])
        ]

    def check_embeddings(self) -> int:
        """Embed a probe string; return the vector size. Raises ``EmbeddingError`` on failure.

        Meant for the health endpoint / startup self-test.
        """
        return len(self._embeddings.embed_query("health check"))


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    """Process-wide singleton (one Chroma client per process)."""
    return VectorStoreService()
