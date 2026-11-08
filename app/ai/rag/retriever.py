"""Multi-query retriever: question in, citation-ready context out.

WHY (architecture decisions)
----------------------------
1. **One question is a weak search probe.** A single phrasing may miss a
   relevant chunk that uses different wording. We ask the local chat model for
   a few paraphrases and, for Arabic questions, an English translation. This
   matters here because ``nomic-embed-text`` is English-centric and many
   business documents mix languages: the translated query can hit chunks the
   Arabic one misses. All variants are searched and the lists are merged.

2. **Custom expander instead of LangChain's ``MultiQueryRetriever``.** The
   built-in class has a fixed English prompt, no translation step, and no
   fallback if the LLM misbehaves. Our ``QueryExpander`` is ~60 lines, fully
   controllable and testable with a fake LLM.

3. **Graceful degradation.** Query rewriting is an *optimisation*. If the chat
   model is down, slow, or returns garbage, we log a warning and search with
   the original question only. (If the *embedding* backend is down, the
   original query fails with ``EmbeddingError``, because nothing works
   without it.)

4. **Reciprocal Rank Fusion (RRF).** Scores from different queries are not
   directly comparable, but ranks are. A chunk found by several variants gets
   a higher fused rank, which is a strong sign of relevance. The reported
   ``score`` of a chunk is still its best raw cosine similarity, so the
   similarity threshold keeps a meaningful, tunable meaning.

5. **Threshold + "no context" result.** Weak matches are discarded. When
   nothing is relevant, ``has_context`` is False so the answering layer can
   say "the documents do not cover this" rather than hallucinate from noise.

6. **Budgeted context and structured sources.** The context string is capped
   (``RAG_MAX_CONTEXT_CHARS``) to protect the LLM's context window, and each
   block is labelled ``[Source n: file, page p]``. ``sources`` mirrors exactly
   the chunks included, in the JSON shape of ``Message.sources`` so it can be
   stored with the assistant's answer.

Security note: retrieved text comes from uploaded files and is UNTRUSTED. It
may contain instructions ("ignore previous rules..."). The system prompt must
tell the model to treat the context as data only.
"""

import logging
import re
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from app.ai.rag.chunking import contains_arabic
from app.ai.rag.loader import normalize_text
from app.ai.rag.vector_store import RetrievedChunk, VectorStoreService, get_vector_store
from app.core.config import Settings, get_settings
from app.core.exceptions import RAGError, RetrievalError

logger = logging.getLogger(__name__)

# Standard RRF constant: damps the advantage of rank 1 over rank 2, so one
# lucky top hit cannot dominate chunks found consistently by many queries.
_RRF_K = 60
_MAX_VARIANT_CHARS = 300
_SNIPPET_CHARS = 200
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]+|\d+\s*[.)\-:])\s*")


# =============================================================================
# Data models
# =============================================================================
class RetrieverConfig(BaseModel):
    """Retrieval behaviour; built from settings, overridable per call/test."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=4, ge=1, le=20)
    fetch_k: int = Field(default=8, ge=1, le=50)
    score_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=200)
    query_max_chars: int = Field(default=2000, ge=50)
    multi_query_enabled: bool = True
    num_paraphrases: int = Field(default=2, ge=0, le=5)
    translate_arabic: bool = True
    strip_diacritics: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrieverConfig":
        return cls(
            top_k=settings.rag_top_k,
            fetch_k=settings.rag_fetch_k,
            score_threshold=settings.rag_score_threshold,
            max_context_chars=settings.rag_max_context_chars,
            query_max_chars=settings.rag_query_max_chars,
            multi_query_enabled=settings.rag_multi_query_enabled,
            num_paraphrases=settings.rag_num_paraphrases,
            translate_arabic=settings.rag_translate_arabic_queries,
            strip_diacritics=settings.strip_arabic_diacritics,
        )


class RetrievalResult(BaseModel):
    """Everything the answering layer needs from one retrieval."""

    model_config = ConfigDict(frozen=True)

    question: str
    queries_used: list[str]
    chunks: list[RetrievedChunk]
    context: str = Field(description="Formatted, budgeted text to place in the LLM prompt.")
    sources: list[dict[str, Any]] = Field(
        description="Citation records (JSON-safe), matching Message.sources."
    )

    @property
    def has_context(self) -> bool:
        return bool(self.chunks)


# =============================================================================
# Query expansion
# =============================================================================
_EXPANDER_SYSTEM_PROMPT = (
    "You rewrite search queries for a document retrieval system. "
    "You never answer the question. You output only search queries, one per line, "
    "with no numbering, no bullets, no quotes and no explanations."
)


class QueryExpander:
    """Uses the local chat model to produce paraphrases and an English translation."""

    def __init__(self, llm: BaseChatModel, config: RetrieverConfig) -> None:
        self._llm = llm
        self._config = config

    def expand(self, question: str) -> list[str]:
        """Return extra query variants (never includes ``question`` itself).

        Never raises: any failure yields an empty list so retrieval continues
        with the original question (see module docstring, point 3).
        """
        wants_translation = self._config.translate_arabic and contains_arabic(question)
        n = self._config.num_paraphrases
        if n == 0 and not wants_translation:
            return []

        lines = []
        if n > 0:
            lines.append(
                f"Write {n} alternative search queries with the same meaning but different "
                "wording or synonyms, in the SAME language as the original."
            )
        if wants_translation:
            lines.append(
                "Add exactly one more line containing a faithful English translation "
                "of the question."
            )
        lines.append("Keep names, numbers and dates unchanged.")
        user_prompt = f"Original question:\n{question}\n\n" + " ".join(lines)

        try:
            response = self._llm.invoke(
                [SystemMessage(content=_EXPANDER_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
            )
            raw = response.content if isinstance(response.content, str) else str(response.content)
        except Exception as exc:  # network, timeout, model missing: all non-fatal here
            logger.warning("Query expansion failed; using the original query only: %s", exc)
            return []

        return self._parse(raw, question, limit=n + (1 if wants_translation else 0))

    def _parse(self, raw: str, original: str, limit: int) -> list[str]:
        """Turn the model's free-form output into a clean, de-duplicated list.

        Small local models often add numbering, intros ("Here are..."), or
        blank lines despite instructions, so parsing is deliberately forgiving.
        """
        seen = {original.casefold()}
        variants: list[str] = []
        for line in raw.splitlines():
            candidate = _LIST_MARKER_RE.sub("", line).strip().strip("\"'`«»“”")
            if not candidate or candidate.endswith(":"):
                continue
            if len(candidate) > _MAX_VARIANT_CHARS:
                continue
            candidate = normalize_text(candidate, self._config.strip_diacritics)
            if not candidate or candidate.casefold() in seen:
                continue
            seen.add(candidate.casefold())
            variants.append(candidate)
            if len(variants) >= limit:
                break
        return variants


def build_query_llm(settings: Optional[Settings] = None) -> BaseChatModel:
    """Chat model used only for query rewriting (deterministic, short output)."""
    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {
        "model": cfg.ollama_chat_model,
        "base_url": cfg.ollama_base_url,
        "temperature": 0.0,
        "num_predict": 200,  # a handful of short queries never needs more
    }
    try:
        return ChatOllama(**kwargs, client_kwargs={"timeout": cfg.ollama_request_timeout})
    except (TypeError, ValueError):
        return ChatOllama(**kwargs)


# =============================================================================
# Retriever
# =============================================================================
class MultiQueryRetriever:
    """Retrieve, fuse and format the most relevant chunks for a question."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        *,
        config: Optional[RetrieverConfig] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        self._store = vector_store
        self._config = config or RetrieverConfig.from_settings(get_settings())
        self._expander: Optional[QueryExpander] = None
        if self._config.multi_query_enabled:
            self._expander = QueryExpander(llm or build_query_llm(), self._config)

    def retrieve(
        self,
        question: str,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Find the best context for ``question``.

        Args:
            doc_ids: Limit the search to these documents.
            top_k: Override the number of chunks to return.

        Raises:
            RetrievalError: Empty question or invalid ``top_k``.
            EmbeddingError / VectorStoreError: Backend failures on the main query.
        """
        cfg = self._config
        k = top_k if top_k is not None else cfg.top_k
        if k < 1:
            raise RetrievalError("top_k must be at least 1.")

        cleaned = normalize_text(question or "", cfg.strip_diacritics)
        if not cleaned:
            raise RetrievalError("Question must not be empty.")
        if len(cleaned) > cfg.query_max_chars:
            logger.warning("Question truncated from %d to %d chars.", len(cleaned), cfg.query_max_chars)
            cleaned = cleaned[: cfg.query_max_chars]

        queries = [cleaned]
        if self._expander is not None:
            queries.extend(self._expander.expand(cleaned))

        ranked_lists: list[list[RetrievedChunk]] = []
        for position, query in enumerate(queries):
            try:
                hits = self._store.similarity_search(query, k=cfg.fetch_k, doc_ids=doc_ids)
            except RAGError:
                if position == 0:
                    raise  # the original question must succeed
                logger.warning("Search with variant failed and was skipped: %r", query)
                continue
            ranked_lists.append([h for h in hits if h.score >= cfg.score_threshold])

        fused = self._fuse(ranked_lists)[:k]
        context, used = self._build_context(fused)
        logger.info(
            "Retrieved %d chunk(s) for %d query variant(s).", len(used), len(queries)
        )
        return RetrievalResult(
            question=cleaned,
            queries_used=queries,
            chunks=used,
            context=context,
            sources=[self._source_record(chunk) for chunk in used],
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fuse(ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        """Merge ranked lists with Reciprocal Rank Fusion; drop duplicate text."""
        fused_scores: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}

        for hits in ranked_lists:
            for rank, hit in enumerate(hits):
                key = hit.chunk_id or hit.text
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
                if key not in best or hit.score > best[key].score:
                    best[key] = hit

        ordered = sorted(
            best, key=lambda key: (fused_scores[key], best[key].score), reverse=True
        )
        result: list[RetrievedChunk] = []
        seen_text: set[str] = set()
        for key in ordered:
            fingerprint = " ".join(best[key].text.split()).casefold()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            result.append(best[key])
        return result

    def _build_context(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk]]:
        """Format chunks into labelled blocks within the character budget."""
        budget = self._config.max_context_chars
        blocks: list[str] = []
        used: list[RetrievedChunk] = []
        remaining = budget

        for chunk in chunks:
            page = f", page {chunk.page}" if chunk.page is not None else ""
            header = f"[Source {len(used) + 1}: {chunk.source}{page}]\n"
            body = chunk.text
            if len(header) + len(body) > remaining:
                if used:
                    break  # keep whole chunks; stop once the next one does not fit
                # A single oversized first chunk is truncated rather than dropped.
                body = body[: max(0, remaining - len(header))].rstrip()
                if not body:
                    break
                chunk = chunk.model_copy(update={"text": body})
            blocks.append(header + body)
            used.append(chunk)
            remaining -= len(header) + len(body) + 2  # +2 for the blank line separator

        return "\n\n".join(blocks), used

    @staticmethod
    def _source_record(chunk: RetrievedChunk) -> dict[str, Any]:
        """JSON-safe citation record stored in ``Message.sources``."""
        snippet = " ".join(chunk.text.split())
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[:_SNIPPET_CHARS].rstrip() + "…"
        return {
            "doc_id": chunk.metadata.get("doc_id"),
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "page": chunk.page,
            "score": round(chunk.score, 4),
            "snippet": snippet,
        }


def get_retriever() -> MultiQueryRetriever:
    """Convenience factory wired to the singleton vector store and settings."""
    return MultiQueryRetriever(get_vector_store())
