#!/usr/bin/env python
"""Arabic dataset preparation for QLoRA fine-tuning (Phase 3, step 1).

ARCHITECTURE
------------
The pipeline is a chain of small, single-purpose classes orchestrated by
``DataPreparer``::

    RecordLoader -> RecordValidator -> (dedupe) -> ChatFormatter -> GroupedSplitter
                                                                   -> DatasetWriter

* ``RecordLoader``     reads JSON / JSONL files; unreadable *files* are fatal,
                       unreadable *records* are rejected and reported.
* ``ArabicNormalizer`` cleans text under an explicit, configurable policy.
* ``RecordValidator``  enforces schema, types, language and safety rules.
* ``ChatFormatter``    builds system/user/assistant messages with the project's
                       real system prompt and the tokenizer's native template.
* ``NearDuplicateGrouper`` + ``GroupedSplitter`` keep related prompts together
                       so the test set never contains a rephrasing of a training prompt.
* ``DatasetWriter``    atomic UTF-8 JSONL output plus a reproducibility manifest.

KEY DECISIONS
-------------
1. **Two normalisation levels.** Text that becomes training data is cleaned
   *lightly* (diacritics, tatweel, invisible characters, whitespace). Alef/Yaa
   folding is opt-in because it changes spelling, and the model must learn to
   write correct Fusha. Duplicate detection uses a separate *aggressive* fold
   (alef, yaa, digits, punctuation) that is never written to the dataset.
   Financial symbols (% ٪ $ € £ ¥ ﷼) are preserved in both.

2. **Compact system prompt, built by the app.** Records use
   ``build_compact_system_prompt()`` from ``app/ai/prompts/system_prompts.py``:
   the same behavioural rules in ~15% of the space, so the 512-token training
   window is spent on the question and answer. The compact builder reuses the
   production builder for the ``<documents>`` block, so that structure cannot
   drift. No prompt text lives in this file; its hash is stored in the manifest.
   ``--prompt-function build_system_prompt`` switches to the full prompt.

3. **Native chat template, verified.** Messages are rendered with
   ``tokenizer.apply_chat_template``. A probe checks the Qwen2.5 ChatML markers
   and that the template did not inject its own default system prompt.

4. **The token budget is enforced, not assumed.** The system prompt is resent
   in every example. If it leaves too little room inside ``max_seq_length``, the
   run stops immediately with exact numbers. Over-long examples are dropped (or
   raise) - never truncated, because truncated answers teach the model to stop
   mid-sentence.

5. **Grouped, deterministic splitting.** Exact duplicates are removed first.
   Records whose prompts match after folding (or are near-identical) form a
   group, and groups are assigned to splits as units (seeded, order-independent).

6. **Dependency injection.** The tokenizer and prompt builder can be injected,
   so the whole pipeline is unit-testable without downloads or LangChain.

TRAIN/SERVE CONVERSATION STRUCTURE (documented differences)
------------------------------------------------------------
Identical to production: ChatML roles, a system message that ends with the same
``<documents>`` block (the standard "no relevant context" notice when a record
has no context; production-format ``[Source n: file, page p]`` blocks when it
has), and citations in the form ``(المصدر 1، ص 3)``. Context is never invented:
blank or missing context produces no source blocks, and a record's context must
be in production format (``context_sources`` are rendered by the shared
``app/ai/rag/context_format.py``; a pre-formatted ``context`` string is validated).

Known differences from production, by design:
  1. The system message starts with the COMPACT rule text, not the full prompt.
  2. Examples are single-turn (no chat history).
  3. ``<session_summary>`` / ``<past_conversations>`` memory sections are absent.
  4. The user message is ``instruction`` plus the optional ``input`` field.

Requirements: standard library, ``transformers`` (tokenizer only, no GPU) and
the app's prompt module (needs ``langchain-core``). Run from the project root.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, IO, Literal, Optional, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow `python training/prepare_data.py` (not only `python -m training.prepare_data`).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shared, standard-library-only definition of the production context format.
from app.ai.rag.context_format import (  # noqa: E402
    format_context,
    validate_formatted_context,
)

logger = logging.getLogger("prepare_data")

# =============================================================================
# Constants
# =============================================================================
DEFAULT_TOKENIZER = "unsloth/Qwen3-8B-bnb-4bit"
DEFAULT_PROMPT_FUNCTION = "build_compact_system_prompt"
FULL_PROMPT_FUNCTION = "build_system_prompt"
PROMPT_MODULE = "app.ai.prompts.system_prompts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "data" / "processed"
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_SEED = 42

# Room that must remain for the user message and the answer after the system prompt.
MIN_RESPONSE_TOKENS = 64
# Below this much room the compact prompt is technically usable but cramped: warn.
RECOMMENDED_ROOM_TOKENS = 224
# Warn when more than this share of records is dropped for length.
OVERLENGTH_WARN_FRACTION = 0.10
# Reference length that is always reported on, whatever --max-seq-length is.
REFERENCE_LENGTH = 512
REPORT_THRESHOLDS = (128, 256, 384, 512, 768, 1024)
# Answers that are placeholders rather than content (compared after folding).
PLACEHOLDER_OUTPUTS = frozenset(
    {"n a", "na", "none", "null", "nil", "undefined", "nan", "tbd", "todo", "empty",
     "no answer", "no output", "lorem ipsum"}
)
# Below this size an empty split is a warning; above it, an error.
MIN_RECORDS_FOR_FULL_SPLIT = 30
# Near-duplicate search compares candidate pairs; skip it beyond this size.
NEAR_DUP_MAX_RECORDS = 20_000
# Ignore shingles shared by more documents than this (very common phrases).
NEAR_DUP_MAX_BUCKET = 200

SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_FIELDS = ("instruction", "output")
OPTIONAL_TEXT_FIELDS = ("input", "context")
CONTEXT_SOURCES_FIELD = "context_sources"
WRAPPER_KEYS = ("data", "records", "items", "examples")
# Text that would corrupt ChatML structure or loss masking if it appeared in data.
CHATML_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
REJECT_PREVIEW_CHARS = 500
MANIFEST_PACKAGES = (
    "transformers", "tokenizers", "datasets", "torch", "unsloth", "trl", "peft", "bitsandbytes",
)


# =============================================================================
# Exceptions
# =============================================================================
class DataPreparationError(Exception):
    """Base class for every expected failure of this pipeline."""


class InputFileError(DataPreparationError):
    """An input file is missing, unreadable, or structurally unusable."""


class TokenizerLoadError(DataPreparationError):
    """The tokenizer could not be loaded."""


class TemplateMismatchError(DataPreparationError):
    """The tokenizer's chat template is not the expected Qwen2.5 ChatML format."""


class PromptBudgetError(DataPreparationError):
    """The system prompt leaves too little room inside ``max_seq_length``."""


class DatasetTooSmallError(DataPreparationError):
    """Too few usable records remain to build the requested splits."""


class DatasetSplitError(DataPreparationError):
    """Records could not be split without leakage or empty splits."""


class OutputWriteError(DataPreparationError):
    """An output file could not be written."""


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class NormalizationPolicy:
    """What the *training text* normaliser may change (see module docstring, 1)."""

    remove_diacritics: bool = True
    remove_tatweel: bool = True
    alef: Literal["keep", "normalize"] = "keep"      # أ إ آ ٱ -> ا
    yaa: Literal["keep", "normalize"] = "keep"       # ى ی -> ي
    digits: Literal["keep", "western"] = "keep"      # ٠-٩ ۰-۹ -> 0-9


@dataclass(frozen=True)
class PreparationConfig:
    input_paths: tuple[Path, ...]
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tokenizer_name: str = DEFAULT_TOKENIZER
    prompt_function: str = DEFAULT_PROMPT_FUNCTION
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    seed: int = DEFAULT_SEED
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    min_output_chars: int = 20
    min_arabic_ratio: float = 0.0  # 0 = language check disabled (default)
    near_duplicate_threshold: float = 0.85
    overlength_policy: Literal["drop", "error"] = "drop"
    normalization: NormalizationPolicy = field(default_factory=NormalizationPolicy)
    dry_run: bool = False

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.validation_ratio

    def validate(self) -> None:
        """Raise ``ValueError`` for settings that can never produce a valid dataset."""
        if not self.input_paths:
            raise ValueError("At least one input path is required.")
        if not (0 < self.train_ratio < 1 and 0 < self.validation_ratio < 1):
            raise ValueError("train_ratio and validation_ratio must be between 0 and 1.")
        if self.test_ratio <= 1e-9:
            raise ValueError("train_ratio + validation_ratio must be less than 1 (test needs data).")
        if not 0.0 <= self.min_arabic_ratio <= 1.0:
            raise ValueError("min_arabic_ratio must be between 0 and 1.")
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0 and 1 (0 disables it).")
        if self.max_seq_length < 128:
            raise ValueError("max_seq_length must be at least 128.")
        if self.min_output_chars < 1:
            raise ValueError("min_output_chars must be at least 1.")


# =============================================================================
# Arabic normalisation
# =============================================================================
_INVISIBLE_RE = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DIACRITICS_RE = re.compile("[\u0610-\u061a\u064b-\u065f\u0670]")
_TATWEEL = "\u0640"
# Arabic Presentation Forms-A/-B (glyph shapes found in PDF-derived text). The
# range deliberately skips U+FDFC, the rial sign "﷼", whose NFKC expansion would
# replace a currency symbol with Persian-script letters.
_PRESENTATION_FORMS_RE = re.compile("[\ufb50-\ufdfb\ufe70-\ufefc]")
_HSPACE_RE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" *\n *")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# Characters that carry business meaning and must survive duplicate folding.
_FOLD_DROP_RE = re.compile("[^\\w\\s%\u066a$\u20ac\u00a3\u00a5\ufdfc]")

_ALEF_MAP = {ord(c): "ا" for c in "أإآٱ"}
_YAA_MAP = {ord(c): "ي" for c in "ىی"}
_DIGIT_MAP = {
    **{0x0660 + i: str(i) for i in range(10)},
    **{0x06F0 + i: str(i) for i in range(10)},
}
_ARABIC_LETTER_RE = re.compile("[\u0621-\u064a\u0671-\u06d3\u06fa-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN_LETTER_RE = re.compile("[A-Za-z]")
_ARABIC_INDIC_DIGIT_RE = re.compile("[\u0660-\u0669\u06f0-\u06f9]")


def arabic_letter_ratio(text: str) -> float:
    """Share of Arabic letters among Arabic + Latin letters (1.0 if there are none)."""
    arabic = len(_ARABIC_LETTER_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    total = arabic + latin
    return arabic / total if total else 1.0


class ArabicNormalizer:
    """Cleans text for training under a ``NormalizationPolicy``; folds text for hashing."""

    def __init__(self, policy: NormalizationPolicy) -> None:
        self._policy = policy

    def normalize(self, text: str) -> str:
        """Light, structure-preserving clean-up used for the actual dataset text.

        Order matters: presentation forms are expanded and the text is NFC-composed
        *before* diacritics are removed, otherwise a decomposed "آ" (alef + madda)
        would lose its madda and silently become "ا".
        """
        policy = self._policy
        text = _PRESENTATION_FORMS_RE.sub(
            lambda match: unicodedata.normalize("NFKC", match.group()), text
        )
        text = unicodedata.normalize("NFC", text)
        text = _CONTROL_RE.sub("", text)
        text = _INVISIBLE_RE.sub("", text)
        if policy.remove_diacritics:
            text = _DIACRITICS_RE.sub("", text)
        if policy.remove_tatweel:
            text = text.replace(_TATWEEL, "")
        if policy.alef == "normalize":
            text = text.translate(_ALEF_MAP)
        if policy.yaa == "normalize":
            text = text.translate(_YAA_MAP)
        if policy.digits == "western":
            text = text.translate(_DIGIT_MAP)
        return self._normalize_whitespace(text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse runs of spaces but keep line breaks (answers contain lists)."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _HSPACE_RE.sub(" ", text)
        text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
        text = _BLANK_LINES_RE.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def fold(text: str) -> str:
        """Aggressive normal form used ONLY for duplicate detection, never stored.

        Spelling variants, digits scripts, case, spacing and incidental punctuation
        are erased. ``%``, ``٪`` and currency symbols are kept so that "5%" and "5"
        are never treated as the same figure.
        """
        text = unicodedata.normalize("NFKC", text)
        text = _INVISIBLE_RE.sub("", text)
        text = _DIACRITICS_RE.sub("", text).replace(_TATWEEL, "")
        text = text.translate({**_ALEF_MAP, **_YAA_MAP, **_DIGIT_MAP}).casefold()
        text = _FOLD_DROP_RE.sub(" ", text)
        return " ".join(text.split())


# =============================================================================
# Records
# =============================================================================
@dataclass(frozen=True)
class RawItem:
    source: str
    position: str
    data: Any


@dataclass(frozen=True)
class CleanRecord:
    instruction: str
    input: str
    output: str
    context: str
    source: str
    position: str


@dataclass(frozen=True)
class RejectedRecord:
    source: str
    position: str
    category: str
    detail: str
    preview: str = ""


@dataclass(frozen=True)
class Example:
    """A validated record rendered as a conversation, ready to be written."""

    record: CleanRecord
    exact_key: str
    prompt_fold: str
    messages: list[dict[str, str]]
    text: str
    num_tokens: int
    group_id: str = ""

    @property
    def example_id(self) -> str:
        return self.exact_key[:16]

    @property
    def has_context(self) -> bool:
        return bool(self.record.context)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "group_id": self.group_id[:16],
            "messages": self.messages,
            "text": self.text,
            "num_tokens": self.num_tokens,
        }


# =============================================================================
# Loading
# =============================================================================
class RecordLoader:
    """Reads ``.json`` / ``.jsonl`` files (or directories of them) into raw items."""

    def load(self, paths: Sequence[Path]) -> tuple[list[RawItem], list[RejectedRecord]]:
        items: list[RawItem] = []
        rejected: list[RejectedRecord] = []
        for path in self._expand(paths):
            suffix = path.suffix.lower()
            text = self._read_text(path)
            if suffix == ".jsonl":
                self._parse_jsonl(path, text, items, rejected)
            else:
                self._parse_json(path, text, items)
            logger.info("Read %s", path)
        if not items and not rejected:
            raise InputFileError("The input files contain no records.")
        return items, rejected

    @staticmethod
    def _expand(paths: Sequence[Path]) -> list[Path]:
        files: list[Path] = []
        for path in paths:
            if path.is_dir():
                found = sorted(p for p in path.iterdir() if p.suffix.lower() in (".json", ".jsonl"))
                if not found:
                    raise InputFileError(f"Directory contains no .json/.jsonl files: {path}")
                files.extend(found)
            elif path.is_file():
                if path.suffix.lower() not in (".json", ".jsonl"):
                    raise InputFileError(f"Unsupported input format '{path.suffix}': {path}")
                files.append(path)
            else:
                raise InputFileError(f"Input path does not exist: {path}")
        return files

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            # utf-8-sig transparently drops the BOM that Windows editors add.
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputFileError(f"{path} is not valid UTF-8: {exc}") from exc
        except OSError as exc:
            raise InputFileError(f"Cannot read {path}: {exc}") from exc

    @staticmethod
    def _parse_jsonl(
        path: Path, text: str, items: list[RawItem], rejected: list[RejectedRecord]
    ) -> None:
        # Split on "\n" only: str.splitlines() would also split on U+2028, which
        # may legally appear inside a JSON string.
        for number, line in enumerate(text.split("\n"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(
                    RejectedRecord(
                        str(path), f"line {number}", "invalid_json", str(exc),
                        line[:REJECT_PREVIEW_CHARS],
                    )
                )
                continue
            items.append(RawItem(str(path), f"line {number}", data))

    @staticmethod
    def _parse_json(path: Path, text: str, items: list[RawItem]) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputFileError(
                f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg}). "
                "Fix the file, or convert it to JSONL so one bad record cannot invalidate the rest."
            ) from exc

        if isinstance(data, dict):
            wrapped = next((data[k] for k in WRAPPER_KEYS if isinstance(data.get(k), list)), None)
            if wrapped is not None:
                data = wrapped
            elif "instruction" in data:
                data = [data]
            else:
                raise InputFileError(
                    f"{path}: expected a list of records, or an object wrapping one under "
                    f"{', '.join(WRAPPER_KEYS)}."
                )
        if not isinstance(data, list):
            raise InputFileError(f"{path}: top-level JSON must be a list of records.")
        for index, element in enumerate(data, start=1):
            items.append(RawItem(str(path), f"item {index}", element))


# =============================================================================
# Validation
# =============================================================================
class RecordValidator:
    """Applies schema, type, content, context and (optional) language rules to raw items."""

    def __init__(self, normalizer: ArabicNormalizer, config: PreparationConfig) -> None:
        self._normalizer = normalizer
        self._config = config
        self.style_warnings: Counter[str] = Counter()

    def validate(self, item: RawItem) -> CleanRecord | RejectedRecord:
        def reject(category: str, detail: str) -> RejectedRecord:
            preview = json.dumps(item.data, ensure_ascii=False, default=str)
            return RejectedRecord(
                item.source, item.position, category, detail, preview[:REJECT_PREVIEW_CHARS]
            )

        data = item.data
        if not isinstance(data, dict):
            return reject("not_an_object", f"record is a {type(data).__name__}, expected an object")

        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            return reject("missing_field", f"missing required field(s): {', '.join(missing)}")

        for name in (*REQUIRED_FIELDS, *OPTIONAL_TEXT_FIELDS):
            value = data.get(name)
            if value is None and name in OPTIONAL_TEXT_FIELDS:
                continue
            if not isinstance(value, str):
                return reject(
                    "bad_type", f"field '{name}' must be a string, got {type(value).__name__}"
                )

        fields = {
            name: self._normalizer.normalize(data.get(name) or "")
            for name in (*REQUIRED_FIELDS, *OPTIONAL_TEXT_FIELDS)
        }
        for name in REQUIRED_FIELDS:
            if not fields[name]:
                return reject("empty_field", f"field '{name}' is empty after cleaning")

        context, problem = self._resolve_context(data, fields["context"])
        if problem:
            return reject(*problem)
        fields["context"] = context

        for name, value in fields.items():
            if any(token in value for token in CHATML_CONTROL_TOKENS):
                return reject(
                    "chatml_control_tokens", f"field '{name}' contains ChatML control tokens"
                )

        output = fields["output"]
        if len(output) < self._config.min_output_chars:
            return reject(
                "output_too_short",
                f"output has {len(output)} characters (minimum {self._config.min_output_chars})",
            )
        if not any(char.isalpha() for char in output):
            return reject("output_no_letters", "output contains no letters (only digits/symbols)")
        if ArabicNormalizer.fold(output) in PLACEHOLDER_OUTPUTS:
            return reject("output_placeholder", f"output is a placeholder ({output!r})")
        if self._config.min_arabic_ratio > 0:
            ratio = arabic_letter_ratio(output)
            if ratio < self._config.min_arabic_ratio:
                return reject(
                    "output_not_arabic",
                    f"output is {ratio:.0%} Arabic letters "
                    f"(minimum {self._config.min_arabic_ratio:.0%})",
                )
        if _ARABIC_INDIC_DIGIT_RE.search(output):
            # Not an error: the system prompt asks for 0-9, so flag it for the summary.
            self.style_warnings["arabic_indic_digits_in_output"] += 1

        return CleanRecord(
            instruction=fields["instruction"],
            input=fields["input"],
            output=output,
            context=context,
            source=item.source,
            position=item.position,
        )

    def _resolve_context(
        self, data: dict[str, Any], formatted: str
    ) -> tuple[str, Optional[tuple[str, str]]]:
        """Return ``(production-format context, None)`` or ``("", (category, detail))``.

        Missing, ``None``, empty or whitespace-only context yields ``""``: no
        source blocks are ever fabricated or left empty.
        """
        sources = data.get(CONTEXT_SOURCES_FIELD)
        if sources in (None, []):
            if not formatted:
                return "", None
            problems = validate_formatted_context(formatted)
            if problems:
                return "", (
                    "context_format",
                    "'context' is not in production format: " + "; ".join(problems)
                    + ". Use 'context_sources' to have it formatted for you.",
                )
            return formatted, None

        if formatted:
            return "", ("context_conflict", "give either 'context' or 'context_sources', not both")
        if not isinstance(sources, list):
            return "", ("bad_type", "field 'context_sources' must be a list of objects")

        cleaned: list[dict[str, Any]] = []
        for number, entry in enumerate(sources, start=1):
            label = f"context_sources[{number}]"
            if not isinstance(entry, dict):
                return "", ("bad_context_source", f"{label} must be an object")
            name, text, page = entry.get("source"), entry.get("text"), entry.get("page")
            if not isinstance(name, str) or not name.strip() or "\n" in name or "]" in name:
                return "", ("bad_context_source",
                            f"{label} needs a one-line string 'source' without ']'")
            if not isinstance(text, str) or not self._normalizer.normalize(text):
                return "", ("empty_context", f"{label} has empty or non-string 'text'")
            if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
                return "", ("bad_context_source", f"{label} 'page' must be a positive integer")
            cleaned.append(
                {"source": name.strip(), "page": page, "text": self._normalizer.normalize(text)}
            )
        return format_context(cleaned), None


# =============================================================================
# Chat formatting (system prompt + native chat template)
# =============================================================================
class ChatTokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this module uses."""

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(self, text: str, *, add_special_tokens: bool) -> Any: ...


PromptBuilder = Callable[..., str]


def load_tokenizer(name: str) -> ChatTokenizer:
    """Load a tokenizer by hub id or local folder (no model weights, no GPU)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerLoadError("The 'transformers' package is required to load the tokenizer.") from exc
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # network, auth, or corrupt cache; re-raised with advice
        raise TokenizerLoadError(
            f"Could not load tokenizer '{name}': {exc}\n"
            "If you are offline, download it once while online or pass a local folder "
            "with --tokenizer."
        ) from exc


def load_prompt_builder(function_name: str) -> PromptBuilder:
    """Import the project's system-prompt function (single source of truth)."""
    try:
        module = importlib.import_module(PROMPT_MODULE)
    except ImportError as exc:
        raise DataPreparationError(
            f"Cannot import {PROMPT_MODULE}: {exc}. Run from the project root in an "
            "environment where the app's light dependencies (langchain-core) are installed."
        ) from exc
    builder = getattr(module, function_name, None)
    if not callable(builder):
        raise DataPreparationError(f"{PROMPT_MODULE} has no callable named '{function_name}'.")
    return builder


class ChatFormatter:
    """Turns clean records into ChatML conversations and counts their tokens."""

    _PROBE_SYSTEM = "SYSTEM_PROBE_TEXT"
    _PROBE_USER = "USER_PROBE_TEXT"
    _PROBE_ASSISTANT = "ASSISTANT_PROBE_TEXT"

    def __init__(self, tokenizer: ChatTokenizer, prompt_builder: PromptBuilder) -> None:
        self._tokenizer = tokenizer
        self._prompt_builder = prompt_builder

    # -- template ---------------------------------------------------------- #
    def verify_template(self) -> None:
        """Fail fast unless the tokenizer renders Qwen2.5-style ChatML as expected."""
        probe = [
            {"role": "system", "content": self._PROBE_SYSTEM},
            {"role": "user", "content": self._PROBE_USER},
            {"role": "assistant", "content": self._PROBE_ASSISTANT},
        ]
        try:
            rendered = self.render(probe)
        except Exception as exc:
            raise TemplateMismatchError(
                f"The tokenizer could not render a chat template: {exc}"
            ) from exc

        # Qwen3 inserts <think>...</think> before the assistant content. Strip it
        # before checking so the same verification works for Qwen2.5 and Qwen3.
        import re as _re
        rendered_clean = _re.sub(r"<think>.*?</think>\s*", "", rendered, flags=_re.DOTALL)

        expected = (
            f"<|im_start|>system\n{self._PROBE_SYSTEM}<|im_end|>",
            f"<|im_start|>user\n{self._PROBE_USER}<|im_end|>",
            f"<|im_start|>assistant\n{self._PROBE_ASSISTANT}<|im_end|>",
        )
        missing = [marker for marker in expected if marker not in rendered_clean]
        if missing:
            raise TemplateMismatchError(
                "The chat template does not match the Qwen2.5 ChatML format. Missing: "
                f"{missing}. Rendered probe: {rendered[:300]!r}"
            )
        if "You are Qwen" in rendered:
            raise TemplateMismatchError(
                "The template injected its default system prompt despite an explicit one."
            )
        if not rendered.rstrip().endswith("<|im_end|>"):
            raise TemplateMismatchError("The rendered conversation does not end with <|im_end|>.")

    def render(self, messages: list[dict[str, str]]) -> str:
        """Render with the tokenizer's native template (training format, no generation prompt)."""
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def count_tokens(self, text: str) -> int:
        """Token count of rendered text; control tokens like <|im_start|> count as one each."""
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    # -- prompt ------------------------------------------------------------ #
    def system_prompt(self, context: str = "") -> str:
        prompt = self._prompt_builder(rag_context=context) if context else self._prompt_builder()
        if not isinstance(prompt, str) or not prompt.strip():
            raise DataPreparationError("The system prompt builder returned an empty prompt.")
        return prompt

    def overhead_tokens(self) -> int:
        """Tokens used by the system prompt and template before any real content."""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": "س"},
            {"role": "assistant", "content": "ج"},
        ]
        return self.count_tokens(self.render(messages))

    # -- records ----------------------------------------------------------- #
    @staticmethod
    def user_message(record: CleanRecord) -> str:
        """Alpaca convention: instruction, then the optional input after a blank line."""
        return f"{record.instruction}\n\n{record.input}" if record.input else record.instruction

    def build_messages(self, record: CleanRecord) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt(record.context)},
            {"role": "user", "content": self.user_message(record)},
            {"role": "assistant", "content": record.output},
        ]


# =============================================================================
# Near-duplicate grouping and splitting
# =============================================================================
class _UnionFind:
    """Union-find whose root is always the smallest index (deterministic)."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def union(self, a: int, b: int) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        low, high = sorted((root_a, root_b))
        self._parent[high] = low
        return True


class NearDuplicateGrouper:
    """Groups records whose prompts are identical or near-identical after folding."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def assign(self, prompt_folds: Sequence[str]) -> tuple[list[str], int]:
        """Return ``(group_id per record, number of near-duplicate merges)``.

        A group id is the smallest prompt hash in the group, so ids do not depend
        on the order of the input file.
        """
        size = len(prompt_folds)
        keys = [hashlib.sha256(fold.encode("utf-8")).hexdigest() for fold in prompt_folds]
        union = _UnionFind(size)

        first_seen: dict[str, int] = {}
        for index, key in enumerate(keys):
            if key in first_seen:
                union.union(first_seen[key], index)
            else:
                first_seen[key] = index

        merges = 0
        if self._threshold > 0:
            if size > NEAR_DUP_MAX_RECORDS:
                logger.warning(
                    "Skipping near-duplicate grouping: %d records exceeds the %d limit. "
                    "Only identical prompts are grouped.", size, NEAR_DUP_MAX_RECORDS,
                )
            else:
                merges = self._merge_near_duplicates(prompt_folds, union)

        smallest: dict[int, str] = {}
        for index, key in enumerate(keys):
            root = union.find(index)
            smallest[root] = min(key, smallest.get(root, key))
        return [smallest[union.find(i)] for i in range(size)], merges

    def _merge_near_duplicates(self, folds: Sequence[str], union: _UnionFind) -> int:
        shingle_sets = [self._shingles(fold) for fold in folds]
        index: dict[str, list[int]] = defaultdict(list)
        merges = 0
        for i, shingles in enumerate(shingle_sets):
            shared: Counter[int] = Counter()
            for shingle in shingles:
                bucket = index.get(shingle)
                if bucket and len(bucket) <= NEAR_DUP_MAX_BUCKET:
                    shared.update(bucket)
            for j, common in shared.items():
                if union.find(i) == union.find(j):
                    continue
                jaccard = common / (len(shingles) + len(shingle_sets[j]) - common)
                if jaccard >= self._threshold and union.union(i, j):
                    merges += 1
            for shingle in shingles:
                index[shingle].append(i)
        return merges

    @staticmethod
    def _shingles(fold: str) -> set[str]:
        words = fold.split()
        if len(words) < 2:
            return {fold}
        return {f"{a} {b}" for a, b in zip(words, words[1:])}


class GroupedSplitter:
    """Assigns whole groups to train/validation/test (seeded, deterministic)."""

    def __init__(self, config: PreparationConfig) -> None:
        self._targets = {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        }
        self._seed = config.seed

    def split(self, examples: Sequence[Example]) -> dict[str, list[Example]]:
        by_group: dict[str, list[Example]] = defaultdict(list)
        for example in examples:
            by_group[example.group_id].append(example)
        if len(by_group) < len(SPLIT_NAMES):
            raise DatasetSplitError(
                f"Only {len(by_group)} distinct prompt group(s) remain; at least "
                f"{len(SPLIT_NAMES)} are needed for train/validation/test."
            )

        rng = random.Random(self._seed)
        groups = [by_group[key] for key in sorted(by_group)]  # order-independent start
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)  # big groups first packs ratios better (stable sort)

        total = len(examples)
        result: dict[str, list[Example]] = {name: [] for name in SPLIT_NAMES}
        for group in groups:
            name = max(
                SPLIT_NAMES,
                key=lambda n: (self._targets[n] * total - len(result[n]), -SPLIT_NAMES.index(n)),
            )
            result[name].extend(group)

        empty = [name for name in SPLIT_NAMES if not result[name]]
        if empty:
            largest = max(len(group) for group in groups)
            message = (
                f"Split(s) {empty} are empty. The largest prompt group holds {largest} of "
                f"{total} records ({largest / total:.0%}); too many near-identical prompts."
            )
            if total >= MIN_RECORDS_FOR_FULL_SPLIT:
                raise DatasetSplitError(message)
            logger.warning("%s The dataset is small, continuing.", message)

        for name in SPLIT_NAMES:
            rng.shuffle(result[name])
        return result

    @staticmethod
    def verify_no_leakage(splits: dict[str, list[Example]]) -> None:
        """Assert no prompt group and no exact record appears in more than one split."""
        seen_groups: dict[str, str] = {}
        seen_records: dict[str, str] = {}
        for name, examples in splits.items():
            for example in examples:
                for registry, key in ((seen_groups, example.group_id), (seen_records, example.exact_key)):
                    owner = registry.setdefault(key, name)
                    if owner != name:
                        raise DatasetSplitError(
                            f"Data leakage: example {example.example_id} appears in both "
                            f"'{owner}' and '{name}'."
                        )


# =============================================================================
# Summary and writing
# =============================================================================
@dataclass
class TokenReport:
    """Sequence-length statistics for the VALID, UNIQUE records.

    Population: records that passed validation AND survived exact deduplication,
    measured BEFORE length filtering. Exact duplicates are excluded on purpose:
    they are removed from the dataset, so counting them would overstate how much
    data exists at each length. ``examples`` is the denominator of every count and
    percentage in this report (``with_context_over_max_seq`` is relative to
    ``with_context``). Invalid records and duplicates are reported separately in the
    summary's record funnel.
    """

    examples: int = 0
    stats: dict[str, float] = field(default_factory=dict)
    answer_stats: dict[str, float] = field(default_factory=dict)
    histogram: dict[str, int] = field(default_factory=dict)
    reference_length: int = REFERENCE_LENGTH
    over_reference: int = 0
    max_seq_length: int = 0
    over_max_seq: int = 0
    with_context: int = 0
    with_context_over_max_seq: int = 0
    prompt_overhead_tokens: int = 0
    room_tokens: int = 0
    reference_full_prompt_tokens: Optional[int] = None

    def render_lines(self) -> list[str]:
        def pct(part: int) -> str:
            return f"{part / self.examples:.0%}" if self.examples else "-"

        lines = [
            f"  Token report: {self.examples} valid unique record(s) "
            "(after validation and exact deduplication, before length filtering)",
            f"    Sequence tokens   : {self.stats or 'n/a'}",
            f"    Answer tokens     : {self.answer_stats or 'n/a'}",
            f"    Histogram         : {self.histogram or 'n/a'}",
            f"    Over {self.reference_length} tokens   : {self.over_reference} of "
            f"{self.examples} ({pct(self.over_reference)})",
            f"    Over max_seq={self.max_seq_length}: {self.over_max_seq} of "
            f"{self.examples} ({pct(self.over_max_seq)}) -> dropped, never truncated",
            f"    With context      : {self.with_context} of {self.examples} "
            f"({pct(self.with_context)}); {self.with_context_over_max_seq} of those "
            "over max_seq",
            f"    Prompt overhead   : {self.prompt_overhead_tokens} tokens "
            f"(room for question+answer at max_seq: {self.room_tokens})",
        ]
        if self.reference_full_prompt_tokens is not None:
            lines.append(
                f"    Full prod prompt  : {self.reference_full_prompt_tokens} tokens (for comparison)"
            )
        return lines


@dataclass
class PreparationSummary:
    original_records: int = 0
    invalid_records: int = 0
    invalid_by_reason: dict[str, int] = field(default_factory=dict)
    duplicate_records: int = 0
    valid_unique_records: int = 0   # original - invalid - duplicates (token-report population)
    near_duplicate_merges: int = 0
    overlength_dropped: int = 0
    final_records: int = 0
    prompt_groups: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    token_stats: dict[str, float] = field(default_factory=dict)
    token_report: TokenReport = field(default_factory=TokenReport)
    system_prompt_tokens: int = 0
    warnings: list[str] = field(default_factory=list)
    output_dir: str = ""

    def verify_accounting(self) -> None:
        """Raise if the record funnel does not add up (guards against counting bugs).

        original = invalid + duplicates + valid_unique, and
        valid_unique = over-length dropped + final; the token report, the reason
        breakdown and the split sizes must agree with those totals.
        """
        problems: list[str] = []
        expected_unique = self.original_records - self.invalid_records - self.duplicate_records
        if self.valid_unique_records != expected_unique:
            problems.append(
                f"valid_unique={self.valid_unique_records} but original-invalid-duplicates="
                f"{expected_unique}"
            )
        if self.valid_unique_records != self.overlength_dropped + self.final_records:
            problems.append(
                f"valid_unique={self.valid_unique_records} but dropped+final="
                f"{self.overlength_dropped + self.final_records}"
            )
        if self.token_report.examples != self.valid_unique_records:
            problems.append(
                f"token report covers {self.token_report.examples} records, expected "
                f"{self.valid_unique_records}"
            )
        if self.token_report.over_max_seq != self.overlength_dropped:
            problems.append("token report and summary disagree on dropped records")
        if sum(self.invalid_by_reason.values()) != self.invalid_records:
            problems.append("invalid-reason breakdown does not sum to the invalid count")
        if self.splits and sum(self.splits.values()) != self.final_records:
            problems.append("split sizes do not sum to the final record count")
        if problems:
            raise DataPreparationError("Internal accounting error: " + "; ".join(problems))

    def render(self) -> str:
        def pct(part: int, whole: int) -> str:
            return f"{part / whole:.0%}" if whole else "-"

        unique = self.valid_unique_records
        lines = [
            "Preparation summary",
            "  Record funnel (original - invalid - duplicates = valid unique; "
            "valid unique - over-length = final)",
            f"  Original records      : {self.original_records}",
            f"  - Invalid             : {self.invalid_records} {self.invalid_by_reason or ''}".rstrip(),
            f"  - Exact duplicates    : {self.duplicate_records}",
            f"  = Valid unique        : {unique}  (population of the token report)",
            f"  - Over-length dropped : {self.overlength_dropped} of {unique} "
            f"({pct(self.overlength_dropped, unique)})",
            f"  = Final records       : {self.final_records} of {unique} in "
            f"{self.prompt_groups} prompt group(s)",
            f"  Train / Val / Test    : {self.splits.get('train', 0)} / "
            f"{self.splits.get('validation', 0)} / {self.splits.get('test', 0)} "
            f"(of {self.final_records} final)",
            f"  System prompt tokens  : {self.system_prompt_tokens}",
            f"  Final-record tokens   : {self.token_stats or 'n/a'}  (kept records only)",
            *self.token_report.render_lines(),
        ]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


class DatasetWriter:
    """Writes outputs atomically (one file at a time) and removes files on request."""

    def __init__(self, output_dir: Path) -> None:
        self._dir = output_dir

    def write_jsonl(self, name: str, rows: Sequence[dict[str, Any]]) -> Path:
        def write(handle: IO[str]) -> None:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        return self._atomic_write(name, write)

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        def write(handle: IO[str]) -> None:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        return self._atomic_write(name, write)

    def _atomic_write(self, name: str, writer: Callable[[IO[str]], None]) -> Path:
        path = self._dir / name
        tmp = path.with_name(path.name + ".tmp")
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            # newline="\n" keeps files byte-identical across Windows and Linux.
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                writer(handle)
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise OutputWriteError(f"Cannot write {path}: {exc}") from exc
        return path

    def remove(self, name: str) -> None:
        """Delete an output file if it exists (used for stale files from earlier runs)."""
        path = self._dir / name
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise OutputWriteError(f"Cannot remove stale file {path}: {exc}") from exc

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in MANIFEST_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _percentile(ordered: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile of an ascending list."""
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _distribution(counts: Sequence[int]) -> dict[str, float]:
    """min / mean / median / p90 / p95 / p99 / max of a list of lengths."""
    if not counts:
        return {}
    ordered = sorted(counts)
    return {
        "min": ordered[0],
        "mean": round(statistics.fmean(ordered), 1),
        "median": statistics.median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _token_stats(counts: Sequence[int]) -> dict[str, float]:
    return _distribution(counts)


def _histogram(counts: Sequence[int]) -> dict[str, int]:
    """Counts per bucket: <=128, <=256, ... and an open-ended last bucket."""
    labels = [f"<={limit}" for limit in REPORT_THRESHOLDS] + [f">{REPORT_THRESHOLDS[-1]}"]
    buckets = dict.fromkeys(labels, 0)
    for count in counts:
        for limit in REPORT_THRESHOLDS:
            if count <= limit:
                buckets[f"<={limit}"] += 1
                break
        else:
            buckets[labels[-1]] += 1
    return buckets


# =============================================================================
# Orchestrator
# =============================================================================
class DataPreparer:
    """Runs the full preparation pipeline and returns a ``PreparationSummary``."""

    def __init__(
        self,
        config: PreparationConfig,
        *,
        tokenizer: Optional[ChatTokenizer] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        reference_prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        config.validate()
        self._config = config
        self._normalizer = ArabicNormalizer(config.normalization)
        self._loader = RecordLoader()
        self._validator = RecordValidator(self._normalizer, config)
        self._tokenizer = (
            tokenizer if tokenizer is not None else load_tokenizer(config.tokenizer_name)
        )
        builder = (
            prompt_builder if prompt_builder is not None
            else load_prompt_builder(config.prompt_function)
        )
        self._formatter = ChatFormatter(self._tokenizer, builder)

        # Informational only: the size of the full production prompt, for comparison.
        if (
            reference_prompt_builder is None
            and prompt_builder is None
            and config.prompt_function != FULL_PROMPT_FUNCTION
        ):
            try:
                reference_prompt_builder = load_prompt_builder(FULL_PROMPT_FUNCTION)
            except DataPreparationError as exc:
                logger.warning("Full-prompt size comparison skipped: %s", exc)
        self._reference_builder = reference_prompt_builder

        self._grouper = NearDuplicateGrouper(config.near_duplicate_threshold)
        self._splitter = GroupedSplitter(config)
        self._writer = DatasetWriter(config.output_dir)
        self._dropped: list[dict[str, Any]] = []

    def prepare(self) -> PreparationSummary:
        cfg = self._config
        summary = PreparationSummary(output_dir=str(cfg.output_dir))
        report = summary.token_report
        report.max_seq_length = cfg.max_seq_length

        self._formatter.verify_template()
        self._check_prompt_budget(report)
        summary.system_prompt_tokens = report.prompt_overhead_tokens

        items, rejected = self._loader.load(cfg.input_paths)
        summary.original_records = len(items) + len(rejected)

        records: list[CleanRecord] = []
        for item in items:
            outcome = self._validator.validate(item)
            if isinstance(outcome, RejectedRecord):
                rejected.append(outcome)
            else:
                records.append(outcome)
        summary.invalid_records = len(rejected)
        summary.invalid_by_reason = dict(Counter(r.category for r in rejected))

        unique, summary.duplicate_records = self._deduplicate(records)
        summary.valid_unique_records = len(unique)
        if not unique:
            raise DatasetTooSmallError("No valid records remain after validation and deduplication.")

        examples = self._format_examples(unique, report)
        summary.overlength_dropped = len(self._dropped)
        if not examples:
            raise DatasetTooSmallError(
                f"Every record exceeds max_seq_length={cfg.max_seq_length} tokens "
                f"(sequence tokens: {report.stats}). Raise --max-seq-length or shorten the data."
            )

        examples, summary.near_duplicate_merges, summary.prompt_groups = self._assign_groups(examples)
        splits = self._splitter.split(examples)
        self._splitter.verify_no_leakage(splits)

        summary.final_records = len(examples)
        summary.splits = {name: len(splits[name]) for name in SPLIT_NAMES}
        summary.token_stats = _token_stats([e.num_tokens for e in examples])
        summary.verify_accounting()
        summary.warnings = self._collect_warnings(summary)

        if cfg.dry_run:
            logger.info("Dry run: no files were written.")
        else:
            self._write_outputs(splits, rejected, summary)
        logger.info("\n%s", summary.render())
        return summary

    # -- steps ------------------------------------------------------------- #
    def _check_prompt_budget(self, report: TokenReport) -> None:
        """Stop early if the system prompt leaves no room for content (see module docstring, 4)."""
        overhead = self._formatter.overhead_tokens()
        room = self._config.max_seq_length - overhead
        report.prompt_overhead_tokens = overhead
        report.room_tokens = room
        if self._reference_builder is not None:
            reference = ChatFormatter(self._tokenizer, self._reference_builder)
            report.reference_full_prompt_tokens = reference.overhead_tokens()

        logger.info(
            "System prompt + template use %d tokens; %d remain of max_seq_length=%d%s.",
            overhead, room, self._config.max_seq_length,
            f" (full production prompt: {report.reference_full_prompt_tokens})"
            if report.reference_full_prompt_tokens is not None else "",
        )
        if room < MIN_RESPONSE_TOKENS:
            raise PromptBudgetError(
                f"The system prompt and chat template use {overhead} tokens, leaving only "
                f"{max(room, 0)} for the question and answer inside max_seq_length="
                f"{self._config.max_seq_length} (need at least {MIN_RESPONSE_TOKENS}). Options: "
                "(1) raise --max-seq-length and verify it fits 8 GB VRAM when training; "
                "(2) shorten SYSTEM_PROMPT_COMPACT further; "
                "(3) accept a smaller training context and shorter answers."
            )

    def _deduplicate(self, records: Sequence[CleanRecord]) -> tuple[list[CleanRecord], int]:
        """Drop exact duplicates after folding; keep the first occurrence."""
        seen: set[str] = set()
        unique: list[CleanRecord] = []
        for record in records:
            key = self._exact_key(record)
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)
        return unique, len(records) - len(unique)

    @staticmethod
    def _exact_key(record: CleanRecord) -> str:
        parts = (record.instruction, record.input, record.context, record.output)
        folded = "\x1f".join(ArabicNormalizer.fold(part) for part in parts)
        return hashlib.sha256(folded.encode("utf-8")).hexdigest()

    @staticmethod
    def _prompt_fold(record: CleanRecord) -> str:
        return ArabicNormalizer.fold(f"{record.instruction}\n{record.input}\n{record.context}")

    def _format_examples(
        self, records: Sequence[CleanRecord], report: TokenReport
    ) -> list[Example]:
        """Render every record, measure it, and keep those that fit ``max_seq_length``.

        Lengths of ALL records are recorded in ``report`` before filtering, so
        the report shows the true distribution, not only the survivors. Records
        that do not fit are dropped (listed in ``self._dropped``) or raise,
        depending on the policy; assistant answers are never truncated.
        """
        limit = self._config.max_seq_length
        examples: list[Example] = []
        all_tokens: list[int] = []
        answer_tokens: list[int] = []
        self._dropped = []

        for record in records:
            messages = self._formatter.build_messages(record)
            text = self._formatter.render(messages)
            tokens = self._formatter.count_tokens(text)
            all_tokens.append(tokens)
            answer_tokens.append(self._formatter.count_tokens(record.output))
            if record.context:
                report.with_context += 1

            if tokens > limit:
                if self._config.overlength_policy == "error":
                    raise DataPreparationError(
                        f"{record.source} ({record.position}) needs {tokens} tokens; "
                        f"max_seq_length is {limit}."
                    )
                self._dropped.append(
                    {
                        "id": self._exact_key(record)[:16],
                        "source": record.source,
                        "position": record.position,
                        "tokens": tokens,
                        "max_seq_length": limit,
                        "has_context": bool(record.context),
                    }
                )
                if record.context:
                    report.with_context_over_max_seq += 1
                continue
            examples.append(
                Example(record, self._exact_key(record), self._prompt_fold(record),
                        messages, text, tokens)
            )

        report.examples = len(records)
        report.stats = _distribution(all_tokens)
        report.answer_stats = _distribution(answer_tokens)
        report.histogram = _histogram(all_tokens)
        report.over_reference = sum(1 for t in all_tokens if t > report.reference_length)
        report.over_max_seq = len(self._dropped)
        if self._dropped:
            logger.warning(
                "Dropped %d of %d record(s) longer than %d tokens (see overlength.jsonl).",
                len(self._dropped), len(records), limit,
            )
        return examples

    def _assign_groups(self, examples: Sequence[Example]) -> tuple[list[Example], int, int]:
        group_ids, merges = self._grouper.assign([e.prompt_fold for e in examples])
        grouped = [
            Example(e.record, e.exact_key, e.prompt_fold, e.messages, e.text, e.num_tokens, gid)
            for e, gid in zip(examples, group_ids)
        ]
        return grouped, merges, len(set(group_ids))

    def _collect_warnings(self, summary: PreparationSummary) -> list[str]:
        warnings: list[str] = []
        report = summary.token_report
        if report.room_tokens < RECOMMENDED_ROOM_TOKENS:
            warnings.append(
                f"Only {report.room_tokens} tokens remain after the system prompt at "
                f"max_seq_length={report.max_seq_length} (recommended >= {RECOMMENDED_ROOM_TOKENS})."
            )
        if report.examples and summary.overlength_dropped / report.examples > OVERLENGTH_WARN_FRACTION:
            warnings.append(
                f"{summary.overlength_dropped} of {report.examples} valid unique records "
                f"({summary.overlength_dropped / report.examples:.0%}) exceed "
                f"{report.max_seq_length} tokens and were DROPPED (never truncated); the "
                "remaining data may be biased toward short answers."
            )
        if report.with_context_over_max_seq:
            warnings.append(
                f"{report.with_context_over_max_seq} of {report.with_context} valid unique "
                "record(s) with retrieved context were dropped for length: grounded/cited "
                "examples need a longer max_seq_length."
            )
        if summary.final_records < 500:
            warnings.append(
                f"Only {summary.final_records} examples: expect limited gains from fine-tuning "
                "(roughly 1,000+ high-quality examples is a sensible target)."
            )
        digits = self._validator.style_warnings.get("arabic_indic_digits_in_output", 0)
        if digits:
            warnings.append(
                f"{digits} output(s) use Arabic-Indic digits but the system prompt asks for "
                "0-9; consider --digits western."
            )
        if summary.near_duplicate_merges:
            warnings.append(
                f"{summary.near_duplicate_merges} near-duplicate prompt pair(s) were grouped "
                "together so they cannot straddle train and test."
            )
        if summary.splits.get("validation", 0) < 20 or summary.splits.get("test", 0) < 20:
            warnings.append(
                "Validation or test set has fewer than 20 examples; metrics will be noisy."
            )
        return warnings

    def _write_outputs(
        self,
        splits: dict[str, list[Example]],
        rejected: Sequence[RejectedRecord],
        summary: PreparationSummary,
    ) -> None:
        # manifest.json is this run's "commit marker". Removing the previous one first
        # means a run that fails half-way leaves a directory WITHOUT a manifest, never
        # new data files sitting next to an old manifest that describes other files.
        self._writer.remove("manifest.json")

        files: dict[str, dict[str, Any]] = {}
        split_files = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
        for name, file_name in split_files.items():
            path = self._writer.write_jsonl(file_name, [e.to_row() for e in splits[name]])
            files[file_name] = {"records": len(splits[name]), "sha256": self._writer.sha256(path)}

        # Audit files are written on EVERY run and are simply empty when there is
        # nothing to report. A rejected.jsonl / overlength.jsonl left over from an
        # earlier run can therefore never be mistaken for this run's result.
        audit_files = {
            "rejected.jsonl": [asdict(r) for r in rejected],
            "overlength.jsonl": list(self._dropped),
        }
        for file_name, rows in audit_files.items():
            path = self._writer.write_jsonl(file_name, rows)
            files[file_name] = {"records": len(rows), "sha256": self._writer.sha256(path)}

        system_prompt = self._formatter.system_prompt()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": json.loads(json.dumps(asdict(self._config), default=str)),
            "tokenizer": self._config.tokenizer_name,
            "system_prompt": {
                "function": f"{PROMPT_MODULE}.{self._config.prompt_function}",
                "sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "tokens_with_template": summary.system_prompt_tokens,
                "reference_full_prompt_tokens": summary.token_report.reference_full_prompt_tokens,
            },
            "summary": asdict(summary),
            "files": files,
            "libraries": _library_versions(),
        }
        self._writer.write_json("manifest.json", manifest)
        logger.info("Wrote dataset to %s", self._config.output_dir)


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an Arabic instruction dataset for Qwen2.5 QLoRA fine-tuning."
    )
    parser.add_argument("--input", nargs="+", required=True, type=Path,
                        help=".json / .jsonl files or directories of them")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="hub id or local folder of the Qwen2.5 tokenizer")
    parser.add_argument("--prompt-function", default=DEFAULT_PROMPT_FUNCTION,
                        help=f"function in {PROMPT_MODULE} that builds the system prompt")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-output-chars", type=int, default=20)
    parser.add_argument("--min-arabic-ratio", type=float, default=0.0,
                        help="minimum Arabic-letter share of each output; 0 (default) disables "
                             "the check so English terms, acronyms and figures are always kept")
    parser.add_argument("--near-dup-threshold", type=float, default=0.85,
                        help="Jaccard similarity to group near-identical prompts (0 disables)")
    parser.add_argument("--overlength-policy", choices=["drop", "error"], default="drop")
    parser.add_argument("--alef", choices=["keep", "normalize"], default="keep")
    parser.add_argument("--yaa", choices=["keep", "normalize"], default="keep")
    parser.add_argument("--digits", choices=["keep", "western"], default="keep")
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--keep-tatweel", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    parser.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PreparationConfig:
    return PreparationConfig(
        input_paths=tuple(args.input),
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        prompt_function=args.prompt_function,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.val_ratio,
        min_output_chars=args.min_output_chars,
        min_arabic_ratio=args.min_arabic_ratio,
        near_duplicate_threshold=args.near_dup_threshold,
        overlength_policy=args.overlength_policy,
        normalization=NormalizationPolicy(
            remove_diacritics=not args.keep_diacritics,
            remove_tatweel=not args.keep_tatweel,
            alef=args.alef,
            yaa=args.yaa,
            digits=args.digits,
        ),
        dry_run=args.dry_run,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        DataPreparer(config_from_args(args)).prepare()
    except (DataPreparationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Unexpected failure during data preparation.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

