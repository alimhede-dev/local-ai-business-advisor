"""Automatic checks applied to every model reply in the evaluation.

WHY (architecture decisions)
----------------------------
1. **Objective checks catch regressions; humans judge quality.** Code can
   reliably verify things like "is the reply Arabic?", "did it cite a source
   that exists?", "did it contain the right figure?". It cannot judge whether
   the Arabic is *natural* Fusha. So the runner pairs these checks with a
   manual-review sheet for the subjective part.

2. **Hard checks vs weak checks.** A *hard* check has a clear pass/fail
   definition and decides whether an item passes. A *weak* check is a keyword
   heuristic (dialect markers, "admits missing information" phrases) that can
   misfire on valid answers; it is reported for human attention but never fails
   an item on its own.

3. **Standard library only.** This module imports nothing from the app or from
   third-party packages, so it can be unit-tested anywhere and reused unchanged
   to score the fine-tuned model later with exactly the same yardstick.

4. **Matching ignores superficial variation.** Text is normalised before
   comparison (diacritics, tatweel, alef/yaa variants, Arabic-Indic digits,
   thousands separators) so "١٦٥٬٠٠٠" in a reply still matches the expected
   "165000". The *style* check for digits is separate, because the system
   prompt asks for Western digits (0-9).
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check. ``weak`` checks are advisory (see module docstring)."""

    name: str
    passed: bool
    detail: str = ""
    weak: bool = False


# =============================================================================
# Normalisation
# =============================================================================
_INVISIBLE_RE = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad\u0640]")
_DIACRITICS_RE = re.compile("[\u0610-\u061a\u064b-\u065f\u0670]")
# Thousands separators: "," or Arabic "٬" between digits followed by exactly 3 digits.
_THOUSANDS_RE = re.compile("(?<=\\d)[,\u066c](?=\\d{3}(?!\\d))")
_SPACES_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Arabic-Indic (U+0660..) and Persian (U+06F0..) digits -> ASCII digits.
_DIGIT_MAP = {
    **{0x0660 + i: ord("0") + i for i in range(10)},
    **{0x06F0 + i: ord("0") + i for i in range(10)},
}
# Fold alef variants and alef maqsura so "إنستغرام"/"انستغرام" and "على"/"علي" match.
_LETTER_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})

_ARABIC_LETTER_RE = re.compile("[\u0621-\u064a\u0671-\u06d3\u06fa-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN_LETTER_RE = re.compile("[A-Za-z]")
_ARABIC_INDIC_DIGIT_RE = re.compile("[\u0660-\u0669\u06f0-\u06f9]")


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


def normalize_for_matching(text: str) -> str:
    """Normalise text for tolerant substring matching (see module docstring, 4)."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _DIACRITICS_RE.sub("", text)
    text = text.translate(_DIGIT_MAP)
    text = text.replace("\u066b", ".")  # Arabic decimal separator
    text = _THOUSANDS_RE.sub("", text)
    text = text.translate(_LETTER_MAP)
    return _SPACES_RE.sub(" ", text).strip().casefold()


# =============================================================================
# Individual checks
# =============================================================================
def arabic_ratio(text: str) -> float:
    """Share of Arabic letters among Arabic + Latin letters (1.0 if no letters)."""
    arabic = len(_ARABIC_LETTER_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    total = arabic + latin
    return arabic / total if total else 1.0


def check_non_empty(reply: str) -> CheckResult:
    return CheckResult("non_empty", bool(reply.strip()), "" if reply.strip() else "empty reply")


def check_not_truncated(done_reason: Optional[str]) -> CheckResult:
    """The model stopped on its own rather than hitting the token limit."""
    truncated = done_reason == "length"
    return CheckResult(
        "not_truncated", not truncated, "hit the max-token limit" if truncated else ""
    )


def check_arabic_ratio(reply: str, minimum: float = 0.85) -> CheckResult:
    """Reply must be (almost) entirely Arabic; acronyms like ROI are tolerated."""
    ratio = arabic_ratio(reply)
    return CheckResult(
        "arabic_ratio", ratio >= minimum, f"arabic letters = {ratio:.0%} (min {minimum:.0%})"
    )


def check_western_digits(reply: str) -> CheckResult:
    """The system prompt asks for 0-9; flag Arabic-Indic digits."""
    found = _ARABIC_INDIC_DIGIT_RE.findall(reply)
    return CheckResult(
        "western_digits", not found, f"{len(found)} Arabic-Indic digit(s)" if found else ""
    )


def check_length(reply: str, min_words: int, max_words: int = 800) -> CheckResult:
    words = len(reply.split())
    ok = min_words <= words <= max_words
    return CheckResult("length", ok, f"{words} words (allowed {min_words}-{max_words})")


_CITATION_RE = re.compile(
    r"(?:المصدر|المصادر|source|sources)\s*(\d+(?:\s*(?:،|,|و|-|–)\s*\d+)*)"
)


def extract_citations(reply: str) -> list[int]:
    """Source numbers cited like ``(المصدر 1، ص 3)`` or ``المصادر 1 و2`` (in order)."""
    normalized = normalize_for_matching(reply)
    numbers: list[int] = []
    for match in _CITATION_RE.finditer(normalized):
        numbers.extend(int(n) for n in re.findall(r"\d+", match.group(1)))
    return numbers


def check_citations(
    reply: str, n_sources: int, policy: str, min_distinct: int = 1
) -> CheckResult:
    """Validate citations against the sources actually provided.

    Policies: ``required`` (>= ``min_distinct`` valid, none invalid),
    ``forbidden`` (none at all - there was nothing to cite) and
    ``optional`` (only invalid numbers are an error).
    """
    cited = extract_citations(reply)
    valid = {n for n in cited if 1 <= n <= n_sources}
    invalid = sorted({n for n in cited if not 1 <= n <= n_sources})

    if policy == "forbidden":
        return CheckResult(
            "citations", not cited, f"unexpected citation(s): {sorted(set(cited))}" if cited else ""
        )
    if invalid:
        return CheckResult(
            "citations", False, f"cites non-existent source(s) {invalid} (only {n_sources} given)"
        )
    if policy == "required" and len(valid) < min_distinct:
        return CheckResult(
            "citations", False, f"{len(valid)} distinct valid citation(s), need {min_distinct}"
        )
    return CheckResult("citations", True, f"cited sources: {sorted(valid)}")


def _alternatives(entry: str) -> list[str]:
    return [normalize_for_matching(part) for part in entry.split("|") if part.strip()]


def check_must_include(reply: str, entries: Sequence[str]) -> CheckResult:
    """Every entry must appear; ``a|b`` means either ``a`` or ``b`` is enough."""
    haystack = normalize_for_matching(reply)
    missing = [e for e in entries if not any(alt in haystack for alt in _alternatives(e))]
    return CheckResult(
        "must_include", not missing, f"missing: {missing}" if missing else f"{len(entries)} found"
    )


def check_must_not_include(reply: str, entries: Sequence[str]) -> CheckResult:
    """None of the entries may appear (forbidden content, leaked text, hallucinations)."""
    haystack = normalize_for_matching(reply)
    present = [e for e in entries if any(alt in haystack for alt in _alternatives(e))]
    return CheckResult("must_not_include", not present, f"found: {present}" if present else "")


_SECTION_KEYWORDS = ("خلاصة", "التحليل", "التوصيات", "المخاطر", "الخطوات")
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*•]\s+)?(?:\*\*)?\d+\s*[.)\-:]")


def check_structure(reply: str) -> CheckResult:
    """Advisory answers need a numbered list and at least two of the prescribed sections."""
    normalized = normalize_for_matching(reply)
    numbered = len(_NUMBERED_ITEM_RE.findall(unicodedata.normalize("NFKC", reply).translate(_DIGIT_MAP)))
    sections = [k for k in _SECTION_KEYWORDS if normalize_for_matching(k) in normalized]
    ok = numbered >= 2 and len(sections) >= 2
    return CheckResult(
        "structure", ok, f"{numbered} numbered item(s); sections found: {sections}"
    )


# --- weak (advisory) heuristics ---------------------------------------------
_DIALECT_MARKERS = (
    "ازاي", "ايش", "شو", "ليش", "دلوقتي", "دلوقت", "عايز", "عاوز", "عايزه", "بدي", "بدك",
    "وش", "مش", "كده", "كدا", "اللي", "يلا", "علشان", "عشان", "لسه", "برضو", "اوي",
)
_DIALECT_RE = re.compile(
    "(?<![\u0600-\u06ff])(?:" + "|".join(_DIALECT_MARKERS) + ")(?![\u0600-\u06ff])"
)


def check_dialect(reply: str) -> CheckResult:
    """Weak: spot common colloquial words that should not appear in Fusha."""
    found = sorted(set(_DIALECT_RE.findall(normalize_for_matching(reply))))
    return CheckResult(
        "no_dialect", not found, f"possible dialect words: {found}" if found else "", weak=True
    )


_ADMISSION_PHRASES = (
    "لا تتضمن", "لا تحتوي", "لا تشمل", "لا تتوفر", "لا تتوافر", "غير متوفر", "غير متاح",
    "لم يرد", "لم ترد", "لم اجد", "لا اجد", "لا توجد معلومات", "لا يوجد", "لا املك",
    "لا تغطي", "لا تكفي", "لا تذكر", "لا تتناول", "لم يسترجع", "لا اتوفر", "لا يمكنني الاطلاع",
    "لا استطيع الاطلاع", "لا تتاح", "ليست لدي", "ليس لدي",
)
_DISCLOSURE_PHRASES = _ADMISSION_PHRASES + (
    "ارشاد عام", "ارشادات عامة", "نصيحة عامة", "توجيه عام", "بشكل عام", "معلومات عامة",
    "غير مستند", "ليست مستندة", "لا تستند", "غير مبني", "اطار عام",
)


def check_admits_missing_info(reply: str) -> CheckResult:
    """Weak: the reply says the documents/data are missing (keyword heuristic)."""
    haystack = normalize_for_matching(reply)
    ok = any(normalize_for_matching(p) in haystack for p in _ADMISSION_PHRASES)
    return CheckResult(
        "admits_missing_info", ok, "" if ok else "no admission phrase found - review manually",
        weak=True,
    )


def check_general_disclosure(reply: str) -> CheckResult:
    """Weak: with no documents, the reply should say its advice is general."""
    haystack = normalize_for_matching(reply)
    ok = any(normalize_for_matching(p) in haystack for p in _DISCLOSURE_PHRASES)
    return CheckResult(
        "general_disclosure", ok, "" if ok else "does not say the advice is general - review",
        weak=True,
    )


# =============================================================================
# Orchestration
# =============================================================================
MIN_WORDS_STRUCTURED = 60
MIN_WORDS_SHORT = 8


def evaluate(
    reply: str,
    expect: Mapping[str, Any],
    n_sources: int,
    *,
    done_reason: Optional[str] = None,
) -> list[CheckResult]:
    """Run every check that applies to one item.

    Args:
        reply: Model output (reasoning tags are stripped here).
        expect: The item's ``expect`` block from ``evaluation_set.json``.
        n_sources: How many context sources were given to the model.
        done_reason: Ollama's stop reason (``"length"`` means truncated).
    """
    text = strip_reasoning(reply)
    structured = bool(expect.get("full_structure", False))

    results = [
        check_non_empty(text),
        check_not_truncated(done_reason),
        check_arabic_ratio(text),
        check_western_digits(text),
        check_length(text, MIN_WORDS_STRUCTURED if structured else MIN_WORDS_SHORT),
        check_citations(
            text,
            n_sources,
            str(expect.get("citations", "optional")),
            int(expect.get("min_distinct_citations", 1)),
        ),
    ]
    if expect.get("must_include"):
        results.append(check_must_include(text, expect["must_include"]))
    if expect.get("must_not_include"):
        results.append(check_must_not_include(text, expect["must_not_include"]))
    if structured:
        results.append(check_structure(text))

    results.append(check_dialect(text))
    if expect.get("no_context_admission"):
        results.append(check_admits_missing_info(text))
    if expect.get("general_disclosure"):
        results.append(check_general_disclosure(text))
    return results


def item_passed(results: Sequence[CheckResult]) -> bool:
    """An item passes when every *hard* check passes (weak checks are advisory)."""
    return all(r.passed for r in results if not r.weak)
