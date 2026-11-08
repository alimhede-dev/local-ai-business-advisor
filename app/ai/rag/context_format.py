"""The one place that defines how retrieved context looks inside a prompt.

WHY (architecture decision)
---------------------------
Retrieval (production), the evaluation runner and fine-tuning data preparation
must show the model *exactly the same* context layout, or the model learns a
format it never sees at inference time. Each source is rendered as::

    [Source 1: report.pdf, page 3]
    <chunk text>

    [Source 2: contract.pdf]
    <chunk text>

and the system prompt tells the model to cite them as "(المصدر 1، ص 3)". Every
consumer imports the helpers below instead of rebuilding the string. The module
is pure standard library, so training environments can use it without LangChain.
"""

import re
from typing import Any, Mapping, Optional, Sequence

# A header line produced by ``format_source_header``.
_HEADER_RE = re.compile(r"^\[Source (\d+): [^\]\n]+\]$", re.MULTILINE)


def format_source_header(number: int, source: str, page: Optional[int] = None) -> str:
    """``[Source n: file, page p]`` (the page part is omitted when unknown)."""
    page_part = f", page {page}" if page is not None else ""
    return f"[Source {number}: {source}{page_part}]"


def format_source_block(
    number: int, source: str, page: Optional[int], text: str
) -> str:
    """One labelled block: the header, a newline, then the chunk text."""
    return f"{format_source_header(number, source, page)}\n{text.strip()}"


def format_context(sources: Sequence[Mapping[str, Any]]) -> str:
    """Join sources (mappings with ``source``, ``text`` and optional ``page``).

    Returns an empty string for no sources: callers must treat that as "no
    context" and never inject an empty block.
    """
    blocks = [
        format_source_block(number, str(s["source"]), s.get("page"), str(s["text"]))
        for number, s in enumerate(sources, start=1)
    ]
    return "\n\n".join(blocks)


def validate_formatted_context(context: str) -> list[str]:
    """Return a list of problems with a pre-formatted context string (empty = valid).

    Checks that the text starts with ``[Source 1: ...]``, that source numbers run
    1..n without gaps, and that every block has non-empty text.
    """
    problems: list[str] = []
    stripped = context.strip()
    if not stripped:
        return ["context is empty"]
    headers = list(_HEADER_RE.finditer(stripped))
    if not headers:
        return ["no '[Source n: file, page p]' header found"]
    if headers[0].start() != 0:
        problems.append("text appears before the first '[Source 1: ...]' header")
    numbers = [int(m.group(1)) for m in headers]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"source numbers must be 1..{len(numbers)} in order, got {numbers}")
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(stripped)
        if not stripped[match.end():end].strip():
            problems.append(f"source {match.group(1)} has no text")
    return problems
