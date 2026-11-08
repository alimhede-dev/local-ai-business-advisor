"""Checks against the REAL Qwen2.5 tokenizer and chat template (no fakes).

Run from the project root, in the training environment::

    python -m pytest tests/integration/test_qwen_tokenizer.py -v -s

``-s`` prints the token statistics. By default the tokenizer comes from
``training.prepare_data.DEFAULT_TOKENIZER``; set ``QWEN_TOKENIZER`` to a hub id
or a local folder to override it. The module is skipped only when
``transformers`` / ``langchain-core`` are not installed; if the tokenizer itself
cannot be loaded the tests FAIL (a skip would hide the problem).
"""

import json
import os
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("langchain_core")

from app.ai.prompts.system_prompts import (  # noqa: E402
    SYSTEM_PROMPT_COMPACT,
    build_compact_system_prompt,
    build_system_prompt,
)
from training.prepare_data import (  # noqa: E402
    DEFAULT_TOKENIZER,
    MIN_RESPONSE_TOKENS,
    RECOMMENDED_ROOM_TOKENS,
    REFERENCE_LENGTH,
    ChatFormatter,
    DataPreparer,
    PreparationConfig,
)

EXAMPLE_FILE = Path(__file__).resolve().parents[2] / "training" / "data" / "raw" / "example.jsonl"


@pytest.fixture(scope="module")
def tokenizer():
    name = os.environ.get("QWEN_TOKENIZER", DEFAULT_TOKENIZER)
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        pytest.fail(f"Could not load the tokenizer {name!r}: {exc}")


@pytest.fixture(scope="module")
def formatter(tokenizer):
    return ChatFormatter(tokenizer, build_compact_system_prompt)





def test_control_tokens_are_single_special_tokens(tokenizer):
    for token in ("<|im_start|>", "<|im_end|>"):
        assert len(tokenizer(token, add_special_tokens=False)["input_ids"]) == 1, token
        assert tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id


def test_token_count_matches_the_template_tokenisation(tokenizer, formatter):
    messages = [
        {"role": "system", "content": build_compact_system_prompt()},
        {"role": "user", "content": "كيف أزيد أرباح متجري؟\n\nمتجر ملابس برأس مال 5000 دولار"},
        {"role": "assistant", "content": "الخلاصة: ابدأ بتحليل هامش الربح لكل منتج."},
    ]
    text = formatter.render(messages)
    counted = formatter.count_tokens(text)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    assert counted == len(ids)


def test_compact_prompt_leaves_room_for_a_real_question_and_answer(tokenizer, formatter):
    compact = formatter.overhead_tokens()
    full = ChatFormatter(tokenizer, build_system_prompt).overhead_tokens()
    room = REFERENCE_LENGTH - compact
    print(f"\nprompt tokens (with template): compact={compact}, full={full}, "
          f"room at {REFERENCE_LENGTH} = {room}")
    assert compact < full
    assert room >= MIN_RESPONSE_TOKENS
    assert room >= RECOMMENDED_ROOM_TOKENS, (
        f"Only {room} tokens remain at {REFERENCE_LENGTH}; shorten SYSTEM_PROMPT_COMPACT "
        f"({len(SYSTEM_PROMPT_COMPACT)} chars) or use a longer max_seq_length."
    )


def test_example_file_report_with_real_tokenizer(tokenizer, tmp_path):
    config = PreparationConfig(
        input_paths=(EXAMPLE_FILE,), output_dir=tmp_path / "out", max_seq_length=REFERENCE_LENGTH
    )
    summary = DataPreparer(config, tokenizer=tokenizer).prepare()
    print("\n" + summary.render())

    report = summary.token_report
    assert report.examples == summary.final_records + summary.overlength_dropped
    assert sum(report.histogram.values()) == report.examples
    assert report.over_max_seq == summary.overlength_dropped

    kept = [
        json.loads(line)
        for name in ("train", "validation", "test")
        for line in (tmp_path / "out" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(kept) == summary.final_records
    for row in kept:
        rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        assert row["text"] == rendered
        assert row["num_tokens"] == len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        assert row["num_tokens"] <= REFERENCE_LENGTH          # nothing over the limit survives
        assert row["messages"][2]["content"].strip()           # answers are complete, never cut

