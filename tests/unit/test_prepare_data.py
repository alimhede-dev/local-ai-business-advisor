"""Tests for training/prepare_data.py (fake tokenizer and prompt builder, no downloads)."""

import hashlib
import json
import random
from pathlib import Path

import pytest

import training.prepare_data as prepare_module

from app.ai.rag.context_format import format_context
from training.prepare_data import (
    DEFAULT_PROMPT_FUNCTION,
    ArabicNormalizer,
    DataPreparer,
    build_parser,
    DatasetSplitError,
    InputFileError,
    NormalizationPolicy,
    OutputWriteError,
    PreparationConfig,
    PromptBudgetError,
    RecordLoader,
    TemplateMismatchError,
    DataPreparationError,
)

VOCAB = [
    "المبيعات", "التسويق", "الميزانية", "العملاء", "الموردين", "المخزون", "الأرباح", "التكاليف",
    "الموظفين", "الأسعار", "المنافسين", "التوسع", "التمويل", "الضرائب", "الجودة", "الإنتاج",
    "التصدير", "المتجر", "الفروع", "الحملات", "الإيرادات", "العقود", "الشراكات", "التحليل",
    "الخطة", "المشروع", "الأسواق", "القروض", "السيولة", "الاستثمار", "الخدمات", "المنتجات",
    "الإدارة", "التشغيل", "الكفاءة", "الأداء", "النمو", "المخاطر", "القرار", "الاستراتيجية",
    "الفرص", "الشركة", "الفريق", "التدريب", "التقارير", "المؤشرات", "الحوافز", "الشحن",
    "التوريد", "الجدوى", "الامتثال", "الابتكار", "الولاء", "التحصيل", "الائتمان", "الهوامش",
    "الطلب", "العرض", "الشراء", "التخزين",
]
OUTPUT = "الخلاصة: ركّز على تحسين الهامش وتقليل التكاليف الثابتة خلال الربع القادم."


class FakeTokenizer:
    """Renders Qwen2.5-style ChatML; one token per whitespace-separated word."""

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        text = ""
        if conversation[0]["role"] != "system":
            text += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.<|im_end|>\n"
        for message in conversation:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": text.split()}


class PlainTokenizer(FakeTokenizer):
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)


class InjectingTokenizer(FakeTokenizer):
    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        return "You are Qwen. " + super().apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )


def short_prompt(rag_context: str = "") -> str:
    return "SYS " + (f"ctx:{rag_context}" if rag_context else "no-ctx")


def long_prompt(rag_context: str = "") -> str:
    return "كلمة " * 600


def make_records(count: int, seed: int = 1) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "instruction": " ".join(rng.sample(VOCAB, 8)),
            "input": "",
            "output": f"{OUTPUT} رقم {i}",
        }
        for i in range(count)
    ]


def write_jsonl(path: Path, records: list) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    return path


def make_config(tmp_path: Path, records: list, **overrides) -> PreparationConfig:
    source = write_jsonl(tmp_path / "in.jsonl", records)
    params = dict(input_paths=(source,), output_dir=tmp_path / "out", max_seq_length=512)
    params.update(overrides)
    return PreparationConfig(**params)


def run(config: PreparationConfig, tokenizer=None, builder=short_prompt):
    return DataPreparer(config, tokenizer=tokenizer or FakeTokenizer(), prompt_builder=builder).prepare()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_normalizer_default_policy_is_light():
    n = ArabicNormalizer(NormalizationPolicy())
    assert n.normalize("الفَرْقُ") == "الفرق"                      # diacritics removed
    assert n.normalize("العـــربية") == "العربية"                   # tatweel removed
    assert n.normalize("أحمد إبراهيم آمن") == "أحمد إبراهيم آمن"    # alef kept by default
    assert n.normalize("على") == "على"                              # yaa kept by default
    assert n.normalize("ربح 5% و٣٠٪ بقيمة 1,000 ﷼ و$200") == "ربح 5% و٣٠٪ بقيمة 1,000 ﷼ و$200"
    assert n.normalize("آ") == "آ"                                  # decomposed madda keeps its meaning


def test_normalizer_optional_policies():
    n = ArabicNormalizer(NormalizationPolicy(alef="normalize", yaa="normalize", digits="western"))
    assert n.normalize("أإآٱ") == "اااا"
    assert n.normalize("على ی") == "علي ي"
    assert n.normalize("٣٠ و۴۵") == "30 و45"
    keep = ArabicNormalizer(NormalizationPolicy(remove_diacritics=False))
    assert keep.normalize("الفَرْق") == "الفَرْق"


def test_normalizer_whitespace_keeps_lists():
    n = ArabicNormalizer(NormalizationPolicy())
    text = "التوصيات:  \n 1.  أولًا\r\n2. ثانيًا\n\n\n\nخاتمة\u200b"
    assert n.normalize(text) == "التوصيات:\n1. أولا\n2. ثانيا\n\nخاتمة"


def test_normalizer_expands_presentation_forms_but_keeps_rial_sign():
    n = ArabicNormalizer(NormalizationPolicy())
    assert n.normalize("\ufee3\ufeae\ufea3\ufe92\ufe8e") == "مرحبا"
    assert n.normalize("100 \ufdfc") == "100 \ufdfc"


def test_fold_is_aggressive_for_hashing_but_keeps_financial_symbols():
    fold = ArabicNormalizer.fold
    assert fold("أحمد، على!") == fold("احمد علي")
    assert fold("ربح ٥٠٪") != fold("ربح ٥٠")
    assert fold("5%") != fold("5")
    assert fold("SALES  Report") == fold("sales report")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_loader_json_formats(tmp_path):
    rec = {"instruction": "س", "output": "ج"}
    (tmp_path / "a.json").write_text(json.dumps([rec, rec]), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"data": [rec]}), encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps(rec), encoding="utf-8")
    items, rejected = RecordLoader().load([tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"])
    assert len(items) == 4 and not rejected


def test_loader_jsonl_bom_blank_lines_and_bad_line(tmp_path):
    path = tmp_path / "x.jsonl"
    good = json.dumps({"instruction": "س", "output": "ج"}, ensure_ascii=False)
    path.write_bytes(("\ufeff" + good + "\n\n{broken json\n" + good + "\n").encode("utf-8"))
    items, rejected = RecordLoader().load([path])
    assert len(items) == 2
    assert len(rejected) == 1 and rejected[0].category == "invalid_json" and "line 3" in rejected[0].position


def test_loader_errors(tmp_path):
    with pytest.raises(InputFileError):
        RecordLoader().load([tmp_path / "missing.jsonl"])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputFileError, match="not valid JSON"):
        RecordLoader().load([bad])
    txt = tmp_path / "a.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(InputFileError, match="Unsupported"):
        RecordLoader().load([txt])
    latin = tmp_path / "latin.jsonl"
    latin.write_bytes(b'{"instruction": "\xe9"}')
    with pytest.raises(InputFileError, match="UTF-8"):
        RecordLoader().load([latin])


def test_loader_directory(tmp_path):
    write_jsonl(tmp_path / "1.jsonl", make_records(2))
    write_jsonl(tmp_path / "2.jsonl", make_records(3, seed=2))
    items, _ = RecordLoader().load([tmp_path])
    assert len(items) == 5


# --------------------------------------------------------------------------- #
# Validation and reporting
# --------------------------------------------------------------------------- #
def test_invalid_records_are_rejected_with_reasons(tmp_path):
    records = make_records(40)
    records += [
        {"instruction": "س"},                                              # missing output
        {"instruction": 5, "output": OUTPUT},                              # bad type
        {"instruction": "   ", "output": OUTPUT},                          # empty
        {"instruction": "ما هو ROI؟", "output": "ROI is net profit divided by cost."},  # not Arabic
        {"instruction": "قصير", "output": "نعم"},                          # too short
        {"instruction": "س <|im_start|>", "output": OUTPUT},               # control tokens
        ["not", "an", "object"],
        {"instruction": "كيف أبدأ؟", "input": None, "output": OUTPUT},    # input None is fine
    ]
    summary = run(make_config(tmp_path, records, min_arabic_ratio=0.6))
    assert summary.original_records == 48
    assert summary.invalid_by_reason == {
        "missing_field": 1, "bad_type": 1, "empty_field": 1, "output_not_arabic": 1,
        "output_too_short": 1, "chatml_control_tokens": 1, "not_an_object": 1,
    }
    assert summary.final_records == 41
    rejected = read_rows(tmp_path / "out" / "rejected.jsonl")
    assert len(rejected) == 7 and all(r["detail"] for r in rejected)


def test_duplicates_removed_after_normalisation(tmp_path):
    records = make_records(40)
    base = {"instruction": "ما الفرق بين الإيراد والربح؟", "input": "", "output": OUTPUT}
    variant = {"instruction": "ما الفَرْقُ بين الإيراد والربح", "input": "", "output": OUTPUT}
    summary = run(make_config(tmp_path, records + [base, variant]))
    assert summary.duplicate_records == 1
    assert summary.final_records == 41


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def test_messages_empty_input_and_context(tmp_path):
    records = make_records(38) + [
        {"instruction": "سؤال بلا مدخل", "input": "", "output": OUTPUT},
        {"instruction": "سؤال مع مدخل", "input": "متجر ملابس", "output": OUTPUT},
        {"instruction": "سؤال مع سياق", "context": "[Source 1: a.pdf, page 2]\nنص", "output": OUTPUT},
    ]
    run(make_config(tmp_path, records))
    rows = [r for name in ("train", "validation", "test") for r in read_rows(tmp_path / "out" / f"{name}.jsonl")]
    by_user = {r["messages"][1]["content"]: r for r in rows}

    plain = by_user["سؤال بلا مدخل"]
    assert [m["role"] for m in plain["messages"]] == ["system", "user", "assistant"]
    assert plain["messages"][0]["content"] == "SYS no-ctx"
    # Default policy strips diacritics (including the shadda in "ركّز").
    assert plain["messages"][2]["content"] == ArabicNormalizer(NormalizationPolicy()).normalize(OUTPUT)
    assert "ّ" not in plain["messages"][2]["content"]
    assert plain["text"].startswith("<|im_start|>system\nSYS no-ctx<|im_end|>\n<|im_start|>user")
    assert plain["text"].rstrip().endswith("<|im_end|>")

    assert "سؤال مع مدخل\n\nمتجر ملابس" in by_user
    grounded = by_user["سؤال مع سياق"]
    assert grounded["messages"][0]["content"].startswith("SYS ctx:[Source 1: a.pdf, page 2]")


def test_template_verification(tmp_path):
    config = make_config(tmp_path, make_records(40))
    for bad in (PlainTokenizer(), InjectingTokenizer()):
        with pytest.raises(TemplateMismatchError):
            run(config, tokenizer=bad)


def test_prompt_budget_is_enforced(tmp_path):
    config = make_config(tmp_path, make_records(40), max_seq_length=512)
    with pytest.raises(PromptBudgetError, match="max_seq_length=512"):
        run(config, builder=long_prompt)
    assert run(make_config(tmp_path, make_records(40), max_seq_length=2048), builder=long_prompt)


def test_overlength_examples_are_dropped_or_raise(tmp_path):
    records = make_records(40)
    records.append({"instruction": "سؤال طويل", "output": "كلمة " * 400})
    summary = run(make_config(tmp_path, records, max_seq_length=256))
    assert summary.overlength_dropped == 1 and summary.final_records == 40
    with pytest.raises(DataPreparationError, match="needs"):
        run(make_config(tmp_path, records, max_seq_length=256, overlength_policy="error"))


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def test_split_ratios_files_and_utf8(tmp_path):
    summary = run(make_config(tmp_path, make_records(100)))
    assert summary.final_records == 100
    assert abs(summary.splits["train"] - 80) <= 1
    assert abs(summary.splits["validation"] - 10) <= 1
    assert abs(summary.splits["test"] - 10) <= 1
    assert sum(summary.splits.values()) == 100
    raw = (tmp_path / "out" / "train.jsonl").read_bytes().decode("utf-8")
    assert "الخلاصة" in raw and "\\u06" not in raw        # real Arabic, not \u escapes
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["train.jsonl"]["records"] == summary.splits["train"]
    assert manifest["system_prompt"]["sha256"] and manifest["libraries"]["python"]


def test_splits_are_deterministic(tmp_path):
    records = make_records(100)
    run(make_config(tmp_path, records, output_dir=tmp_path / "o1"))
    run(make_config(tmp_path, records, output_dir=tmp_path / "o2"))
    for name in ("train", "validation", "test"):
        assert (tmp_path / "o1" / f"{name}.jsonl").read_bytes() == (tmp_path / "o2" / f"{name}.jsonl").read_bytes()


def test_no_leakage_and_near_duplicates_stay_together(tmp_path):
    records = make_records(80)
    near = []
    for record in records[:12]:
        near.append({**record, "instruction": record["instruction"] + " الآن", "output": OUTPUT + " نسخة"})
    summary = run(make_config(tmp_path, records + near))
    assert summary.near_duplicate_merges >= 12

    where = {}
    for name in ("train", "validation", "test"):
        for row in read_rows(tmp_path / "out" / f"{name}.jsonl"):
            where[row["messages"][1]["content"]] = name
    for record in records[:12]:
        assert where[record["instruction"]] == where[record["instruction"] + " الآن"]

    groups = {}
    for name in ("train", "validation", "test"):
        for row in read_rows(tmp_path / "out" / f"{name}.jsonl"):
            assert groups.setdefault(row["group_id"], name) == name


def test_too_few_groups_raises(tmp_path):
    with pytest.raises(DatasetSplitError):
        run(make_config(tmp_path, make_records(2)))


def test_dry_run_writes_nothing(tmp_path):
    summary = run(make_config(tmp_path, make_records(40), dry_run=True))
    assert summary.final_records == 40
    assert not (tmp_path / "out").exists()


def test_config_validation():
    with pytest.raises(ValueError):
        PreparationConfig(input_paths=(Path("x"),), train_ratio=0.9, validation_ratio=0.1).validate()
    with pytest.raises(ValueError):
        PreparationConfig(input_paths=()).validate()


# --------------------------------------------------------------------------- #
# Language check is opt-in; English terms and figures are preserved
# --------------------------------------------------------------------------- #
MIXED_OUTPUT = (
    "الخلاصة: ارتفع هامش EBITDA إلى 18% على إيرادات بلغت $1,250,000 (SAR 4.7M)، "
    "وتتبع مؤشرات KPI شهريًا، وحقق الـ ROI نحو 2.9x خلال Q3."
)
ENGLISH_OUTPUT = "ROI is the ratio of net profit to the cost of the investment, as a percentage."


def test_language_check_is_disabled_by_default_and_terms_survive(tmp_path):
    assert PreparationConfig(input_paths=(Path("x"),)).min_arabic_ratio == 0.0
    assert build_parser().parse_args(["--input", "x"]).min_arabic_ratio == 0.0

    records = make_records(40) + [
        {"instruction": "ما أثر الهامش؟", "input": "", "output": MIXED_OUTPUT},
        {"instruction": "ما هو العائد على الاستثمار؟", "input": "", "output": ENGLISH_OUTPUT},
    ]
    summary = run(make_config(tmp_path, records))
    assert summary.final_records == 42 and not summary.invalid_by_reason
    rows = [r for n in ("train", "validation", "test") for r in read_rows(tmp_path / "out" / f"{n}.jsonl")]
    answer = next(r["messages"][2]["content"] for r in rows if r["messages"][1]["content"] == "ما أثر الهامش؟")
    for term in ("EBITDA", "18%", "$1,250,000", "SAR 4.7M", "KPI", "ROI", "2.9x", "Q3"):
        assert term in answer

    strict = run(make_config(tmp_path, records, min_arabic_ratio=0.6, output_dir=tmp_path / "strict"))
    assert strict.invalid_by_reason == {"output_not_arabic": 1}


def test_empty_and_invalid_answers_are_still_rejected(tmp_path):
    records = make_records(40) + [
        {"instruction": "سؤال أول", "output": ""},
        {"instruction": "سؤال ثان", "output": "   \n  "},
        {"instruction": "سؤال ثالث", "output": "12345 6789 %"},
        {"instruction": "سؤال رابع", "output": "N/A"},
        {"instruction": "سؤال خامس", "output": None},
    ]
    summary = run(make_config(tmp_path, records, min_output_chars=3))
    assert summary.invalid_by_reason == {
        "empty_field": 2, "output_no_letters": 1, "output_placeholder": 1, "bad_type": 1,
    }
    assert summary.final_records == 40


# --------------------------------------------------------------------------- #
# Retrieved context: production format, never empty or invented
# --------------------------------------------------------------------------- #
def system_prompt_of(tmp_path, record_overrides: dict) -> str:
    records = make_records(40) + [{"instruction": "سؤال الاختبار", "output": OUTPUT, **record_overrides}]
    run(make_config(tmp_path, records))
    rows = [r for n in ("train", "validation", "test") for r in read_rows(tmp_path / "out" / f"{n}.jsonl")]
    return next(r["messages"][0]["content"] for r in rows if r["messages"][1]["content"] == "سؤال الاختبار")


def test_context_sources_use_the_production_format(tmp_path):
    sources = [
        {"source": "a.pdf", "page": 3, "text": "  نص أول  "},
        {"source": "b.pdf", "text": "نص ثان"},
    ]
    prompt = system_prompt_of(tmp_path, {"context_sources": sources})
    expected = format_context([{"source": "a.pdf", "page": 3, "text": "نص أول"},
                               {"source": "b.pdf", "text": "نص ثان"}])
    assert prompt == f"SYS ctx:{expected}"
    assert expected == "[Source 1: a.pdf, page 3]\nنص أول\n\n[Source 2: b.pdf]\nنص ثان"


def test_preformatted_context_is_accepted_only_in_production_format(tmp_path):
    good = "[Source 1: a.pdf, page 2]\nنص"
    assert system_prompt_of(tmp_path, {"context": good}) == f"SYS ctx:{good}"


def test_blank_or_missing_context_injects_nothing(tmp_path):
    for override in ({}, {"context": None}, {"context": "   "}, {"context_sources": []},
                     {"context_sources": None}):
        out = tmp_path / f"c{len(list(tmp_path.iterdir()))}"
        out.mkdir()
        assert system_prompt_of(out, override) == "SYS no-ctx"


def test_invalid_context_is_rejected(tmp_path):
    bad = {
        "context_format": {"context": "نص بلا ترويسة"},
        "context_conflict": {"context": "[Source 1: a.pdf]\nنص", "context_sources": [{"source": "a", "text": "ن"}]},
        "empty_context": {"context_sources": [{"source": "a.pdf", "text": "   "}]},
        "bad_context_source": {"context_sources": [{"source": "a.pdf", "page": 0, "text": "نص"}]},
        "bad_type": {"context_sources": "not a list"},
    }
    for category, override in bad.items():
        records = make_records(40) + [{"instruction": "س", "output": OUTPUT, **override}]
        out = tmp_path / category
        out.mkdir()
        summary = run(make_config(out, records))
        assert summary.invalid_by_reason == {category: 1}, (category, summary.invalid_by_reason)


# --------------------------------------------------------------------------- #
# Real application prompt module (compact builder), fake tokenizer
# --------------------------------------------------------------------------- #
def test_pipeline_with_the_real_compact_prompt(tmp_path):
    pytest.importorskip("langchain_core")
    from app.ai.prompts.system_prompts import (
        NO_CONTEXT_NOTICE,
        SYSTEM_PROMPT_COMPACT,
        SYSTEM_PROMPT_CORE,
        build_compact_system_prompt,
    )

    assert DEFAULT_PROMPT_FUNCTION == "build_compact_system_prompt"
    records = make_records(40) + [
        {"instruction": "سؤال مع سياق", "output": OUTPUT,
         "context_sources": [{"source": "a.pdf", "page": 1, "text": "نص المستند"}]},
    ]
    run(make_config(tmp_path, records, max_seq_length=2048), builder=build_compact_system_prompt)
    rows = [r for n in ("train", "validation", "test") for r in read_rows(tmp_path / "out" / f"{n}.jsonl")]
    plain = next(r for r in rows if r["messages"][1]["content"] != "سؤال مع سياق")
    grounded = next(r for r in rows if r["messages"][1]["content"] == "سؤال مع سياق")

    for row in rows:
        system = row["messages"][0]["content"]
        assert system.startswith(SYSTEM_PROMPT_COMPACT)
        assert SYSTEM_PROMPT_CORE not in system          # the full prompt is never used for training
        assert "<documents>\n\n</documents>" not in system  # no empty context block
    assert plain["messages"][0]["content"] == build_compact_system_prompt()
    assert NO_CONTEXT_NOTICE in plain["messages"][0]["content"]
    assert "<documents>\n[Source 1: a.pdf, page 1]\nنص المستند\n</documents>" in grounded["messages"][0]["content"]
    assert NO_CONTEXT_NOTICE not in grounded["messages"][0]["content"]


# --------------------------------------------------------------------------- #
# Token report and budget
# --------------------------------------------------------------------------- #
def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def length_records() -> list[dict]:
    records = []
    for index, size in enumerate([8] * 30 + [300] * 10 + [600] * 4 + [1100]):
        records.append({
            "instruction": " ".join(random.Random(index).sample(VOCAB, 8)),
            "output": "ملخص " + words(f"ك{index}_", size),
        })
    return records


def test_token_report_distribution_counts_and_audit(tmp_path):
    summary = run(make_config(tmp_path, length_records(), max_seq_length=1024))
    report = summary.token_report

    assert report.examples == 45
    assert sum(report.histogram.values()) == 45
    assert set(report.stats) == {"min", "mean", "median", "p90", "p95", "p99", "max"}
    assert report.stats["min"] <= report.stats["median"] <= report.stats["p95"] <= report.stats["max"]
    assert report.over_reference == 5            # 4 long + 1 very long exceed 512
    assert report.over_max_seq == 1 == summary.overlength_dropped   # only the 1,100-word record exceeds 1024
    assert report.histogram[">1024"] == 1
    assert summary.final_records == 44

    dropped = read_rows(tmp_path / "out" / "overlength.jsonl")
    assert len(dropped) == 1 and dropped[0]["tokens"] > 1024 and dropped[0]["max_seq_length"] == 1024
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["token_report"]["over_reference"] == 5

    # Surviving answers are complete, never truncated.
    expected = {ArabicNormalizer(NormalizationPolicy()).normalize(r["output"]) for r in length_records()}
    for name in ("train", "validation", "test"):
        for row in read_rows(tmp_path / "out" / f"{name}.jsonl"):
            assert row["messages"][2]["content"] in expected
            assert row["num_tokens"] <= 1024


def test_at_512_long_records_are_reported_and_dropped_not_truncated(tmp_path):
    summary = run(make_config(tmp_path, length_records(), max_seq_length=512))
    assert summary.token_report.over_max_seq == 5 == summary.overlength_dropped
    assert any("DROPPED" in w and "never truncated" in w for w in summary.warnings) is (5 / 45 > 0.10)
    assert "Over 512 tokens" in summary.render()


def test_room_below_recommended_warns_but_above_minimum_runs(tmp_path):
    def mid_prompt(rag_context: str = "") -> str:
        return "كلمة " * 300

    summary = run(make_config(tmp_path, make_records(40), max_seq_length=512), builder=mid_prompt)
    assert 64 <= summary.token_report.room_tokens < 256
    assert any("recommended" in w for w in summary.warnings)

    roomy = run(make_config(tmp_path, make_records(40), max_seq_length=512, output_dir=tmp_path / "roomy"))
    assert roomy.token_report.room_tokens >= 256
    assert not any("recommended" in w for w in roomy.warnings)


def test_reference_full_prompt_tokens_are_reported_when_given(tmp_path):
    config = make_config(tmp_path, make_records(40), max_seq_length=2048)
    summary = DataPreparer(config, tokenizer=FakeTokenizer(), prompt_builder=short_prompt,
                           reference_prompt_builder=long_prompt).prepare()
    assert summary.token_report.reference_full_prompt_tokens > summary.token_report.prompt_overhead_tokens


# --------------------------------------------------------------------------- #
# Fix 2: stale output files across consecutive runs
# --------------------------------------------------------------------------- #
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
OVERLONG = {"instruction": "سؤال طويل", "output": "كلمة " * 400}
INVALID = {"instruction": "سؤال بلا إجابة"}


def read_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_audit_files_follow_the_current_run_across_consecutive_runs(tmp_path):
    out = tmp_path / "out"

    # Run 1: invalid AND over-length records -> both audit files have content.
    first = run(make_config(tmp_path, make_records(40) + [INVALID, OVERLONG], max_seq_length=256))
    assert (first.invalid_records, first.overlength_dropped) == (1, 1)
    assert len(read_rows(out / "rejected.jsonl")) == 1
    assert len(read_rows(out / "overlength.jsonl")) == 1

    # Run 2, same directory: clean data -> both files still exist, but are EMPTY.
    second = run(make_config(tmp_path, make_records(40), max_seq_length=256))
    assert (second.invalid_records, second.overlength_dropped) == (0, 0)
    for name in ("rejected.jsonl", "overlength.jsonl"):
        assert (out / name).exists() and (out / name).read_bytes() == b"", name
    files = read_manifest(out)["files"]
    assert files["rejected.jsonl"] == {"records": 0, "sha256": EMPTY_SHA256}
    assert files["overlength.jsonl"] == {"records": 0, "sha256": EMPTY_SHA256}

    # Run 3: only invalid records -> rejected has content, overlength stays empty.
    third = run(make_config(tmp_path, make_records(40) + [INVALID], max_seq_length=256))
    assert third.invalid_records == 1 and third.overlength_dropped == 0
    assert len(read_rows(out / "rejected.jsonl")) == 1
    assert (out / "overlength.jsonl").read_bytes() == b""

    # Run 4: only an over-length record -> rejected is emptied again, overlength has content.
    fourth = run(make_config(tmp_path, make_records(40) + [OVERLONG], max_seq_length=256))
    assert fourth.invalid_records == 0 and fourth.overlength_dropped == 1
    assert (out / "rejected.jsonl").read_bytes() == b""
    assert len(read_rows(out / "overlength.jsonl")) == 1

    # The data files and the manifest always describe the latest run.
    for name, count in fourth.splits.items():
        assert len(read_rows(out / f"{name}.jsonl")) == count
        assert read_manifest(out)["files"][f"{name}.jsonl"]["records"] == count
    assert set(read_manifest(out)["files"]) == {
        "train.jsonl", "validation.jsonl", "test.jsonl", "rejected.jsonl", "overlength.jsonl",
    }


def test_junk_audit_files_from_an_older_run_are_overwritten(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "rejected.jsonl").write_text('{"stale": "from an older version"}\n', encoding="utf-8")
    (out / "overlength.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
    run(make_config(tmp_path, make_records(40)))
    assert (out / "rejected.jsonl").read_bytes() == b""
    assert (out / "overlength.jsonl").read_bytes() == b""


def test_dry_run_leaves_existing_outputs_untouched(tmp_path):
    out = tmp_path / "out"
    run(make_config(tmp_path, make_records(40) + [INVALID]))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    run(make_config(tmp_path, make_records(40, seed=7), dry_run=True))
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    assert before == after


def test_failed_run_never_leaves_an_old_manifest_next_to_new_files(tmp_path):
    out = tmp_path / "out"
    run(make_config(tmp_path, make_records(40)))
    assert (out / "manifest.json").exists()

    original = prepare_module.DatasetWriter.write_jsonl

    def failing(self, name, rows):
        if name == "validation.jsonl":
            raise OutputWriteError("disk full (simulated)")
        return original(self, name, rows)

    prepare_module.DatasetWriter.write_jsonl = failing
    try:
        with pytest.raises(OutputWriteError):
            run(make_config(tmp_path, make_records(40, seed=9)))
    finally:
        prepare_module.DatasetWriter.write_jsonl = original

    assert not (out / "manifest.json").exists()      # no manifest = incomplete run, detectable
    run(make_config(tmp_path, make_records(40, seed=9)))
    assert (out / "manifest.json").exists()          # a successful run restores it


# --------------------------------------------------------------------------- #
# Fix 3: token-report population, wording and consistent denominators
# --------------------------------------------------------------------------- #
def test_token_report_describes_valid_unique_records_after_exact_deduplication(tmp_path):
    long_record = {"instruction": "سؤال طويل جدا", "output": "ملخص " + words("ط", 600)}
    records = make_records(40) + [long_record] * 5 + [INVALID]
    summary = run(make_config(tmp_path, records, max_seq_length=512))
    report = summary.token_report

    assert summary.original_records == 46
    assert summary.invalid_records == 1
    assert summary.duplicate_records == 4                  # five identical copies -> one kept
    assert summary.valid_unique_records == 41 == report.examples
    # The duplicated long record is counted ONCE, not five times.
    assert report.over_reference == 1 and report.over_max_seq == 1
    assert sum(report.histogram.values()) == 41
    assert summary.overlength_dropped == 1 and summary.final_records == 40


def test_record_funnel_is_consistent(tmp_path):
    base = make_records(60)
    records = base + [base[0]] * 3 + [INVALID, OVERLONG]
    summary = run(make_config(tmp_path, records, max_seq_length=256))

    assert summary.original_records == (
        summary.invalid_records + summary.duplicate_records + summary.valid_unique_records
    )
    assert summary.valid_unique_records == summary.overlength_dropped + summary.final_records
    assert sum(summary.splits.values()) == summary.final_records
    assert sum(summary.invalid_by_reason.values()) == summary.invalid_records
    assert read_manifest(tmp_path / "out")["summary"]["valid_unique_records"] == summary.valid_unique_records


def test_rendered_report_states_its_population_and_uses_one_denominator(tmp_path):
    summary = run(make_config(tmp_path, length_records(), max_seq_length=512))
    report = summary.token_report
    total = report.examples
    text = summary.render()

    assert "valid unique" in text
    assert "(after validation and exact deduplication, before length filtering)" in text
    line = next(l for l in text.splitlines() if l.strip().startswith("Over 512 tokens"))
    assert f"{report.over_reference} of {total} ({report.over_reference / total:.0%})" in line
    line = next(l for l in text.splitlines() if l.strip().startswith("Over max_seq=512"))
    assert f"{report.over_max_seq} of {total} ({report.over_max_seq / total:.0%})" in line
    funnel = next(l for l in text.splitlines() if "Over-length dropped" in l)
    assert f"{summary.overlength_dropped} of {summary.valid_unique_records}" in funnel
    assert any(l.strip().startswith("= Valid unique") for l in text.splitlines())


def test_context_counts_use_the_same_denominator(tmp_path):
    grounded = [
        {"instruction": f"سؤال مع سياق {i} " + " ".join(random.Random(i).sample(VOCAB, 6)),
         "output": OUTPUT, "context_sources": [{"source": "a.pdf", "page": i + 1, "text": f"نص {i}"}]}
        for i in range(3)
    ]
    summary = run(make_config(tmp_path, make_records(40) + grounded))
    report = summary.token_report
    assert report.with_context == 3 and report.examples == 43
    line = next(l for l in summary.render().splitlines() if l.strip().startswith("With context"))
    assert "3 of 43" in line and f"({3 / 43:.0%})" in line


def test_warning_text_uses_valid_unique_denominator(tmp_path):
    summary = run(make_config(tmp_path, length_records(), max_seq_length=256))
    assert summary.overlength_dropped == 15 and summary.token_report.examples == 45
    warning = next(w for w in summary.warnings if "DROPPED" in w)
    assert "15 of 45 valid unique records (33%)" in warning


def test_accounting_guard_detects_inconsistent_totals(tmp_path):
    summary = run(make_config(tmp_path, make_records(40)))
    summary.verify_accounting()                      # consistent: no error
    summary.final_records += 1
    with pytest.raises(DataPreparationError, match="accounting"):
        summary.verify_accounting()
    summary.final_records -= 1
    summary.valid_unique_records += 1
    with pytest.raises(DataPreparationError, match="accounting"):
        summary.verify_accounting()
