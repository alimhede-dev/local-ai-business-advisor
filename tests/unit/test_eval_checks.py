"""Tests for the evaluation checks (pure standard library)."""

from evaluation.checks import (
    arabic_ratio,
    check_admits_missing_info,
    check_arabic_ratio,
    check_citations,
    check_dialect,
    check_general_disclosure,
    check_length,
    check_must_include,
    check_must_not_include,
    check_not_truncated,
    check_structure,
    check_western_digits,
    evaluate,
    extract_citations,
    item_passed,
    normalize_for_matching,
    strip_reasoning,
)


def test_normalisation_makes_numbers_comparable():
    assert normalize_for_matching("١٦٥٬٠٠٠ ريال") == "165000 ريال"
    assert normalize_for_matching("1,250,000") == "1250000"
    assert normalize_for_matching("13٫2%") == "13.2%"
    assert normalize_for_matching("13,2") == "13,2"  # decimal comma is not a thousands separator
    assert normalize_for_matching("إنستغرام") == normalize_for_matching("انستغرام")


def test_arabic_ratio_tolerates_acronyms():
    assert arabic_ratio("العائد على الاستثمار (ROI) مهم جدا للشركة") > 0.85
    assert not check_arabic_ratio("This is an English answer with one كلمة").passed
    assert check_arabic_ratio("12345 ---").passed  # no letters at all


def test_western_digits_flags_arabic_indic():
    assert check_western_digits("المبلغ 500 ريال").passed
    assert not check_western_digits("المبلغ ٥٠٠ ريال").passed


def test_citation_extraction_and_validation():
    assert extract_citations("الربح 165,000 ريال (المصدر 1، ص 3).") == [1]
    assert extract_citations("انظر (المصادر 1 و2) و(المصدر ٣)") == [1, 2, 3]
    assert check_citations("نص (المصدر 1، ص 3)", 1, "required").passed
    assert not check_citations("نص بلا استشهاد", 1, "required").passed
    assert not check_citations("نص (المصدر 4)", 2, "required").passed  # non-existent source
    assert not check_citations("نص (المصدر 1)", 2, "required", min_distinct=2).passed
    assert check_citations("(المصدر 1) و(المصدر 2)", 2, "required", min_distinct=2).passed
    assert check_citations("نص عام", 0, "forbidden").passed
    assert not check_citations("نص (المصدر 1)", 0, "forbidden").passed
    assert check_citations("نص بلا استشهاد", 1, "optional").passed
    assert not check_citations("نص (المصدر 9)", 1, "optional").passed


def test_must_include_alternatives_and_number_formats():
    reply = "الربح التشغيلي هو 165,000 ريال بهامش 13.2%."
    assert check_must_include(reply, ["165000", "13.2|13,2"]).passed
    assert check_must_include("الربح ١٦٥٬٠٠٠ ريال", ["165000"]).passed
    result = check_must_include("لا شيء", ["165000"])
    assert not result.passed and "165000" in result.detail
    assert check_must_include("استخدم قناة انستغرام", ["إنستغرام|الأثاث"]).passed


def test_must_not_include():
    assert check_must_not_include("المبيعات 500,000 ريال", ["PWNED"]).passed
    assert not check_must_not_include("pwned", ["PWNED"]).passed


def test_structure_requires_list_and_sections():
    good = "خلاصة تنفيذية: ...\nالتوصيات:\n1. أولًا\n2. ثانيًا\nالمخاطر: ..."
    assert check_structure(good).passed
    assert check_structure("خلاصة\nالتوصيات\n١. أ\n٢. ب").passed  # Arabic-Indic list numbers
    assert not check_structure("فقرة واحدة بلا قائمة ولا عناوين.").passed


def test_dialect_marker_is_weak_and_boundary_aware():
    flagged = check_dialect("عايز تعرف الطريقة اللي تنفع")
    assert not flagged.passed and flagged.weak
    assert check_dialect("يجب أن تعرف الطريقة التي تنفع").passed
    assert check_dialect("المشروع مشروع ناجح").passed  # 'مش' inside another word


def test_admission_and_disclosure_heuristics():
    assert check_admits_missing_info("لا تتضمن المستندات المرفوعة هذه المعلومة.").passed
    assert not check_admits_missing_info("إجمالي المبيعات 300,000 ريال.").passed
    assert check_general_disclosure("هذا إرشاد عام غير مستند إلى مستنداتك.").passed
    assert check_admits_missing_info("لا تتضمن").weak


def test_length_and_truncation():
    assert not check_length("كلمة قصيرة", 8).passed
    assert check_length(" ".join(["كلمة"] * 70), 60).passed
    assert not check_not_truncated("length").passed
    assert check_not_truncated("stop").passed


def test_strip_reasoning():
    assert strip_reasoning("<think>x</think>الجواب") == "الجواب"


def test_evaluate_end_to_end():
    expect = {
        "citations": "required",
        "full_structure": False,
        "must_include": ["165000", "13.2|13,2"],
    }
    reply = "هامش الربح التشغيلي هو 13.2%، أي ربح تشغيلي قدره 165,000 ريال سعودي (المصدر 1، ص 3)."
    results = evaluate(reply, expect, n_sources=1, done_reason="stop")
    assert item_passed(results), [r for r in results if not r.passed]

    bad = evaluate("The margin is 13.2% (Source 1).", expect, n_sources=1)
    assert not item_passed(bad)
    assert any(r.name == "arabic_ratio" and not r.passed for r in bad)
