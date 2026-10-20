import pytest

from src.variants import (
    anchor_term, check_rewrite, check_rows, content_words, features, read_sheet, standalone_numbers,
    tokens, write_sheet,
)

# Toy corpus statistics: "coronavirus" is everywhere, "remdesivir" is rare.
DF = {"coronavirus": 900, "remdesivir": 5, "origin": 40, "antiviral": 30, "gs": 2, "5734": 2,
      "sars": 500, "cov": 500, "2": 600, "source": 80, "drug": 200, "trial": 150}
N = 1000


def test_tokens_and_content_words():
    assert tokens("SARS-CoV-2 spike") == ["sars", "cov", "2", "spike"]
    assert content_words("how does the coronavirus spread") == {"coronavirus", "spread"}


def test_anchor_is_rarest_content_word():
    assert anchor_term("coronavirus remdesivir", DF, N) == "remdesivir"
    assert anchor_term("coronavirus origin", DF, N) == "origin"


def test_features():
    f = features("coronavirus remdesivir", "SARS-CoV-2 GS-5734", DF, N)
    assert f["anchor"] == "remdesivir" and f["anchor_removed"]
    assert f["content_overlap"] == 0.0
    assert f["n_words"] == 2 and f["length_change"] == 0  # "SARS-CoV-2" is one word
    assert f["n_acronyms"] == 2                            # SARS-CoV-2, GS-5734
    assert features("coronavirus", "mRNA vaccine", DF, N)["n_acronyms"] == 1


def test_good_rewrite_passes_cleanly():
    assert check_rewrite("coronavirus remdesivir", "SARS-CoV-2 GS-5734", "alias", "a", DF, N) == ([], [])


def test_standalone_numbers_ignore_names():
    assert standalone_numbers("COVID-19 deaths in 2020 among over 65s") == {"2020"}
    assert standalone_numbers("SARS-CoV-2 GS-5734 dose 3.5 mg") == {"3.5"}
    _, warnings = check_rewrite("vaccine trials 2020", "vaccine studies 2021", "paraphrase", "a", DF, N)
    assert any("numbers differ" in w for w in warnings)


@pytest.mark.parametrize("rewrite, phenomenon, problem", [
    ("", "alias", "missing"),
    ("coronavirus remdesivir", "alias", "same words"),
    ("coronavirus antiviral", "synonym", "not one of"),
    ("coronavirus without remdesivir", "paraphrase", "negation"),
])
def test_errors(rewrite, phenomenon, problem):
    errors, _ = check_rewrite("coronavirus remdesivir", rewrite, phenomenon, "a", DF, N)
    assert any(problem in e for e in errors)


@pytest.mark.parametrize("rewrite, problem", [
    ("remdesivir for SARS-CoV-2 patients", "anchor"),
    ("coronavirus remdesivir drug", "overlap"),
    ("what does the antiviral drug from Gilead do to coronavirus patients", "length"),
    ("does antiviral GS-5734 help", "question"),
])
def test_warnings(rewrite, problem):
    _, warnings = check_rewrite("coronavirus remdesivir", rewrite, "paraphrase", "a", DF, N)
    assert any(problem in w for w in warnings)


def row(qid, a, b, verified="yes", original="coronavirus remdesivir"):
    return {"qid": qid, "original": original, "question": "", "narrative": "", "anchor": "",
            "rewrite_a": a, "phenomenon_a": "alias", "rewrite_b": b, "phenomenon_b": "deanchoring",
            "verified": verified, "notes": ""}


def test_sheet_level_checks():
    rows = [row("1", "SARS-CoV-2 GS-5734", "SARS-CoV-2 GS-5734"), row("1", "x y", "y z", verified="")]
    errors, warnings = check_rows(rows, {"1", "2"}, DF, N)
    assert "topic missing from the sheet" in errors["2"]
    assert any("appears 2 times" in e for e in errors["1"])
    assert any("are the same" in e for e in errors["1"])
    assert any("not verified" in w for w in warnings["1"])


def test_sheet_round_trip_with_excel_bom(tmp_path):
    path = write_sheet([row("1", "a b", "c d")], tmp_path / "s.csv")
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())  # Excel "CSV UTF-8" adds this
    assert read_sheet(path)[0]["rewrite_a"] == "a b"


def test_sheet_saved_as_windows_1252(tmp_path):
    path = write_sheet([row("1", "Gilead’s antiviral", "c d")], tmp_path / "s.csv")
    path.write_bytes(path.read_text(encoding="utf-8").encode("cp1252"))  # Excel plain "CSV"
    assert read_sheet(path)[0]["rewrite_a"] == "Gilead’s antiviral"
