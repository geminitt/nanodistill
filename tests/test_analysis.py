import json
import math

import pytest

from nanodistill.analysis import compare, holm, load_run, t_test


def test_t_test_matches_the_t_table():
    # [1, 2, 3]: t = 2 / (1 / sqrt 3) = 3.464 with 2 degrees of freedom -> two-sided p = 0.0742
    assert t_test([1.0, 2.0, 3.0]) == pytest.approx(0.0742, abs=5e-4)
    assert t_test([1.0, 1.0, 1.0]) == 0.0


def test_holm_is_monotone_and_capped():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == [1.0, 1.0]


def fake_bfcl(tmp_path, correct: dict):
    """A BFCL folder with one category: `correct` maps item id -> bool."""
    res = tmp_path / "result" / "m" / "non_live"
    sco = tmp_path / "score" / "m" / "non_live"
    res.mkdir(parents=True); sco.mkdir(parents=True)
    (res / "BFCL_v4_simple_python_result.json").write_text(
        "".join(json.dumps({"id": i, "result": "", "output_token_count": 10}) + "\n" for i in correct))
    failed = [i for i, ok in correct.items() if not ok]
    (sco / "BFCL_v4_simple_python_score.json").write_text(
        json.dumps({"accuracy": 1 - len(failed) / len(correct)}) + "\n" +
        "".join(json.dumps({"id": i, "error_type": "ast_decoder:decoder_failed"}) + "\n" for i in failed))
    return str(tmp_path)


def test_load_run_reads_failures_from_the_score_file(tmp_path):
    run = load_run(fake_bfcl(tmp_path, {"a": True, "b": False, "c": True}))
    assert run["simple_python"]["correct"] == {"a": True, "b": False, "c": True}
    assert run["simple_python"]["unparsed"] == 1


def test_compare_finds_a_clear_difference():
    ids = [str(i) for i in range(200)]
    good = {"simple_python": {"correct": {i: True for i in ids}}}
    bad = {"simple_python": {"correct": {i: int(i) % 2 == 0 for i in ids}}}
    r = compare([good, good, good], [bad, bad, bad], ["simple_python"], n_boot=500)
    assert r["diff"] == pytest.approx(0.5) and r["ci95"][0] > 0


def test_non_inferiority_is_tested_against_the_margin():
    ids = [str(i) for i in range(200)]
    same = {"simple_python": {"correct": {i: int(i) % 2 == 0 for i in ids}}}
    r = compare([same, same], [same, same], ["simple_python"], null=-0.1, n_boot=500)
    assert r["diff"] == 0 and r["p_boot"] < 0.05 and r["p_seed"] == 0.0     # 0 is clearly above -10 points
    assert compare([same, same], [same, same], ["simple_python"], n_boot=500)["p_boot"] == 1.0


def test_report_runs_the_seven_tests_with_one_correction():
    from nanodistill.analysis import FAMILY, GROUPS, report
    ids = [str(i) for i in range(50)]
    run = {c: {"correct": {i: True for i in ids}, "tokens": {i: 10 for i in ids}, "unparsed": 0}
           for cats in GROUPS.values() for c in cats}
    conditions = {x for t in FAMILY for x in t[1:3]}
    text = report({c: {0: run, 1: run, 2: run} for c in conditions})
    table = text.split("## Pre-registered tests")[1].split("## Secondary")[0]
    assert len(FAMILY) == 7 and sum(line.startswith("| ") for line in table.splitlines()) == 1 + 7
    assert "not final" not in table
