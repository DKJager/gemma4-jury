import sys, os, json, tempfile, pathlib, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.g4paper import data, sample, judge, analysis

# ----------------------------------------------------------------------
def test_load_submission_labels_and_skips_empty():
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        # Create all_preds.jsonl
        preds = [
            {"instance_id": 1, "model_patch": "code1"},
            {"instance_id": 2, "model_patch": ""},
            {"instance_id": 3, "model_patch": "code3"},
            {"instance_id": 4}  # missing model_patch
        ]
        with (td_path / "all_preds.jsonl").open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        # Create results.json
        results = {"resolved": [1, 3, 5]}
        results_dir = td_path / "results"
        results_dir.mkdir()
        with (results_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f)

        out = data.load_submission(str(td_path))
        # Should contain only 1 and 3
        assert set(out.keys()) == {1, 3}, "load_submission should skip empty or missing patches"
        assert out[1]["patch"] == "code1", "patch for instance 1 should be retained"
        assert out[3]["patch"] == "code3", "patch for instance 3 should be retained"
        assert out[1]["resolved"] is True and out[3]["resolved"] is True, "resolved flag should follow results.json"

        # ----- load_issues -----
        issues_path = td_path / "issues.jsonl"
        issues = [
            {"instance_id": 10, "repo": "repoA", "problem_statement": "stmtA"},
            {"instance_id": 20, "repo": "repoB", "problem_statement": "stmtB"},
        ]
        with issues_path.open("w", encoding="utf-8") as f:
            for i in issues:
                f.write(json.dumps(i) + "\n")
        issues_out = data.load_issues(str(issues_path))
        assert set(issues_out.keys()) == {10, 20}, "load_issues should map all instance_ids"
        assert issues_out[10]["repo"] == "repoA", "repo field should be retained"
        assert issues_out[20]["problem_statement"] == "stmtB", "problem_statement should be retained"


# ----------------------------------------------------------------------
def test_balanced_sample_is_balanced_deterministic_and_unique():
    items = [
        {"instance_id": 1, "submission": "sub1", "repo": "R1", "resolved": True},
        {"instance_id": 2, "submission": "sub2", "repo": "R2", "resolved": True},
        {"instance_id": 3, "submission": "sub3", "repo": "R1", "resolved": True},
        {"instance_id": 4, "submission": "sub4", "repo": "R3", "resolved": False},
        {"instance_id": 5, "submission": "sub5", "repo": "R2", "resolved": False},
        {"instance_id": 6, "submission": "sub6", "repo": "R3", "resolved": False},
    ]
    n = 4
    seed = 12345
    first = sample.balanced_sample(items, n, seed)
    second = sample.balanced_sample(items, n, seed)

    assert len(first) == n, "balanced_sample must return exactly n items"
    # Check balanced labels
    resolved_cnt = sum(1 for x in first if x["resolved"])
    assert resolved_cnt == n // 2, "must have n/2 resolved items"
    # Check uniqueness of (submission, instance_id)
    pairs = {(x["submission"], x["instance_id"]) for x in first}
    assert len(pairs) == n, "no duplicate (submission, instance_id) pairs allowed"
    # Deterministic ordering
    assert first == second, "same seed must give identical ordering"

    # Round‑robin repo allocation sanity check
    # For each label, compute max allowed per repo
    for label in (True, False):
        labelled = [x for x in first if x["resolved"] == label]
        repos = {x["repo"] for x in labelled}
        max_per_repo = (n // 2) // len(repos) + 1
        repo_counts = {r: sum(1 for x in labelled if x["repo"] == r) for r in repos}
        for rc in repo_counts.values():
            assert rc <= max_per_repo, "repo allocation exceeds round‑robin limit"


# ----------------------------------------------------------------------
def test_balanced_sample_raises_when_short():
    items = [
        {"instance_id": 1, "submission": "s1", "repo": "R1", "resolved": True},
        {"instance_id": 2, "submission": "s2", "repo": "R1", "resolved": False},
    ]
    try:
        sample.balanced_sample(items, 4, seed=0)
    except ValueError:
        pass
    else:
        assert False, "balanced_sample should raise ValueError when insufficient items"


# ----------------------------------------------------------------------
def test_build_messages_truncates_and_asks_json():
    long_text = "x" * 8000
    msgs = judge.build_messages(long_text, long_text, max_chars=12000)
    assert isinstance(msgs, list) and len(msgs) == 2, "build_messages must return two messages"
    system, user = msgs
    for m in (system, user):
        assert "role" in m and "content" in m, "each message must have role and content"
        assert isinstance(m["content"], str), "content must be a string"
    # Problem and patch live in the user message, so the truncation marker belongs there
    assert user["content"].count("...[truncated]...") >= 2, "truncation marker missing for problem and patch"
    # User message should request JSON output
    assert "p_resolved" in user["content"], "user message should ask for JSON with p_resolved"


# ----------------------------------------------------------------------
def test_parse_judgment_clamps_and_handles_garbage():
    # Clamp >1
    txt1 = '{"p_resolved": 1.7, "reason":"over"}'
    p, r = judge.parse_judgment(txt1)
    assert p == 1.0, "p_resolved should be clamped to 1.0"
    assert r == "over", "reason should be extracted correctly"

    # Prose before JSON
    txt2 = "Here is my answer:\n\n" + json.dumps({"p_resolved": 0.3, "reason": "ok"})
    p2, r2 = judge.parse_judgment(txt2)
    assert p2 == 0.3 and r2 == "ok", "parse_judgment must ignore preceding text"

    # No JSON at all
    p3, r3 = judge.parse_judgment("no json here")
    assert p3 is None and r3 == "unparseable", "should return (None, 'unparseable') when no JSON"


# ----------------------------------------------------------------------
def test_run_judgments_resumes_and_records_errors():
    items = [
        {"submission": "subA", "instance_id": 1, "repo": "R", "resolved": True, "patch": "codeA"},
        {"submission": "subB", "instance_id": 2, "repo": "R", "resolved": False, "patch": "codeB"},
    ]
    judges = ["judge_good", "judge_bad"]
    call_counts = {"judge_good": 0, "judge_bad": 0}

    def good_ask(messages):
        call_counts["judge_good"] += 1
        # Always return a valid JSON with p_resolved 0.8
        return json.dumps({"p_resolved": 0.8, "reason": "ok"})

    def bad_ask(messages):
        call_counts["judge_bad"] += 1
        raise RuntimeError("simulated failure")

    ask_fns = {"judge_good": good_ask, "judge_bad": bad_ask}
    rate_limits = {"judge_good": 0.0, "judge_bad": 0.0}
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "out.jsonl")
        # First run
        judge.run_judgments(items, judges, ask_fns, out_path, rate_limits, sleep=fake_sleep)

        # Verify file contents
        with open(out_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 4, "should have one row per (item, judge) pair"
        # Two rows should have errors
        errors = [r for r in rows if r["error"]]
        assert len(errors) == 2, "exactly two rows should contain errors"
        for e in errors:
            assert e["p"] is None, "error rows must have p null"
            assert "simulated failure" in e["error"], "error message should be captured"

        # Reset counters and run again – no new calls should be made
        call_counts["judge_good"] = 0
        call_counts["judge_bad"] = 0
        judge.run_judgments(items, judges, ask_fns, out_path, rate_limits, sleep=fake_sleep)
        assert call_counts["judge_good"] == 0 and call_counts["judge_bad"] == 0, "second run must be resumable with no extra calls"

        # File should still contain exactly the original rows
        with open(out_path, "r", encoding="utf-8") as f:
            rows2 = [json.loads(line) for line in f if line.strip()]
        assert len(rows2) == 4, "file size must remain unchanged after resumable run"


# ----------------------------------------------------------------------
def test_auroc_known_values():
    au = analysis.auroc
    # Perfect separation (positives higher)
    assert au([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0, "AUROC should be 1.0 for perfect ordering"
    # Perfect inverse
    assert au([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0, "AUROC should be 0.0 for reverse ordering"
    # Ties giving 0.5
    assert au([0.5, 0.5], [1, 0]) == 0.5, "AUROC should handle ties correctly"
    # Single‑class case returns None
    assert au([0.4], [1]) is None, "AUROC must return None when a class is empty"


# ----------------------------------------------------------------------
def test_agreement_curve_counts():
    # Build rows for 4 items, 3 judges
    items = [
        {"submission": "S1", "instance_id": 1},
        {"submission": "S1", "instance_id": 2},
        {"submission": "S2", "instance_id": 3},
        {"submission": "S2", "instance_id": 4},
    ]
    # Labels: items 1 and 3 are resolved
    labels = {
        ("S1", 1): 1,
        ("S1", 2): 0,
        ("S2", 3): 1,
        ("S2", 4): 0,
    }
    judges = ["J1", "J2", "J3"]
    rows = []
    # Assign p values: resolved items get high (>0.5), unresolved low (<0.5)
    for item in items:
        for j in judges:
            if labels[(item["submission"], item["instance_id"])] == 1:
                p = 0.9  # high
            else:
                p = 0.1  # low
            rows.append({
                "submission": item["submission"],
                "instance_id": item["instance_id"],
                "judge": j,
                "p": p,
                "reason": "",
                "error": ""
            })
    curve = analysis.agreement_curve(rows, labels, judges, threshold=0.5)
    # Expect for k=1,2,3: coverage 0.5 (2 of 4 items accepted), precision 1.0
    expected = [
        {"k": 1, "coverage": 0.5, "precision": 1.0},
        {"k": 2, "coverage": 0.5, "precision": 1.0},
        {"k": 3, "coverage": 0.5, "precision": 1.0},
    ]
    assert curve == expected, "agreement_curve should match expected coverage/precision values"


# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_load_submission_labels_and_skips_empty()
    test_balanced_sample_is_balanced_deterministic_and_unique()
    test_balanced_sample_raises_when_short()
    test_build_messages_truncates_and_asks_json()
    test_parse_judgment_clamps_and_handles_garbage()
    test_run_judgments_resumes_and_records_errors()
    test_auroc_known_values()
    test_agreement_curve_counts()
    print("PASS")
