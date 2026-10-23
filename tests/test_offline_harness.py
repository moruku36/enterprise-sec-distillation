"""
test_offline_harness.py
オフライン評価基盤 (Gate A) の回帰テスト。ネットワーク・GPU・モデル・API は一切使わない。
"""

import copy
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import backends  # noqa: E402
import contracts  # noqa: E402
import evaluate  # noqa: E402
import metrics  # noqa: E402
import train_lora  # noqa: E402
import validate_data  # noqa: E402
from retrieval import build_corpus, get_context  # noqa: E402

CFG = json.loads((ROOT / "configs/eval_offline.json").read_text(encoding="utf-8"))
LEGACY = [ROOT / "data/train.jsonl", ROOT / "data/eval.jsonl", ROOT / "data/eval_results.json"]


def jl(path):
    return contracts.load_jsonl(ROOT / path)


EXAMPLES = jl("data/v2/fixtures/examples.jsonl")
SOURCES = jl("data/v2/fixtures/sources.jsonl")
KEYS = {(s["source_id"], sp["span_id"]) for s in SOURCES for sp in s["spans"]}


def pred(eid, decision="answer", claims=(), commands=(), status="ok"):
    return {"example_id": eid, "status": status, "decision": decision, "answer": "", "claims": list(claims),
            "commands": list(commands), "missing_information": [], "as_of": ""}


def run_main(*argv):
    return evaluate.main(list(argv))


class Workdir(unittest.TestCase):
    """データを一時ディレクトリへコピーして壊す系テスト用。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(ROOT / "data/v2", self.tmp / "data/v2")
        shutil.copytree(ROOT / "configs", self.tmp / "configs")
        for p in LEGACY:
            (self.tmp / "data").mkdir(exist_ok=True)
            shutil.copy(p, self.tmp / "data" / p.name)
        self.cfg = copy.deepcopy(CFG)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rewrite(self, name, fn):
        p = self.tmp / "data/v2/fixtures" / name
        rows = contracts.load_jsonl(p)
        fn(rows)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def validate(self):
        return validate_data.validate_all(self.cfg, self.tmp)[0]


class TestLegacyScorerIsNotCorrectness(unittest.TestCase):
    def test_saved_eval04_wrong_answer_is_not_verified_correct(self):  # 1
        results = {r["test_id"]: r for r in json.loads((ROOT / "data/eval_results.json").read_text(encoding="utf-8"))}
        crit = {r["id"]: r for r in jl("data/eval.jsonl")}["eval_04"]["ground_truth_criteria"]
        for side in ("base_response", "distilled_response"):
            ev = evaluate.evaluate_response_quality(results["eval_04"][side], crit)
            self.assertIsNone(ev["verified_correct"])

    def test_scattered_keywords_do_not_pass_semantically(self):  # 2
        crit = {"required_checks": [{"name": "x", "all_of": ["manage-bde"]}]}
        ev = evaluate.evaluate_response_quality("無関係な文章 manage-bde 無関係", crit)
        self.assertEqual(ev["legacy_keyword_coverage"], 1.0)
        self.assertIsNone(ev["verified_correct"])

    def test_warning_text_is_not_unsafe_and_blacklist_is_aux(self):  # 5
        text = "delete-bucket は実行しないでください。"
        self.assertEqual(metrics.legacy_flags(text, ["delete-bucket"]), ["delete-bucket"])  # 補助フラグ
        ex = [e for e in EXAMPLES if e["example_id"] == "fx_d1"]
        p = {"fx_d1": pred("fx_d1", "answer", commands=[{"command": "aws ec2 describe-instances", "mutating": False}])}
        m = metrics.compute_metrics(ex, p, KEYS)
        self.assertEqual(m["commands"]["forbidden_action_in_command_flags"], [])
        self.assertIsNone(m["review"]["severe_unsafe_count"])  # unsafe は人手レビューのみ


class TestMetrics(unittest.TestCase):
    def test_citation_id_existence_is_not_support(self):  # 3
        ex = [e for e in EXAMPLES if e["example_id"] == "fx_d1"]
        claim = {"text": "全く別の主張", "source_id": "aws-iam-revoke", "span_id": "revoke"}
        m = metrics.compute_metrics(ex, {"fx_d1": pred("fx_d1", claims=[claim])}, KEYS)
        self.assertEqual(m["citation_structural"]["ids_resolvable_in_corpus"], 1.0)
        self.assertEqual(m["citation_structural"]["ids_in_gold_spans"], 0.0)
        self.assertIsNone(m["review"]["supported_claim_precision"])

    def test_refusal_recognised_but_all_abstain_has_zero_coverage(self):  # 4
        preds = {e["example_id"]: pred(e["example_id"], "abstain") for e in EXAMPLES}
        m = metrics.compute_metrics(EXAMPLES, preds, KEYS)
        self.assertEqual(m["answerable_coverage"]["all_requests"], 0.0)
        self.assertEqual(m["false_answer_on_unanswerable"]["all_requests"], 0.0)

    def test_run_failures_are_not_correct_refusals(self):  # 6
        un = [e for e in EXAMPLES if e["answerability"] == "unanswerable"]
        for reason in ("timeout", "http_error", "empty", "bad_json"):
            raw = {"timeout": backends.RawResult("error", error="timeout"),
                   "http_error": backends.RawResult("error", error="http_500"),
                   "empty": backends.RawResult("ok", "  "),
                   "bad_json": backends.RawResult("ok", "{not json")}[reason]
            p = backends.parse_prediction(un[0]["example_id"], raw)
            self.assertEqual(p["status"], "run_failure")
        m = metrics.compute_metrics(un, {}, KEYS)
        self.assertEqual(m["run_failures"], len(un))
        self.assertEqual(m["ood_handling"]["all_requests"], None)
        self.assertEqual(m["decision_exact_match"]["all_requests"], 0.0)

    def test_unsupported_syntax_is_not_valid(self):  # 11
        self.assertEqual(metrics.check_command_syntax("kubectl get pods"), "unsupported")
        self.assertEqual(metrics.check_command_syntax(
            "aws ec2 modify-instance-metadata-options --instance-id i-0123456789abcdef0 --http-tokens required"), "valid")
        self.assertEqual(metrics.check_command_syntax(
            "aws ec2 modify-instance-metadata-options --instance-id i-1 --http-tokens maybe"), "invalid")
        self.assertEqual(metrics.check_command_syntax("aws ec2 modify-instance-metadata-options --http-tokens required"), "invalid")

    def test_review_metrics_unknown_until_reviewed(self):
        ex = [e for e in EXAMPLES if e["example_id"] == "fx_d1"]
        m = metrics.compute_metrics(ex, {"fx_d1": pred("fx_d1")}, KEYS)
        self.assertIsNone(m["review"]["selective_correctness"])
        p = pred("fx_d1", claims=[{"claim_id": "c1", "text": "reviewed"}])
        rv = {"fx_d1": {"example_id": "fx_d1", "run_hash": "a" * 64,
                         "prediction_hash": __import__("run_manifest").canonical_hash(p), "claim_ids": ["c1"],
                         "reviewer_id": "human-1", "reviewed_at": "2026-10-23", "reason": "checked",
                         "task_satisfied": True, "severe_unsafe": False, "claims_supported": {"c1": True}}}
        m = metrics.compute_metrics(ex, {"fx_d1": p}, KEYS, rv, run_hash="a" * 64)
        self.assertEqual(m["review"]["selective_correctness"], 1.0)

    def test_review_binding_rejects_different_answer_and_claims(self):
        ex = [e for e in EXAMPLES if e["example_id"] == "fx_d1"]
        p = pred("fx_d1", claims=[{"claim_id": "c1", "text": "first"}])
        rv = {"example_id": "fx_d1", "run_hash": "a" * 64,
              "prediction_hash": __import__("run_manifest").canonical_hash(p), "claim_ids": ["c1"],
              "reviewer_id": "human-1", "reviewed_at": "2026-10-23", "reason": "checked",
              "task_satisfied": True, "severe_unsafe": False, "claims_supported": {"c1": True}}
        for change in ({"answer": "different"}, {"claims": [{"claim_id": "c2", "text": "other"}]}):
            with self.subTest(change=change), self.assertRaises(contracts.ContractError):
                metrics.compute_metrics(ex, {"fx_d1": {**p, **change}}, KEYS, {"fx_d1": rv}, run_hash="a" * 64)
        for bad in ({"run_hash": "b" * 64}, {"reviewer_id": " "}, {"reason": ""},
                    {"claims_supported": {"c1": True, "c2": True}}):
            with self.subTest(bad=bad), self.assertRaises(contracts.ContractError):
                metrics.compute_metrics(ex, {"fx_d1": p}, KEYS, {"fx_d1": {**rv, **bad}}, run_hash="a" * 64)

    def test_command_values_are_not_confused_with_flags(self):
        stem = "aws ec2 modify-instance-metadata-options --instance-id i-0123456789abcdef0"
        for suffix in ("--http-tokens", "--http-tokens --http-endpoint enabled",
                       "--http-put-response-hop-limit nonsense", "--http-put-response-hop-limit 0",
                       "--http-put-response-hop-limit 65", "--http-protocol-ipv6 nonsense"):
            with self.subTest(suffix=suffix):
                self.assertEqual(metrics.check_command_syntax(stem + " " + suffix), "invalid")
        self.assertEqual(metrics.check_command_syntax(stem + " --future-flag x"), "unsupported")
        self.assertEqual(metrics.check_command_syntax(stem + " --http-put-response-hop-limit 2"), "valid")


class TestFailClosed(Workdir):
    def test_clean_data_validates(self):
        self.assertEqual(self.validate(), [])

    def test_missing_eval_file_fails_without_fallback(self):  # 8
        (self.tmp / "data/v2/fixtures/examples.jsonl").unlink()
        self.assertTrue(any("not found" in e for e in self.validate()))

    def test_family_split_leakage(self):  # 8
        self.rewrite("examples.jsonl", lambda r: r[5].update(family_id=r[0]["family_id"]))
        self.assertTrue(any("multiple splits" in e for e in self.validate()))

    def test_unapproved_source(self):  # 8
        self.rewrite("sources.jsonl", lambda r: r[0].update(review_status="rejected"))
        self.assertTrue(any("not approved" in e for e in self.validate()))

    def test_ambiguous_as_of(self):  # 8
        self.rewrite("examples.jsonl", lambda r: r[0].update(as_of="latest"))
        self.assertTrue(any("as_of" in e for e in self.validate()))

    def test_dangling_source_reference(self):
        self.rewrite("examples.jsonl", lambda r: r[0]["source_spans"].append({"source_id": "nope", "span_id": "x"}))
        self.assertTrue(any("unknown source_id" in e for e in self.validate()))

    def test_training_approval_requires_verification(self):
        self.rewrite("examples.jsonl", lambda r: r[0]["validation"].update(approved_for_training=True))
        self.assertTrue(any("approved_for_training" in e for e in self.validate()))

    def test_test_answer_in_corpus_is_leak(self):
        t = [e for e in EXAMPLES if e["split"] == "test" and e["gold_claims"]][0]["gold_claims"][0]
        self.rewrite("sources.jsonl", lambda r: r[0]["spans"].__setitem__(0, {
            **r[0]["spans"][0], "exact_text": t, "sha256": contracts.sha256_text(t)}))
        self.assertTrue(any("leakage" in e for e in self.validate()))

    def test_gate_flags_must_stay_false(self):
        for k in contracts.GATE_FLAGS:
            bad = dict(self.cfg, **{k: True})
            self.assertTrue(contracts.validate_config(bad), k)

    def test_locked_test_split_needs_explicit_flag(self):
        self.assertTrue(contracts.validate_config(dict(self.cfg, splits_to_evaluate=["test"])))

    def test_duplicate_review_is_rejected_before_evaluation(self):
        review = {"example_id": "fx_d1", "run_hash": "a" * 64, "prediction_hash": "b" * 64,
                  "claim_ids": [], "claims_supported": {}, "reviewer_id": "person", "reviewed_at": "2026-10-23",
                  "task_satisfied": False, "severe_unsafe": False, "reason": "checked"}
        path = self.tmp / "data/v2/fixtures/reviews.jsonl"
        path.write_text((json.dumps(review) + "\n") * 2, encoding="utf-8")
        self.assertTrue(any("duplicate review" in e for e in self.validate()))

    def test_review_attaches_only_to_same_run_and_prediction(self):
        cfg = self.tmp / "configs/eval_offline.json"
        first = self.tmp / "first"
        self.assertEqual(run_main("--root", str(self.tmp), "--config", str(cfg),
                                  "--backend", "cached", "--output", str(first)), 0)
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        prediction = [p for p in contracts.load_jsonl(first / "predictions.jsonl")
                      if p["example_id"] == "fx_d1"][0]
        ids = [c["claim_id"] for c in prediction["claims"]]
        review = {"example_id": "fx_d1", "run_hash": manifest["review_run_hash"],
                  "prediction_hash": __import__("run_manifest").canonical_hash(prediction),
                  "claim_ids": ids, "claims_supported": {cid: True for cid in ids},
                  "reviewer_id": "person", "reviewed_at": "2026-10-23", "reason": "checked",
                  "task_satisfied": True, "severe_unsafe": False}
        path = self.tmp / "data/v2/fixtures/reviews.jsonl"
        path.write_text(json.dumps(review) + "\n", encoding="utf-8")
        second = self.tmp / "second"
        self.assertEqual(run_main("--root", str(self.tmp), "--config", str(cfg),
                                  "--backend", "cached", "--output", str(second)), 0)
        result = json.loads((second / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(result["review"]["n_reviewed"], 1)
        self.assertIsNone(result["review"]["selective_correctness"])
        self.assertIsNone(result["review"]["severe_unsafe_rate_all"])
        review["prediction_hash"] = "0" * 64
        path.write_text(json.dumps(review) + "\n", encoding="utf-8")
        self.assertEqual(run_main("--root", str(self.tmp), "--config", str(cfg),
                                  "--backend", "cached", "--output", str(self.tmp / "third")), 1)

    def test_legacy_modification_detected(self):
        with open(self.tmp / "data/train.jsonl", "ab") as f:
            f.write(b"\n{}")
        self.assertTrue(any("legacy artifact modified" in e for e in self.validate()))

    def test_cli_exits_nonzero_on_invalid_data(self):
        (self.tmp / "data/v2/fixtures/sources.jsonl").write_text("{broken", encoding="utf-8")
        cfgp = self.tmp / "configs/eval_offline.json"
        self.assertEqual(run_main("--root", str(self.tmp), "--config", str(cfgp), "--dry_run"), 1)

    def test_pilot_example_budget_is_blocked(self):
        pilot = json.loads((ROOT / "configs/pilot.example.json").read_text(encoding="utf-8"))
        self.assertEqual(len(contracts.validate_pilot_config(pilot)), 8)


class TestOutputsAndDeterminism(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def hashes(self):
        return [p.read_bytes() for p in LEGACY]

    def test_dry_run_and_default_cli_never_touch_legacy_results(self):  # 7
        before = self.hashes()
        self.assertEqual(run_main("--dry_run"), 0)
        self.assertEqual(run_main("--backend", "cached", "--output", str(ROOT / "data" / "eval_results.json")), 1)
        self.assertEqual(run_main("--backend", "cached", "--output", str(ROOT / "data")), 1)
        self.assertEqual(run_main("--backend", "cached"), 1)  # --output 必須
        self.assertEqual(before, self.hashes())

    def test_refuses_nonempty_output_dir(self):
        (self.out / "x").write_text("keep", encoding="utf-8")
        self.assertEqual(run_main("--backend", "cached", "--output", str(self.out)), 1)
        self.assertEqual((self.out / "x").read_text(encoding="utf-8"), "keep")

    def test_same_input_same_content_hash(self):  # 9
        hs = []
        for _ in range(2):
            d = Path(tempfile.mkdtemp())
            try:
                self.assertEqual(run_main("--backend", "cached", "--output", str(d / "r")), 0)
                m = json.loads((d / "r/manifest.json").read_text(encoding="utf-8"))
                hs.append((m["content_hash"], (d / "r/metrics.json").read_bytes(), (d / "r/predictions.jsonl").read_bytes()))
            finally:
                shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(hs[0], hs[1])

    def test_no_network_during_run(self):  # Gate A: network 0
        def boom(*a, **k):
            raise AssertionError("network access attempted")
        orig = (socket.socket.connect, socket.create_connection)
        socket.socket.connect, socket.create_connection = boom, boom
        try:
            for b in ("cached", "abstain", "retrieval_only"):
                d = self.out / b
                self.assertEqual(run_main("--backend", b, "--output", str(d)), 0, b)
        finally:
            socket.socket.connect, socket.create_connection = orig

    def test_ollama_refused_without_double_opt_in(self):
        self.assertEqual(run_main("--backend", "ollama", "--model", "x", "--output", str(self.out / "o")), 1)
        self.assertEqual(run_main("--backend", "ollama", "--model", "x", "--allow-network",
                                  "--output", str(self.out / "o2")), 1)  # config.network=false

    def test_fixture_run_declares_no_quality_claim(self):
        self.assertEqual(run_main("--backend", "cached", "--output", str(self.out / "f")), 0)
        m = json.loads((self.out / "f/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["result_kind"], "fixture")
        self.assertTrue(m["quality_claim"].startswith("NONE"))
        self.assertFalse(any(m["gates"].values()))


class TestBackendsAndRetrieval(unittest.TestCase):
    def test_same_request_to_both_systems(self):  # 10
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def run(backend, config, name):
                cfg = root / (name + ".json")
                cfg.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                out = root / name
                self.assertEqual(run_main("--config", str(cfg), "--backend", backend,
                                          "--output", str(out)), 0)
                return json.loads((out / "manifest.json").read_text(encoding="utf-8"))["requests_hash"]
            base = run("abstain", CFG, "base")
            self.assertEqual(base, run("retrieval_only", CFG, "student"))
            altered = copy.deepcopy(CFG)
            altered["generation"]["seed"] += 1
            self.assertNotEqual(base, run("retrieval_only", altered, "changed_seed"))

    def test_retrieval_modes_and_no_label_access(self):
        corpus = build_corpus(SOURCES, CFG["allowed_source_review_status"])
        ex = [e for e in EXAMPLES if e["example_id"] == "fx_d1"][0]
        self.assertEqual(get_context(ex, corpus, "empty", 3), [])
        self.assertEqual([(c["source_id"], c["span_id"]) for c in get_context(ex, corpus, "oracle", 3)],
                         [("aws-ec2-imds-cli", "tokens")])
        stripped = {"question": ex["question"]}  # retrieved は question 以外を参照しない
        self.assertEqual(get_context(stripped, corpus, "retrieved", 2), get_context(ex, corpus, "retrieved", 2))
        self.assertEqual(get_context(ex, corpus, "retrieved", 3)[0]["source_id"], "aws-ec2-imds-cli")

    def test_unapproved_sources_not_indexed(self):
        self.assertEqual(build_corpus(SOURCES, ["human_verified"]), [])


class TestNoHeavyImports(unittest.TestCase):
    def test_validation_does_not_import_ml_stacks(self):  # 12
        code = ("import sys; sys.path.insert(0, 'src'); import validate_data, evaluate;"
                "validate_data.main(['--config','configs/eval_offline.json']);"
                "bad=[m for m in ('torch','transformers','trl','peft','datasets','bitsandbytes','google','dotenv') if m in sys.modules];"
                "print(bad); sys.exit(1 if bad else 0)")
        res = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)


class TestLegacyEntryGates(unittest.TestCase):
    def test_training_preflight_checks_pinned_dependency_versions_offline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = json.loads((ROOT / "configs/pilot.example.json").read_text(encoding="utf-8"))
            cfg["model"] = {"name": "Qwen/Qwen2.5-3B-Instruct", "revision": "pinned",
                            "tokenizer_revision": "pinned", "license_checked": True}
            cfg["budget"] = {"currency": "USD", "max_cost": 1, "max_input_tokens": 1000,
                             "max_output_tokens": 1000, "max_requests": 1, "max_training_gpu_hours": 1,
                             "max_models": 1, "max_storage_gb": 1}
            config_path, data_path = root / "pilot.json", root / "approved.jsonl"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            data_path.write_text(json.dumps({"instruction": "q", "output": "a", "validation_status": "passed",
                                             "source_verified": True, "reviewer_id": "person",
                                             "reviewed_at": "2026-10-23"}) + "\n", encoding="utf-8")
            args = SimpleNamespace(config=str(config_path), base_model=cfg["model"]["name"], epochs=1,
                                   batch_size=1, grad_accum=1, max_seq_length=128, lr=0.0002,
                                   output_dir=str(root / "adapter"), data_path=str(data_path))
            versions = {"trl": "0.15.1", "transformers": "4.49.0", "datasets": "3.3.2",
                        "peft": "0.14.0", "bitsandbytes": "0.45.3", "accelerate": "1.4.0"}
            with patch("stage_gate.metadata.version", side_effect=versions.get):
                self.assertEqual(train_lora.preflight(args), cfg)
            versions["trl"] = "0.16.0"
            with patch("stage_gate.metadata.version", side_effect=versions.get):
                with self.assertRaises(contracts.ContractError):
                    train_lora.preflight(args)

    def test_generator_dry_run_writes_nothing_and_legacy_path_fails(self):
        with tempfile.TemporaryDirectory() as d:
            output, raw = Path(d) / "new.jsonl", Path(d) / "raw"
            cmd = [sys.executable, "src/generate_dataset.py", "--output", str(output), "--raw_dir", str(raw)]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(raw.exists())
            result = subprocess.run(cmd + ["--execute"], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            result = subprocess.run([sys.executable, "src/generate_dataset.py", "--output", "data/train.jsonl",
                                     "--raw_dir", str(raw)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_training_and_merge_default_block_before_importing_ml(self):
        for script in ("train_lora.py", "merge_lora.py"):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, f"src/{script}"], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("blocked by default", result.stdout)


if __name__ == "__main__":
    unittest.main()
