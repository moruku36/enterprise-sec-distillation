# Offline CPU harness proof (2026-10-23)

Base: `main` at `103238ed9c5bccd3a5c3551a7cae54efe9a3885b`. Tested change: `efecbde7de11ed551f87c4db46b7260f0dc4eb00`. Environment: local Windows, Python 3.11.9, standard-library offline harness; no credentials, model, GPU, or network use. All data used by these checks is the repository's synthetic fixture. Temporary evaluation outputs were removed after inspection. Legacy data files were not changed.

## Reproduce

From the repository root, run:

```powershell
python src/validate_data.py
python src/evaluate.py --config configs/eval_offline.json --dry_run
python -m unittest tests.test_offline_harness.TestFailClosed.test_review_attaches_only_to_same_run_and_prediction -v
python -m unittest tests.test_offline_harness.TestMetrics.test_command_values_are_not_confused_with_flags -v
python -m unittest tests.test_offline_harness.TestLegacyEntryGates -v
python -m unittest discover -s tests -p test_offline_harness.py -q
python -m unittest discover -s tests -p test_smoke.py -q
```

For a saved fixture run, pass a **new** temporary directory to `python src/evaluate.py --backend cached --output <new-directory>`, then inspect `metrics.json` and `manifest.json` there. The harness refuses existing nonempty output paths.

## Observed locally

| Check | Observation |
|---|---|
| Data contracts and legacy hashes | 0 errors, 0 warnings |
| Cached fixture run | exit 0; `result_kind=fixture`; 5 examples; 0 run failures |
| Human-reviewed quality | 0 reviews; `selective_correctness=null`; manifest quality claim says `NONE` |
| Matching synthetic review | Accepted for the same run, prediction hash, and claim IDs |
| Changed prediction hash | Exit 1 with `review[fx_d1]: prediction hash mismatch`; no result directory written |
| CLI syntax check | `--http-tokens required` valid; `--http-tokens --http-endpoint enabled` invalid; nonsensical hop limit invalid; unknown flag unsupported. Commands were parsed, never executed. |
| Generator safety gate | Dry run exited 0 and wrote no files. `--execute` with the unset pilot template failed. Legacy `data/train.jsonl` output was refused. |
| Training and merge entry points | Both exited nonzero by default before importing the ML stack. |
| Offline harness and smoke tests | 38 and 5 tests passed, respectively. |

The full local schema suite could not start because this Python environment lacks `pydantic`; CI installs it and runs the complete suite on Python 3.10 and 3.11. This proof does not include human authenticity verification, actual model inference, training, model download, GPU use, paid API use, or a hallucination-reduction finding. The synthetic review is test data only.
