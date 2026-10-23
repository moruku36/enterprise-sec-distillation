# Offline evaluation protocol (Phase 1 / Gate A)

This harness decides *whether* retraining is worth doing. It does not train, download models, or call APIs.
Plan source: `enterprise-sec-distillation-retry-plan-2026-10-23.md` (Phase 1, Gate A).

## Run

```bash
python -m unittest discover -s tests -v
python src/validate_data.py --config configs/eval_offline.json
python src/evaluate.py --config configs/eval_offline.json --dry_run
python src/evaluate.py --config configs/eval_offline.json --backend cached --output runs/offline-review/
```

`--output` must be a new or empty directory outside `data/`. Existing outputs are never overwritten.
Missing/invalid inputs exit non-zero (fail-closed); there is no fallback to a smaller scenario set.

## What the numbers mean

- `data/v2/fixtures/` is **not a benchmark**: every record has `provenance: fixture`, sources are
  `fixture_unreviewed` with fixture-authored paraphrases (`retrieved_at: not_retrieved`). Metrics from it
  verify the pipeline only; manifests carry `quality_claim: NONE`.
- Structural metrics (decision match, coverage, citation id resolvability, command syntax) are deterministic.
  *Citation id resolvable* does not mean the span supports the claim.
- Quality metrics (selective correctness, supported-claim precision, severe unsafe) come only from
  `reviews.jsonl` (human `ReviewRecord`s). Without reviews they are `null` = unknown, never 0 or pass.
- `run_failure` (timeout, HTTP error, empty, invalid JSON/schema) is counted separately from refusals;
  rates are reported over all requests and excluding failures.
- Command syntax is `valid | invalid | unsupported`; unsupported is never valid. Only
  `aws ec2 modify-instance-metadata-options` has a spec table so far.
- Legacy known-bad strings are auxiliary flags. A warning like "do not run delete-bucket" matches them too.
- Baselines: `abstain` (answerable coverage must be 0) and `retrieval_only` (no generation).

## Legacy artifacts

`data/train.jsonl`, `data/eval.jsonl`, `data/eval_results.json` are immutable; `configs/legacy_hashes.json`
holds line-ending-normalised SHA-256s checked by `validate_data.py`. The 10 legacy questions are
regression/dev only, never the final test.

## Not in this phase

Source licence review, expert gold review, the 60 dev / 120 test set, B0–B3 runs, SFT, export/quantization
parity. `OllamaBackend` exists but needs `config.network=true` **and** `--allow-network`; it was not run.
