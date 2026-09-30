# Enterprise Security LLM Distillation

[English](README.md) | [日本語](README.ja.md)

An experiment transferring enterprise-security defensive knowledge from Gemini-generated synthetic data into Qwen2.5-3B with 4-bit QLoRA, then merging and exporting GGUF for Ollama evaluation.

## Pipeline

Generate provenance-tagged synthetic defensive scenarios with Gemini, train Qwen2.5-3B-Instruct using 4-bit QLoRA, merge the adapter, convert to GGUF Q8_0, register with Ollama, and evaluate against the same-size baseline.

The five domains are SOC/SIEM/EDR, zero trust/IAM, incident response, cloud security, and network perimeter defense. The documented training environment uses an RTX 3070 with 8GB VRAM.

## Setup and validation

```bash
python -m venv .venv
# Activate the virtual environment for your shell.
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/evaluate.py --dry_run
```

## Interpretation

The experiment observed more concrete operational commands, but also confident hallucinations. The ten-scenario evaluation uses strict coverage and known-bad pattern checks; those automated scores do not establish production correctness. The Japanese guide retains the training commands, metrics, benchmark results, and limitations.

Inspired by [KnowledgeDistillation2026](https://github.com/furuhashilab/KnowledgeDistillation2026).


## Contents

- [data/](data)
- [docs/](docs)
- [ollama/](ollama)
- [src/](src)
- [tests/](tests)

## Detailed documentation

The [Japanese guide](README.ja.md) retains the complete original setup instructions, configuration, examples, project status, and limitations. Supporting documents keep their existing language.
