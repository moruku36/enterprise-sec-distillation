"""
merge_lora.py
学習済みの LoRA アダプターをベースモデル (Qwen2.5-3B-Instruct) に結合し、
Hugging Face 形式のマージ済みモデルを出力するスクリプト。
（GGUF変換は本スクリプト実行後、llama.cpp の convert_hf_to_gguf.py を利用します）
"""

import os
import argparse
from pathlib import Path
from contracts import ContractError
from stage_gate import approved, new_output, pilot_config


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--adapter_dir", default="models/lora_adapter", help="Directory containing LoRA weights")
    parser.add_argument("--output_dir", default="models/merged_model", help="Directory to save merged model")
    parser.add_argument("--config", default="configs/pilot.example.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-merge", action="store_true")
    args = parser.parse_args()

    if not args.execute and not args.validate_only:
        print("Merge blocked by default. Use --validate-only for offline preflight.")
        return 1
    try:
        config = pilot_config(args.config)
        if args.base_model != config["model"]["name"]:
            raise ContractError("base model differs from approved pilot config")
        new_output(args.output_dir)
        if not Path(args.adapter_dir).is_dir():
            raise ContractError("adapter directory is missing")
        if not args.execute:
            print("Offline merge validation passed; no model loaded.")
            return 0
        approved(config, "merge", args.approve_merge)
    except ContractError as exc:
        print(f"ERROR: {exc}")
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"=== Merging LoRA weights ===")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter Dir: {args.adapter_dir}")
    print(f"Output Dir: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=config["model"]["tokenizer_revision"],
                                              local_files_only=True, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=config["model"]["revision"], local_files_only=True,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPUマージでVRAMを消費しない
        trust_remote_code=False,
    )

    print("Loading and attaching LoRA weights...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging model weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("=== Merge Completed Successfully ===")
    print("\n[Next Steps for GGUF Conversion]")
    print(f"1. Convert to GGUF (FP16):")
    print(f"   python convert_hf_to_gguf.py {args.output_dir} --outfile models/sec-defense-f16.gguf")
    print(f"2. Quantize to Q8_0:")
    print(f"   llama-quantize models/sec-defense-f16.gguf models/sec-defense-q8_0.gguf Q8_0")
    print(f"3. Register to Ollama:")
    print(f"   ollama create sec-defense -f ollama/Modelfile\n")


if __name__ == "__main__":
    raise SystemExit(main())
