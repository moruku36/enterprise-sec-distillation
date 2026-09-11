"""
merge_and_export.py
学習済みの LoRA アダプターをベースモデル (Qwen2.5-3B-Instruct) に結合し、
マージ済みモデルを出力するスクリプト。
"""

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--adapter_dir", default="models/lora_adapter", help="Directory containing LoRA weights")
    parser.add_argument("--output_dir", default="models/merged_model", help="Directory to save merged model")
    args = parser.parse_args()

    print(f"=== Merging LoRA weights ===")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter Dir: {args.adapter_dir}")
    print(f"Output Dir: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPUマージでVRAMを消費しない
        trust_remote_code=True,
    )

    print("Loading and attaching LoRA weights...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging model weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("=== Merge Completed Successfully ===")


if __name__ == "__main__":
    main()
