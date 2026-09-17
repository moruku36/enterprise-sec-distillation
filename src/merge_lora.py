"""
merge_lora.py
学習済みの LoRA アダプターをベースモデル (Qwen2.5-3B-Instruct) に結合し、
Hugging Face 形式のマージ済みモデルを出力するスクリプト。
（GGUF変換は本スクリプト実行後、llama.cpp の convert_hf_to_gguf.py を利用します）
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--adapter_dir", default="models/lora_adapter", help="Directory containing LoRA weights")
    parser.add_argument("--output_dir", default="models/merged_model", help="Directory to save merged model")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

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
    print("\n[Next Steps for GGUF Conversion]")
    print(f"1. Convert to GGUF (FP16):")
    print(f"   python convert_hf_to_gguf.py {args.output_dir} --outfile models/sec-defense-f16.gguf")
    print(f"2. Quantize to Q8_0:")
    print(f"   llama-quantize models/sec-defense-f16.gguf models/sec-defense-q8_0.gguf Q8_0")
    print(f"3. Register to Ollama:")
    print(f"   ollama create sec-defense -f ollama/Modelfile\n")


if __name__ == "__main__":
    main()
