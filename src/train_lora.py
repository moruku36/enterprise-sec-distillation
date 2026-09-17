"""
train_lora.py
Qwen2.5-3B-Instruct をベースに、生成したセキュリティデータセットを用いて
QLoRA (4-bit) によるファインチューニングを行うスクリプト。
RTX 3070 (8GB VRAM) に最適化された省メモリ設計。
"""

import os
import json
import argparse
from pathlib import Path


def load_qa_dataset(jsonl_path: str, tokenizer, max_length: int = 1536):
    from datasets import Dataset

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # ChatML 形式に変換
                prompt = item["instruction"]
                if item.get("input"):
                    prompt += f"\n\n【追加情報】\n{item['input']}"

                messages = [
                    {
                        "role": "system",
                        "content": "あなたはエンタープライズ環境のサイバーセキュリティ防御を専門とする上級セキュリティエンジニアです。実践的かつ体系的なSOC運用・インシデント対応・ゼロトラスト設計の観点で具体的かつ正確に回答してください。",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": item["output"]},
                ]
                formatted_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                data.append({"text": formatted_text})

    print(f"Loaded {len(data)} training samples from {jsonl_path}")
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="QLoRA training for Enterprise Security LLM")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--data_path", default="data/train.jsonl", help="Training JSONL dataset path")
    parser.add_argument("--output_dir", default="models/lora_adapter", help="Directory to save LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1536, help="Maximum sequence length")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from trl import SFTTrainer

    print(f"=== Starting QLoRA Training ===")
    print(f"Base Model: {args.base_model}")
    print(f"Output Dir: {args.output_dir}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 1. 4-bit 量子化設定 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Tokenizer & Model 読み込み
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )

    # 3. LoRA 設定 (SFTTrainer が内部で適用)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. データセット読み込み
    train_dataset = load_qa_dataset(args.data_path, tokenizer, args.max_seq_length)

    from trl import SFTConfig, SFTTrainer

    # 5. SFTConfig 設定 (8GB VRAM 最適化)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_steps=2,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    # 6. Trainer 初期化と実行
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # 7. アダプターの保存
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("=== Training Completed Successfully ===")


if __name__ == "__main__":
    main()
