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
from contracts import ContractError, load_jsonl, is_iso_date
from stage_gate import approved, new_output, pilot_config, require_version


def preflight(args):
    config = pilot_config(args.config)
    if args.base_model != config["model"]["name"]:
        raise ContractError("base model differs from approved pilot config")
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.max_seq_length < 1 or args.lr <= 0:
        raise ContractError("training hyperparameters must be positive")
    new_output(args.output_dir)
    rows = load_jsonl(Path(args.data_path))
    if not rows:
        raise ContractError("training data is empty")
    for i, row in enumerate(rows, 1):
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("instruction", "output")):
            raise ContractError(f"training row {i}: missing instruction/output")
        if row.get("validation_status") != "passed" or row.get("source_verified") is not True or not (
                isinstance(row.get("reviewer_id"), str) and row["reviewer_id"].strip()
                and is_iso_date(row.get("reviewed_at"))):
            raise ContractError(f"training row {i}: human source review/approval is missing")
    for package, version in (("trl", "0.15.1"), ("transformers", "4.49.0"),
                             ("datasets", "3.3.2"), ("peft", "0.14.0"),
                             ("bitsandbytes", "0.45.3"), ("accelerate", "1.4.0")):
        require_version(package, version)
    return config


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
    parser.add_argument("--config", default="configs/pilot.example.json")
    parser.add_argument("--validate-only", action="store_true", help="Offline preflight; never loads a model")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-training", action="store_true")
    args = parser.parse_args()

    if not args.execute and not args.validate_only:
        print("Training blocked by default. Use --validate-only for offline preflight.")
        return 1
    try:
        config = preflight(args)
        if not args.execute:
            print("Offline validation passed; no model loaded and no training started.")
            return 0
        approved(config, "training", args.approve_training)
    except ContractError as exc:
        print(f"ERROR: {exc}")
        return 1

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
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=config["model"]["tokenizer_revision"],
                                              local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=config["model"]["revision"], local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=False,
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
        max_seq_length=args.max_seq_length,
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
    raise SystemExit(main())
