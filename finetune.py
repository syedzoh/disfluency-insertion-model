#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
import numpy as np
import argparse
import random
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


class Document:
    """Class to format input-output pairs for disfluency insertion task."""

    def __init__(self, input_text, output_text, tokenizer, disfluency_type):
        self.input_text = input_text
        self.output_text = output_text
        self.tokenizer = tokenizer
        self.disfluency_type = disfluency_type

    def to_text(self):
        """Format the document as instruction text."""
        if self.disfluency_type == 0:
            result = "<<SYS>>Add silence pauses to the input.<</SYS>>\n"
            result += f"[INST]Add silence pauses[/INST]\n"
            prompt = [result, self.input_text, self.output_text, "sil"]
            result += f"###Input:  {self.input_text}\n###Output: {self.output_text}"
        else:
            result = "<<SYS>>Add the disfluency to the input.<</SYS>>\n"
            result += f"[INST]Add disfluency[/INST]\n"
            prompt = [result, self.input_text, self.output_text, "disfl"]
            result += f"###Input:  {self.input_text}\n###Output: {self.output_text}"
        return result, prompt

    def token_len(self):
        """Get the length of the document in tokens."""
        return len(self.tokenizer.encode(self.to_text()[0]))


def create_dataset(train_path, test_path, tokenizer):
    """Create datasets from CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_documents = []
    test_documents = []

    # Process training data
    for _, row in train_df.iterrows():
        doc = Document(
            row["Input"], row["Output"], tokenizer, 0 if row["Type"] == "sil" else 1
        )
        document, _ = doc.to_text()
        train_documents.append(document)

    # Process test data
    for _, row in test_df.iterrows():
        doc = Document(
            row["Input"], row["Output"], tokenizer, 0 if row["Type"] == "sil" else 1
        )
        document, _ = doc.to_text()
        test_documents.append(document)

    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_documents})
    test_dataset = Dataset.from_dict({"text": test_documents})

    return DatasetDict({"train": train_dataset, "test": test_dataset})


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-2 for disfluency insertion"
    )

    # Model and data arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model name or path",
    )
    parser.add_argument(
        "--train_file", type=str, default="train.csv", help="Path to training data CSV"
    )
    parser.add_argument(
        "--test_file", type=str, default="test.csv", help="Path to test data CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/disfluency-model",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=200, help="Maximum sequence length"
    )

    # LoRA parameters
    parser.add_argument(
        "--lora_alpha", type=int, default=64, help="LoRA alpha parameter"
    )
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA r parameter")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout rate"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accumulation", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save model every X steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate model every X steps"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log metrics every X steps"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="disfluency-run",
        help="Name for this training run",
    )
    parser.add_argument(
        "--do_merge",
        action="store_true",
        help="Merge LoRA weights with base model after training",
    )

    return parser.parse_args()


def main():
    """Main function to run the fine-tuning process."""
    args = parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    merged_model_dir = os.path.join(args.output_dir, "merged_model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add special tokens for disfluencies if needed
    label_tokens = ["<sil>"]
    tokenizer.add_tokens(label_tokens)

    # Load and prepare the dataset
    print("Loading dataset...")
    dataset = create_dataset(args.train_file, args.test_file, tokenizer)
    print(
        f"Dataset loaded: {len(dataset['train'])} training examples, {len(dataset['test'])} test examples"
    )

    # Configure quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )

    # Resize token embeddings to account for new tokens
    base_model.resize_token_embeddings(len(tokenizer))

    # Disable caching for training
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Define LoRA configuration
    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.grad_accumulation,
        optim="paged_adamw_32bit",
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name=args.run_name,
    )

    # Initialize Trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {final_model_dir}")
    trainer.model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Optionally merge weights
    if args.do_merge:
        print("Cleaning up CUDA memory before merging...")
        del base_model
        del trainer
        torch.cuda.empty_cache()

        print("Loading model with LoRA weights...")
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            final_model_dir, device_map="auto", torch_dtype=torch.bfloat16
        )

        print("Merging weights...")
        model = model.merge_and_unload()

        print(f"Saving merged model to {merged_model_dir}")
        os.makedirs(merged_model_dir, exist_ok=True)
        model.save_pretrained(merged_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_model_dir)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
