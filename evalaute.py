#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
from bert_score import score as bert_score
import numpy as np
import json
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate disfluency generation model")

    # Model and data arguments
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test_file", type=str, default="test.csv", help="Path to test data CSV"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="model_predictions.csv",
        help="Path to save model predictions",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )

    return parser.parse_args()


def calculate_disfluency_rate(text):
    """Calculate approximate disfluency rate in text."""
    # Count typical disfluency markers
    disfluency_markers = [
        "um",
        "uh",
        "like",
        "you know",
        "<sil>",
        "i mean",
        "actually",
        "well",
        "so",
        "kind of",
        "sort of",
        "i think",
    ]

    words = text.lower().split()
    total_words = len(words)
    if total_words == 0:
        return 0

    # Count word repetitions (e.g., "the the")
    repetitions = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repetitions += 1

    # Count disfluency markers
    marker_count = 0
    for marker in disfluency_markers:
        marker_words = marker.split()
        if len(marker_words) == 1:
            marker_count += words.count(marker)
        else:
            # For multi-word markers
            for i in range(len(words) - len(marker_words) + 1):
                if words[i : i + len(marker_words)] == marker_words:
                    marker_count += 1

    # Calculate rate
    disfluency_count = repetitions + marker_count
    disfluency_rate = disfluency_count / total_words

    return disfluency_rate


def generate_predictions(model, tokenizer, test_df, max_samples=None):
    """Generate model predictions for test data."""
    if max_samples:
        test_df = test_df.sample(min(max_samples, len(test_df)), random_state=42)

    predictions = []

    for _, row in tqdm(
        test_df.iterrows(), total=len(test_df), desc="Generating predictions"
    ):
        input_text = row["Input"]
        disfluency_type = 0 if row["Type"] == "sil" else 1

        # Format prompt
        if disfluency_type == 0:
            prompt = "<<SYS>>Add silence pauses to the input.<</SYS>>\n"
            prompt += f"[INST]Add silence pauses.[/INST]\n"
        else:
            prompt = "<<SYS>>Add the disfluency to the input.<</SYS>>\n"
            prompt += f"[INST]Add disfluency.[/INST]\n"

        prompt += f"###Input: {input_text}\n###Output:"

        # Generate prediction
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Calculate appropriate token limits
            input_tokens = len(inputs.input_ids[0])
            expected_output_tokens = input_tokens * 1.5  # Rough estimate

            outputs = model.generate(
                **inputs,
                max_new_tokens=int(expected_output_tokens),
                num_beams=5,
                no_repeat_ngram_size=6,
                return_dict_in_generate=True,
            )

            sequence = outputs["sequences"][0]
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True)

            # Extract just the output part
            output_text = generated_text.split("###Output:")[-1].strip()
            predictions.append(output_text)

        except Exception as e:
            print(f"Error generating for input '{input_text}': {e}")
            predictions.append("")

    test_df["model_output"] = predictions
    return test_df


def evaluate_model(predictions_df):
    """Calculate evaluation metrics for model predictions."""
    # Prepare references and predictions
    references = predictions_df["Output"].tolist()
    predictions = predictions_df["model_output"].tolist()

    # Remove empty predictions
    valid_pairs = [(ref, pred) for ref, pred in zip(references, predictions) if pred]
    if not valid_pairs:
        return {
            "bleu": 0,
            "bertscore_precision": 0,
            "bertscore_recall": 0,
            "bertscore_f1": 0,
            "avg_disfluency_rate": 0,
        }

    valid_refs, valid_preds = zip(*valid_pairs)

    # Calculate BLEU score
    try:
        bleu = load_metric("bleu")
        bleu_score = bleu.compute(
            predictions=valid_preds, references=[[ref] for ref in valid_refs]
        )["bleu"]
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        bleu_score = 0

    # Calculate BERTScore
    try:
        P, R, F1 = bert_score(valid_preds, valid_refs, lang="en")
        bertscore_precision = P.mean().item()
        bertscore_recall = R.mean().item()
        bertscore_f1 = F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        bertscore_precision = bertscore_recall = bertscore_f1 = 0

    # Calculate disfluency rates
    disfluency_rates = [calculate_disfluency_rate(text) for text in valid_preds]
    avg_disfluency_rate = np.mean(disfluency_rates)

    return {
        "bleu": bleu_score,
        "bertscore_precision": bertscore_precision,
        "bertscore_recall": bertscore_recall,
        "bertscore_f1": bertscore_f1,
        "avg_disfluency_rate": avg_disfluency_rate,
    }


def main():
    """Main function to run the evaluation."""
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading test data from {args.test_file}")
    test_df = pd.read_csv(args.test_file)
    print(f"Loaded {len(test_df)} test samples")

    print("Generating predictions...")
    predictions_df = generate_predictions(model, tokenizer, test_df, args.max_samples)

    print(f"Saving predictions to {args.predictions_file}")
    predictions_df.to_csv(args.predictions_file, index=False)

    print("Calculating evaluation metrics...")
    metrics = evaluate_model(predictions_df)

    print("\nEvaluation Results:")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"BERTScore Precision: {metrics['bertscore_precision']:.4f}")
    print(f"BERTScore Recall: {metrics['bertscore_recall']:.4f}")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"Average Disfluency Rate: {metrics['avg_disfluency_rate']:.4f}")

    # Save metrics to file
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
