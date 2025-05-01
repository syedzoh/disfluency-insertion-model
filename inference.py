#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate disfluent versions of input text"
    )

    # Model and generation parameters
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--input_text", type=str, required=True, help="Input text to transform"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["disfluency", "silence"],
        default="disfluency",
        help="Type of transformation to apply",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=None,
        help="Minimum number of tokens to generate",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num_beams", type=int, default=5, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=6,
        help="Size of n-grams to avoid repeating",
    )

    return parser.parse_args()


def generate_disfluent_text(
    model,
    tokenizer,
    input_text,
    mode="disfluency",
    min_tokens=None,
    max_tokens=None,
    num_beams=5,
    no_repeat_ngram_size=6,
):
    """
    Generate disfluent version of input text.

    Args:
        model: The fine-tuned language model
        tokenizer: The tokenizer for the model
        input_text: The input text to transform
        mode: 'disfluency' or 'silence' for the type of transformation
        min_tokens: Minimum tokens to generate (calculated if None)
        max_tokens: Maximum tokens to generate (calculated if None)
        num_beams: Number of beams for beam search
        no_repeat_ngram_size: Size of n-grams to avoid repeating

    Returns:
        Generated text with added disfluencies
    """
    # Format the prompt based on the mode
    if mode == "silence":
        prompt = "<<SYS>>Add silence pauses to the input.<</SYS>>\n"
        prompt += f"[INST]Add silence pauses.[/INST]\n"
    else:
        prompt = "<<SYS>>Add the disfluency to the input.<</SYS>>\n"
        prompt += f"[INST]Add disfluency.[/INST]\n"

    prompt += f"###Input: {input_text}\n###Output:"

    # Calculate token counts
    input_tokens = tokenizer.encode(prompt)
    prompt_length = len(input_tokens)

    # Calculate min and max tokens if not provided
    if min_tokens is None or max_tokens is None:
        instruction_length = 42  # Approximate instruction length
        input_length = prompt_length - instruction_length

        if min_tokens is None:
            min_tokens = prompt_length + int(input_length * 0.25) + input_length

        if max_tokens is None:
            output_length = int(input_length * 0.5) + input_length
            max_tokens = prompt_length + output_length

    # Generate with the model
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        sequence = outputs["sequences"][0]
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)

        # Extract just the output part
        output_text = generated_text.split("###Output:")[-1].strip()

        return output_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


def main():
    """Main function to run inference."""
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Generating {args.mode} for input: {args.input_text}")
    output = generate_disfluent_text(
        model,
        tokenizer,
        args.input_text,
        mode=args.mode,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    if output:
        print("\nGenerated output:")
        print("-----------------")
        print(output)
    else:
        print("Failed to generate output.")


if __name__ == "__main__":
    main()
