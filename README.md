# Disfluency Insertion Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for fine-tuning large language models to insert disfluencies into text, making LLM-generated utterances sound more natural and spontaneous for speech synthesis. This is the official implementation of the paper "Enhancing Naturalness in LLM-Generated Utterances through Disfluency Insertion".

## Overview

Disfluencies are natural features of spontaneous human speech (hesitations, fillers, repetitions, etc.) but are typically absent from LLM outputs. This project demonstrates how the insertion of disfluencies can enhance the perceived naturalness of synthesized speech.

The approach involves:
1. Fine-tuning an LLM (Llama-2-7b) with Low-Rank Adaptation (LoRA) to incorporate various types of disfluencies
2. Generating disfluent versions of input text
3. Using Text-to-Speech models to synthesize more natural-sounding speech

Our user studies show that this approach significantly increases the perceived spontaneity of generated speech, with only a slight reduction in intelligibility.

## Repository Structure

```
├── finetune.py          # Script for fine-tuning the model
├── inference.py         # Script for generating disfluent text
├── evaluate.py          # Script for evaluating model performance
├── requirements.txt     # Dependencies
├── data/                # Directory for datasets
│   ├── train.csv        # Training data
│   └── test.csv         # Test data
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/disfluency-insertion.git
cd disfluency-insertion

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The model is trained on data derived from the Switchboard corpus with disfluency annotations. The datasets include:
- Pairs of fluent and disfluent utterances
- Annotations for both typical disfluencies (fillers, pauses) and atypical disfluencies (repetitions, substitutions, insertions)

The training data format consists of CSV files with the following columns:
- `Prompt`: The system instructions and prompt template
- `Input`: The fluent text input
- `Output`: The disfluent text output
- `Type`: Type of disfluency ('sil' for silence pauses, 'disfl' for other disfluencies)

## Usage

### Fine-tuning

To fine-tune the Llama-2-7b model with the provided dataset:

```bash
python finetune.py \
  --base_model "meta-llama/Llama-2-7b-chat-hf" \
  --train_file "data/train.csv" \
  --test_file "data/test.csv" \
  --output_dir "results/disfluency-model" \
  --batch_size 2 \
  --grad_accumulation 4 \
  --num_epochs 2 \
  --run_name "disfluency-run" \
  --do_merge
```

Key parameters:
- `--base_model`: Specifies the base model to fine-tune (requires Hugging Face access)
- `--do_merge`: Optional flag to merge LoRA weights with base model after training
- `--run_name`: Name for this training run (used for WandB logging)

### Inference

To generate disfluent versions of input text:

```bash
python inference.py \
  --model_path "results/disfluency-model/merged_model" \
  --input_text "As a Norwegian company, we understand firsthand the pressing need for powerful language models." \
  --mode "disfluency"
```

The `--mode` parameter can be either "disfluency" (for um, uh, repetitions, etc.) or "silence" (for silent pauses).

### Evaluation

To evaluate the model on the test set:

```bash
python evaluate.py \
  --model_path "results/disfluency-model/merged_model" \
  --test_file "data/test.csv" \
  --output_file "results/evaluation_results.json" \
  --predictions_file "results/model_predictions.csv"
```

This will calculate BLEU scores, BERTScore, and disfluency rates for the model's outputs.

## Speech Synthesis

While this repository focuses on text disfluency generation, our paper demonstrates that using Bark TTS for speech synthesis works well for rendering disfluencies. After generating disfluent text, you can use Bark or another TTS model to convert it to speech.

Example (using Bark directly):

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
import soundfile as sf

preload_models()

# Generate disfluent text using our model
disfluent_text = "Um, I'm not really uh sure that I even understand why the administration is proposing for instance tax reductions..."

# Generate audio with Bark
audio = generate_audio(disfluent_text)

# Save audio
sf.write("disfluent_speech.wav", audio, SAMPLE_RATE)
```

## Ethical Guidelines

To promote responsible use of this technology, please adhere to the following guidelines:

1. **Transparency**: Always clearly disclose when disfluent speech is AI-generated. Users interacting with systems using this technology must be informed they are not communicating with a human.

2. **Bias Monitoring**: Implement mechanisms to monitor and mitigate potential biases in disfluency patterns across different demographic groups, especially for speech disorders, neurodivergent individuals, non-native speakers, and minorities.

3. **Context Restrictions**: Do not deploy this technology in high-stakes contexts like legal proceedings, job interviews, or medical consultations without appropriate human oversight and explicit disclosure.

4. **Documentation**: Document potential limitations and ethical considerations in any derivative applications or research using this code.

5. **Avoid Deception**: Do not use disfluency insertion to manipulate perceptions of a speaker's credibility, emotional state, or cognitive abilities.

Our aim is to enhance human-computer interaction, not to deceive or manipulate users into believing they are interacting with humans when they are not.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This research used the Switchboard corpus with disfluency annotations by Zayats et al. (2019)
- We thank the NXT Switchboard Corpus for annotations about pitch contours, pauses, and other acoustic features
