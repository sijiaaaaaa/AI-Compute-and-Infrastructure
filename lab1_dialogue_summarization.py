"""
Lab 1: Dialogue Summarization with FLAN-T5

Course context:
DeepLearning.AI / AWS — Generative AI with Large Language Models

This script demonstrates:
1. Loading the DialogSum dataset
2. Loading the FLAN-T5 model and tokenizer
3. Summarizing dialogue without prompt engineering
4. Zero-shot prompting
5. FLAN-T5-style prompt template
6. One-shot prompting
7. Few-shot prompting
8. Generation configuration parameters
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


# -----------------------------
# Config
# -----------------------------

MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "knkarthick/dialogsum"

EXAMPLE_INDICES = [40, 200]
DASH_LINE = "-" * 100


# -----------------------------
# Helper Functions
# -----------------------------

def load_resources(model_name: str = MODEL_NAME, dataset_name: str = DATASET_NAME):
    """
    Load dataset, tokenizer, and model.
    """

    print("Loading dataset...")
    dataset = load_dataset(dataset_name)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Using device: {device}")

    return dataset, tokenizer, model, device


def generate_text(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 50,
    generation_config: GenerationConfig | None = None,
):
    """
    Generate text from a prompt.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        if generation_config is not None:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                generation_config=generation_config,
            )
        else:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_dialogue_examples(dataset, example_indices):
    """
    Print sample dialogues and human-written summaries.
    """

    for i, index in enumerate(example_indices):
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        print(DASH_LINE)
        print(f"Example {i + 1}")
        print(DASH_LINE)
        print("INPUT DIALOGUE:")
        print(dialogue)
        print(DASH_LINE)
        print("BASELINE HUMAN SUMMARY:")
        print(summary)
        print(DASH_LINE)
        print()


def test_tokenizer(tokenizer):
    """
    Quick tokenizer sanity check.
    """

    sentence = "What time is it, Tom?"

    sentence_encoded = tokenizer(sentence, return_tensors="pt")
    sentence_decoded = tokenizer.decode(
        sentence_encoded["input_ids"][0],
        skip_special_tokens=True,
    )

    print(DASH_LINE)
    print("TOKENIZER TEST")
    print(DASH_LINE)
    print("ENCODED SENTENCE:")
    print(sentence_encoded["input_ids"][0])
    print()
    print("DECODED SENTENCE:")
    print(sentence_decoded)
    print(DASH_LINE)
    print()


def make_prompt(dataset, example_indices_full, example_index_to_summarize):
    """
    Build one-shot or few-shot prompt.

    example_indices_full:
        Dialogue-summary examples included before the target dialogue.

    example_index_to_summarize:
        Target dialogue to summarize.
    """

    prompt = ""

    for index in example_indices_full:
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        prompt += f"""
Dialogue:

{dialogue}

What was going on?
{summary}


"""

    target_dialogue = dataset["test"][example_index_to_summarize]["dialogue"]

    prompt += f"""
Dialogue:

{target_dialogue}

What was going on?
"""

    return prompt


# -----------------------------
# Lab Sections
# -----------------------------

def summarize_without_prompt_engineering(dataset, tokenizer, model, device):
    """
    Use the raw dialogue as input, without task instruction.
    """

    print()
    print("=" * 100)
    print("1. Summarization Without Prompt Engineering")
    print("=" * 100)

    for i, index in enumerate(EXAMPLE_INDICES):
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        output = generate_text(
            prompt=dialogue,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=50,
        )

        print(DASH_LINE)
        print(f"Example {i + 1}")
        print(DASH_LINE)
        print("INPUT PROMPT:")
        print(dialogue)
        print(DASH_LINE)
        print("BASELINE HUMAN SUMMARY:")
        print(summary)
        print(DASH_LINE)
        print("MODEL GENERATION - WITHOUT PROMPT ENGINEERING:")
        print(output)
        print()


def zero_shot_instruction_prompt(dataset, tokenizer, model, device):
    """
    Add a direct instruction prompt.
    """

    print()
    print("=" * 100)
    print("2. Zero-Shot Inference with an Instruction Prompt")
    print("=" * 100)

    for i, index in enumerate(EXAMPLE_INDICES):
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

        output = generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=50,
        )

        print(DASH_LINE)
        print(f"Example {i + 1}")
        print(DASH_LINE)
        print("INPUT PROMPT:")
        print(prompt)
        print(DASH_LINE)
        print("BASELINE HUMAN SUMMARY:")
        print(summary)
        print(DASH_LINE)
        print("MODEL GENERATION - ZERO SHOT:")
        print(output)
        print()


def zero_shot_flan_template(dataset, tokenizer, model, device):
    """
    Use a FLAN-T5-style prompt template.
    """

    print()
    print("=" * 100)
    print("3. Zero-Shot Inference with FLAN-T5-Style Template")
    print("=" * 100)

    for i, index in enumerate(EXAMPLE_INDICES):
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        prompt = f"""
Dialogue:

{dialogue}

What was going on?
"""

        output = generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=50,
        )

        print(DASH_LINE)
        print(f"Example {i + 1}")
        print(DASH_LINE)
        print("INPUT PROMPT:")
        print(prompt)
        print(DASH_LINE)
        print("BASELINE HUMAN SUMMARY:")
        print(summary)
        print(DASH_LINE)
        print("MODEL GENERATION - ZERO SHOT:")
        print(output)
        print()


def one_shot_inference(dataset, tokenizer, model, device):
    """
    Provide one full dialogue-summary example before the target dialogue.
    """

    print()
    print("=" * 100)
    print("4. One-Shot Inference")
    print("=" * 100)

    example_indices_full = [40]
    example_index_to_summarize = 200

    one_shot_prompt = make_prompt(
        dataset=dataset,
        example_indices_full=example_indices_full,
        example_index_to_summarize=example_index_to_summarize,
    )

    summary = dataset["test"][example_index_to_summarize]["summary"]

    output = generate_text(
        prompt=one_shot_prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=50,
    )

    print(DASH_LINE)
    print("ONE-SHOT PROMPT:")
    print(one_shot_prompt)
    print(DASH_LINE)
    print("BASELINE HUMAN SUMMARY:")
    print(summary)
    print(DASH_LINE)
    print("MODEL GENERATION - ONE SHOT:")
    print(output)
    print()


def few_shot_inference(dataset, tokenizer, model, device):
    """
    Provide multiple dialogue-summary examples before the target dialogue.
    """

    print()
    print("=" * 100)
    print("5. Few-Shot Inference")
    print("=" * 100)

    example_indices_full = [40, 80, 120]
    example_index_to_summarize = 200

    few_shot_prompt = make_prompt(
        dataset=dataset,
        example_indices_full=example_indices_full,
        example_index_to_summarize=example_index_to_summarize,
    )

    summary = dataset["test"][example_index_to_summarize]["summary"]

    output = generate_text(
        prompt=few_shot_prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=50,
    )

    print(DASH_LINE)
    print("FEW-SHOT PROMPT:")
    print(few_shot_prompt)
    print(DASH_LINE)
    print("BASELINE HUMAN SUMMARY:")
    print(summary)
    print(DASH_LINE)
    print("MODEL GENERATION - FEW SHOT:")
    print(output)
    print()

    return few_shot_prompt, summary


def generation_config_experiment(
    few_shot_prompt,
    summary,
    tokenizer,
    model,
    device,
):
    """
    Experiment with generation configuration parameters.
    """

    print()
    print("=" * 100)
    print("6. Generation Configuration Parameters")
    print("=" * 100)

    generation_configs = {
        "default_max_50": GenerationConfig(max_new_tokens=50),
        "short_max_10": GenerationConfig(max_new_tokens=10),
        "sample_temp_0_1": GenerationConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0.1,
        ),
        "sample_temp_0_5": GenerationConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0.5,
        ),
        "sample_temp_1_0": GenerationConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=1.0,
        ),
    }

    for config_name, generation_config in generation_configs.items():
        output = generate_text(
            prompt=few_shot_prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            generation_config=generation_config,
        )

        print(DASH_LINE)
        print(f"GENERATION CONFIG: {config_name}")
        print(DASH_LINE)
        print("MODEL GENERATION:")
        print(output)
        print()
        print("BASELINE HUMAN SUMMARY:")
        print(summary)
        print()


# -----------------------------
# Main
# -----------------------------

def main():
    dataset, tokenizer, model, device = load_resources()

    print()
    print("=" * 100)
    print("Sample Dialogues")
    print("=" * 100)
    print_dialogue_examples(dataset, EXAMPLE_INDICES)

    test_tokenizer(tokenizer)

    summarize_without_prompt_engineering(dataset, tokenizer, model, device)
    zero_shot_instruction_prompt(dataset, tokenizer, model, device)
    zero_shot_flan_template(dataset, tokenizer, model, device)
    one_shot_inference(dataset, tokenizer, model, device)

    few_shot_prompt, summary = few_shot_inference(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    generation_config_experiment(
        few_shot_prompt=few_shot_prompt,
        summary=summary,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    main()
