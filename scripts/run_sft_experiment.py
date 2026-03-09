"""SFT a model and evaluate sycophancy across test sets."""

from pathlib import Path
import json
import random
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from arcadia_interview_sft import load_train_data, load_prompts, sft_step, generate_completions, score_sycophancy


def avg_length(completions: list[str]) -> float:
    return sum(len(c) for c in completions) / len(completions)


def evaluate_all(model, tokenizer, test_dir, label):
    """Evaluate on all test sets and return results."""
    output_dir = Path(f"scratch/{label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_groups = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    results = {}
    lengths = {}
    for group_dir in test_groups:
        prompts = load_prompts(group_dir / "prompts.jsonl")
        completions = generate_completions(model, tokenizer, prompts)
        score = score_sycophancy(completions)
        length = avg_length(completions)
        results[group_dir.name] = score
        lengths[group_dir.name] = length
        print(f"  {group_dir.name}: sycophancy = {score:.2f}, avg_length = {length:.0f}")

        with open(output_dir / f"{group_dir.name}.jsonl", "w") as f:
            for prompt, completion in zip(prompts, completions):
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
    return results, lengths


def main():
    model_name = "google/gemma-3-4b-it"
    num_steps = 50
    batch_size = 5

    test_dir = Path("datasets/test")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

    print("\nEvaluating BASE model...")
    base_results, base_lengths = evaluate_all(model, tokenizer, test_dir, "base")

    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading training data...")
    train_data = load_train_data("datasets/train_2x2.jsonl")

    print(f"Fine-tuning for {num_steps} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(num_steps):
        batch = random.sample(train_data, batch_size)
        loss = sft_step(model, tokenizer, batch, optimizer)
        if step % 10 == 0:
            print(f"  Step {step}: loss = {loss:.4f}")

    print("\nEvaluating SFT model...")
    sft_results, sft_lengths = evaluate_all(model, tokenizer, test_dir, "sft")

    print("\nPlotting comparison...")
    names = list(base_results.keys())
    x = range(len(names))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar([i - w/2 for i in x], [base_results[n] for n in names], w, label="Base", color="steelblue")
    ax1.bar([i + w/2 for i in x], [sft_results[n] for n in names], w, label="SFT", color="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Sycophancy Score")
    ax1.set_title("Sycophancy: Base vs SFT")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.legend()

    ax2.bar([i - w/2 for i in x], [base_lengths[n] for n in names], w, label="Base", color="steelblue")
    ax2.bar([i + w/2 for i in x], [sft_lengths[n] for n in names], w, label="SFT", color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Avg Response Length (chars)")
    ax2.set_title("Response Length: Base vs SFT")
    ax2.legend()

    plt.tight_layout()
    Path("scratch").mkdir(exist_ok=True)
    plt.savefig("scratch/sft_evaluation.png", dpi=150)
    print("Saved to scratch/sft_evaluation.png")


if __name__ == "__main__":
    main()
