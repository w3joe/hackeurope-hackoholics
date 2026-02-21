"""
Fast inference for fine-tuned model (optimized for speed).

Usage:
    python inference_fast.py --model_path ./qwen-test
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json


def load_model(model_path, base_model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load model with optimizations."""
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    # Merge LoRA weights for faster inference
    print("Merging LoRA weights (speeds up inference)...")
    model = model.merge_and_unload()

    print("✓ Model ready!")
    return model, tokenizer


def generate_fast(model, tokenizer, user_prompt):
    """Fast generation with optimized settings."""
    system_prompt = "You are an expert epidemiological forecasting assistant. Analyze disease surveillance data and provide structured risk assessments with 12-week forecasts in JSON format."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Reduced from 512
            temperature=0.1,  # Lower = faster, more deterministic
            do_sample=False,  # Greedy = faster than sampling
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main(args):
    print("=" * 80)
    print("Fast Inference Mode")
    print("=" * 80)

    model, tokenizer = load_model(args.model_path)

    prompt = """Analyze the following epidemiological data and provide a risk assessment:

**Pathogen:** Influenza
**Country:** Austria

**Recent Observation (Last 6 Weeks):**
Weeks: 2026-W1, 2026-W2, 2026-W3, 2026-W4, 2026-W5, 2026-W6
Cases per 100k: 0.0, 0.0, 0.0, 0.4, 0.4, 1.6

**Statistical Forecast Signals:**
- Holt-Winters forecast (next 12w): 6.4, 4.8, 12.7, 4.9, -3.0, -6.4
- Risk threshold: 18.8% per 100k
- High-risk weeks predicted: 0 weeks

**Anomaly Detection (TDA):**
- Status: normal
- Trend: H0_rising,H1_falling
- Number of anomalies: 8
- H0 z-score: -0.01, H1 z-score: -0.36

**Seasonality:**
- Historical peak weeks (ISO): W1, W26, W46, W47, W49

**Pattern Analysis:**
- Last observation week: 2026-W6
- Residual z-score: -1.57
- PCA position: normal
- Signal convergence: 0/3

Provide a comprehensive 12-week forecast with risk assessment."""

    print("\nGenerating prediction (fast mode)...")
    import time
    start = time.time()
    response = generate_fast(model, tokenizer, prompt)
    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print(f"\n⚡ Generation time: {elapsed:.1f} seconds")

    # Try to parse JSON
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > 0:
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            print("\n" + "=" * 80)
            print("PARSED:")
            print("=" * 80)
            print(json.dumps(parsed, indent=2))
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
