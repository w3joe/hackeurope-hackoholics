"""
Quickest way to run predictions with your fine-tuned model.
Optimized for speed on CPU.

Usage:
    python quick_predict.py --model_path ./qwen-test
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import time


def load_model_fast(model_path):
    """Load model optimized for fast inference."""
    print("⚡ Loading model (optimized for speed)...")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=True
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load and merge LoRA weights (faster inference)
    model = PeftModel.from_pretrained(base_model, model_path)
    print("  Merging LoRA weights for faster inference...")
    model = model.merge_and_unload()  # This makes it faster!
    model.eval()

    print("✓ Ready!\n")
    return model, tokenizer


def predict_fast(model, tokenizer, user_input):
    """Generate prediction with speed optimizations."""
    system = "You are an expert epidemiological forecasting assistant. Analyze disease surveillance data and provide structured risk assessments with 12-week forecasts in JSON format."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,      # Enough for JSON response
            temperature=0.2,          # Low = faster + more consistent
            do_sample=True,           # Small sampling for quality
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - start

    return response, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./qwen-test")
    args = parser.parse_args()

    # Load model once
    model, tokenizer = load_model_fast(args.model_path)

    # Example prompt
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

    print("=" * 80)
    print("Running prediction...")
    print("=" * 80)

    response, time_taken = predict_fast(model, tokenizer, prompt)

    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(response)
    print(f"\n⚡ Time: {time_taken:.1f} seconds")

    # Parse JSON
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > 0:
            parsed = json.loads(response[start_idx:end_idx])

            print("\n" + "=" * 80)
            print("SUMMARY:")
            print("=" * 80)

            if "RiskAssessment" in parsed:
                ra = parsed["RiskAssessment"]
                print(f"Country: {ra.get('country')}")
                print(f"Risk Level: {ra.get('risk_level')}")
                print(f"Spread Likelihood: {ra.get('spread_likelihood')}")

                forecast = ra.get('twelve_week_forecast', {})
                if forecast:
                    cases = forecast.get('weekly_cases_per_100k', [])
                    if cases:
                        print(f"\nForecast (12 weeks):")
                        print(f"  Weeks 1-4:  {cases[:4]}")
                        print(f"  Weeks 5-8:  {cases[4:8]}")
                        print(f"  Weeks 9-12: {cases[8:12]}")
                        print(f"  Peak: {max(cases):.1f} cases/100k")
    except Exception as e:
        print(f"\n⚠️  Could not parse JSON: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
