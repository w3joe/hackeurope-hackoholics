"""
Quick test script for your fine-tuned model.
Usage: python test_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json


def load_model(model_path="./qwen-test"):
    """Load fine-tuned model."""
    print("Loading model...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        trust_remote_code=True
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    print("✓ Model loaded!")
    return model, tokenizer


def generate_prediction(model, tokenizer, user_prompt):
    """Generate a prediction."""
    system_prompt = "You are an expert epidemiological forecasting assistant. Analyze disease surveillance data and provide structured risk assessments with 12-week forecasts in JSON format."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Format with chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    print("\nGenerating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def main():
    print("=" * 80)
    print("Testing Fine-Tuned Qwen2.5-3B Epidemiological Forecasting Model")
    print("=" * 80)

    # Load model
    model, tokenizer = load_model()

    # Example prompt
    prompt = """Analyze the following epidemiological data and provide a risk assessment:

**Pathogen:** Influenza
**Country:** Austria

**Recent Observation (Last 6 Weeks):**
Weeks: 2026-W1, 2026-W2, 2026-W3, 2026-W4, 2026-W5, 2026-W6
Cases per 100k: 0.0, 0.0, 0.0, 0.4, 0.4, 1.6

**Statistical Forecast Signals:**
- Holt-Winters forecast (next 12w): 6.4, 4.8, 12.7, 4.9, -3.0, -6.4... (truncated)
- Risk threshold: 18.8% per 100k
- High-risk weeks predicted: 0 weeks

**Anomaly Detection (TDA):**
- Status: normal
- Trend: H0_rising,H1_falling
- Number of anomalies: 8
- Lead time: 7 weeks
- H0 z-score: -0.01, H1 z-score: -0.36

**Seasonality:**
- Historical peak weeks (ISO): W1, W26, W46, W47, W49...

**Pattern Analysis:**
- Last observation week: 2024-W49
- Residual z-score: -1.57
- Forecast surprised by actual: False
- PCA distance z-score: -1.49
- PCA position: normal
- Signal convergence: 0/3

Provide a comprehensive 12-week forecast with risk assessment."""

    # Generate prediction
    response = generate_prediction(model, tokenizer, prompt)

    print("\n" + "=" * 80)
    print("MODEL RESPONSE:")
    print("=" * 80)
    print(response)

    # Try to parse JSON
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > 0:
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)

            print("\n" + "=" * 80)
            print("PARSED JSON (Pretty):")
            print("=" * 80)
            print(json.dumps(parsed, indent=2))

            # Extract key info
            if "RiskAssessment" in parsed:
                ra = parsed["RiskAssessment"]
                print("\n" + "=" * 80)
                print("KEY FINDINGS:")
                print("=" * 80)
                print(f"Country: {ra.get('country', 'N/A')}")
                print(f"Risk Level: {ra.get('risk_level', 'N/A')}")
                print(f"Spread Likelihood: {ra.get('spread_likelihood', 'N/A')}")

                forecast = ra.get('twelve_week_forecast', {})
                if forecast:
                    cases = forecast.get('weekly_cases_per_100k', [])
                    if cases:
                        print(f"\n12-Week Forecast Preview:")
                        print(f"  Week 1-3: {cases[:3]}")
                        print(f"  Peak: {max(cases):.1f} cases per 100k")
                        print(f"  Start Week: {forecast.get('forecast_start_week', 'N/A')}")
    except Exception as e:
        print(f"\nNote: Could not parse JSON ({e})")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
