"""
Inference script for fine-tuned Qwen2.5-3B-Instruct epidemiological forecasting model.

Usage:
    python inference.py --model_path ./qwen-epi-forecast --interactive
    python inference.py --model_path ./qwen-epi-forecast --test_file test_input.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_and_tokenizer(model_path, base_model_name="Qwen/Qwen2.5-3B-Instruct"):
    """Load fine-tuned model and tokenizer."""
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    print(f"Loading base model from {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def generate_forecast(model, tokenizer, user_prompt, max_new_tokens=512, temperature=0.3):
    """Generate forecast from user prompt."""
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

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the input)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return generated_text


def parse_json_response(response_text):
    """Extract and parse JSON from model response."""
    try:
        # Find JSON object in response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return None


def create_example_prompt():
    """Create an example prompt for demonstration."""
    return """Analyze the following epidemiological data and provide a risk assessment:

**Pathogen:** Influenza
**Country:** Austria

**Recent Observation (Last 6 Weeks):**
Weeks: 2024-W44, 2024-W45, 2024-W46, 2024-W47, 2024-W48, 2024-W49
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


def interactive_mode(model, tokenizer):
    """Run interactive inference loop."""
    print("\n" + "="*80)
    print("Interactive Inference Mode")
    print("="*80)
    print("Commands:")
    print("  'example' - Load example prompt")
    print("  'quit' - Exit")
    print("  Or paste your own prompt\n")

    while True:
        print("\n" + "-"*80)
        user_input = input("Enter prompt (or command): ").strip()

        if user_input.lower() == 'quit':
            print("Exiting...")
            break

        if user_input.lower() == 'example':
            user_input = create_example_prompt()
            print("\nUsing example prompt:")
            print(user_input[:200] + "...\n")

        if not user_input:
            continue

        print("\nGenerating forecast...")
        response = generate_forecast(model, tokenizer, user_input)

        print("\n" + "="*80)
        print("MODEL RESPONSE:")
        print("="*80)
        print(response)

        # Try to parse and pretty-print JSON
        parsed = parse_json_response(response)
        if parsed:
            print("\n" + "="*80)
            print("PARSED JSON:")
            print("="*80)
            print(json.dumps(parsed, indent=2))


def batch_inference(model, tokenizer, test_file, output_file):
    """Run batch inference on test file."""
    print(f"\nLoading test data from {test_file}...")
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]

    print(f"Processing {len(test_data)} examples...")
    results = []

    for i, example in enumerate(test_data):
        print(f"\nProcessing {i+1}/{len(test_data)}...")

        # Extract user prompt from messages
        user_prompt = None
        for msg in example.get("messages", []):
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break

        if not user_prompt:
            print(f"Warning: No user prompt found in example {i}")
            continue

        # Generate
        response = generate_forecast(model, tokenizer, user_prompt)
        parsed = parse_json_response(response)

        result = {
            "index": i,
            "user_prompt": user_prompt[:100] + "...",  # truncate for readability
            "model_response": response,
            "parsed_json": parsed,
        }

        # Include ground truth if available
        for msg in example.get("messages", []):
            if msg["role"] == "assistant":
                try:
                    result["ground_truth"] = json.loads(msg["content"])
                except:
                    pass

        results.append(result)

    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Batch inference complete! Results saved to {output_file}")


def main(args):
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.test_file:
        output_file = args.output_file or "inference_results.json"
        batch_inference(model, tokenizer, args.test_file, output_file)
    else:
        # Single example
        prompt = create_example_prompt()
        print("\nUsing example prompt...")
        response = generate_forecast(model, tokenizer, prompt)

        print("\n" + "="*80)
        print("MODEL RESPONSE:")
        print("="*80)
        print(response)

        parsed = parse_json_response(response)
        if parsed:
            print("\n" + "="*80)
            print("PARSED JSON:")
            print("="*80)
            print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="JSONL file with test examples"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for batch results (default: inference_results.json)"
    )

    args = parser.parse_args()
    main(args)
