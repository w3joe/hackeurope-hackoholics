"""
Evaluation script for fine-tuned epidemiological forecasting model.
Computes metrics: RMSE, MAE, MAPE, accuracy for risk level classification.

Usage:
    python evaluate.py --model_path ./qwen-epi-forecast --test_file training_data/test.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


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
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer


def generate_forecast(model, tokenizer, user_prompt, max_new_tokens=512, temperature=0.1):
    """Generate forecast from user prompt with low temperature for evaluation."""
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

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text


def parse_json_response(response_text):
    """Extract and parse JSON from model response."""
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            return None

        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        return None


def evaluate_forecasts(predictions, ground_truths):
    """Compute regression metrics for 12-week forecasts."""
    pred_values = []
    true_values = []

    for pred, truth in zip(predictions, ground_truths):
        if pred is None or truth is None:
            continue

        try:
            pred_forecast = pred.get("RiskAssessment", {}).get("twelve_week_forecast", {}).get("weekly_cases_per_100k", [])
            true_forecast = truth.get("RiskAssessment", {}).get("twelve_week_forecast", {}).get("weekly_cases_per_100k", [])

            if len(pred_forecast) == 12 and len(true_forecast) == 12:
                pred_values.extend(pred_forecast)
                true_values.extend(true_forecast)
        except:
            continue

    if len(pred_values) == 0:
        return None

    pred_values = np.array(pred_values)
    true_values = np.array(true_values)

    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    mae = mean_absolute_error(true_values, pred_values)

    # MAPE (Mean Absolute Percentage Error) - avoid division by zero
    mask = true_values != 0
    mape = np.mean(np.abs((true_values[mask] - pred_values[mask]) / true_values[mask])) * 100 if mask.any() else None

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape) if mape is not None else None,
        "n_forecasts": len(pred_values) // 12,
        "n_weeks": len(pred_values)
    }


def evaluate_risk_classification(predictions, ground_truths):
    """Compute classification metrics for risk level."""
    pred_risks = []
    true_risks = []

    for pred, truth in zip(predictions, ground_truths):
        if pred is None or truth is None:
            continue

        try:
            pred_risk = pred.get("RiskAssessment", {}).get("risk_level")
            true_risk = truth.get("RiskAssessment", {}).get("risk_level")

            if pred_risk and true_risk:
                pred_risks.append(pred_risk)
                true_risks.append(true_risk)
        except:
            continue

    if len(pred_risks) == 0:
        return None

    accuracy = accuracy_score(true_risks, pred_risks)
    report = classification_report(true_risks, pred_risks, output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "n_samples": len(pred_risks)
    }


def evaluate_spread_likelihood(predictions, ground_truths):
    """Compute regression metrics for spread likelihood."""
    pred_likelihoods = []
    true_likelihoods = []

    for pred, truth in zip(predictions, ground_truths):
        if pred is None or truth is None:
            continue

        try:
            pred_likelihood = pred.get("RiskAssessment", {}).get("spread_likelihood")
            true_likelihood = truth.get("RiskAssessment", {}).get("spread_likelihood")

            if pred_likelihood is not None and true_likelihood is not None:
                pred_likelihoods.append(pred_likelihood)
                true_likelihoods.append(true_likelihood)
        except:
            continue

    if len(pred_likelihoods) == 0:
        return None

    pred_likelihoods = np.array(pred_likelihoods)
    true_likelihoods = np.array(true_likelihoods)

    rmse = np.sqrt(mean_squared_error(true_likelihoods, pred_likelihoods))
    mae = mean_absolute_error(true_likelihoods, pred_likelihoods)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "n_samples": len(pred_likelihoods)
    }


def main(args):
    print("="*80)
    print("Epidemiological Forecasting Model Evaluation")
    print("="*80)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file) as f:
        test_data = [json.loads(line) for line in f]
    print(f"Loaded {len(test_data)} test examples")

    # Run inference
    print("\nRunning inference on test set...")
    predictions = []
    ground_truths = []
    failed_parses = 0

    for example in tqdm(test_data):
        # Extract user prompt
        user_prompt = None
        ground_truth = None

        for msg in example.get("messages", []):
            if msg["role"] == "user":
                user_prompt = msg["content"]
            elif msg["role"] == "assistant":
                try:
                    ground_truth = json.loads(msg["content"])
                except:
                    pass

        if not user_prompt or not ground_truth:
            continue

        # Generate prediction
        response = generate_forecast(model, tokenizer, user_prompt)
        parsed = parse_json_response(response)

        if parsed is None:
            failed_parses += 1

        predictions.append(parsed)
        ground_truths.append(ground_truth)

    print(f"\nInference complete!")
    print(f"  Successfully parsed: {len(predictions) - failed_parses}/{len(predictions)}")
    print(f"  Failed to parse: {failed_parses}/{len(predictions)}")

    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # 1. Forecast accuracy (12-week predictions)
    print("\n[1] 12-Week Forecast Accuracy:")
    forecast_metrics = evaluate_forecasts(predictions, ground_truths)
    if forecast_metrics:
        print(f"  RMSE: {forecast_metrics['rmse']:.2f} cases per 100k")
        print(f"  MAE:  {forecast_metrics['mae']:.2f} cases per 100k")
        if forecast_metrics['mape'] is not None:
            print(f"  MAPE: {forecast_metrics['mape']:.2f}%")
        print(f"  Evaluated on {forecast_metrics['n_forecasts']} forecasts ({forecast_metrics['n_weeks']} weeks total)")
    else:
        print("  No valid forecasts to evaluate")

    # 2. Risk level classification
    print("\n[2] Risk Level Classification:")
    risk_metrics = evaluate_risk_classification(predictions, ground_truths)
    if risk_metrics:
        print(f"  Accuracy: {risk_metrics['accuracy']:.3f}")
        print(f"  Samples:  {risk_metrics['n_samples']}")
        print("\n  Per-class metrics:")
        for risk_level, metrics in risk_metrics['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"    {risk_level:12s}: Precision={metrics.get('precision', 0):.3f}, Recall={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}")
    else:
        print("  No valid risk classifications to evaluate")

    # 3. Spread likelihood
    print("\n[3] Spread Likelihood Prediction:")
    likelihood_metrics = evaluate_spread_likelihood(predictions, ground_truths)
    if likelihood_metrics:
        print(f"  RMSE: {likelihood_metrics['rmse']:.3f}")
        print(f"  MAE:  {likelihood_metrics['mae']:.3f}")
        print(f"  Samples: {likelihood_metrics['n_samples']}")
    else:
        print("  No valid spread likelihoods to evaluate")

    # Save detailed results
    if args.save_results:
        results = {
            "forecast_metrics": forecast_metrics,
            "risk_classification_metrics": risk_metrics,
            "spread_likelihood_metrics": likelihood_metrics,
            "predictions": [
                {
                    "prediction": pred,
                    "ground_truth": truth
                }
                for pred, truth in zip(predictions, ground_truths)
            ]
        }

        output_path = Path(args.model_path) / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="JSONL file with test examples"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save detailed results to JSON"
    )

    args = parser.parse_args()
    main(args)
