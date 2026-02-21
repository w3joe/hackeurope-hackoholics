"""
Prepare training data for Qwen2.5-3B-Instruct fine-tuning.
Converts llm_risk_training.csv into instruction-tuning format with structured JSON output.

Usage: python prepare_training_data.py
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def calculate_risk_level(target_values, threshold_pct):
    """Calculate overall risk level based on forecasted values."""
    if not target_values or all(v is None for v in target_values):
        return "LOW"

    valid_values = [v for v in target_values if v is not None]
    max_val = max(valid_values)
    avg_val = sum(valid_values) / len(valid_values)

    # Risk classification based on threshold
    if max_val > threshold_pct * 2:
        return "CRITICAL"
    elif max_val > threshold_pct * 1.5:
        return "HIGH"
    elif max_val > threshold_pct:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_spread_likelihood(tda_status, hw_surprised, convergence, pca_position):
    """Calculate spread likelihood score [0, 1]."""
    score = 0.5  # baseline

    # TDA status contribution
    if tda_status == "anomaly":
        score += 0.3
    elif tda_status == "elevated":
        score += 0.15

    # Holt-Winters surprise
    if hw_surprised:
        score += 0.1

    # Convergence signal
    score += convergence * 0.05

    # PCA position
    if pca_position == "far":
        score += 0.1
    elif pca_position == "elevated":
        score += 0.05

    return min(1.0, max(0.0, score))


def create_instruction_prompt(row):
    """Create user instruction from row data."""
    # Parse pipe-separated fields
    trailing_weeks = row['trailing_weeks'].split('|')
    trailing_values = [float(x) for x in row['trailing_values'].split('|')]
    forecast_hw_values = [float(x) for x in row['forecast_hw_values'].split('|')]
    high_risk_weeks = row['forecast_hw_high_risk_weeks'].split('|') if row['forecast_hw_high_risk_weeks'] else []
    seasonal_iso_weeks = [int(x) for x in row['tda_seasonal_iso_weeks'].split('|')] if row['tda_seasonal_iso_weeks'] else []

    prompt = f"""Analyze the following epidemiological data and provide a risk assessment:

**Pathogen:** {row['pathogen']}
**Country:** {row['country']}

**Recent Observation (Last 6 Weeks):**
Weeks: {', '.join(trailing_weeks)}
Cases per 100k: {', '.join(str(v) for v in trailing_values)}

**Statistical Forecast Signals:**
- Holt-Winters forecast (next 12w): {', '.join(f'{v:.1f}' for v in forecast_hw_values[:6])}... (truncated)
- Risk threshold: {row['forecast_hw_threshold_pct']}% per 100k
- High-risk weeks predicted: {len(high_risk_weeks)} weeks

**Anomaly Detection (TDA):**
- Status: {row['tda_status']}
- Trend: {row['tda_trend']}
- Number of anomalies: {row['tda_n_anomalies']}
- Lead time: {row['tda_lead_weeks']} weeks
- H0 z-score: {row['tda_H0_z']}, H1 z-score: {row['tda_H1_z']}

**Seasonality:**
- Historical peak weeks (ISO): {', '.join(f'W{w}' for w in seasonal_iso_weeks[:5])}{'...' if len(seasonal_iso_weeks) > 5 else ''}

**Pattern Analysis:**
- Last observation week: {row['hw_last_obs_week']}
- Residual z-score: {row['hw_residual_z']}
- Forecast surprised by actual: {row['hw_surprised']}
- PCA distance z-score: {row['pca_dist_z']}
- PCA position: {row['pca_position']}
- Signal convergence: {row['convergence']}/3

Provide a comprehensive 12-week forecast with risk assessment."""

    return prompt


def create_assistant_response(row):
    """Create structured JSON response from row data."""
    # Parse target values
    target_values = []
    if 'target_values' in row and pd.notna(row['target_values']):
        target_values = [float(x) if x else 0.0 for x in row['target_values'].split('|')]
    else:
        # Fallback to HW forecast if no targets
        target_values = [float(x) for x in row['forecast_hw_values'].split('|')]

    # Ensure exactly 12 values
    target_values = (target_values + [0.0] * 12)[:12]

    # Calculate risk level
    risk_level = calculate_risk_level(target_values, row['forecast_hw_threshold_pct'])

    # Calculate spread likelihood
    spread_likelihood = calculate_spread_likelihood(
        row['tda_status'],
        row['hw_surprised'],
        row['convergence'],
        row['pca_position']
    )

    # Generate reasoning
    trend_desc = row['tda_trend'].replace('_', ' ').replace(',', ' and ')
    reasoning = f"Based on {row['tda_status']} TDA status with {trend_desc} trends, "

    if row['convergence'] >= 2:
        reasoning += "strong signal convergence detected. "
    elif row['convergence'] == 1:
        reasoning += "partial signal convergence. "
    else:
        reasoning += "low signal convergence. "

    if row['hw_surprised']:
        reasoning += "Recent observations deviate from forecast expectations. "

    if row['pca_position'] == 'far':
        reasoning += "Trajectory pattern significantly differs from historical norms. "

    reasoning += f"Forecasting {max(target_values):.1f} peak cases per 100k over the next 12 weeks."

    # Determine forecast start week
    forecast_start_week = row.get('cutoff_week', row['hw_last_obs_week'])
    if 'forecast_hw_weeks' in row:
        forecast_weeks = row['forecast_hw_weeks'].split('|')
        if forecast_weeks:
            forecast_start_week = forecast_weeks[0]

    response = {
        "RiskAssessment": {
            "country": row['country'],
            "risk_level": risk_level,
            "spread_likelihood": round(spread_likelihood, 3),
            "reasoning": reasoning,
            "recommended_disease_focus": [row['pathogen']],
            "twelve_week_forecast": {
                "weekly_cases_per_100k": [round(v, 2) for v in target_values],
                "forecast_start_week": forecast_start_week
            }
        }
    }

    return json.dumps(response)


def prepare_dataset(csv_path, output_dir, test_split=0.15, val_split=0.05):
    """Convert CSV to instruction-tuning JSONL format."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter to records with target values (for training)
    df_with_targets = df[df['target_values'].notna()].copy()
    print(f"Total records with targets: {len(df_with_targets)}")

    # Create instruction tuning examples
    examples = []
    for idx, row in df_with_targets.iterrows():
        try:
            user_prompt = create_instruction_prompt(row)
            assistant_response = create_assistant_response(row)

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert epidemiological forecasting assistant. Analyze disease surveillance data and provide structured risk assessments with 12-week forecasts in JSON format."
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
            }
            examples.append(example)
        except Exception as e:
            print(f"Skipping row {idx}: {e}")
            continue

    print(f"Created {len(examples)} training examples")

    # Split into train/val/test
    train_examples, test_examples = train_test_split(
        examples, test_size=test_split, random_state=42
    )
    train_examples, val_examples = train_test_split(
        train_examples, test_size=val_split/(1-test_split), random_state=42
    )

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    # Save to JSONL
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for split_name, split_data in [
        ('train', train_examples),
        ('val', val_examples),
        ('test', test_examples)
    ]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, 'w') as f:
            for example in split_data:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {split_name} to {output_path}")

    # Save a sample for inspection
    sample_path = output_dir / "sample.json"
    with open(sample_path, 'w') as f:
        json.dump(examples[0], f, indent=2)
    print(f"\nSample example saved to {sample_path}")

    return len(train_examples), len(val_examples), len(test_examples)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "llm_risk_training.csv"
    output_dir = Path(__file__).parent / "training_data"

    prepare_dataset(csv_path, output_dir)
    print("\nData preparation complete!")
