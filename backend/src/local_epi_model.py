"""
Local epidemiological forecasting model (merged Qwen fine-tune).
Loads merged model and generates RiskAssessment JSON from epidemiological prompts.

Used by Module 1A when USE_LOCAL_EPI_MODEL=true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Lazy load to avoid import errors when torch/transformers not installed
_model_cache = None
_tokenizer_cache = None


def _parse_json_response(response_text: str) -> Optional[dict]:
    """Extract and parse JSON from model response."""
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx == 0:
            return None
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception:
        return None


def load_model(model_path: Optional[str] = None):
    """Load merged model and tokenizer. Uses cache on subsequent calls."""
    global _model_cache, _tokenizer_cache
    if _model_cache is not None:
        return _model_cache, _tokenizer_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_path is None:
        # Default: llm_finetune/qwen-epi-forecast-merged relative to project root
        backend_root = Path(__file__).resolve().parent.parent
        project_root = backend_root.parent
        model_path = project_root / "llm_finetune" / "qwen-epi-forecast-merged"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. "
            "Run: cd llm_finetune && python merge_adapter.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    _model_cache = model
    _tokenizer_cache = tokenizer
    return model, tokenizer


def generate_forecast(
    user_prompt: str,
    model_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Optional[dict]:
    """
    Generate RiskAssessment JSON from epidemiological prompt.
    Returns parsed dict with RiskAssessment key, or None on failure.
    """
    model, tokenizer = load_model(model_path)

    system_prompt = (
        "You are an expert epidemiological forecasting assistant. "
        "Analyze disease surveillance data and provide structured risk assessments "
        "with 12-week forecasts in JSON format."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    import torch

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(outputs[0][gen_len:], skip_special_tokens=True)
    return _parse_json_response(generated_text)
