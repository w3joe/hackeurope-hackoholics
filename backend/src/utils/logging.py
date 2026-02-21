"""Structured JSON logging for LLM calls."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class LLMCallLog:
    """Structured log for LLM invocation."""

    module: str
    timestamp: str
    input_token_count: int | None = None
    output_token_count: int | None = None
    latency_ms: int | None = None
    retry_triggered: bool = False
    validation_errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "module": self.module,
                "timestamp": self.timestamp,
                "input_token_count": self.input_token_count,
                "output_token_count": self.output_token_count,
                "latency_ms": self.latency_ms,
                "retry_triggered": self.retry_triggered,
                "validation_errors": self.validation_errors,
            },
            default=str,
        )


def log_llm_call(
    module: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
    retry_triggered: bool = False,
    validation_errors: list[str] | None = None,
) -> None:
    """Log an LLM call to stdout as structured JSON."""
    log = LLMCallLog(
        module=module,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        latency_ms=latency_ms,
        retry_triggered=retry_triggered,
        validation_errors=validation_errors or [],
    )
    print(log.to_json(), flush=True)


class Timer:
    """Context manager for measuring latency."""

    def __init__(self):
        self.start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = int((time.perf_counter() - self.start) * 1000)
