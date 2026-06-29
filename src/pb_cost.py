"""Token usage accumulator and cost reporter for Anthropic API calls."""

PRICING = {
    "claude-opus-4-8": {
        "input":        5.00 / 1_000_000,
        "output":      25.00 / 1_000_000,
        "cache_write": 10.00 / 1_000_000,
        "cache_read":   0.50 / 1_000_000,
    },
    "claude-sonnet-4-6": {
        "input":        3.00 / 1_000_000,
        "output":      15.00 / 1_000_000,
        "cache_write":  6.00 / 1_000_000,
        "cache_read":   0.30 / 1_000_000,
    },
}

_ZERO_COUNTS = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}


class CostTracker:
    def __init__(self):
        self._counts: dict[str, dict[str, int]] = {}

    def record(self, model: str, usage) -> None:
        bucket = self._counts.setdefault(model, dict(_ZERO_COUNTS))
        bucket["input"]       += getattr(usage, "input_tokens", 0) or 0
        bucket["output"]      += getattr(usage, "output_tokens", 0) or 0
        bucket["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        bucket["cache_read"]  += getattr(usage, "cache_read_input_tokens", 0) or 0

    def totals_for(self, model: str) -> dict[str, int]:
        return dict(self._counts.get(model, _ZERO_COUNTS))

    def total_cost(self) -> float:
        total = 0.0
        for model, counts in self._counts.items():
            rates = PRICING.get(model, {})
            for token_type, count in counts.items():
                total += count * rates.get(token_type, 0.0)
        return total

    def print_report(self) -> None:
        print("\n" + "=" * 52)
        print("  TOKEN USAGE & ESTIMATED COST")
        print("=" * 52)
        for model, counts in self._counts.items():
            rates = PRICING.get(model, {})
            model_cost = sum(counts[t] * rates.get(t, 0.0) for t in counts)
            print(f"\n  {model}")
            print(f"    Input tokens:       {counts['input']:>10,}")
            print(f"    Output tokens:      {counts['output']:>10,}")
            print(f"    Cache write tokens: {counts['cache_write']:>10,}")
            print(f"    Cache read tokens:  {counts['cache_read']:>10,}")
            if model in PRICING:
                print(f"    Subtotal:           ${model_cost:>10.4f}")
            else:
                print(f"    Subtotal:           (unknown model, no pricing data)")
        print(f"\n  TOTAL COST:           ${self.total_cost():>10.4f}")
        print("=" * 52 + "\n")
