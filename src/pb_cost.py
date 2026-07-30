"""Token usage accumulator and cost reporter for Anthropic API calls."""

PRICING = {
    "claude-opus-4-8": {
        "input":        5.00 / 1_000_000,
        "output":      25.00 / 1_000_000,
        "cache_write": 10.00 / 1_000_000,
        "cache_read":   0.50 / 1_000_000,
    },
    # Standard (post-introductory) rate; Anthropic offers $2/$10 intro pricing
    # through 2026-08-31, but the tracker uses the durable post-intro rate.
    "claude-sonnet-5": {
        "input":        3.00 / 1_000_000,
        "output":      15.00 / 1_000_000,
        "cache_write":  6.00 / 1_000_000,
        "cache_read":   0.30 / 1_000_000,
    },
    # o3-mini pricing as of 2026-07-29 (used only by the coverage-judge checkpoint).
    # OpenAI pricing changes independently of this codebase's release cadence — verify
    # against platform.openai.com/pricing before relying on cost totals for budgeting.
    "o3-mini": {
        "input":        1.10 / 1_000_000,
        "output":       4.40 / 1_000_000,
        "cache_write":  0.00,
        "cache_read":   0.55 / 1_000_000,
    },
    # text-embedding-3-small pricing as of 2026-07-29 (used only by the Phase 4
    # embedding-based weight rescale). Embeddings have no output/cache tokens.
    "text-embedding-3-small": {
        "input":        0.02 / 1_000_000,
        "output":       0.00,
        "cache_write":  0.00,
        "cache_read":   0.00,
    },
}

_ZERO_COUNTS = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}

_PROVIDER_OF = {
    "claude-opus-4-8": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "o3-mini": "OpenAI",
    "text-embedding-3-small": "OpenAI",
}


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
        provider_totals: dict[str, float] = {}
        for model, counts in self._counts.items():
            rates = PRICING.get(model, {})
            provider = _PROVIDER_OF.get(model, "Unknown")
            provider_totals[provider] = provider_totals.get(provider, 0.0) + sum(
                counts[token_type] * rates.get(token_type, 0.0) for token_type in counts
            )
        print()
        for provider, cost in provider_totals.items():
            print(f"  {provider} subtotal:       ${cost:>10.4f}")
        print(f"\n  TOTAL COST:           ${self.total_cost():>10.4f}")
        print("=" * 52 + "\n")
