"""API-equivalent Claude pricing, used to report what the labelling calls cost.

Extend RATES when a new model ships; an unknown model is reported as unpriced."""

# $/1M input, $/1M output
RATES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-mythos-5-1": (10.0, 50.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_READ_MULTIPLIER_BY_MODEL = {
    "claude-opus-5-5": 0.05,
    "claude-fable-5-1": 0.025,
    "claude-mythos-5-1": 0.025,
}
# Prompts (input + cache) above the threshold bill on a second rate card.
LONG_PROMPT_RATES = {
    "claude-haiku-5-5": (100_000, (0.50, 2.50)),
}
# Harness-generated records that no API call billed.
UNBILLED_MODELS = {"<synthetic>", "unknown"}
CACHE_WRITE_MULTIPLIER = 2.0  # 1h TTL, which is what this harness uses


def base_model(model):
    """Strip a context-window variant suffix, e.g. claude-opus-5[1m]."""
    return model.split("[")[0] if model else ""


def is_priced(model):
    return base_model(model) in RATES


def cost_usd(model, cache_read=0, cache_write=0, input_tokens=0, output=0):
    model = base_model(model)
    rate = RATES.get(model)
    if rate is None:
        return 0.0
    input_rate, output_rate = rate
    long_card = LONG_PROMPT_RATES.get(model)
    if long_card and cache_read + cache_write + input_tokens > long_card[0]:
        input_rate, output_rate = long_card[1]
    read_multiplier = CACHE_READ_MULTIPLIER_BY_MODEL.get(model, CACHE_READ_MULTIPLIER)
    return (cache_read / 1e6 * input_rate * read_multiplier
            + cache_write / 1e6 * input_rate * CACHE_WRITE_MULTIPLIER
            + input_tokens / 1e6 * input_rate
            + output / 1e6 * output_rate)
