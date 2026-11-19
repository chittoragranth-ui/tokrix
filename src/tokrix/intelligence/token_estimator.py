"""Approximate token-count estimation using character length."""


def estimate_tokens(text: str) -> int:
    """Estimate tokens at approximately one token per four characters.

    This deterministic character-based approximation is not actual output from
    a tokenizer. The result is rounded up, with non-empty input estimated as
    at least one token.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty or whitespace-only")

    return max(1, (len(text) + 3) // 4)