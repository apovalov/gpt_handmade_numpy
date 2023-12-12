from typing import Callable
from typing import List

import numpy as np


def generate(
    llm: Callable[[List[int]], List[float]],
    prompt: List[str],
    n_tokens: int,
    vocab: List[str],
    top_k: int = 1,
    top_p: float = 0.75,
    temperature: float = 1.0,
    random_state: int = 0,
) -> List[str]:
    """Generate a sequence of tokens from a prompt using a language model."""
    np.random.seed(random_state)

    # Convert prompt to token IDs
    input_ids = [vocab.index(token) for token in prompt]
    generated_tokens = []

    # Auto-regressive decode loop
    for _ in range(n_tokens):
        # Obtain token probabilities using the callable object
        # and apply temperature scaling to the logits
        logits = np.array(llm(input_ids))
        logits = logits / temperature
        probs = np.exp(logits) / np.exp(logits).sum()

        # Apply top-k filtering
        indices = np.argsort(probs)[-top_k:]
        probs = probs[indices] / probs[indices].sum()

        # Sort token IDs by probability
        order = np.argsort(probs)[::-1]
        indices = indices[order]
        probs = probs[order]

        # Apply top-p filtering
        cum_probs = np.cumsum(probs)
        mask = cum_probs <= np.min(cum_probs[cum_probs >= top_p])
        probs = probs[mask] / probs[mask].sum()
        indices = indices[mask]

        # Sample next token ID from the filtered distribution
        next_id = np.random.choice(indices, p=probs)
        generated_tokens.append(vocab[next_id])
        input_ids.append(int(next_id))

    return generated_tokens
