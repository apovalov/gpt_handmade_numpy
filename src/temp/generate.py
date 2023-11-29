from typing import Callable, List
import numpy as np

def generate(
    llm: Callable[[List[int]], List[float]],
    prompt: List[str],
    n_tokens: int,
    vocab: List[str],
    top_k: int = 50,
    top_p: float = 0.75,
    temperature: float = 1.1,
    random_state: int = 0,
) -> List[str]:
    np.random.seed(random_state)

    # Convert prompt to token IDs
    input_ids = [vocab.index(token) for token in prompt if token in vocab]
    generated_tokens = []

    for _ in range(n_tokens):
        # Obtain token probabilities using the callable object
        logits = llm(input_ids)

        # Convert logits to a NumPy array and apply temperature
        logits = np.array(logits)
        probs = np.exp(logits / temperature)
        probs /= probs.sum()

        # Apply top-k filtering: choose top k tokens
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_probs = probs[top_k_indices]

        # Apply top-p filtering on top-k tokens
        cumulative_probs = np.cumsum(top_k_probs)
        # Find the index where the cumulative probability exceeds top_p for the first time
        top_p_idx = np.searchsorted(cumulative_probs, top_p, side='right')
        top_p_indices = top_k_indices[:top_p_idx + 1]

        # Re-normalize the probabilities of the selected tokens after top-p filtering
        final_probs = probs[top_p_indices]
        final_probs /= final_probs.sum()

        # Sample the next token
        next_token_id = np.random.choice(top_p_indices, p=final_probs)
        generated_tokens.append(next_token_id)  # Store token index

        # Update input_ids for the next iteration
        input_ids.append(next_token_id)  # Correctly adding the index

    # Convert indices to tokens for the final output
    final_output = [vocab[token_id] for token_id in generated_tokens]
    return final_output

