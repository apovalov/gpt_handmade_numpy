from typing import Callable, List
import numpy as np
import torch

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
    vocab_to_id = {word: i for i, word in enumerate(vocab)}
    input_ids = [vocab_to_id[word] for word in prompt]
    generated_tokens = []

    for _ in range(n_tokens):
        # Obtain token probabilities
        logits = llm(input_ids)

        # Convert logits to a NumPy array
        logits = np.array(logits)

        # Apply temperature
        logits /= temperature

        # Apply top-k
        indices_to_remove = logits < np.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

        # Apply top-p
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

        # Sample the next token
        next_token = np.random.choice(len(logits), p=F.softmax(logits, dim=-1).numpy())
        generated_tokens.append(vocab[next_token])
        input_ids.append(next_token)

    return generated_tokens




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
        mask = cumulative_probs <= top_p
        top_p_indices = top_k_indices[mask]

        # Check if any indices are left after top-p filtering
        if top_p_indices.size > 0:
            # Re-normalize the probabilities of the selected tokens after top-p filtering
            final_probs = top_k_probs[mask]
            final_probs /= final_probs.sum()
            indices_to_sample = top_p_indices
        else:
            # If no indices are left after top-p filtering, fallback to the original distribution
            indices_to_sample = np.arange(len(probs))
            final_probs = probs

        # Sample the next token
        next_token_id = np.random.choice(indices_to_sample, p=final_probs)
        generated_tokens.append(next_token_id)  # Store token index

        # Update input_ids for the next iteration
        input_ids.append(next_token_id)  # Correctly adding the index

    # Convert indices to tokens for the final output
    final_output = [vocab[token_id] for token_id in generated_tokens]
    return final_output
