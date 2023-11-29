import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    # We subtract max(x) for numerical stability
    # https://jaykmody.com/blog/stable-softmax/
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(q, k, v, mask):
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k) + mask  # Добавление маски к scores
    weights = softmax(scores)
    return weights @ v

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    n_seq, n_embd = x.shape

    # split into qkv
    qkv = np.split(x, 3, axis=-1)

    # split into heads and reshape
    q, k, v = [np.concatenate(np.split(head, n_head, axis=-1), axis=0) for head in qkv]

    causal_mask = (1 - np.tri(n_seq)) * -1e10

    # perform attention over each head
    out_heads = [attention(q[i*n_seq:(i+1)*n_seq], k[i*n_seq:(i+1)*n_seq], v[i*n_seq:(i+1)*n_seq], causal_mask) for i in range(n_head)]

    # merge heads
    x = np.concatenate(out_heads, axis=-1)

    # out projection
    x = linear(x, **c_proj)
    return x





def causal_self_attention(x, c_attn, c_proj):
    x = linear(x, **c_attn)
    q, k, v = np.split(x, 3, axis=-1)

    causal_mask = (1 - np.tri(x.shape[0])) * -1e10

    x = attention(q, k, v, causal_mask)
    x = linear(x, **c_proj)
    return x