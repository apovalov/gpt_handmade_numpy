import numpy as np

def linear(x, w, b):
    return x @ w + b

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# def attention(q, k, v):
#     d_k = q.shape[-1]
#     scores = q @ k.transpose(-2, -1) / np.sqrt(d_k)
#     weights = softmax(scores)
#     return weights @ v

def attention(q, k, v):
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)  # Используем k.T для транспонирования
    weights = softmax(scores)
    return weights @ v
def self_attention(x, c_attn, c_proj):
    qkv = linear(x, **c_attn)
    n_seq, n_embd_three = qkv.shape
    n_embd = n_embd_three // 3

    # Корректное разделение qkv на q, k, v без создания нового измерения
    q, k, v = np.split(qkv, 3, axis=-1)
    # Теперь q, k и v имеют размерность [n_seq, n_embd]

    x = attention(q, k, v)

    x = linear(x, **c_proj)
    return x






















