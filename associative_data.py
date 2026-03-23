import numpy as np

def generate_associative_kv_pairs(N=1000, D=100, seed=42):
    """
    Generate synthetic Key-Value pairs for testing 100% retrieval precision.
    Keys are random embeddings, values are random phase targets (conceptually).
    """
    rng = np.random.default_rng(seed)
    # Keys (Embeddings)
    keys = rng.standard_normal((N, D))
    keys /= (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12)

    # Values (Indices or unique identifiers)
    values = np.arange(N)

    return keys, values

def generate_corrupted_query(keys, noise_level=0.1, seed=43):
    """
    Add noise to keys to simulate corrupted queries for retrieval.
    """
    rng = np.random.default_rng(seed)
    noisy_keys = keys + noise_level * rng.standard_normal(keys.shape)
    noisy_keys /= (np.linalg.norm(noisy_keys, axis=1, keepdims=True) + 1e-12)
    return noisy_keys
