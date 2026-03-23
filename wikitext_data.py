import numpy as np
from datasets import load_dataset

def load_wikitext2_bytelevel():
    """
    Load Wikitext-2 and prepare for byte-level next-token prediction.
    """
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tokenize(text):
        # Byte-level encoding (ASCII 0-255)
        return np.array([ord(c) for c in text if ord(c) < 256], dtype=np.uint8)

    train_data = tokenize(" ".join(dataset['train']['text']))
    val_data   = tokenize(" ".join(dataset['validation']['text']))

    return train_data, val_data

def get_ntp_batches(data, batch_size=32, seq_len=64):
    """
    Generator for Next-Token Prediction batches.
    """
    n = len(data)
    while True:
        idxs = np.random.randint(0, n - seq_len - 1, batch_size)
        x = np.stack([data[i:i+seq_len] for i in idxs])
        y = np.stack([data[i+1:i+seq_len+1] for i in idxs])

        # One-hot or Embedding?
        # For PhaseLLM, we need D-dim embeddings.
        # Let's use a simple learnable byte-embedding table inside the train loop.
        yield x, y

class ByteEmbedding:
    def __init__(self, D, seed=42):
        rng = np.random.default_rng(seed)
        self.table = rng.standard_normal((256, D)) * 0.1
        self.opt = None # To be initialized in train
