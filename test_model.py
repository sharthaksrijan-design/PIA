import numpy as np
import time
import os
from data import load_clinc150, load_glove, build_embeddings
from associative_data import generate_associative_kv_pairs, generate_corrupted_query
from model import PhaseEncoderV2, PhaseLLM
from train import train, associative_pretrain

LOG_FILE = "model_test_log.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def test_convergence_and_accuracy():
    log("=== TEST: Convergence and Accuracy (MLP Head + Sharpness) ===")
    splits, idx_to_intent, _ = load_clinc150()
    glove, D = load_glove()
    train_embs = build_embeddings(splits['train'][0], glove, D)
    val_embs   = build_embeddings(splits['validation'][0], glove, D)
    train_labels = np.array(splits['train'][1])
    val_labels   = np.array(splits['validation'][1])

    K = 320; N_INTENTS = len(idx_to_intent)
    enc = PhaseEncoderV2(D, K)

    log(f"Training. K={K}, D={D}, Intents={N_INTENTS}")
    enc, weights = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=300, lam_sharp=0.01)

    W_hid, b_hid = weights['W_hid'], weights['b_hid']
    W_cls, b_cls = weights['W_cls'], weights['b_cls']
    phi_val = enc.phi(val_embs.astype(np.float64))
    h = phi_val @ W_hid.T + b_hid[None, :]
    val_logits = np.maximum(0, h) @ W_cls.T + b_cls[None, :]
    acc = np.mean(np.argmax(val_logits, axis=1) == val_labels)
    log(f"300 Epoch Val Accuracy: {acc:.4f}")

def test_associative_retrieval():
    log("\n=== TEST: 100% Associative Retrieval Goal ===")
    N, D, K = 500, 100, 320
    keys, values = generate_associative_kv_pairs(N, D)
    enc = PhaseEncoderV2(D, K)

    log(f"Pre-training for exact retrieval of {N} KV pairs...")
    enc, W_assoc = associative_pretrain(enc, keys, values, epochs=1000, lam_sharp=0.05)

    # Eval retrieval
    phi_keys = enc.phi(keys)
    retrieval_logits = phi_keys @ W_assoc.T
    retrieval_preds = np.argmax(retrieval_logits, axis=1)
    acc = np.mean(retrieval_preds == values)
    log(f"Exact Retrieval Accuracy (Clean): {acc:.4%}")

    # Test Robustness (Noisy Retrieval)
    noisy_keys = generate_corrupted_query(keys, noise_level=0.05)
    phi_noisy = enc.phi(noisy_keys)
    noisy_preds = np.argmax(phi_noisy @ W_assoc.T, axis=1)
    noisy_acc = np.mean(noisy_preds == values)
    log(f"Noisy Retrieval Accuracy (5% noise): {noisy_acc:.4%}")

    if acc > 0.99:
        log("SUCCESS: Reached near-100% retrieval accuracy on clean keys.")
    else:
        log(f"FAIL: Retrieval accuracy {acc:.4f} is below 100% goal.")

if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    log("Starting Part 3 Verification...")
    test_associative_retrieval()
    test_convergence_and_accuracy()
    log("\nVerification Complete.")
