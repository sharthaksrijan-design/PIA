import numpy as np
import time
import os
from data import load_clinc150, load_glove, build_embeddings
from model import PhaseEncoderV2, PhaseLLM
from train import train

LOG_FILE = "model_test_log.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def test_convergence_and_accuracy():
    log("=== TEST: Convergence and Accuracy ===")
    splits, idx_to_intent, _ = load_clinc150()
    glove, D = load_glove()
    train_embs = build_embeddings(splits['train'][0], glove, D)
    val_embs   = build_embeddings(splits['validation'][0], glove, D)
    train_labels = np.array(splits['train'][1])
    val_labels   = np.array(splits['validation'][1])

    K = 320; N_INTENTS = len(idx_to_intent)
    enc = PhaseEncoderV2(D, K, lr=2e-3)

    log(f"Training to verify convergence. K={K}, D={D}, Intents={N_INTENTS}")

    prev_acc = 0.0
    for ep in range(50, 251, 50):
        t_start = time.time()
        enc, weights = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=50)

        W_hid, b_hid = weights['W_hid'], weights['b_hid']
        W_cls, b_cls = weights['W_cls'], weights['b_cls']

        phi_val = enc.phi(val_embs.astype(np.float64))
        h = phi_val @ W_hid.T + b_hid[None, :]
        val_logits = np.maximum(0, h) @ W_cls.T + b_cls[None, :]
        current_acc = np.mean(np.argmax(val_logits, axis=1) == val_labels)

        log(f"Epoch {ep}: Accuracy={current_acc:.4f}, Time={time.time()-t_start:.1f}s")

        if current_acc < prev_acc:
            log(f"WARNING: ACCURACY FALL DETECTED at epoch {ep}! Prev={prev_acc:.4f}, Current={current_acc:.4f}")
        prev_acc = current_acc

    log(f"Final Validation Accuracy: {current_acc:.4f}")

def test_parameter_stability():
    log("\n=== TEST: Parameter Stability ===")
    D = 100; K = 320
    enc = PhaseEncoderV2(D, K)

    if np.any(np.isnan(enc.W)):
        log("ERROR: NaN detected in initial weights W!")
    if np.any(np.isinf(enc.W)):
        log("ERROR: Inf detected in initial weights W!")

    w_norm = np.linalg.norm(enc.W)
    log(f"Initial Weight W Norm: {w_norm:.4f}")
    log(f"Omega Range: [{np.min(enc.omega):.4f}, {np.max(enc.omega):.4f}]")

def test_hierarchical_forward():
    log("\n=== TEST: Hierarchical PhaseLLM Forward Pass ===")
    B, L, D = 2, 10, 100
    K = 320
    llm = PhaseLLM(D, K, n_layers=3)
    dummy_input = np.random.randn(B, L, D)

    try:
        t_start = time.time()
        out = llm.forward(dummy_input, causal=True)
        log(f"Forward Pass Success. Output shape: {out.shape}, Time={time.time()-t_start:.4f}s")
    except Exception as e:
        log(f"CRITICAL ERROR in forward pass: {str(e)}")

if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    log("Starting Model Extensive Tests...")
    test_parameter_stability()
    test_hierarchical_forward()
    test_convergence_and_accuracy()
    log("\nTests Complete.")
