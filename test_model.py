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

    log(f"Training for 100 epochs to verify convergence. K={K}, D={D}, Intents={N_INTENTS}")

    # Run training for 100 epochs to check initial convergence
    # Capture accuracy trends
    prev_acc = 0.0
    for ep in range(50, 151, 50):
        t_start = time.time()
        enc, W_cls = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=50)
        phi_val = enc.phi(val_embs.astype(np.float64)).astype(np.float32)
        val_logits = phi_val @ W_cls.T.astype(np.float32)
        val_preds = np.argmax(val_logits, axis=1)
        current_acc = np.mean(val_preds == val_labels)

        log(f"Epoch {ep}: Accuracy={current_acc:.4f}, Time={time.time()-t_start:.1f}s")

        if current_acc < prev_acc:
            log(f"WARNING: ACCURACY FALL DETECTED at epoch {ep}! Prev={prev_acc:.4f}, Current={current_acc:.4f}")
        prev_acc = current_acc

    log(f"Final Validation Accuracy: {current_acc:.4f}")
    if current_acc < 0.4:
         log("ALERT: Convergence is slower than expected (<40% at 150 epochs).")

def test_parameter_stability():
    log("\n=== TEST: Parameter Stability ===")
    D = 100; K = 320
    enc = PhaseEncoderV2(D, K)

    # Check for NaNs or Infs in weights
    if np.any(np.isnan(enc.W)):
        log("ERROR: NaN detected in initial weights W!")
    if np.any(np.isinf(enc.W)):
        log("ERROR: Inf detected in initial weights W!")

    # Check weight norms
    w_norm = np.linalg.norm(enc.W)
    log(f"Initial Weight W Norm: {w_norm:.4f}")

    # Verify omega ranges
    log(f"Omega Range: [{np.min(enc.omega):.4f}, {np.max(enc.omega):.4f}]")
    if np.any(enc.omega < 0.25) or np.any(enc.omega > 4.0):
        log("WARNING: Omega parameters are outside the suggested [0.25, 4.0] stable range.")

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
        if out.shape != (B, L, K):
            log(f"ERROR: Incorrect output shape! Expected {(B, L, K)}, got {out.shape}")
        if np.any(np.isnan(out)):
            log("ERROR: NaN detected in forward pass output!")
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
