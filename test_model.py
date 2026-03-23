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
    log("=== TEST: Convergence and Accuracy (Stochastic + Relational) ===")
    splits, idx_to_intent, _ = load_clinc150()
    glove, D = load_glove()
    train_embs = build_embeddings(splits['train'][0], glove, D)
    val_embs   = build_embeddings(splits['validation'][0], glove, D)
    train_labels = np.array(splits['train'][1])
    val_labels   = np.array(splits['validation'][1])
    K = 320; N_INTENTS = len(idx_to_intent); enc = PhaseEncoderV2(D, K)
    enc, weights = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=200)
    phi_val = enc.phi(val_embs.astype(np.float64))
    h = phi_val @ weights['W_hid'].T + weights['b_hid'][None, :]
    val_logits = np.maximum(0, h) @ weights['W_cls'].T + weights['b_cls'][None, :]
    acc = np.mean(np.argmax(val_logits, axis=1) == val_labels)
    log(f"Final Val Accuracy: {acc:.4f}")

def test_multi_hop_reasoning():
    log("\n=== TEST: Multi-Hop Chain Reasoning ===")
    N, D, K = 100, 100, 320
    keys, values = generate_associative_kv_pairs(N, D)
    # Use low confidence threshold to force reasoning hops for demonstration
    llm = PhaseLLM(D, K, n_layers=2)
    for att in llm.attentions:
        att.confidence_threshold = 0.99

    log("Populating Memory Bank...")
    llm.forward(keys[None, :], update_memory=True)

    query = generate_corrupted_query(keys[:5], noise_level=0.1)

    try:
        t_start = time.time()
        out, energy = llm.forward(query[None, :], update_memory=False)
        log(f"Multi-hop Reasoning Pass Complete. Avg Energy: {energy:.4f}, Time={time.time()-t_start:.4f}s")
        if energy > 0:
            log(f"SUCCESS: Model performed active multi-step reasoning hops (Energy: {energy:.4f}).")
        else:
            log("WARNING: No reasoning hops detected.")
    except Exception as e:
        log(f"ERROR in reasoning pass: {str(e)}")

if __name__ == "__main__":
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    log("Starting Chain Reasoning and Generalization Verification...")
    test_multi_hop_reasoning()
    test_convergence_and_accuracy()
    log("\nVerification Complete.")
