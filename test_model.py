import numpy as np
import time
import os
from data import load_clinc150, load_glove, build_embeddings
from associative_data import generate_associative_kv_pairs, generate_corrupted_query
from model import PhaseEncoderV2, PhaseLLM
from train import train, associative_pretrain, train_ntp
from wikitext_data import load_wikitext2_bytelevel, ByteEmbedding

LOG_FILE = "model_test_log.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def test_slm_viability():
    log("\n=== TEST: SLM Viability (Wikitext-2 NTP) ===")
    train_data, val_data = load_wikitext2_bytelevel()
    D, K = 100, 320
    byte_emb = ByteEmbedding(D)
    enc = PhaseEncoderV2(D, K)

    log(f"Training SLM for 100 steps. Vocab=256, D={D}, K={K}")
    t_start = time.time()
    # Using small epochs for verification
    enc, byte_emb, weights = train_ntp(enc, byte_emb, train_data, val_data, K, D, epochs=100)
    log(f"SLM Training Step Complete. Time={time.time()-t_start:.1f}s")

    # Simple generation test
    prompt = "The quick brown"
    log(f"Generation test with prompt: '{prompt}'")
    x = np.array([ord(c) for c in prompt])[None, :]
    E = byte_emb.table[x]
    # PhaseLLM forward
    llm = PhaseLLM(D, K, n_layers=2)
    llm.layers[0].W = enc.W # Patch trained weights

    out_phi, _ = llm.forward(E, causal=True)
    # MLP projection to vocab
    h = out_phi[:, -1, :] @ weights['W_hid'].T + weights['b_hid']
    logits = np.maximum(0, h) @ weights['W_cls'].T + weights['b_cls']
    next_char = chr(np.argmax(logits))
    log(f"Predicted next character: '{next_char}'")

def test_needle_in_haystack():
    log("\n=== TEST: Needle-In-A-Haystack Retrieval Viability ===")
    N_HAYSTACK = 500
    D, K = 100, 320
    keys, values = generate_associative_kv_pairs(N_HAYSTACK, D)
    enc = PhaseEncoderV2(D, K)

    log(f"Encoding {N_HAYSTACK} patterns into PhaseLLM memory...")
    llm = PhaseLLM(D, K, n_layers=1)
    llm.forward(keys[None, :], update_memory=True)

    # Retrieval
    needle_idx = 42
    needle_query = generate_corrupted_query(keys[needle_idx:needle_idx+1], noise_level=0.01)

    t_start = time.time()
    out, energy = llm.forward(needle_query[None, :], update_memory=False)
    log(f"Retrieval pass complete. Avg Energy: {energy:.4f}, Time={time.time()-t_start:.4f}s")

    # Check if retrieval preserved pattern identity (coherence check)
    sim = np.cos(out[0, 0, :] - llm.memory_banks[0].memory[needle_idx])
    coherence = np.mean(sim)
    log(f"Retrieval Coherence with target: {coherence:.4f}")
    if coherence > 0.9:
        log("SUCCESS: High-fidelity retrieval verified.")
    else:
        log("WARNING: Retrieval fidelity low.")

if __name__ == "__main__":
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    log("Starting Part 4/5 Viability Benchmarks...")
    test_needle_in_haystack()
    test_slm_viability()
    log("\nBenchmarks Complete.")
