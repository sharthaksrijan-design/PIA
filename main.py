import numpy as np
import time
from data import load_clinc150, load_glove, build_embeddings, build_domain_structure, build_balanced_quads
from model import PhaseEncoderV2, RecursiveEnergyAttention, PhaseLLM
from train import train

def main():
    print("Loading data...")
    splits, idx_to_intent, intent_to_idx = load_clinc150()
    glove, D = load_glove()

    train_texts, train_intents = splits['train']
    val_texts,   val_intents   = splits['validation']
    test_texts,  test_intents  = splits['test']

    train_labels = np.array(train_intents)
    val_labels   = np.array(val_intents)
    test_labels  = np.array(test_intents)

    train_embs = build_embeddings(train_texts, glove, D)
    val_embs   = build_embeddings(val_texts,   glove, D)
    test_embs  = build_embeddings(test_texts,  glove, D)

    # Build Semantic Relational Quads
    print("Building semantic relational structures...")
    domain_to_intents = build_domain_structure(intent_to_idx)
    quads = build_balanced_quads(domain_to_intents, train_labels, max_per_family=50)
    print(f"Total relational quads: {len(quads)}")

    N_INTENTS = len(idx_to_intent)
    K = 512

    enc = PhaseEncoderV2(D, K, lr=5e-3)

    # Train with CE + Sharpness + Relational Loss
    trained_enc, best_weights = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=600, quads=quads)

    # Eval
    W_hid, b_hid = best_weights['W_hid'], best_weights['b_hid']
    W_cls, b_cls = best_weights['W_cls'], best_weights['b_cls']
    phi_test = trained_enc.phi(test_embs.astype(np.float64))
    h = phi_test @ W_hid.T + b_hid[None, :]
    test_logits = np.maximum(0, h) @ W_cls.T + b_cls[None, :]
    test_acc = np.mean(np.argmax(test_logits, axis=1) == test_labels)
    print(f"Relational Trained Test Accuracy: {test_acc:.4f}")

    # Advanced PhaseLLM Inference with Memory
    print("\nRunning Advanced PhaseLLM Inference (Context + Memory)...")
    llm = PhaseLLM(D, K, n_layers=2)
    llm.layers[0].W = trained_enc.W
    llm.layers[0].omega = trained_enc.omega

    # Simulate a stream of queries to build memory
    for i in range(0, 500, 100):
        seq_E = test_embs[i:i+100][None, :, :]
        out_phi, energy = llm.forward(seq_E, causal=True, update_memory=True)
        print(f"Sequence {i//100 + 1}: Avg Energy: {energy:.4f}")

    print("Memory Bank Status: Ready.")

if __name__ == "__main__":
    main()
