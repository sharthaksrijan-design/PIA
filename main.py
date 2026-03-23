import numpy as np
import time
from data import load_clinc150, load_glove, build_embeddings
from model import PhaseEncoderV2, WhisperProtocolAttention, PhaseLLM
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

    N_INTENTS = len(idx_to_intent)
    K = 512 # Increased oscillator count

    enc = PhaseEncoderV2(D, K, sampled=True, lr=5e-3)

    # Train for 1500 epochs with upgraded MLP head
    trained_enc, best_weights = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=1500)

    # Eval
    W_hid = best_weights['W_hid']
    b_hid = best_weights['b_hid']
    W_cls = best_weights['W_cls']
    b_cls = best_weights['b_cls']

    phi_test = trained_enc.phi(test_embs.astype(np.float64))
    h = phi_test @ W_hid.T + b_hid[None, :]
    test_logits = np.maximum(0, h) @ W_cls.T + b_cls[None, :]
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_preds == test_labels)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Demonstrate Hierarchical PhaseLLM
    print("\nDemonstrating PhaseLLM with Causal Whisper Attention...")
    llm = PhaseLLM(D, K, n_layers=2)
    seq_E = test_embs[:5][None, :, :]
    out_phi = llm.forward(seq_E, causal=True)
    print(f"Output phase shape: {out_phi.shape}")

if __name__ == "__main__":
    main()
