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
    K = 320

    enc = PhaseEncoderV2(D, K, sampled=True, lr=2e-3)

    # Train
    trained_enc, W_cls = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=600)

    # Eval
    phi_test = trained_enc.phi(test_embs.astype(np.float64)).astype(np.float32)
    test_logits = phi_test @ W_cls.T.astype(np.float32)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_preds == test_labels)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Demonstrate Hierarchical PhaseLLM and Whisper Protocol
    print("\nDemonstrating PhaseLLM with Causal Whisper Attention...")
    llm = PhaseLLM(D, K, n_layers=2)
    # Add a sequence dimension for testing (B, L, D)
    seq_E = test_embs[:5][None, :, :] # (1, 5, 100)
    out_phi = llm.forward(seq_E, causal=True)
    print(f"Input shape: {seq_E.shape}")
    print(f"Output phase shape: {out_phi.shape}")
    print("Hierarchical Forward pass complete.")

if __name__ == "__main__":
    main()
