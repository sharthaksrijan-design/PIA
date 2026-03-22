import numpy as np
import time
from data import load_clinc150, load_glove, build_embeddings, build_domain_structure, build_balanced_quads
from model import PhaseEncoderV2, WhisperProtocolAttention
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

    # Train for 600 epochs
    trained_enc, W_cls = train(enc, train_embs, train_labels, val_embs, val_labels, N_INTENTS, K, D, epochs=600)

    # Eval
    phi_test = trained_enc.phi(test_embs.astype(np.float64)).astype(np.float32)
    test_logits = phi_test @ W_cls.T.astype(np.float32)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_preds == test_labels)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Demonstrate Whisper Protocol Attention
    neighbor_map = trained_enc.get_neighbor_map()
    whisper_att = WhisperProtocolAttention(K, D, neighbor_map=neighbor_map)

    # Example: Attention with Whisper
    Q = phi_test[:10]
    K_ref = phi_test[10:20]
    V = phi_test[20:30]
    scores, final_V = whisper_att.compute_attention(Q, K_ref, V)
    print("Whisper Protocol attention computed.")
    print(f"Scores shape: {scores.shape}, Final V shape: {final_V.shape}")

if __name__ == "__main__":
    main()
