import numpy as np
import time
from model import Adam, PhaseEncoderV2

def ce_forward_and_grads(E_b, labels, W_enc, omega, W_cls, K):
    from scipy.special import expit as sigmoid
    B = len(E_b)
    Ed = E_b.astype(np.float64)
    z = Ed @ W_enc.T
    sig = sigmoid(z)
    phi = 2 * np.pi * sig * omega[None, :]
    logits = phi @ W_cls.T
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(B), labels] + 1e-12))
    delta = probs.copy()
    delta[np.arange(B), labels] -= 1.0
    delta /= B
    grad_W_cls = delta.T @ phi
    d_phi = delta @ W_cls
    sp = sig * (1 - sig)
    scale = 2 * np.pi * omega[None, :] * d_phi * sp
    grad_W_enc = scale.T @ Ed
    return loss, grad_W_enc, grad_W_cls

def train(enc, train_embs, train_labels, val_embs, val_labels,
          N_INTENTS, K, D, epochs=300, batch_size=256,
          lam_ce=1.0, quads_pos=None, quads_neg=None):

    W_cls = np.random.randn(N_INTENTS, K).astype(np.float64) * 0.01
    opt_cls = Adam((N_INTENTS, K), lr=5e-3)
    opt_enc_ce = Adam((K, D), lr=2e-3)

    rng_batch = np.random.default_rng(42)
    t0 = time.time()
    best_val_acc = 0.0
    best_weights = {}

    print(f"Training {epochs} epochs...")

    for ep in range(1, epochs + 1):
        idx = rng_batch.choice(len(train_embs), batch_size, replace=False)
        E_b = train_embs[idx]
        labs = train_labels[idx]

        ce_loss, gW_enc, gW_cls = ce_forward_and_grads(E_b, labs, enc.W, enc.omega, W_cls, K)

        enc.W -= opt_enc_ce.step(lam_ce * gW_enc)
        W_cls -= opt_cls.step(lam_ce * gW_cls)

        # Geometric step
        geo = enc.step(train_embs)

        if ep % 50 == 0 or ep == epochs:
            phi_val = enc.phi(val_embs.astype(np.float64)).astype(np.float32)
            val_logits = phi_val @ W_cls.T.astype(np.float32)
            val_preds = np.argmax(val_logits, axis=1)
            val_acc = np.mean(val_preds == val_labels)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {
                    'W_enc': enc.W.copy(),
                    'W_cls': W_cls.copy(),
                    'omega': enc.omega.copy(),
                    'ep': ep
                }
            print(f"  ep={ep:>3}  ce={ce_loss:.4f}  metric={geo['metric']:.4f}  val_acc={val_acc:.4f} {'*best' if val_acc == best_val_acc else ''}")

    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_weights.get('ep')}")
    enc.W = best_weights['W_enc']
    enc.omega = best_weights['omega']
    return enc, best_weights['W_cls']
