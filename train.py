import numpy as np
import time

def ce_forward_and_grads(E_b, labels, W_enc, omega, W_cls, K):
    """
    Revised CE for complex-valued Phase Encoder.
    Using tanh of magnitude to gate the phase, preventing instability.
    Includes proper chain rule for angle and magnitude.
    """
    B = len(E_b)
    Ed = E_b.astype(np.float64)
    z = Ed @ W_enc.T
    mag = np.abs(z) + 1e-12
    gate = np.tanh(mag)
    phi = np.angle(z) * omega[None, :] * gate

    logits = phi @ W_cls.T
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(B), labels] + 1e-12))

    delta = probs.copy()
    delta[np.arange(B), labels] -= 1.0
    delta /= B

    grad_W_cls = (delta.T @ phi).astype(np.float64)
    d_phi = delta @ W_cls

    # phi = angle(z) * omega * tanh(|z|)
    # grad_phi_z = (d_angle * gate + np.angle(z) * d_tanh) * omega
    # d_angle = (1j * z / (2 * mag ** 2))
    # d_tanh = (1.0 - gate ** 2) * (z / (2 * mag))

    d_angle = (1j * z / (2 * mag ** 2))
    d_tanh = (1.0 - gate ** 2) * (z / (2 * mag))

    grad_phi_z = (d_angle * gate + np.angle(z) * d_tanh) * omega[None, :]
    # (B, K) * (B, K) -> (B, K). Then (K, B) @ (B, D) -> (K, D)
    grad_W_enc = (d_phi * grad_phi_z).T @ Ed

    return loss, grad_W_enc, grad_W_cls

def train(enc, train_embs, train_labels, val_embs, val_labels,
          N_INTENTS, K, D, epochs=300, batch_size=256,
          lam_ce=1.0):

    from model import Adam
    # Initialize W_cls with proper variance
    W_cls = (np.random.randn(N_INTENTS, K) * np.sqrt(1.0 / K)).astype(np.float64)

    class RealAdam(Adam):
        def __init__(self, shape, lr=5e-3):
            super().__init__(shape, lr)
            self.m = self.m.real; self.v = self.v.real

    opt_cls = RealAdam((N_INTENTS, K), lr=1e-2)
    opt_enc_ce = Adam((K, D), lr=2e-3)

    rng_batch = np.random.default_rng(42)
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
            print(f"  ep={ep:>3}  ce={ce_loss:.4f}  val_acc={val_acc:.4f} {'*best' if val_acc == best_val_acc else ''}")

    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_weights.get('ep')}")
    enc.W = best_weights['W_enc']
    enc.omega = best_weights['omega']
    return enc, best_weights['W_cls']
