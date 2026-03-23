import numpy as np
import time

def ce_forward_and_grads(E_b, labels, W_enc, omega, W_cls, b_cls, W_hid, b_hid, K):
    B = len(E_b)
    Ed = E_b.astype(np.float64)
    z = Ed @ W_enc.T
    mag = np.abs(z) + 1e-12
    gate = np.tanh(mag)
    phi = np.angle(z) * omega[None, :] * gate

    h = phi @ W_hid.T + b_hid[None, :]
    h_act = np.maximum(0, h)

    logits = h_act @ W_cls.T + b_cls[None, :]
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
    loss = -np.mean(np.log(probs[np.arange(B), labels] + 1e-12))

    delta = probs.copy()
    delta[np.arange(B), labels] -= 1.0
    delta /= B

    grad_W_cls = delta.T @ h_act
    grad_b_cls = np.sum(delta, axis=0)

    d_h = delta @ W_cls
    d_h[h <= 0] = 0
    grad_W_hid = d_h.T @ phi
    grad_b_hid = np.sum(d_h, axis=0)

    d_phi = d_h @ W_hid
    d_angle_dz_conj = -1j * z / (2 * mag ** 2)
    d_mag_dz_conj = z / (2 * mag)
    d_gate_dz_conj = (1.0 - gate ** 2) * d_mag_dz_conj
    grad_phi_z_conj = (d_angle_dz_conj * gate + np.angle(z) * d_gate_dz_conj) * omega[None, :]
    grad_W_enc = (d_phi * grad_phi_z_conj).T @ Ed

    return loss, grad_W_enc, grad_W_hid, grad_b_hid, grad_W_cls, grad_b_cls

def train(enc, train_embs, train_labels, val_embs, val_labels,
          N_INTENTS, K, D, epochs=2000, batch_size=256):

    from model import Adam
    H = 1024

    W_hid = np.random.randn(H, K) * np.sqrt(2.0 / K)
    b_hid = np.zeros(H)
    W_cls = np.random.randn(N_INTENTS, H) * np.sqrt(2.0 / H)
    b_cls = np.zeros(N_INTENTS)

    class RealAdam(Adam):
        def __init__(self, shape, lr=5e-3):
            super().__init__(shape, lr)
            self.m = self.m.real; self.v = self.v.real

    opt_enc = Adam((K, D), lr=1e-3) # Smaller learning rate for encoder stability
    opt_hid = RealAdam((H, K), lr=1e-3)
    opt_b_hid = RealAdam((H,), lr=1e-3)
    opt_cls = RealAdam((N_INTENTS, H), lr=1e-3)
    opt_b_cls = RealAdam((N_INTENTS,), lr=1e-3)

    rng_batch = np.random.default_rng(42)
    best_val_acc = 0.0
    best_weights = {}

    print(f"Targeting 97% Accuracy. Training {epochs} epochs...")

    for ep in range(1, epochs + 1):
        idx = rng_batch.choice(len(train_embs), batch_size, replace=False)
        E_b = train_embs[idx]
        labs = train_labels[idx]

        loss, gW_enc, gW_hid, gb_hid, gW_cls, gb_cls = ce_forward_and_grads(
            E_b, labs, enc.W, enc.omega, W_cls, b_cls, W_hid, b_hid, K)

        enc.W -= opt_enc.step(gW_enc)
        W_hid -= opt_hid.step(gW_hid)
        b_hid -= opt_b_hid.step(gb_hid)
        W_cls -= opt_cls.step(gW_cls)
        b_cls -= opt_b_cls.step(gb_cls)

        if ep % 500 == 0:
            opt_enc.lr *= 0.5
            opt_hid.lr *= 0.5
            opt_b_hid.lr *= 0.5
            opt_cls.lr *= 0.5
            opt_b_cls.lr *= 0.5

        if ep % 100 == 0 or ep == epochs:
            phi_val = enc.phi(val_embs.astype(np.float64))
            h = phi_val @ W_hid.T + b_hid[None, :]
            val_logits = np.maximum(0, h) @ W_cls.T + b_cls[None, :]
            val_acc = np.mean(np.argmax(val_logits, axis=1) == val_labels)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {
                    'W_enc': enc.W.copy(), 'omega': enc.omega.copy(),
                    'W_hid': W_hid.copy(), 'b_hid': b_hid.copy(),
                    'W_cls': W_cls.copy(), 'b_cls': b_cls.copy(),
                    'ep': ep
                }
            print(f"  ep={ep:>4}  loss={loss:.4f}  val_acc={val_acc:.4f} {'*best' if val_acc == best_val_acc else ''}")

    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_weights.get('ep')}")
    enc.W = best_weights['W_enc']
    enc.omega = best_weights['omega']
    return enc, best_weights
