import numpy as np
import time

def sharpness_regularization(phi, strength=0.01):
    norm_phi = (phi / (2 * np.pi)) % 1.0
    penalty = strength * np.mean(np.sin(2 * np.pi * norm_phi) ** 2)
    grad_phi = strength * 2 * np.sin(4 * np.pi * norm_phi)
    return penalty, grad_phi

def ce_forward_and_grads(E_b, labels, W_enc, omega, W_cls, b_cls, W_hid, b_hid, K, lam_sharp=0.01, dropout_rate=0.1):
    # E_b: (B, L, D) or (N, D)
    if len(E_b.shape) == 3:
        B, L, D = E_b.shape
        E_flat = E_b.reshape(-1, D)
    else:
        E_flat = E_b
        D = E_b.shape[-1]

    Ed = E_flat.astype(np.float64)
    z = Ed @ W_enc.T
    mag = np.abs(z) + 1e-12; gate = np.tanh(mag)
    phi_flat = np.angle(z) * omega[None, :] * gate

    noise_mask = 1.0
    if dropout_rate > 0:
        noise_mask = np.random.binomial(1, 1 - dropout_rate, phi_flat.shape)
        phi_flat = phi_flat * noise_mask

    sharp_loss, grad_sharp_phi = sharpness_regularization(phi_flat, strength=lam_sharp)

    h = phi_flat @ W_hid.T + b_hid[None, :]
    h_act = np.maximum(0, h)
    logits_flat = h_act @ W_cls.T + b_cls[None, :]

    ex = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
    probs = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)
    labels_flat = labels.reshape(-1)
    loss = -np.mean(np.log(probs[np.arange(len(probs)), labels_flat] + 1e-12)) + sharp_loss

    delta = probs.copy(); delta[np.arange(len(probs)), labels_flat] -= 1.0; delta /= len(probs)

    grad_W_cls = delta.T @ h_act; grad_b_cls = np.sum(delta, axis=0)
    d_h = delta @ W_cls; d_h[h <= 0] = 0
    grad_W_hid = d_h.T @ phi_flat; grad_b_hid = np.sum(d_h, axis=0)

    d_phi = (d_h @ W_hid + grad_sharp_phi) * noise_mask
    d_angle = -1j * z / (2 * mag ** 2); d_mag = z / (2 * mag); d_gate = (1.0 - gate ** 2) * d_mag
    grad_phi_z_conj = (d_angle * gate + np.angle(z) * d_gate) * omega[None, :]
    grad_W_enc = (d_phi * grad_phi_z_conj).T @ Ed

    # grad_E: (N, D)
    grad_E = (d_phi * grad_phi_z_conj).real @ W_enc.real

    return loss, grad_W_enc, grad_W_hid, grad_b_hid, grad_W_cls, grad_b_cls, grad_E

def train_ntp(enc, byte_emb, train_data, val_data, K, D, epochs=100, batch_size=16, seq_len=32):
    from model import Adam; from wikitext_data import get_ntp_batches
    H = 512; N_VOCAB = 256
    W_hid = np.random.randn(H, K) * np.sqrt(2.0/K); b_hid = np.zeros(H)
    W_cls = np.random.randn(N_VOCAB, H) * np.sqrt(2.0/H); b_cls = np.zeros(N_VOCAB)
    class RealAdam(Adam):
        def __init__(self, shape, lr=1e-3): super().__init__(shape, lr); self.m = self.m.real; self.v = self.v.real
    opt_enc = Adam((K, D), lr=1e-3); opt_hid = RealAdam((H, K), lr=2e-3); opt_b_hid = RealAdam((H,), lr=2e-3)
    opt_cls = RealAdam((N_VOCAB, H), lr=2e-3); opt_b_cls = RealAdam((N_VOCAB,), lr=2e-3)
    opt_emb = RealAdam((256, D), lr=2e-3)
    data_gen = get_ntp_batches(train_data, batch_size, seq_len)
    print(f"SLM NTP Training. {epochs} epochs. seq_len={seq_len}")
    for ep in range(1, epochs + 1):
        x, y = next(data_gen)
        E_b = byte_emb.table[x]
        loss, gW_enc, gW_hid, gb_hid, gW_cls, gb_cls, gE = ce_forward_and_grads(
            E_b, y, enc.W, enc.omega, W_cls, b_cls, W_hid, b_hid, K, dropout_rate=0.0)
        enc.W -= opt_enc.step(gW_enc); W_hid -= opt_hid.step(gW_hid); b_hid -= opt_b_hid.step(gb_hid)
        W_cls -= opt_cls.step(gW_cls); b_cls -= opt_b_cls.step(gb_cls)
        gE_table = np.zeros_like(byte_emb.table)
        np.add.at(gE_table, x.reshape(-1), gE.reshape(-1, D))
        byte_emb.table -= opt_emb.step(gE_table)
        if ep % 20 == 0: print(f"  ep={ep:>4}  loss={loss:.4f}")
    return enc, byte_emb, {'W_hid':W_hid, 'b_hid':b_hid, 'W_cls':W_cls, 'b_cls':b_cls}

def train(enc, train_embs, train_labels, val_embs, val_labels,
          N_INTENTS, K, D, epochs=600, batch_size=256, lam_sharp=0.01, quads=None, dropout_rate=0.1):
    from model import Adam
    H = 1024; W_hid = np.random.randn(H, K) * np.sqrt(2.0/K); b_hid = np.zeros(H)
    W_cls = np.random.randn(N_INTENTS, H) * np.sqrt(2.0/H); b_cls = np.zeros(N_INTENTS)
    class RealAdam(Adam):
        def __init__(self, shape, lr=5e-3): super().__init__(shape, lr); self.m = self.m.real; self.v = self.v.real
    opt_enc = Adam((K, D), lr=2e-3); opt_hid = RealAdam((H, K), lr=5e-3); opt_b_hid = RealAdam((H,), lr=5e-3)
    opt_cls = RealAdam((N_INTENTS, H), lr=5e-3); opt_b_cls = RealAdam((N_INTENTS,), lr=5e-3)
    rng = np.random.default_rng(42); best_val_acc = 0.0; best_weights = {}
    for ep in range(1, epochs + 1):
        idx = rng.choice(len(train_embs), batch_size, replace=False); E_b = train_embs[idx]; labs = train_labels[idx]
        loss, gW_enc, gW_hid, gb_hid, gW_cls, gb_cls, _ = ce_forward_and_grads(
            E_b[:, None, :], labs, enc.W, enc.omega, W_cls, b_cls, W_hid, b_hid, K, lam_sharp=lam_sharp, dropout_rate=dropout_rate)
        if quads:
            batch_quads = [quads[i] for i in rng.choice(len(quads), min(32, len(quads)), replace=False)]
            from train import transfer_relational_grad
            x_loss, gxW_enc = transfer_relational_grad(enc, train_embs, batch_quads); gW_enc += gxW_enc
        enc.W -= opt_enc.step(gW_enc); W_hid -= opt_hid.step(gW_hid); b_hid -= opt_b_hid.step(gb_hid); W_cls -= opt_cls.step(gW_cls); b_cls -= opt_b_cls.step(gb_cls)
        if ep % 100 == 0 or ep == epochs:
            phi_val = enc.phi(val_embs.astype(np.float64))
            h = phi_val @ W_hid.T + b_hid[None, :]; val_logits = np.maximum(0, h) @ W_cls.T + b_cls[None, :]
            val_acc = np.mean(np.argmax(val_logits, axis=1) == val_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {'W_enc': enc.W.copy(), 'omega': enc.omega.copy(), 'W_hid': W_hid.copy(), 'b_hid': b_hid.copy(), 'W_cls': W_cls.copy(), 'b_cls': b_cls.copy(), 'ep': ep}
            print(f"  ep={ep:>4}  loss={loss:.4f}  val_acc={val_acc:.4f} {'*best' if val_acc == best_val_acc else ''}")
    enc.W = best_weights['W_enc']; enc.omega = best_weights['omega']
    return enc, best_weights

def transfer_relational_grad(enc, E_all, quads, lam_xfer=0.1):
    if not quads: return 0.0, np.zeros_like(enc.W)
    indices = list(set([idx for q in quads for idx in q])); idx_map = {g: l for l, g in enumerate(indices)}; E_loc = E_all[indices]
    z = E_loc @ enc.W.T; mag = np.abs(z) + 1e-12; gate = np.tanh(mag); phi = np.angle(z) * enc.omega[None, :] * gate
    L = 0.0; G_phi = np.zeros_like(phi)
    for a, b, c, d in quads:
        la, lb, lc, ld = idx_map[a], idx_map[b], idx_map[c], idx_map[d]
        theta = phi[lc] + phi[lb] - phi[la] - phi[ld]; L += float(np.mean(-np.cos(theta))); g = np.sin(theta) / (enc.K * len(quads)); G_phi[lb] += g; G_phi[lc] += g; G_phi[la] -= g; G_phi[ld] -= g
    d_angle = -1j * z / (2 * mag ** 2); d_mag = z / (2 * mag); d_gate = (1.0 - gate ** 2) * d_mag
    grad_phi_z_conj = (d_angle * gate + np.angle(z) * d_gate) * enc.omega[None, :]
    grad_W_enc = (G_phi * grad_phi_z_conj).T @ E_loc
    return L / len(quads), lam_xfer * grad_W_enc

def associative_pretrain(enc, keys, values, epochs=500, batch_size=64, lam_sharp=0.05):
    from model import Adam; K, D = enc.K, enc.D; N = len(keys); W_assoc = np.random.randn(N, K) * np.sqrt(1.0 / K)
    class RealAdam(Adam):
        def __init__(self, shape, lr=5e-3): super().__init__(shape, lr); self.m = self.m.real; self.v = self.v.real
    opt_enc = Adam((K, D), lr=5e-3); opt_assoc = RealAdam((N, K), lr=1e-2); rng = np.random.default_rng(42)
    for ep in range(1, epochs + 1):
        idx = rng.choice(N, batch_size, replace=False); E_b = keys[idx]; labs = values[idx]
        z = E_b @ enc.W.T; mag = np.abs(z) + 1e-12; gate = np.tanh(mag); phi = np.angle(z) * enc.omega[None, :] * gate
        norm_phi = (phi / (2 * np.pi)) % 1.0; sharp_loss = lam_sharp * np.mean(np.sin(2 * np.pi * norm_phi) ** 2); g_sharp_phi = lam_sharp * 2 * np.sin(4 * np.pi * norm_phi)
        logits = phi @ W_assoc.T ; ex = np.exp(logits - np.max(logits, axis=1, keepdims=True)); probs = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12); delta = probs.copy(); delta[np.arange(batch_size), labs] -= 1.0; delta /= batch_size
        gW_assoc = delta.T @ phi; d_phi = delta @ W_assoc + g_sharp_phi; d_angle = -1j * z / (2 * mag ** 2); d_mag = z / (2 * mag); d_gate = (1.0 - gate ** 2) * d_mag; g_phi_z_conj = (d_angle * gate + np.angle(z) * d_gate) * enc.omega[None, :]
        gW_enc = (d_phi * g_phi_z_conj).T @ E_b; enc.W -= opt_enc.step(gW_enc); W_assoc -= opt_assoc.step(gW_assoc)
    return enc, W_assoc
