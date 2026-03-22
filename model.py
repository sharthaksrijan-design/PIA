import numpy as np
from scipy.special import expit as sigmoid

class Adam:
    def __init__(self, shape, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m  = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat  = self.m / (1 - self.b1 ** self.t)
        v_hat  = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class WhisperProtocolAttention:
    """
    Implements individual oscillator matching logic.
    Neighborhood defined by embedding similarity/closeness.
    Every model (oscillator) matches the query to its own weights.
    If it doesn't satisfy (low confidence), it passes the query to its neighborhood.
    """
    def __init__(self, K, D, neighbor_map=None, confidence_threshold=0.8):
        self.K = K
        self.D = D
        self.neighbor_map = neighbor_map # Map oscillator k -> neighbors [k1, k2, ...]
        self.confidence_threshold = confidence_threshold

    def compute_attention(self, Q_phase, K_phase, V_phase):
        """
        Modified attention using Whisper Protocol.
        Q_phase, K_phase, V_phase: (B, K) phase representations.
        """
        B = Q_phase.shape[0]
        # Traditional cosine similarity base for scores
        # scores[b, k] = mean_j cos(Q_phase[b, k] - K_phase[j, k])
        # In simple single-token retrieval, Q, K, V are same length

        # 1. Match own phase to query
        # For each batch b and oscillator k, calculate coherence between Q and K
        # diag_diff: difference between query phase and key phase for each oscillator
        diag_diff = Q_phase - K_phase # (B, K)
        # confidence[b, k]: 1 if phases match exactly, < 1 otherwise
        # We'll use a local window or just the point-wise difference
        confidence = np.cos(diag_diff) # (B, K)

        final_V = V_phase.copy()

        # Whisper Protocol Logic
        # We iterate to allow "whispering" to propagate through neighborhood
        # neighborhood awareness added to neighbors
        for _ in range(2): # 2 steps of propagation
            next_V = final_V.copy()
            if self.neighbor_map:
                for k in range(self.K):
                    # Check confidence for each batch element
                    low_conf_mask = confidence[:, k] < self.confidence_threshold
                    if np.any(low_conf_mask):
                        neighbors = self.neighbor_map.get(k, [])
                        if neighbors:
                            # Neighborhood awareness: neighbors provide their values
                            # Update only those with low confidence
                            avg_neighbor_v = np.mean(final_V[:, neighbors], axis=1)
                            next_V[low_conf_mask, k] = avg_neighbor_v[low_conf_mask]
            final_V = next_V

        # Scores are kept for attention weights (needed for routing/layer selection)
        scores = np.cos(Q_phase[:, :, None] - K_phase[:, None, :]) # (B, K, K)

        return scores, final_V

class PhaseEncoderV2:
    def __init__(self, D, K, sampled=True, batch_size=256, hard_ratio=0.5,
                 freq_bands=True, train_omega=True, partition=True, pos_frac=0.7,
                 lam_metric=1.0, lam_xfer=0.6, lam_rel=0.15, lr=5e-3, seed=42):
        self.D = D; self.K = K
        self.sampled = sampled; self.batch_size = batch_size; self.hard_ratio = hard_ratio
        self.freq_bands = freq_bands; self.train_omega = train_omega and freq_bands
        self.partition = partition
        self.lam_metric = lam_metric; self.lam_xfer = lam_xfer; self.lam_rel = lam_rel

        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((K, D)) * 1.5
        self.opt = Adam((K, D), lr=lr)

        if freq_bands:
            log_lo = np.log(0.5); log_hi = np.log(2.0)
            self.omega = np.exp(rng.uniform(log_lo, log_hi, K)).astype(np.float64)
            if self.train_omega:
                self.opt_omega = Adam((K,), lr=lr * 0.1)
        else:
            self.omega = np.ones(K)

        if partition:
            self.K_pos = int(pos_frac * K)
            self.K_rel = K - self.K_pos
        else:
            self.K_pos = K; self.K_rel = 0

        self._rng = np.random.default_rng(seed + 1)

    def phi(self, E):
        return 2 * np.pi * sigmoid(E @ self.W.T) * self.omega[None, :]

    def _metric_grad(self, E_b, D_emb_b):
        B = E_b.shape[0]; K = self.K
        z = E_b @ self.W.T
        sig = sigmoid(z)
        Phi = 2 * np.pi * sig * self.omega[None, :]
        C = np.cos(Phi); S = np.sin(Phi)
        SimM = (C @ C.T + S @ S.T) / K
        D_phase = 1.0 - SimM

        TI, TJ = np.triu_indices(B, k=1)
        diff = D_phase[TI, TJ] - D_emb_b[TI, TJ]
        abs_diff = np.abs(diff)

        n_keep = max(1, int(self.hard_ratio * len(TI)))
        hard = np.argsort(-abs_diff)[:n_keep]
        TI_h = TI[hard]; TJ_h = TJ[hard]

        loss = float(np.mean(abs_diff[hard]))

        sign_M = np.zeros((B, B))
        sign_M[TI_h, TJ_h] = np.sign(diff[hard])
        sign_M[TJ_h, TI_h] = np.sign(diff[hard])

        SC = sign_M @ C; SS = sign_M @ S
        G_phi = (SC * S - SS * C) / (K * B ** 2)
        sp = sig * (1 - sig)
        scale = 2 * np.pi * self.omega[None, :] * G_phi * sp
        grad_W = scale.T @ E_b

        if self.partition:
            grad_W[self.K_pos:, :] = 0.0

        return loss, grad_W

    def step(self, E, E_dist=None):
        N = E.shape[0]
        idx = self._rng.choice(N, min(self.batch_size, N), replace=False)
        E_b = E[idx].astype(np.float64)

        E_b_n = E_b / (np.linalg.norm(E_b, axis=1, keepdims=True) + 1e-12)
        D_emb_b = 1.0 - E_b_n @ E_b_n.T

        loss, grad_W = self._metric_grad(E_b, D_emb_b)
        self.W -= self.opt.step(self.lam_metric * grad_W)

        return {"metric": loss}

    def get_neighbor_map(self):
        # Neighborhood defined by oscillator weight similarity
        W_norm = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12)
        sim = W_norm @ W_norm.T
        neighbor_map = {}
        for k in range(self.K):
            # neighborhood is closeness
            idxs = np.argsort(-sim[k])[:6]
            neighbor_map[k] = [i for i in idxs if i != k]
        return neighbor_map
