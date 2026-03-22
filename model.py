import numpy as np
from scipy.special import expit as sigmoid

class Adam:
    def __init__(self, shape, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m  = np.zeros(shape, dtype=np.complex128)
        self.v  = np.zeros(shape, dtype=np.float64)
        self.t  = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * np.abs(g) ** 2
        m_hat  = self.m / (1 - self.b1 ** self.t)
        v_hat  = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class WhisperProtocolAttention:
    """
    Implements individual oscillator matching logic with CAUSAL MASKING.
    """
    def __init__(self, K, D, neighbor_map=None, confidence_threshold=0.8):
        self.K = K
        self.D = D
        self.neighbor_map = neighbor_map
        self.confidence_threshold = confidence_threshold

    def compute_attention(self, Q_phase, K_phase, V_phase, mask=None):
        """
        Q, K, V: (B, L, K)
        mask: (L, L) causal mask if provided
        """
        B, L, K = Q_phase.shape

        # 1. Match own phase to query
        diff = Q_phase - K_phase
        confidence = np.cos(diff) # (B, L, K)

        final_V = V_phase.copy()

        for _ in range(2):
            next_V = final_V.copy()
            if self.neighbor_map:
                for k in range(self.K):
                    low_conf_mask = confidence[:, :, k] < self.confidence_threshold
                    if np.any(low_conf_mask):
                        neighbors = self.neighbor_map.get(k, [])
                        if neighbors:
                            avg_neighbor_v = np.mean(final_V[:, :, neighbors], axis=2)
                            next_V[low_conf_mask, k] = avg_neighbor_v[low_conf_mask]
            final_V = next_V

        # 2. Causal Dot-Product Attention (Phase-based)
        Q_p = Q_phase.transpose(0, 2, 1)[:, :, :, None]
        K_p = K_phase.transpose(0, 2, 1)[:, :, None, :]

        attn_logits = np.cos(Q_p - K_p)

        if mask is not None:
            attn_logits = attn_logits + (mask[None, None, :, :] * -1e9)

        ex = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
        attn_weights = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)

        V_p = final_V.transpose(0, 2, 1)[:, :, :, None]
        out = np.matmul(attn_weights, V_p)
        return out.squeeze(-1).transpose(0, 2, 1)

class PhaseEncoderV2:
    def __init__(self, D, K, sampled=True, batch_size=256, hard_ratio=0.5,
                 freq_bands=True, train_omega=True, partition=True, pos_frac=0.7,
                 lam_metric=1.0, lr=5e-3, seed=42):
        self.D = D; self.K = K
        self.sampled = sampled; self.batch_size = batch_size; self.hard_ratio = hard_ratio
        self.freq_bands = freq_bands; self.train_omega = train_omega
        self.partition = partition
        self.lam_metric = lam_metric

        rng = np.random.default_rng(seed)
        r = rng.standard_normal((K, D)) * 0.1
        theta = rng.uniform(-np.pi, np.pi, (K, D))
        self.W = (r * np.exp(1j * theta)).astype(np.complex128)
        self.opt = Adam((K, D), lr=lr)

        if freq_bands:
            self.omega = np.exp(rng.uniform(np.log(0.5), np.log(2.0), K)).astype(np.float64)
            if self.train_omega:
                self.opt_omega = Adam((K,), lr=lr * 0.1)
        else:
            self.omega = np.ones(K)

        self.K_pos = int(pos_frac * K) if partition else K
        self._rng = np.random.default_rng(seed + 1)

    def phi(self, E):
        z = E @ self.W.T
        mag = np.abs(z) + 1e-12
        gate = np.tanh(mag)
        phase = np.angle(z) * self.omega[None, ...] * gate
        return phase

    def get_neighbor_map(self):
        W_flat = self.W.reshape(self.K, -1)
        W_norm = W_flat / (np.linalg.norm(W_flat, axis=1, keepdims=True) + 1e-12)
        sim = np.abs(W_norm @ W_norm.conj().T)
        neighbor_map = {}
        for k in range(self.K):
            idxs = np.argsort(-sim[k])[:6]
            neighbor_map[k] = [i for i in idxs if i != k]
        return neighbor_map

class PhaseLLM:
    """
    Hierarchical Phase Model (Stacked Layers).
    Each layer refines the semantic phase of the sequence.
    """
    def __init__(self, D, K, n_layers=3, seed=42):
        self.layers = []
        self.attentions = []
        for i in range(n_layers):
            in_dim = D if i == 0 else K
            self.layers.append(PhaseEncoderV2(in_dim, K, seed=seed+i))
            # Every layer has neighborhood-aware attention
            self.attentions.append(WhisperProtocolAttention(K, K))

    def forward(self, E, causal=True):
        x = E
        L = x.shape[1] if len(x.shape) == 3 else 1
        mask = np.triu(np.ones((L, L)), k=1) if causal and L > 1 else None

        for i in range(len(self.layers)):
            # Encoder projection
            x = self.layers[i].phi(x)
            # Neighborhood-aware attention (routing/refinement)
            self.attentions[i].neighbor_map = self.layers[i].get_neighbor_map()
            x = self.attentions[i].compute_attention(x, x, x, mask=mask)
        return x
