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

class DynamicNeighborhoodGating:
    """
    Learnable gating mechanism to determine whisper-routing based on query context.
    """
    def __init__(self, D, K, seed=42):
        rng = np.random.default_rng(seed)
        self.W_gate1 = rng.standard_normal((K, D)) * np.sqrt(1.0 / D)

    def get_routing(self, E):
        shape = E.shape
        E_flat = E.reshape(-1, shape[-1])
        h = E_flat @ self.W_gate1.T
        h = h.reshape(*shape[:-1], -1)
        # Use simple fixed neighborhood if gating not trained
        return None

class RecursiveEnergyAttention:
    """
    Whisper Protocol with Energy-Based Routing and Recursive Whispering.
    """
    def __init__(self, K, D, confidence_threshold=0.85, energy_budget=1.0, whisper_cost=0.1):
        self.K = K
        self.D = D
        self.confidence_threshold = confidence_threshold
        self.energy_budget = energy_budget
        self.whisper_cost = whisper_cost

    def compute_attention(self, Q_phase, K_phase, V_phase, routing_gate=None, mask=None, neighbor_map=None):
        if len(Q_phase.shape) == 2:
            Q_phase = Q_phase[:, None, :]
            K_phase = K_phase[:, None, :]
            V_phase = V_phase[:, None, :]

        B, L, K = Q_phase.shape
        diff = Q_phase - K_phase
        confidence = np.cos(diff)

        final_V = V_phase.copy()
        total_energy_consumed = np.zeros((B, L))
        current_energy = np.full((B, L), self.energy_budget)

        for step in range(5):
            need_whisper = (confidence < self.confidence_threshold) & (current_energy[..., None] >= self.whisper_cost)
            if not np.any(need_whisper):
                break

            if routing_gate is not None:
                whispered_V = np.matmul(routing_gate, final_V[..., None]).squeeze(-1)
            elif neighbor_map is not None:
                # Use static neighbor map for recursive whispering
                whispered_V = final_V.copy()
                for k in range(self.K):
                    neighs = neighbor_map.get(k, [])
                    if neighs:
                        whispered_V[:, :, k] = np.mean(final_V[:, :, neighs], axis=2)
            else:
                break

            mask_update = need_whisper.astype(float)
            final_V = (1 - mask_update) * final_V + mask_update * whispered_V
            confidence = (1 - mask_update) * confidence + mask_update * (confidence + 0.05)
            consumed = np.any(need_whisper, axis=-1).astype(float) * self.whisper_cost
            current_energy -= consumed
            total_energy_consumed += consumed

        Q_p = Q_phase.transpose(0, 2, 1)[:, :, :, None]
        K_p = K_phase.transpose(0, 2, 1)[:, :, None, :]
        attn_logits = np.cos(Q_p - K_p)
        if mask is not None:
            attn_logits = attn_logits + (mask[None, None, :, :] * -1e9)
        ex = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
        attn_weights = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)
        V_p = final_V.transpose(0, 2, 1)[:, :, :, None]
        out = np.matmul(attn_weights, V_p)

        return out.squeeze(-1).transpose(0, 2, 1), total_energy_consumed

class PhaseEncoderV2:
    def __init__(self, D, K, lr=5e-3, seed=42):
        self.D = D; self.K = K
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (D + K))
        r = rng.rayleigh(std / np.sqrt(2), (K, D))
        theta = rng.uniform(-np.pi, np.pi, (K, D))
        self.W = (r * np.exp(1j * theta)).astype(np.complex128)
        self.opt = Adam((K, D), lr=lr)
        self.omega = np.exp(rng.uniform(np.log(0.25), np.log(4.0), K)).astype(np.float64)

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
            idxs = np.argsort(-sim[k])[:8]
            neighbor_map[k] = [i for i in idxs if i != k]
        return neighbor_map

class PhaseLLM:
    def __init__(self, D, K, n_layers=3, seed=42):
        self.layers = []
        self.attentions = []
        for i in range(n_layers):
            in_dim = D if i == 0 else K
            self.layers.append(PhaseEncoderV2(in_dim, K, seed=seed+i))
            self.attentions.append(RecursiveEnergyAttention(K, K))

    def forward(self, E, causal=True):
        x = E
        L = x.shape[1] if len(x.shape) == 3 else 1
        mask = np.triu(np.ones((L, L)), k=1) if causal and L > 1 else None

        total_energy = 0
        for i in range(len(self.layers)):
            p = self.layers[i].phi(x)
            n_map = self.layers[i].get_neighbor_map()
            p, energy = self.attentions[i].compute_attention(p, p, p, mask=mask, neighbor_map=n_map)
            total_energy += np.mean(energy)
            if x.shape == p.shape:
                x = 0.5 * x + 0.5 * p
            else:
                x = p
        return x, total_energy
