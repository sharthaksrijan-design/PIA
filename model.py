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
        m_hat  = self.m / (1 - self.b1 ** self.t); v_hat  = self.v / (1 - self.b2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def hillis_steele_scan(x):
    B, L, K = x.shape; num_steps = int(np.ceil(np.log2(L))); res = x.copy()
    for i in range(num_steps):
        stride = 2**i
        if stride >= L: break
        shifted = np.zeros_like(res); shifted[:, stride:, :] = res[:, :-stride, :]
        res = res + shifted
    return res

class HippocampalMemoryBank:
    def __init__(self, K, capacity=1024):
        self.K = K; self.capacity = capacity; self.memory = np.zeros((capacity, K)); self.ptr = 0; self.is_full = False
    def write(self, phase_vec):
        flat = phase_vec.reshape(-1, self.K)
        for v in flat:
            self.memory[self.ptr] = v
            self.ptr = (self.ptr + 1) % self.capacity
            if self.ptr == 0: self.is_full = True
    def read(self, query_phase):
        B, L, K = query_phase.shape; limit = self.capacity if self.is_full else self.ptr
        if limit == 0: return np.zeros_like(query_phase)
        curr = self.memory[:limit]; q_flat = query_phase.reshape(-1, K)
        sim = np.cos(q_flat[:, None, :] - curr[None, :, :])
        coh = np.mean(sim, axis=-1); w = np.exp(coh * 5.0)
        w /= (np.sum(w, axis=-1, keepdims=True) + 1e-12)
        return (w @ curr).reshape(B, L, K)

class RecursiveEnergyAttention:
    """
    Advanced Whisper Protocol with Hardware Optimization (Quantization & Sparsity).
    """
    def __init__(self, K, D, confidence_threshold=0.85, energy_budget=1.0, whisper_cost=0.1, quantization_bits=8):
        self.K = K; self.D = D; self.confidence_threshold = confidence_threshold
        self.energy_budget = energy_budget; self.whisper_cost = whisper_cost
        self.q_bits = quantization_bits

    def quantize_phase(self, phi):
        """Hardware optimization: Map continuous phase to discrete bins."""
        if self.q_bits is None: return phi
        bins = 2**self.q_bits
        # Map (-pi, pi) to (0, bins-1)
        q = np.round(((phi + np.pi) / (2 * np.pi)) * (bins - 1))
        # Map back to (-pi, pi)
        return (q / (bins - 1)) * (2 * np.pi) - np.pi

    def compute_attention(self, Q_p, K_p, V_p, mask=None, neighbor_map=None, gpc=None, memory_bank=None, sparse_threshold=0.1):
        if len(Q_p.shape) == 2: Q_p, K_p, V_p = Q_p[:, None, :], K_p[:, None, :], V_p[:, None, :]
        B, L, K = Q_p.shape

        # 1. Hardware Optimization: Phase Quantization
        Q_p = self.quantize_phase(Q_p)
        K_p = self.quantize_phase(K_p)

        if gpc is not None: Q_p = Q_p + 0.05 * gpc
        fV = V_p.copy(); energy = np.zeros((B, L)); curr_e = np.full((B, L), self.energy_budget)

        for step in range(3):
            conf = np.cos(Q_p - K_p)
            need = (conf < self.confidence_threshold) & (curr_e[..., None] >= self.whisper_cost)

            # 2. Hardware Optimization: Sparse Updates
            # Skip oscillators that are already confident or out of energy
            if not np.any(need): break

            wV = fV.copy()
            if neighbor_map:
                for k in range(self.K):
                    # Sparse skip: Only update oscillators that actually need whispering
                    if not np.any(need[:, :, k]): continue
                    n = neighbor_map.get(k, [])
                    if n: wV[:, :, k] = np.mean(fV[:, :, n], axis=2)

            if memory_bank:
                mem_hop = memory_bank.read(Q_p)
                Q_p = Q_p + 0.1 * mem_hop

            mU = need.astype(float); fV = (1-mU)*fV + mU*wV
            consumed = np.any(need, axis=-1).astype(float) * self.whisper_cost
            curr_e -= consumed; energy += consumed

        Q, K = Q_p.transpose(0,2,1)[:,:,:,None], K_p.transpose(0,2,1)[:,:,None,:]
        logits = np.cos(Q-K)
        if mask is not None: logits += (mask[None,None,:,:] * -1e9)
        ex = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        w = ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)
        out = np.matmul(w, fV.transpose(0,2,1)[:,:,:,None])
        return out.squeeze(-1).transpose(0,2,1), energy

class PhaseEncoderV2:
    def __init__(self, D, K, lr=5e-3, seed=42):
        self.D = D; self.K = K; rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (D + K)); r = rng.rayleigh(std / np.sqrt(2), (K, D))
        th = rng.uniform(-np.pi, np.pi, (K, D)); self.W = (r * np.exp(1j * th)).astype(np.complex128)
        self.opt = Adam((K, D), lr=lr); self.omega = np.exp(rng.uniform(np.log(0.25), np.log(4.0), K)).astype(np.float64)

    def phi(self, E, stochastic=False, dropout_rate=0.1):
        z = E @ self.W.T; mag = np.abs(z) + 1e-12; gate = np.tanh(mag); phase = np.angle(z) * self.omega[None, ...] * gate
        if stochastic:
            mask = np.random.binomial(1, 1 - dropout_rate, phase.shape); phase = phase * mask
        return phase
    def get_neighbor_map(self):
        W_n = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12); sim = np.abs(W_n @ W_n.conj().T); neighbor_map = {}
        for k in range(self.K):
            idxs = np.argsort(-sim[k])[:8]; neighbor_map[k] = [i for i in idxs if i != k]
        return neighbor_map

class PhaseLLM:
    def __init__(self, D, K, n_layers=3, seed=42):
        self.layers, self.attentions, self.memory_banks = [], [], []
        for i in range(n_layers):
            in_dim = D if i == 0 else K
            self.layers.append(PhaseEncoderV2(in_dim, K, seed=seed+i))
            self.attentions.append(RecursiveEnergyAttention(K, K))
            self.memory_banks.append(HippocampalMemoryBank(K))

    def forward(self, E, causal=True, use_scan=True, update_memory=True, stochastic=False):
        x = E if len(E.shape) == 3 else E[:, None, :]
        L = x.shape[1]; mask = np.triu(np.ones((L, L)), k=1) if causal and L > 1 else None
        total_e = 0
        for i in range(len(self.layers)):
            p = self.layers[i].phi(x, stochastic=stochastic); gpc = np.mean(p, axis=1, keepdims=True)
            if use_scan and L > 1: p = hillis_steele_scan(p)
            n_map = self.layers[i].get_neighbor_map()
            p, e = self.attentions[i].compute_attention(p, p, p, mask=mask, neighbor_map=n_map, gpc=gpc, memory_bank=self.memory_banks[i])
            if update_memory: self.memory_banks[i].write(p)
            total_e += np.mean(e)
            x = 0.5 * x + 0.5 * p if x.shape == p.shape else p
        return x, total_e
