# LLM Expansion Roadmap for Phase-SNN + Whisper Protocol

This document outlines the strategy for scaling the current Phase-SNN and Whisper Protocol architecture into a full Large Language Model (LLM) capable of 100% retrieval accuracy and generative reasoning.

## 1. Architectural Scaling
- **From Encoder to Decoder**: Implement a causal mask in the `WhisperProtocolAttention` to enable autoregressive generation.
- **Hierarchical Phase Layers**: Instead of a single PhaseEncoder, stack multiple layers where each layer's output phase serves as the input to the next, allowing for abstract semantic hierarchies.
- **Complex-Valued Oscillators**: Transition to full complex-valued weights $W \in \mathbb{C}^{K \times D}$ to represent both magnitude (importance) and phase (semantic content) in the hidden state.

## 2. Advanced Whisper Protocol
- **Recursive Whispering**: Implement multi-step "whispering" where low-confidence oscillators query their neighborhood recursively. This mimics the brain's hippocampal-cortical retrieval loops.
- **Dynamic Neighborhoods**: Replace static similarity-based neighborhoods with learnable gating mechanisms that "whisper" to different sub-networks based on the query context.
- **Energy-Based Routing**: Introduce a "whisper energy" budget. Queries that require more passing (low confidence) consume more energy, naturally prioritizing "sharp" matches for 100% accuracy in known KV-pairs.

## 3. Training for 100% Retrieval
- **Associative Memory Pre-training**: Use the Whisper Protocol to pre-train the model on massive Key-Value pairs (e.g., dictionary definitions, code snippets) with a contrastive loss that penalizes neighborhood "blur".
- **Sharpness Regularization**: Add a loss term that encourages oscillators to either be extremely confident or pass to a neighbor, minimizing "undecided" semantic states.

## 4. Hardware Optimization
- **Phase-Quantization**: Since the model relies on phase, we can quantize the weights into discrete phase-steps (e.g., 8-bit or 4-bit phases), significantly reducing memory footprint while maintaining semantic resolution.
- **Sparse Oscillator Updates**: During inference, only update/compute oscillators that are "whispered" to, leveraging the sparsity of the hippocampal memory bank.

## 5. Next Steps
1. Implement a **Small Language Model (SLM)** prototype on the Wikitext-2 dataset using Byte-Level tokenization.
2. Integrate the **Parallel Scan** (Hillis-Steele) from the PIA architecture for $O(\log L)$ sequence processing.
3. Benchmarking against standard Transformer-based LLMs on the **Needle-In-A-Haystack** test to verify 100% retrieval.
