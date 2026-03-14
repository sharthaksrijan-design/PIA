# PIA - Paired Interference Architecture

This repository contains an optimized implementation of the Paired Interference Architecture (PIA) language model.

## Features
- **Parallel Scan**: Uses a Hillis-Steele inclusive prefix scan for fast $O(\log L)$ state-space model (SSM) computations.
- **Hippocampal Memory**: Implements a sparse-addressing Memory Bank for long-term retention.
- **DLiNOSS SSM**: Gated complex SSM with lossless and decaying state bands.

## Files
- `pia_torch_optimized.py`: Core model implementation and training script.
- `verify_scan.py`: Unit test verifying the mathematical equivalence of the parallel scan.
- `prepare_data.py`: Script to download and prepare the Wikitext-2 dataset.
- `eval_capabilities.py`: Comprehensive benchmarking for memory, context, and reasoning.
- `requirements.txt`: Python dependencies.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Architecture Verification
Ensure the optimized parallel scan matches the sequential reference:
```bash
python verify_scan.py
```

### 2. Data Preparation
Prepare the Wikitext-2 dataset:
```bash
python prepare_data.py
```

### 3. Training
Train the model on the prepared dataset:
```bash
python pia_torch_optimized.py --local_train wikitext2_train.txt --local_val wikitext2_validation.txt --steps 1000 --batch_size 4 --seq_len 128 --d_model 128 --n_layers 2 --d_state 32 --n_heads 2 --device cpu
```

### 4. Evaluation
Evaluate a trained checkpoint's cognitive abilities:
```bash
python eval_capabilities.py --ckpt ckpt/best.pt
```
