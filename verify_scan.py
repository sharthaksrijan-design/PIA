import torch
from pia_torch_optimized import DLiNOSSLayer

def test_scan_vs_sequential():
    device = torch.device("cpu")
    d_model = 64
    d_state = 32
    n_heads = 4
    seq_len = 32
    batch_size = 2

    layer = DLiNOSSLayer(d_model, d_state, n_heads, use_scan=True).to(device)
    layer.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    h0 = layer.init_state(batch_size, device)

    # Run scan
    y_scan, h_scan = layer._forward_scan(x, h0)

    # Run sequential
    y_seq, h_seq = layer._forward_sequential(x, h0)

    # Compare
    y_diff = (y_scan - y_seq).abs().max().item()
    h_diff = (h_scan - h_seq).abs().max().item()

    print(f"Max Y diff: {y_diff}")
    print(f"Max H diff: {h_diff}")

    assert y_diff < 1e-4, f"Y diff too large: {y_diff}"
    assert h_diff < 1e-4, f"H diff too large: {h_diff}"
    print("Verification successful!")

if __name__ == "__main__":
    test_scan_vs_sequential()
