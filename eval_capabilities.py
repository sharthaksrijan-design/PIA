import torch
import argparse
from pia_torch_optimized import PIAModel, VOCAB

def evaluate_model(ckpt_path):
    device = torch.device("cpu")
    # Using weights_only=False because PIAModel is a custom class not in a package
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scfg_dict = ckpt.get("cfg", {})

    model = PIAModel(
        vocab         = VOCAB,
        d_model       = scfg_dict.get("d_model", 128),
        n_layers      = scfg_dict.get("n_layers", 2),
        d_state       = scfg_dict.get("d_state", 32),
        n_heads       = scfg_dict.get("n_heads", 2),
        K             = scfg_dict.get("K", 4),
        mem_slots     = scfg_dict.get("mem_slots", 64),
        slow_frac     = scfg_dict.get("slow_frac", 0.25),
        mem_top_k     = scfg_dict.get("mem_top_k", 4),
        use_scan      = False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tests = [
        ("Memory/Recall", "The special code is: 9-7-2-1. Question: What is the special code? Answer: The special code is", 10),
        ("Context", "Once upon a time, there was a small cat named Luna. Luna loved to climb trees. One day, Luna climbed the tallest tree in the", 10),
        ("Reasoning", "A is bigger than B. B is bigger than C. Therefore, A is bigger than", 5),
    ]

    print(f"{'Test Type':<15} | {'Prompt':<80} | {'Generated'}")
    print("-" * 120)
    for name, prompt, n in tests:
        generated = model.generate(prompt, n=n, temp=0.1, device=device)
        # Extract only the newly generated part
        new_text = generated[len(prompt):]
        print(f"{name:<15} | {prompt[:80]:<80} | {new_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate_model(args.ckpt)
