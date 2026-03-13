from datasets import load_dataset
import os

def prepare_wikitext2():
    """Download and save Wikitext-2 as local text files."""
    for split in ["train", "validation"]:
        print(f"Preparing {split} split...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        filename = f"wikitext2_{split}.txt"
        with open(filename, "w") as f:
            for ex in ds:
                f.write(ex["text"] + "\n")
        print(f"Saved to {filename}")

if __name__ == "__main__":
    prepare_wikitext2()
