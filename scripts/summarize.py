"""Print reproduced Table 1 rows next to the numbers reported in the paper.

Usage: python scripts/summarize.py [results_dir]
"""
import json
import os
import sys

MODELS = ["llama2-7b", "llama2-13b", "llama3.2-3b", "llama3-8b", "llama3-70b"]
RUNS = [
    ("4-4-16", "Ours f.p. calib.", "fp_calib_w4a4kv16"),
    ("4-4-16", "Ours quant. calib.", "quant_calib_w4a4kv16"),
    ("4-4-4", "Ours f.p. calib.", "fp_calib_w4a4kv4"),
    ("4-4-4", "Ours quant. calib.", "quant_calib_w4a4kv4"),
]

# Table 1 of the paper: (avg PPL over WikiText-2/PTB/C4, avg 0-shot accuracy over 9 tasks)
PAPER = {
    "fp_calib_w4a4kv16": {"llama2-7b": (19.28, 58.13), "llama2-13b": (22.78, 62.13), "llama3.2-3b": (13.29, 56.13),
                          "llama3-8b": (10.70, 61.80), "llama3-70b": (8.25, 68.17)},
    "quant_calib_w4a4kv16": {"llama2-7b": (18.77, 58.28), "llama2-13b": (23.24, 62.43), "llama3.2-3b": (13.21, 56.79),
                             "llama3-8b": (10.62, 62.51), "llama3-70b": (8.29, 67.63)},
    "fp_calib_w4a4kv4": {"llama2-7b": (19.15, 57.99), "llama2-13b": (23.02, 61.89), "llama3.2-3b": (13.74, 54.87),
                         "llama3-8b": (10.88, 61.40), "llama3-70b": (8.40, 67.79)},
    "quant_calib_w4a4kv4": {"llama2-7b": (18.64, 57.89), "llama2-13b": (23.12, 62.24), "llama3.2-3b": (13.68, 55.77),
                            "llama3-8b": (10.81, 61.67), "llama3-70b": (8.49, 68.13)},
}


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")

    header = f"{'Model':<12} {'Bits':<7} {'Method':<19} {'Wiki2':>7} {'PTB':>7} {'C4':>7} {'PPL':>7} {'(paper)':>8} {'0-shot':>7} {'(paper)':>8}"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        for bits, method, run in RUNS:
            path = os.path.join(results_dir, model, run, "results.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                res = json.load(f)
            ppl, acc = res.get("ppl", {}), res.get("acc", {})
            paper_ppl, paper_acc = PAPER[run][model]
            print(f"{model:<12} {bits:<7} {method:<19} {fmt(ppl.get('wikitext2')):>7} {fmt(ppl.get('ptb')):>7} "
                  f"{fmt(ppl.get('c4')):>7} {fmt(ppl.get('avg')):>7} {fmt(paper_ppl):>8} "
                  f"{fmt(acc.get('avg')):>7} {fmt(paper_acc):>8}")


if __name__ == "__main__":
    main()
