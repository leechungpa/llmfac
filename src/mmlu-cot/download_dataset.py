
#!/usr/bin/env python3
import argparse, os, json
from mmlu_tools.datasets import load_all_test_and_split, save_jsonl
from dataclasses import asdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data/mmlu`")
    ap.add_argument("--train_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    train, test = load_all_test_and_split(args.train_size, args.test_size, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_p = os.path.join(args.out_dir, f"train_n{args.train_size}_seed{args.seed}.jsonl")
    test_p  = os.path.join(args.out_dir, f"test_n{args.test_size}_seed{args.seed}.jsonl")

    save_jsonl(train_p, (asdict(s) for s in train))
    save_jsonl(test_p,  (asdict(s) for s in test))

    print(f"saved {len(train)} -> {train_p}")
    print(f"saved {len(test)}  -> {test_p}")

if __name__ == "__main__":
    main()
