import argparse
from src.datasets import read_jsonl_samples, save_jsonl
from src.formatting import make_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_path", required=True)
    ap.add_argument("--org_is_cot", action="store_true")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--n_shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dataset = read_jsonl_samples(args.org_path, with_cot_answer=args.org_is_cot)
    rows = list(make_jsonl(dataset, with_cot_answer=args.org_is_cot, n_shots=args.n_shots, seed=args.seed))
    save_jsonl(args.out_path, rows)
    print(f"saved {len(rows)} -> {args.out_path}")


if __name__ == "__main__":
    main()