import json
import random
import argparse
from collections import defaultdict, Counter


def subset_jsonl(input_path, output_path, subset_n=100, seed=0):
    random.seed(seed)

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    categories = ["STEM", "Social Sciences", "Humanities", "Other"]
    grouped = defaultdict(list)
    for item in data:
        if item["category"] in categories:
            grouped[item["category"]].append(item)

    target_per_cat = subset_n // len(categories)

    result = []
    stats = defaultdict(list)

    for cat in categories:
        items = grouped[cat]
        subjects = defaultdict(list)
        for x in items:
            subjects[x["subject"]].append(x)

        chosen = []
        subject_keys = list(subjects.keys())
        while len(chosen) < target_per_cat and any(subjects.values()):
            for key in subject_keys:
                if subjects[key] and len(chosen) < target_per_cat:
                    chosen.append(subjects[key].pop(random.randrange(len(subjects[key]))))
                    stats[cat].append(key)

        result.extend(chosen)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total sampled: {len(result)}")
    for cat in categories:
        subj_counts = Counter(stats[cat])
        print(f"\nCategory: {cat} ({sum(subj_counts.values())} samples)")
        for subj, cnt in subj_counts.items():
            print(f"  {subj}: {cnt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--subset_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    subset_jsonl(args.org_path, args.out_path, args.subset_n, args.seed)

if __name__ == "__main__":
    main()