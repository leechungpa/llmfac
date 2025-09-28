import json
import os
import argparse
from collections import defaultdict


def convert_choices(data, rename_keys=None):
    letters = ["A", "B", "C", "D"]
    if isinstance(data.get("choices"), list):
        for i, choice in enumerate(data["choices"]):
            data[letters[i]] = choice
    if isinstance(data.get("answer_idx"), int):
        data["answer_idx"] = letters[data["answer_idx"]]
    if rename_keys:
        for old_key, new_key in rename_keys.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)
    return data

def process_file(input_path, output_dir, rename_keys=None):
    os.makedirs(output_dir, exist_ok=True)
    file_handles = {}
    counts = defaultdict(int)

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            data = convert_choices(data, rename_keys)

            subject = data.get("subject", "unknown")
            out_path = os.path.join(output_dir, f"{subject}.jsonl")

            if subject not in file_handles:
                file_handles[subject] = open(out_path, "w", encoding="utf-8")

            file_handles[subject].write(json.dumps(data, ensure_ascii=False) + "\n")
            counts[subject] += 1

    for f in file_handles.values():
        f.close()

    print(f"== {input_path} ==")
    for subject, cnt in counts.items():
        print(f"{subject}: {cnt}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_path", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    rename_map = {"cot_answer": "answer"}

    process_file(args.org_path, args.output_dir, rename_map)



if __name__ == "__main__":
    main()
