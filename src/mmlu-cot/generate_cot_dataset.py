
import argparse, json
from mmlu_tools.datasets import read_jsonl_samples
from mmlu_tools.generate_cot_answer import generate_cot_answer



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--model_name", default="gpt-4.1-mini")
    args = ap.parse_args()

    verbose = True
    max_retries = 5

    org_dataset = read_jsonl_samples(args.org_path)
    failure_cases = []

    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(org_dataset):
            cot_answer, failure_case = generate_cot_answer(s, model_name=args.model_name, max_retries=max_retries, verbose=verbose, return_log=True)

            if failure_case is not None:
                failure_cases.append(f"|{i}"+failure_case)

            row = {
                "subject": s.subject,
                "category": s.category,
                "question": s.question,
                "choices": s.choices,
                "answer_idx": s.answer_idx,
                "cot_answer": cot_answer,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Failure Cases ===")
    print("\n".join(failure_cases))

    print("=== Summary ===")
    print(f"wrote {len(org_dataset)} with cot_answers -> {args.out_path}")

    with open(args.out_path.removesuffix(".jsonl") + ".log", "w", encoding="utf-8") as f:
        f.write("=== Failure Cases ===\n")
        f.write("\n".join(failure_cases))
        f.write("\n\n=== Summary ===\n")
        f.write(f"wrote {len(org_dataset)} with cot_answers -> {args.out_path}\n")


if __name__ == "__main__":
    main()
