import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
from datasets import load_dataset


SUBJECT_TO_CAT = {
    "abstract_algebra": "STEM",
    "anatomy": "Other",
    "astronomy": "STEM",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "Other",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "Humanities",
    "global_facts": "Other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Other",
    "human_sexuality": "Social Sciences",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "machine_learning": "STEM",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "Other",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "Other",
    "professional_law": "Humanities",
    "professional_medicine": "Other",
    "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "virology": "Other",
    "world_religions": "Humanities",
}


@dataclass
class Sample:
    subject: str
    category: str
    question: str
    choices: List[str]
    answer_idx: int

@dataclass
class SampleCoT:
    subject: str
    category: str
    question: str
    choices: List[str]
    answer_idx: int
    cot_answer: str


def save_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl_samples(path: str, with_cot_answer: bool=False) -> List[Sample|SampleCoT]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if with_cot_answer:
                rows.append(SampleCoT(**x))
            else:
                rows.append(Sample(**x))
    return rows


def norm_sample(raw: dict, subject_to_cat: Dict[str, str]) -> Sample:
    subject = raw.get("subject", "unknown")
    category = subject_to_cat.get(subject, "Other")

    return Sample(
        subject=subject,
        category=category,
        question=raw["question"],
        choices=list(raw["choices"]),
        answer_idx=int(raw["answer"]),
    )

def _get_balanced_n(n_total: int, categorys: list, by_cat: Dict) -> Dict[str, int]:
    k = len(categorys)
    base = n_total // k
    rem = n_total % k
    targets = {c: base for c in categorys}

    for c in random.sample(categorys, rem):
        targets[c] += 1
    for c in categorys:
        targets[c] = min(targets[c], len(by_cat[c]))
    deficit = n_total - sum(targets.values())

    if deficit > 0:
        pool = [c for c in categorys if len(by_cat[c]) > targets[c]]
        while deficit > 0 and pool:
            c = random.choice(pool)
            if targets[c] < len(by_cat[c]):
                targets[c] += 1
                deficit -= 1
            else:
                pool.remove(c)
    return targets


def split_balanced_by_category(
    all_samples: List[Sample],
    train_size: int = 1000,
    test_size: int = 1000,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample]]:
    random.seed(seed)
    
    categorys = ['STEM', 'Social Sciences', 'Humanities', 'Other']
    by_cat: Dict[str, List[Sample]] = {}
    for s in all_samples:
        by_cat.setdefault(s.category, []).append(s)

    # sample train
    train_n_by_cat = _get_balanced_n(train_size, categorys, by_cat)

    train: List[Sample] = []
    remaining: Dict[str, List[Sample]] = {}
    for c in categorys:
        cand = by_cat[c][:]
        random.shuffle(cand)
        k = train_n_by_cat[c]
        train.extend(cand[:k])
        remaining[c] = cand[k:]

    # sample test
    test_n_by_cat = _get_balanced_n(test_size, categorys, remaining)

    test: List[Sample] = []
    for c in categorys:
        k = test_n_by_cat.get(c, 0)
        test.extend(remaining[c][:k])

    return train, test


def load_all_test_and_split(
    train_size: int = 1000,
    test_size: int = 1000,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample]]:
    ds = load_dataset("cais/mmlu", "all")["test"]
    samples = [norm_sample(x, SUBJECT_TO_CAT) for x in ds]
    return split_balanced_by_category(samples, train_size, test_size, seed)
