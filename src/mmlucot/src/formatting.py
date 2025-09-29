import random
from typing import List, Dict, Iterable

from . import LETTER, SYSTEM_PROMPT
from .datasets import Sample, SampleCoT


def render_qa(s: Sample|SampleCoT, with_answer: bool = True, with_cot_answer: bool = False) -> str:
    lines = [f"Subject: {s.subject}", f"Question: {s.question}"]
    for i, c in enumerate(s.choices):
        lines.append(f"{LETTER[i]}) {c}")
    if with_answer:
        if with_cot_answer and isinstance(s, SampleCoT):
            lines.append(s.cot_answer)
        else:
            lines.append(f"Answer: {LETTER[s.answer_idx]}")
    return "\n".join(lines)


def make_record(
    target: Sample|SampleCoT,
    shots: List[Sample|SampleCoT],
    with_cot_answer: bool,
    system_msg: str,
    instruction_msg: str,
) -> Dict:
    input_msg_parts = []
    for ex in shots:
        input_msg_parts.append(f"[Question]\n" + render_qa(ex, with_answer=True, with_cot_answer=with_cot_answer))
    input_msg_parts.append("[Question]\n" + render_qa(target, with_answer=False, with_cot_answer=with_cot_answer))
    input_msg = "\n\n".join(input_msg_parts)

    if with_cot_answer:
        output_msg = target.cot_answer
    else:
        output_msg = f"Answer: {LETTER[target.answer_idx]}"

    return {
        "system": system_msg,
        "instruction": instruction_msg,
        "input": input_msg,
        "output": output_msg,
    }


def make_jsonl(
    dataset: List[Sample | SampleCoT],
    with_cot_answer: bool,
    n_shots: int,
    seed: int = 0,
) -> Iterable[Dict]:
    system_msg, instruction_msg = SYSTEM_PROMPT.split("\n", 1)
    random.seed(seed)
    for t in dataset:
        shots: List[Sample | SampleCoT] = []
        if n_shots > 0:
            # # same subject shots
            # same_subject = [s for s in dataset if s.subject == t.subject and s.question != t.question]
            # pool = same_subject if same_subject else [s for s in dataset if s.question != t.question]
            # same category shots
            same_category = [s for s in dataset if s.category == t.category and s.question != t.question]
            pool = same_category if same_category else [s for s in dataset if s.question != t.question]
            random.shuffle(pool)
            shots = pool[: min(n_shots, len(pool))]

        yield make_record(t, shots, with_cot_answer, system_msg, instruction_msg)