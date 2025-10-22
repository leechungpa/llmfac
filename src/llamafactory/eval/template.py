import re

from dataclasses import dataclass

from ..data import Role
from ..extras.constants import CHOICES


SYSTEM_PROMPT = (
    "You are solving the multiple-choice question. For each question:\n"
    "- Show your reasoning first, then give the final answer on a new line in this format: 'Answer: <choice>', where <choice> is one of A, B, C, or D.\n"
    "- If sample queries and answers are provided, study them first and match their reasoning style, structure, and level of detail in your responses.\n"
    "\n"
)

# SYSTEM_PROMPT_COUNT_ALPHABETS = (
#     "You are analyzing text statistics. For each question:\n"
#     "- Read the text following 'Question:'. Count only the alphabetic characters (A–Z, a–z) in that text.\n"
#     "- Give the final answer on a new line in this format: 'Answer: <number>', where <number> is the count of alphabetic characters.\n"
#     "- If sample queries and answers are provided, study them first and match their reasoning style, structure, and level of detail in your responses.\n"
#     "\n"
# )

SYSTEM_PROMPT_COUNT_WORDS = (
    "You are analyzing text statistics. For each question:\n"
    "- Read the text following 'Question:'. Count the number of words in that text.\n"
    "- Give the final answer on a new line in this format: 'Answer: <number>', where <number> is the count of words.\n"
    "- If sample queries and answers are provided, study them first and match their reasoning style, structure, and level of detail in your responses.\n"
    "\n"
)


def count_words(text):
    return len(re.sub(r'[^\w\s]', '', text).strip().split())

@dataclass
class EvalTemplate:
    name: str

    def __post_init__(self):
        self.system_prompt = SYSTEM_PROMPT if self.name == "en" else SYSTEM_PROMPT_COUNT_WORDS

    def _parse_example(self, example: dict[str, str]) -> tuple[str, str]:
        if self.name == "en":
            prompt = (
                "[Query]\n" # Change to [Query]
                + "Subject: " + example["subject"].replace("_", " ") + ".\n"
                + "Question: " + example["question"] + "\n"
                + "".join(f"{choice}) {example[choice]}\n" for choice in CHOICES if choice in example)
                + "\n"
            )
            response = example["answer"] # Chain-of-thougt answer
        elif self.name == "count_words":
            prompt = (
                "[Query]\n"
                + "Question: " + example["question"] + "\n"
                + "\n"
            )
            response = f"Answer: {count_words(example['question'])}"
        # elif self.name == "count_alphabets":
        #     prompt = (
        #         "[Query]\n"
        #         # + "Subject: " + example["subject"].replace("_", " ") + ".\n"
        #         + "Question: " + example["question"] + "\n"
        #         # + "".join(f"{choice}) {example[choice]}\n" for choice in CHOICES if choice in example)
        #         + "\n"
        #     )
        #     response = f"Answer: {len(re.findall(r'[A-Za-z]', example['question']))}"

        return prompt, response

    def format_example(
        self, target_data: dict[str, str], support_set: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        r"""Convert dataset examples to messages."""
        messages = []
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k])
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": ""})

        messages[0]["content"] = self.system_prompt + messages[0]["content"]
        return messages

def get_eval_template(name: str) -> "EvalTemplate":
    assert name in ['en', 'count_words', 'count_alphabets'], f"Template {name} does not exist."

    return EvalTemplate(name=name)