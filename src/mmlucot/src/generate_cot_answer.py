import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # 추가

from . import LETTER, SYSTEM_PROMPT_BASE
from .datasets import Sample
from .formatting import render_qa

from openai import OpenAI


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    last_line = text.strip().splitlines()[-1]

    m = re.search(r'(?i)answer\s*:\s*([A-D])\b', last_line)
    if m:
        return m.group(1)
    m2 = re.search(r'(?i)answer\s*:\s*(.+)$', last_line, flags=re.MULTILINE)
    return m2.group(1).strip() if m2 else ""


def ask_llm(
    system_prompt,
    user_prompt,
    model_name:str = "gpt-4.1-mini",
    temperature:float = 0.7,
    max_retries:int = 5,
    verbose:bool = False,
):
    model_prefix = model_name.lower().split("-")[0]

    if model_prefix == "gpt":
        api_base = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
    elif model_prefix == "gemini":
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai"
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = model_name.split(":", 1)
    else:
        raise ValueError(f"Unsupported model prefix '{model_prefix}'. Please specify a model starting with 'gpt-' or 'gemini-'.")

    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    api_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "timeout": 180.0,
    }
    content = ""

    for attempt in range(max_retries):
        if attempt > 0 and verbose:
            print(f"attempt {attempt}", flush=True)

        try:
            response = client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content
            break
        except Exception as e:
            if verbose:
                print(f"[{attempt}/{max_retries}] LLM did not respond: {e}", flush=True)
            time.sleep(2)

    if verbose:
        print("┌──LLM input───┐")
        print(messages[1]["content"])
        print("├──LLM output──┤")
        print(content)
        print("└──────────────┘")

    return content


def generate_cot_answer(
    sample: Sample,
    model_name: str = "gpt-4.1-mini",
    rate_limit_per_sec: float = 1.0,
    max_retries:int = 5,
    verbose: bool = False,
    return_log: bool = True,
):
    delay = 1.0 / max(rate_limit_per_sec, 1e-6)

    user_prompt = render_qa(sample, with_answer=False)
    true_answer = LETTER[sample.answer_idx]

    for attempt in range(max_retries):
        llm_output = ask_llm(
            SYSTEM_PROMPT_BASE,
            user_prompt,
            model_name=model_name,
            verbose=verbose,
        )
        time.sleep(delay)

        pred = _extract_answer(llm_output)
        if pred == true_answer:
            failure_case = None
            break
        if verbose:
            print(f"[{attempt+1}try] pred({pred})!=true({true_answer})")
    else:
        failure_case = f"|Failed| p:{pred} != t:{true_answer}\n - Input: {user_prompt}\n - Output: {llm_output}"
        if verbose:
            print(failure_case)

    lines = llm_output.strip().splitlines()
    llm_output = "\n".join(lines[:-1] + [f"Answer: {true_answer}"])

    if return_log:
        return llm_output, failure_case
    else:
        return llm_output


# 병렬 처리 추가: samples 리스트를 받아 동시에 실행
def generate_cot_answers(
    samples: list[Sample],
    model_name: str = "gpt-4.1-mini",
    rate_limit_per_sec: float = 8.0,   # 워커 수 추정에 사용. 계정 한도에 맞게 조정
    max_workers: int | None = None,    # 명시 시 우선
    max_retries: int = 5,
    verbose: bool = False,
    return_log: bool = True,
):
    workers = max_workers or max(1, int(rate_limit_per_sec))

    def _one(s: Sample):
        # 기존 단일 함수 재사용. 딜레이가 있으니 rate_limit_per_sec를 충분히 크게 설정.
        return generate_cot_answer(
            s,
            model_name=model_name,
            rate_limit_per_sec=rate_limit_per_sec,
            max_retries=max_retries,
            verbose=verbose,
            return_log=return_log,
        )

    results = [None] * len(samples)
    failures = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2idx = {ex.submit(_one, s): i for i, s in enumerate(samples)}
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            out, fail = fut.result()
            results[i] = out
            if fail:
                failures.append((i, fail))

    return results, failures
