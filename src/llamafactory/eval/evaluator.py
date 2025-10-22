# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the Dan's test library.
# https://github.com/hendrycks/test/blob/master/evaluate_flan.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from typing import TYPE_CHECKING, Any, Optional
import re

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template, count_words

if TYPE_CHECKING:
    from numpy.typing import NDArray

VERBOSE = False
NUM_RETURN_SEQUENCES = 10

class Evaluator:
    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "left"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]


    def _parse_answer(self, text: str) -> str:
        try:
            for line in reversed(text.splitlines()):
                match = re.match(r"^\s*(?:Answer|Alphabets)\s*:\s*([A-D]|\d+)", line, flags=re.IGNORECASE)
                if match:
                    if self.eval_args.lang == "count_words":
                        return int(match.group(1))
                    else:
                        return match.group(1)
            return None
        except Exception:
            return None
        
    @torch.inference_mode()
    def batch_inference(self, batch_input: dict[str, "torch.Tensor"]) -> list[str]:
        gen_kwargs = {
            "tokenizer": self.tokenizer,
            "stop_strings": ["[Question]", "[Query]", "Answer: A", "Answer: B", "Answer: C", "Answer: D"],
            "repetition_penalty": 1.0,
            "temperature": self.eval_temperature,
            "top_k": 20,
            "top_p": 0.95,
            "num_return_sequences": NUM_RETURN_SEQUENCES,
            "do_sample": True,
        }

        batch_size = batch_input["input_ids"].shape[0]
        input_len = batch_input["input_ids"].shape[1]
        gen_kwargs["max_length"] = input_len+2000

        gen_ids = self.model.generate(**batch_input, **gen_kwargs)
        grouped = [
            gen_ids[i*gen_kwargs["num_return_sequences"]:(i+1)*gen_kwargs["num_return_sequences"]] for i in range(batch_size)
        ]
        texts = [
            [self.tokenizer.decode(ids[input_len:], skip_special_tokens=True) for ids in group]
            for group in grouped
        ]

        if VERBOSE:
            print("[----output----]")
            for gi, group in enumerate(texts):
                print(f"[sample {gi}]")
                for t in group:
                    print(t); print("---------")

        return [[self._parse_answer(t) for t in group] for group in texts]

    def eval(self) -> None:
        eval_task, train_split, eval_split, eval_temperature = self.eval_args.task.split("_")
        self.eval_temperature = int(eval_temperature[1:]) / 100

        test_dir = self.eval_args.task_dir

        mapping = cached_file(
            path_or_repo_id= f"{test_dir}/{eval_task}",
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, encoding="utf-8") as f:
            categorys: dict[str, dict[str, str]] = json.load(f)

        # if self.eval_args.lang == "count_words":
        #     categorys = {k: v for k, v in categorys.items() if v['category'] in ["Humanities", "Social Sciences", "Other"]}

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            dataset = load_dataset(
                "json",
                data_files={
                    "train": f"{test_dir}/{eval_task}/train/{train_split}/{subject}.jsonl",
                    "test": f"{test_dir}/{eval_task}/test/{eval_split}/{subject}.jsonl"
                },
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=self.model_args.trust_remote_code,
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset["test"]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset["test"][i],
                    support_set=support_set,
                )

                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                if self.eval_args.lang == "en":
                    labels.append(dataset["test"][i]['answer_idx'])
                elif self.eval_args.lang == "count_words":
                    labels.append(count_words(dataset['test'][i]['question']))
                # elif self.eval_args.lang == "count_alphabets":
                #     labels.append(len(re.findall(r'[A-Za-z]', dataset['test'][i]['question'])))

                if VERBOSE:
                    if i==0:
                        print("[----inputs----]")
                        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        print("".join(texts))
                        print("[----answer----]")
                        print(labels[i])
                        print("[--------------]")

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                pred_groups = self.batch_inference(batch_input)


                for j, preds in enumerate(pred_groups):
                    label = labels[i + j]
                    if self.eval_args.lang == "en":
                        acc = float(np.mean([p == label for p in preds]) > 0.5)
                    elif self.eval_args.lang == "count_words":
                        acc = float(np.mean([p == label for p in preds]))
                    outputs.append(acc)

                if VERBOSE:
                    print("[----acc per sample----]")
                    print(pred_groups, labels[i:i + self.eval_args.batch_size])
                    print(outputs[-len(pred_groups):])
                    print("[--------------]")

            corrects = np.array(outputs, dtype=float)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(self, category_corrects: dict[str, "NDArray"], results: dict[str, dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                f"{category_name:>15}: {100 * np.mean(category_correct):.2f}\n{category_name:>15}_std: {100 * np.std(category_correct):.2f}"
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def run_eval() -> None:
    Evaluator().eval()
