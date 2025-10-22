
### Installation

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 2.0.0   | 2.6.0     |
| torchvision  | 0.15.0  | 0.21.0    |
| transformers | 4.49.0  | 4.50.0    |
| datasets     | 2.16.0  | 3.2.0     |
| accelerate   | 0.34.0  | 1.2.1     |
| peft         | 0.14.0  | 0.15.1    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.8.2     |
| flash-attn   | 2.5.6   | 2.7.2     |


```bash
conda create -n llmfac python=3.10.0
conda activate llmfac

pip install -e ".[torch,metrics]" --no-build-isolation
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print(torch.cuda.is_available())"
```

### MMLU Pipeline

0. set the seed
```bash
seed=0
```

1. split datasets
```bash
python3 src/mmlucot/download_dataset.py \
  --out_dir data/mmlu \
  --train_size 10000 \
  --test_size 5000 \
  --seed $seed
```

2. generate chain-of-thought answer
```bash
export OPENAI_API_KEY=sk-####

python src/mmlucot/generate_cot_dataset.py \
  --org_path "data/mmlu/train_n10000_seed${seed}.jsonl" \
  --out_path "data/mmlu/train_n10000_seed${seed}_cot.jsonl" \
  --model gpt-4.1-mini
```

3. generate datasets
```bash
# 3-1. Train dataset
for n in 1000 4000 8000; do
python src/mmlucot/subset_jsonl.py \
  --org_path "data/mmlu/train_n9311_seed${seed}_cot.jsonl" \
  --out_path "data/mmlu/train_n${n}_seed${seed}_cot.jsonl" \
  --subset_n ${n} \
  --seed $seed
done

for n in 0 1 3 5 7 10; do
python src/mmlucot/generate_fewshot_dataset.py \
  --org_path "data/mmlu/train_n1000_seed${seed}_cot.jsonl" \
  --org_is_cot \
  --out_path "data/mmlu/cot/train_n1000_seed${seed}_cot_shot${n}.jsonl" \
  --n_shots ${n} \
  --seed $seed
for cat in "STEM" "Social Sciences" "Humanities" "Other" ; do
python src/mmlucot/generate_fewshot_dataset.py \
  --org_path "data/mmlu/train_n4000_seed${seed}_cot.jsonl" \
  --org_is_cot \
  --out_path "data/mmlu/cot/train_n4000_seed${seed}_cot_shot${n}_${cat}.jsonl" \
  --n_shots ${n} \
  --subset_category "${cat}" \
  --seed $seed
done
done

# 3-2. Validation dataset
for cat in "STEM" "Social Sciences" "Humanities" "Other" ; do
python src/mmlucot/generate_fewshot_dataset.py \
  --org_path "data/mmlu/train_n9311_seed${seed}_cot.jsonl" \
  --org_is_cot \
  --out_path "data/mmlu/cot/validation_cot_shot0_${cat}.jsonl" \
  --n_shots 0 \
  --subset_category "${cat}" \
  --seed $seed
done

# 3-3. Test dataset
# shot examples from train dataset
for n in 1000 4000; do
  python src/mmlucot/modify_jsonl_for_evaluation.py \
    --org_path "data/mmlu/train_n${n}_seed0_cot.jsonl" \
    --output_dir "evaluation/mmlucot/train/n${n}"
done

# test datset
for n in 100 200 500 1000 2000; do
  python src/mmlucot/subset_jsonl.py \
    --org_path "data/mmlu/test_n2328_seed${seed}.jsonl" \
    --out_path "data/mmlu/test_n${n}_seed${seed}.jsonl" \
    --subset_n "$n" \
  --seed $seed

  python src/mmlucot/modify_jsonl_for_evaluation.py \
    --org_path "data/mmlu/test_n${n}_seed${seed}.jsonl" \
    --output_dir "evaluation/mmlucot/test/n${n}"
done
```


## Forked from:

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```