
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
# pip install deepspeed==0.13.2

python -c "import torch; print(torch.cuda.is_available())"

# pip install -U datasets openai google-generativeai
```

### Data Preparation

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can use datasets on HuggingFace / ModelScope / Modelers hub, load the dataset in local disk, or specify a path to s3/gcs cloud storage.

Please update `data/dataset_info.json` to use your custom dataset.


### MMLU Pipeline

1. split datasets
```bash
python3 src/mmlucot/download_dataset.py \
  --out_dir data/mmlu \
  --train_size 1000 \
  --test_size 1000 \
  --seed 0
```

2. generate chain-of-thought answer
```bash
export OPENAI_API_KEY= ######

python src/mmlucot/generate_cot_dataset.py \
  --org_path data/mmlu/train_n1000_seed0.jsonl \
  --out_path data/mmlu/train_n1000_seed0_cot.jsonl \
  --model gpt-4.1-mini

python src/mmlucot/generate_cot_dataset.py \
  --org_path data/mmlu/test_n1000_seed0.jsonl \
  --out_path data/mmlu/test_n1000_seed0_cot.jsonl \
  --model gpt-4.1-mini
```

3. generate zero-shot / few-shot JSONL
```bash
# train
for n in {0..5}; do
  python src/mmlucot/generate_fewshot_dataset.py \
    --org_path data/mmlu/train_n1000_seed0_cot.jsonl \
    --org_is_cot \
    --out_path "data/mmlu/cot/train_n1000_seed0_shot${n}.jsonl" \
    --n_shots ${n} \
    --seed 0
done

# test
python src/mmlucot/generate_fewshot_dataset.py \
  --org_path data/mmlu/test_n1000_seed0_cot.jsonl \
  --org_is_cot \
  --out_path data/mmlu/cot/test_n1000_seed0_shot0.jsonl \
  --n_shots 0 \
  --seed 0
```

4. modify jsonl for evaluation
```bash
python src/mmlucot/modify_jsonl_for_evaluation.py \
  --org_path data/mmlu/train_n1000_seed0_cot.jsonl \
  --output_dir train

python src/mmlucot/modify_jsonl_for_evaluation.py \
  --org_path data/mmlu/test_n1000_seed0_cot.jsonl \
  --output_dir test
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