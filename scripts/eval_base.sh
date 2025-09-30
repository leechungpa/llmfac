export HF_HOME=/data/cl/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

OUTDIR="results"
model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"

testset_size=100
shots=(0 5 10)

for shot in "${shots[@]}"; do
llamafactory-cli eval scripts/eval.yaml \
    task="mmlucot_n${testset_size}" \
    model_name_or_path=$model_name_or_path \
    save_dir="${OUTDIR}/eval/${model_name_or_path}/base/n${testset_size}_s${shot}" \
    n_shot=$shot
done
