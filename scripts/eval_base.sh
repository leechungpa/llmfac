export HF_HOME=/data/cl/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0
# min_count=9999
# CUDA_VISIBLE_DEVICES=-1
# for id in 0 1; do
#   count=$(nvidia-smi -i $id | grep -c " C ")
#   echo "- GPU $id has $count processes"
#   if [ "$count" -lt "$min_count" ]; then
#     min_count=$count
#     export CUDA_VISIBLE_DEVICES=$id
#   fi
# done
# echo "Selected GPU: $CUDA_VISIBLE_DEVICES"

set -euo pipefail

OUTDIR="./saves"
model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-7B-Instruct"

max_samples=100


for shot in 0 3 5; do
llamafactory-cli eval scripts/eval.yaml \
    model_name_or_path=$model_name_or_path \
    save_dir="${OUTDIR}/${model_name_or_path}/base/n${max_samples}_s${shot}" \
    n_shot=$shot \
    max_samples=$max_samples
done
