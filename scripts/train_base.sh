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

model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-7B-Instruct"
epochs_list=(6 10)
ranks_list=(4 8)
declare -A targets=(
  ["q_proj,k_proj"]="qk"
  ["q_proj"]="q"
  ["k_proj"]="k"
  ["all"]="all"
)

for rank in "${ranks_list[@]}"; do
for epochs in "${epochs_list[@]}"; do
for target in "${!targets[@]}"; do

OUTDIR="./results/${model_name_or_path}/"
suffix="${targets[$target]}_r${rank}_epoch${epochs}"

# train
llamafactory-cli train scripts/train.yaml \
  model_name_or_path="$model_name_or_path" \
  lora_target="$target" \
  num_train_epochs="$epochs" \
  lora_rank="$rank" \
  output_dir="${OUTDIR}${suffix}" \
  dataset="mmlu_train_cot_s0"

# llamafactory-cli train scripts/train.yaml \
#   model_name_or_path="$model_name_or_path" \
#   lora_target="$target" \
#   num_train_epochs="$epochs" \
#   lora_rank="$rank" \
#   output_dir="${OUTDIR}${suffix}_s5" \
#   dataset="mmlu_train_cot_s5"

# eval
max_samples=100

for shot in 0 3 5; do
llamafactory-cli eval scripts/eval.yaml \
    model_name_or_path="$model_name_or_path" \
    adapter_name_or_path="${OUTDIR}${suffix}" \
    save_dir="${OUTDIR}${suffix}/eval_s${shot}" \
    n_shot=$shot \
    max_samples=$max_samples
done


done
done
done