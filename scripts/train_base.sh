export HF_HOME=/data/cl/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

OUTDIR="results"
model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"

epochs_list=(6)
ranks_list=(128)

declare -A targets=(
  ["q_proj"]="q"
  ["k_proj"]="k"
  ["v_proj"]="v"
  # ["q_proj,v_proj"]="qv"
  # ["k_proj,v_proj"]="kv"
  ["q_proj,k_proj,v_proj"]="qkv"
  # ["all"]="all"
)

for rank in "${ranks_list[@]}"; do
for epochs in "${epochs_list[@]}"; do
for target in "${!targets[@]}"; do

suffix="${targets[$target]}_r${rank}_epoch${epochs}"

# train
llamafactory-cli train scripts/train.yaml \
  model_name_or_path="$model_name_or_path" \
  num_train_epochs="$epochs" \
  lora_target="$target" \
  lora_rank="$rank" \
  output_dir="${OUTDIR}/${model_name_or_path}/${suffix}" \
  dataset="mmlu_train_cot_s0"

# eval
testset_size=100
shots=(0 5 10)

for shot in "${shots[@]}"; do
llamafactory-cli eval scripts/eval.yaml \
    task="mmlucot_n${testset_size}" \
    model_name_or_path="$model_name_or_path" \
    adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}" \
    save_dir="${OUTDIR}_eval/${model_name_or_path}/${suffix}/eval_s${shot}" \
    n_shot=$shot
done


done
done
done