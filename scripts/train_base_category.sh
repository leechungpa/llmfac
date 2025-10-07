export HF_HOME=/data/leechungpa/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail


# train
epochs_list=(6 10)
ranks_list=(128 64)
declare -A targets=(
  ["q_proj"]="q"
  ["k_proj"]="k"
  ["v_proj"]="v"
  ["q_proj,v_proj"]="qv"
  ["k_proj,v_proj"]="kv"
  ["q_proj,k_proj,v_proj"]="qkv"
  ["all"]="all"
)
train_dataset_shot=0
category="STEM"

# evaluation
trainset_size=1000
testset_size=200
shots=(10 5 0)


# model and path
model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"
OUTDIR="/data/leechungpa/results/${category}"
EVAL_OUTDIR="results_eval/${category}"

for rank in "${ranks_list[@]}"; do
for epochs in "${epochs_list[@]}"; do
for target in "${!targets[@]}"; do

suffix="${targets[$target]}_r${rank}_epoch${epochs}_trains${train_dataset_shot}"


llamafactory-cli train scripts/train.yaml \
  model_name_or_path="$model_name_or_path" \
  num_train_epochs="$epochs" \
  lora_target="$target" \
  lora_rank="$rank" \
  output_dir="${OUTDIR}/${model_name_or_path}/${suffix}" \
  dataset="mmlu_train_cot_s${train_dataset_shot}_${category}"

for shot in "${shots[@]}"; do
llamafactory-cli eval scripts/eval.yaml \
    task="mmlucot_n${trainset_size}_n${testset_size}" \
    model_name_or_path="$model_name_or_path" \
    adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}" \
    save_dir="${EVAL_OUTDIR}/${model_name_or_path}/${suffix}/n${testset_size}_s${shot}" \
    n_shot=$shot
done

done
done
done