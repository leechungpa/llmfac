# !/usr/bin/env bash


source scripts/env.sh

set -euo pipefail

##################################################
# model
model_name="Qwen/Qwen2.5-3B-Instruct" # 3B 7B 1.5B
# model_name="meta-llama/Llama-3.2-3B-Instruct"
# model_name="meta-llama/Llama-3.1-8B-Instruct"

##################################################
# train
train_yaml="scripts/train.yaml"
epochs=3
rank=128
lr=1.0e-5

# target="all"
# target_name="all"
target="q_proj,k_proj,v_proj"
target_name="qkv"
# target="v_proj"
# target_name="v"
# target="q_proj"
# target_name="q"
# target="k_proj"
# target_name="k"

# category="STEM"
category="Humanities"

##################################################
# evaluation
eval_yaml="scripts/eval.yaml"
trainset_size=1000
testset_size=1000
temperature=0.05
eval_seed=0

shots="5 0 3 1 7"

calc_eval_batch_size() {
  case "$1" in
    0) echo 10 ;;
    1) echo 8 ;;
    3) echo 3 ;;
    *) echo 1 ;;
  esac
}

##################################################
# dataset
train_dataset="mmlucot_train_n${trainset_size}_s0_${category}"
test_dataset="mmlucot_n${trainset_size}_n${testset_size}_t${temperature}"

# directories
suffix="${model_name}/${category}/epoch${epochs}_lr${lr}_${target_name}_r${rank}"

base_eval_suffix="${model_name}/base"

username=$(whoami)

model_dir="/data/${username}/llmfac/results"
eval_dir="/home/${username}/llmfac/results_eval"

##################################################
# Train
check_and_set_gpu

llamafactory-cli train "$train_yaml" \
  model_name_or_path="$model_name" \
  num_train_epochs="$epochs" \
  learning_rate="$lr" \
  lora_target="$target" \
  lora_rank="$rank"  \
  output_dir="${model_dir}/${suffix}" \
  dataset="$train_dataset" \
  report_to="wandb" \
  run_name="${suffix}" \
  eval_dataset="mmlucot_val_s0_${category}" \
  per_device_eval_batch_size="4" \
  eval_strategy="steps" \
  eval_steps="1"

sleep 10

##################################################
# Evaluation
for shot in $shots; do
  eval_suffix="t${temperature}_n${testset_size}_s${shot}_seed${eval_seed}"

  # Evaluate the base model
  check_and_set_gpu

  llamafactory-cli eval "$eval_yaml" \
    model_name_or_path="$model_name" \
    task="${test_dataset}" \
    save_dir="${eval_dir}/${base_eval_suffix}/log/checkpoint-0_${eval_suffix}" \
    n_shot=$shot \
    seed=$eval_seed \
    batch_size="$(calc_eval_batch_size "$shot")"

  # Evaluate fintuned models
  check_and_set_gpu

  llamafactory-cli eval "$eval_yaml" \
    model_name_or_path="$model_name" \
    task="${test_dataset}" \
    adapter_name_or_path="${model_dir}/${suffix}" \
    save_dir="${eval_dir}/${suffix}/log/checkpoint-${epochs}00_${eval_suffix}" \
    n_shot=$shot \
    seed=$eval_seed \
    batch_size="$(calc_eval_batch_size "$shot")"
done

sleep 10

mapfile -d '' ckpt_dirs < <(find "${model_dir}/${suffix}" -maxdepth 1 -type d -name 'checkpoint*' -print0 | sort -z)

for shot in $shots; do
  for ckpt in "${ckpt_dirs[@]}"; do
    ckpt="${ckpt%/}"
    ckpt_name="$(basename "$ckpt")"

    eval_suffix="t${temperature}_n${testset_size}_s${shot}_seed${eval_seed}"

    # Evaluate checkpoints
    check_and_set_gpu

    llamafactory-cli eval "$eval_yaml" \
      model_name_or_path="$model_name" \
      task="${test_dataset}" \
      adapter_name_or_path="${ckpt}" \
      save_dir="${eval_dir}/${suffix}/log/${ckpt_name}_${eval_suffix}" \
      n_shot=$shot \
      seed=$eval_seed \
      batch_size="$(calc_eval_batch_size "$shot")"
  done
done

sleep 10

##################################################
# Summarize results

python src/utils/summarize.py  \
  --base_dir "${eval_dir}/${base_eval_suffix}/log" "${eval_dir}/${suffix}/log" \
  --output_dir "${eval_dir}/${suffix}"
