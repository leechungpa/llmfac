# !/usr/bin/env bash

set -euo pipefail

source scripts/env.sh
log_start

##################################################
# model
model_name="Qwen/Qwen2.5-3B-Instruct" # 3B 7B 1.5B
# model_name="meta-llama/Llama-3.2-3B-Instruct"
# model_name="meta-llama/Llama-3.1-8B-Instruct"

##################################################
# train
train_yaml="scripts/train.yaml"
rank=128
lr=1.0e-5

epochs=5

declare -A interleave_probs

for ((i=1; i<=epochs; i++)); do
  p1=$(awk -v i=$i -v n=$epochs 'BEGIN{printf "%.3f", 0.5 + (i-1)*(0.5/(n-1))}')
  p2=$(awk -v p1=$p1 'BEGIN{printf "%.3f", 1-p1}')
  interleave_probs[$i]="$p1,$p2"
done


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
train_dataset="mmlucot_train_n${trainset_size}_s0_${category},mmlucot_train_n${trainset_size}_s5_${category}"
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

i=1

llamafactory-cli train "$train_yaml" \
  model_name_or_path="$model_name" \
  num_train_epochs="1" \
  learning_rate="$lr" \
  lora_target="$target" \
  lora_rank="$rank" \
  output_dir="${model_dir}/${suffix}/epoch${i}" \
  dataset="$train_dataset" \
  interleave_probs="${interleave_probs[$i]}" \
  mix_strategy="interleave_under" \
  report_to="wandb" \
  run_name="${suffix}"

for i in $(seq 2 $epochs); do
  llamafactory-cli train "$train_yaml" \
    model_name_or_path="$model_name" \
    adapter_name_or_path="${model_dir}/${suffix}/epoch$((i-1))" \
    num_train_epochs="1" \
    learning_rate="$lr" \
    lora_target="$target" \
    lora_rank="$rank" \
    output_dir="${model_dir}/${suffix}/epoch${i}" \
    dataset="$train_dataset" \
    interleave_probs="${interleave_probs[$i]}" \
    mix_strategy="interleave_under" \
    report_to="wandb" \
    run_name="${suffix}"
done

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
    adapter_name_or_path="${model_dir}/${suffix}/epoch${epochs}" \
    save_dir="${eval_dir}/${suffix}/log/checkpoint-${epochs}00_${eval_suffix}" \
    n_shot=$shot \
    seed=$eval_seed \
    batch_size="$(calc_eval_batch_size "$shot")"
done

##################################################
# Summarize results

sleep 5

python src/utils/summarize.py  \
  --base_dir "${eval_dir}/${base_eval_suffix}/log" "${eval_dir}/${suffix}/log" \
  --output_dir "${eval_dir}/${suffix}"

log_stop