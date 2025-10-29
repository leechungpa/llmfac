#!/usr/bin/env bash

model_name="Qwen/Qwen2.5-3B-Instruct"
category="Humanities"

eval_dir="/home/$(whoami)/llmfac/results_eval/${model_name}"


##################################################
# Summarize the main models.
eval_dir_base="${eval_dir}/base"
base_model="${eval_dir}/${category}/qkv_r128_epoch5_lr1.0e-5"
compare_model="${eval_dir}/${category}/reduce_qkv_r128_epoch5_lr1.0e-5"
plot_output="/home/$(whoami)/llmfac/results_eval/main_plots"

python src/utils/summarize.py  \
  --base_dir "${eval_dir_base}/log" "${base_model}/log" \
  --compare_dir "${eval_dir_base}/log" "${compare_model}/log" \
  --output_dir $plot_output \
  --shots 0 3 5

##################################################
# Summarize MMLU evaluation results for each finetuned run.
eval_dir_base="${eval_dir}/base"
eval_dir_finetuned="${eval_dir}/${category}"

for eval_finetuned in "${eval_dir_finetuned}"/*; do
  if [ -d "$eval_finetuned" ]; then
    echo "Processing $eval_finetuned..."
    python src/utils/summarize.py \
      --base_dir "${eval_dir_base}/log" "${eval_finetuned}/log" \
      --output_dir "${eval_finetuned}"
  fi
done

##################################################
# Summarize word counting results for each finetuned run.
eval_dir_base="${eval_dir}/counts/base"
eval_dir_finetuned="${eval_dir}/counts/${category}"

for eval_finetuned in "${eval_dir_finetuned}"/*; do
  if [ -d "$eval_finetuned" ]; then
    echo "Processing $eval_finetuned..."
    python src/utils/summarize.py \
      --base_dir "${eval_dir_base}/log" "${eval_finetuned}/log" \
      --output_dir "${eval_finetuned}" \
      --shots 1 3 5 7 9 10
  fi
done