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
ranks_list=(4 8)
declare -A targets=(
  ["q_proj,k_proj"]="qk"
  ["q_proj"]="q"
  ["k_proj"]="k"
  ["all"]="all"
)

max_samples=100

for rank in "${ranks_list[@]}"; do
for target in "${!targets[@]}"; do

OUTDIR="./results/${model_name_or_path}/"

# Reduce the number of shots from 5 to 0 over 6 epochs
suffix="reduce-shot_${targets[$target]}_r${rank}"

llamafactory-cli train scripts/train.yaml \
  model_name_or_path="$model_name_or_path" \
  lora_target="$target" \
  num_train_epochs=1 \
  lora_rank="$rank" \
  output_dir="${OUTDIR}${suffix}_epoch1" \
  dataset="mmlu_train_cot_s5"

for shot in 0 3 5; do
llamafactory-cli eval scripts/eval.yaml \
    adapter_name_or_path="${OUTDIR}${suffix}_epoch1" \
    save_dir="${OUTDIR}${suffix}_epoch1/eval_s${shot}" \
    n_shot=$shot \
    max_samples=$max_samples
done


for i in 2 3 4 5 6; do
  prev=$((i-1))
  next=$((i))
  nshot=$((6-i))
  
  llamafactory-cli train scripts/train.yaml \
    model_name_or_path="$model_name_or_path" \
    adapter_name_or_path="${OUTDIR}${suffix}_epoch${prev}" \
    lora_target="$target" \
    num_train_epochs=1 \
    lora_rank="$rank" \
    output_dir="${OUTDIR}${suffix}_epoch${next}" \
    dataset="mmlu_train_cot_s${nshot}"

  for shot in 0 3 5; do
  llamafactory-cli eval scripts/eval.yaml \
      adapter_name_or_path="${OUTDIR}${suffix}_epoch${next}" \
      save_dir="${OUTDIR}${suffix}_epoch${next}/eval_s${shot}" \
      n_shot=$shot \
      max_samples=$max_samples
  done
done

# Reduce the weight of 5 shots over 6 epochs
suffix="reduce-weight_${targets[$target]}_r${rank}"

llamafactory-cli train scripts/train.yaml \
  model_name_or_path="$model_name_or_path" \
  lora_target="$target" \
  num_train_epochs=1 \
  lora_rank="$rank" \
  output_dir="${OUTDIR}${suffix}_epoch1" \
  dataset="mmlu_train_cot_s5,mmlu_train_cot_s0" \
  interleave_probs="0.5,0.5" \
  mix_strategy="interleave_under"

for shot in 0 3 5; do
llamafactory-cli eval scripts/eval.yaml \
    adapter_name_or_path="${OUTDIR}${suffix}_epoch1" \
    save_dir="${OUTDIR}${suffix}_epoch1/eval_s${shot}" \
    n_shot=$shot \
    max_samples=$max_samples
done

for i in 2 3 4 5 6; do
  prev=$((i-1))
  next=$((i))
  prob_s5=$(awk "BEGIN {print (6-$i)/10}")   # 0.4 → 0.9
  prob_s0=$(awk "BEGIN {print (4+$i)/10}")   # 0.6 → 1.0

  llamafactory-cli train scripts/train.yaml \
    model_name_or_path="$model_name_or_path" \
    adapter_name_or_path="${OUTDIR}${suffix}_epoch${prev}" \
    lora_target="$target" \
    num_train_epochs=1 \
    lora_rank="$rank" \
    output_dir="${OUTDIR}${suffix}_epoch${next}" \
    dataset="mmlu_train_cot_s5,mmlu_train_cot_s0" \
    interleave_probs="${prob_s5},${prob_s0}" \
    mix_strategy="interleave_under"

  for shot in 0 3 5; do
  llamafactory-cli eval scripts/eval.yaml \
      adapter_name_or_path="${OUTDIR}${suffix}_epoch${next}" \
      save_dir="${OUTDIR}${suffix}_epoch${next}/eval_s${shot}" \
      n_shot=$shot \
      max_samples=$max_samples
  done
done

done
done