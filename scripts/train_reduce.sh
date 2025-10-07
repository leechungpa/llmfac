export HF_HOME=/data/leechungpa/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail


# train
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
train_dataset_shot=5

# evaluation
trainset_size=1000
testset_size=200
shots=(10 5 0)

# model and path
OUTDIR="/data/leechungpa/results"
EVAL_OUTDIR="results_eval"
model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"



for rank in "${ranks_list[@]}"; do
for target in "${!targets[@]}"; do

  # Reduce the weight of 5 shots over 6 epochs
  suffix="reduce-weight_${targets[$target]}_r${rank}"

  llamafactory-cli train scripts/train.yaml \
    model_name_or_path="$model_name_or_path" \
    num_train_epochs=1 \
    lora_target="$target" \
    lora_rank="$rank" \
    output_dir="${OUTDIR}/${model_name_or_path}/${suffix}_epoch1" \
    dataset="mmlu_train_cot_s5,mmlu_train_cot_s0" \
    interleave_probs="0.5,0.5" \
    mix_strategy="interleave_under"

  for shot in "${shots[@]}"; do
  llamafactory-cli eval scripts/eval.yaml \
      task="mmlucot_n${trainset_size}_n${testset_size}" \
      model_name_or_path="$model_name_or_path" \
      adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}_epoch1" \
      save_dir="${OUTDIR}_eval/${model_name_or_path}/${suffix}_epoch1/n${testset_size}_s${shot}" \
      n_shot=$shot
  done

  for i in 2 3 4 5 6; do
    prev=$((i-1))
    next=$((i))
    prob_s5=$(awk "BEGIN {print (6-$i)/10}")   # 0.4 → 0.9
    prob_s0=$(awk "BEGIN {print (4+$i)/10}")   # 0.6 → 1.0

    llamafactory-cli train scripts/train.yaml \
      model_name_or_path="$model_name_or_path" \
      adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}_epoch${prev}" \
      num_train_epochs=1 \
      lora_target="$target" \
      lora_rank="$rank" \
      output_dir="${OUTDIR}/${model_name_or_path}/${suffix}_epoch${next}" \
      dataset="mmlu_train_cot_s5,mmlu_train_cot_s0" \
      interleave_probs="${prob_s5},${prob_s0}" \
      mix_strategy="interleave_under"

    for shot in "${shots[@]}"; do
    llamafactory-cli eval scripts/eval.yaml \
        task="mmlucot_n${trainset_size}_n${testset_size}" \
        model_name_or_path="$model_name_or_path" \
        adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}_epoch${next}" \
        save_dir="${EVAL_OUTDIR}/${model_name_or_path}/${suffix}_epoch${next}/n${testset_size}_s${shot}" \
        n_shot=$shot
    done
  done

  # Reduce the number of shots from 5 to 0 over 6 epochs
  suffix="reduce-shot_${targets[$target]}_r${rank}"

  llamafactory-cli train scripts/train.yaml \
    model_name_or_path="$model_name_or_path" \
    num_train_epochs=1 \
    lora_target="$target" \
    lora_rank="$rank" \
    output_dir="${OUTDIR}/${model_name_or_path}/${suffix}_epoch1" \
    dataset="mmlu_train_cot_s5"

  for shot in "${shots[@]}"; do
  llamafactory-cli eval scripts/eval.yaml \
      task="mmlucot_n${trainset_size}_n${testset_size}" \
      model_name_or_path="$model_name_or_path" \
      adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}_epoch1" \
      save_dir="${EVAL_OUTDIR}/${model_name_or_path}/${suffix}_epoch1/n${testset_size}_s${shot}" \
      n_shot=$shot
  done

  for i in 2 3 4 5 6; do
    prev=$((i-1))
    next=$((i))
    nshot=$((6-i))
    
    llamafactory-cli train scripts/train.yaml \
      model_name_or_path="$model_name_or_path" \
      adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}_epoch${prev}" \
      num_train_epochs=1 \
      lora_target="$target" \
      lora_rank="$rank" \
      output_dir="${OUTDIR}/${model_name_or_path}/${suffix}_epoch${next}" \
      dataset="mmlu_train_cot_s${nshot}"

    for shot in "${shots[@]}"; do
    llamafactory-cli eval scripts/eval.yaml \
        task="mmlucot_n${trainset_size}_n${testset_size}" \
        model_name_or_path="$model_name_or_path" \
        adapter_name_or_path="${OUTDIR}/${model_name_or_path}/${suffix}_epoch${next}" \
        save_dir="${EVAL_OUTDIR}/${model_name_or_path}/${suffix}_epoch${next}/n${testset_size}_s${shot}" \
        n_shot=$shot
    done
  done

done
done