username=$(whoami)

export HF_HOME="/data/${username}/hf"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

model_name="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"

# train
train_yaml="scripts/train.yaml"
epochs=1
rank=128
target="all" # "q_proj,k_proj,v_proj"
target_name="all" # "qkv"

category="STEM" # "Humanities"

# evaluation
eval_yaml="scripts/eval.yaml"
trainset_size=1000
testset_size=200
shots=(10 5 0)

# dataset
train_dataset="mmlu_train_cot_s0_${category}"
test_dataset="mmlucot_n${trainset_size}_n${testset_size}"

# Directories
MODEL_OUTDIR="/data/${username}/results/${category}"
EVAL_OUTDIR="/home/${username}/llmfac/results/${category}"
suffix="${target_name}_r${rank}_epoch${epochs}"


##################################################
# Train
llamafactory-cli train "$train_yaml" \
  model_name_or_path="$model_name" \
  num_train_epochs="$epochs" \
  lora_target="$target" \
  lora_rank="$rank" \
  output_dir="${MODEL_OUTDIR}/${suffix}" \
  dataset="$train_dataset"

##################################################
# Evaluation
mapfile -d '' ckpt_dirs < <(find "${MODEL_OUTDIR}/${suffix}" -maxdepth 1 -type d -name 'checkpoint*' -print0 | sort -z)

for shot in "${shots[@]}"; do
    llamafactory-cli eval "$eval_yaml" \
        model_name_or_path="$model_name" \
        task="${test_dataset}" \
        save_dir="${EVAL_OUTDIR}/${suffix}/log/checkpoint-0-n${testset_size}_s${shot}" \
        n_shot=$shot

    for ckpt in "${ckpt_dirs[@]}"; do
        ckpt="${ckpt%/}"
        ckpt_name="$(basename "$ckpt")"

        llamafactory-cli eval "$eval_yaml" \
            model_name_or_path="$model_name" \
            task="${test_dataset}" \
            adapter_name_or_path="${ckpt}" \
            save_dir="${EVAL_OUTDIR}/${suffix}/log/${ckpt_name}-n${testset_size}_s${shot}" \
            n_shot=$shot
        done
done

##################################################
# Summarize results
python src/summarize.py  \
  --base_dir "${EVAL_OUTDIR}/${suffix}/log" \
  --output_dir "${EVAL_OUTDIR}/${suffix}"
