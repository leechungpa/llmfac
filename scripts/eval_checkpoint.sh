export HF_HOME=/data/cl/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=1

set -euo pipefail

max_samples=100
model_name_or_path="Qwen/Qwen2.5-7B-Instruct" # "Qwen/Qwen2.5-7B-Instruct"

OUTDIR="./results"
CKPT_ROOT="/home/cl/llmfac/results/Qwen/Qwen2.5-7B-Instruct/all_r4_epoch6"

shots=(0 5)

# Collect checkpoint directories
mapfile -d '' ckpt_dirs < <(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint*' -print0 | sort -z)

if [[ ${#ckpt_dirs[@]} -eq 0 ]]; then
  echo "No checkpoint directories found under $CKPT_ROOT"
  exit 1
fi

for ckpt in "${ckpt_dirs[@]}"; do
  ckpt="${ckpt%/}"
  name="$(basename "$ckpt")" 
  echo ">> Evaluating: $name"

  for shot in "${shots[@]}"; do
    llamafactory-cli eval scripts/eval.yaml \
      model_name_or_path="$model_name_or_path" \
      adapter_name_or_path="${ckpt}" \
      save_dir="${OUTDIR}/eval/${model_name_or_path}/${name}/n${max_samples}_s${shot}" \
      n_shot="$shot" \
      max_samples="$max_samples"
  done
done

