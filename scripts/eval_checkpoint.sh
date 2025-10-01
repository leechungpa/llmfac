export HF_HOME=/data/cl/hf
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

OUTDIR="results"
model_name_or_path="Qwen/Qwen2.5-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"  "Qwen/Qwen2.5-7B-Instruct"
suffix="all_r128_epoch6"

testset_size=400
shots=(0 5 10)

mapfile -d '' ckpt_dirs < <(find "${OUTDIR}/${model_name_or_path}/${suffix}" -maxdepth 1 -type d -name 'checkpoint*' -print0 | sort -z)

for ckpt in "${ckpt_dirs[@]}"; do
  ckpt="${ckpt%/}"
  name="$(basename "$ckpt")" 
  echo ">> Evaluating: $name"

  for shot in "${shots[@]}"; do
    llamafactory-cli eval scripts/eval.yaml \
      task="mmlucot_n${testset_size}" \
      model_name_or_path="$model_name_or_path" \
      adapter_name_or_path="${ckpt}" \
      save_dir="${OUTDIR}_eval/${model_name_or_path}/${suffix}/${name}-n${testset_size}_s${shot}" \
      n_shot="$shot"
  done
done
