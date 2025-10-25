# !/usr/bin/env bash

export HF_HOME="/data/$(whoami)/hf"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


check_and_set_gpu() {
  local REQ_GPUS=${1:-1}        # number of GPUs required (default: 1)
  local GPUS_TO_CHECK=(${2:-})  # user-specified GPU list, e.g., "0 1 3 5" (default: all GPUs)
  local WAIT_TIME=${3:-60}      # wait time between checks in seconds (default: 60)

  local NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  local TARGET_GPUS=()
  local FREE_GPUS=()

  # validate GPU list
  if [ -n "${GPUS_TO_CHECK[*]}" ]; then
    for gpu in "${GPUS_TO_CHECK[@]}"; do
      if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[GPU Manager] Error: GPU IDs must be numbers (e.g., '0 1 2')."
        return 1
      fi
      if (( gpu < 0 || gpu >= NUM_GPUS )); then
        echo "[GPU Manager] Error: GPU ID $gpu out of range (0–$((NUM_GPUS-1)))."
        return 1
      fi
    done
  fi

  # default GPU list when GPUS_TO_CHECK=""
  if [ ${#GPUS_TO_CHECK[@]} -eq 0 ]; then
    GPUS_TO_CHECK=($(seq 0 $((NUM_GPUS-1))))
  fi

  # check and set gpu
  while true; do
    FREE_GPUS=()

    for gpu in "${GPUS_TO_CHECK[@]}"; do
      if ! nvidia-smi -i "$gpu" | grep -qE "python"; then
        FREE_GPUS+=("$gpu")
      fi
    done

    if (( ${#FREE_GPUS[@]} >= REQ_GPUS )); then
      TARGET_GPUS=("${FREE_GPUS[@]:0:$REQ_GPUS}")
      export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TARGET_GPUS[*]}")
      echo "[GPU Manager] Assigned GPU(s): $CUDA_VISIBLE_DEVICES"
      return 0
    fi

    echo "[GPU Manager] Not enough free GPUs. Retrying in ${WAIT_TIME}s..."
    sleep $WAIT_TIME
  done
}
