# !/usr/bin/env bash

##################################################
# setting API keys
export HF_TOKEN="hf_"
export OPENAI_API_KEY="sk-"

##################################################
# HF cache location
export HF_HOME="/data/$(whoami)/hf"

##################################################
# NCCL flags
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

##################################################
# logging helpers
log_start() {
  local log_dir="${1:-./logs}"
  mkdir -p "$log_dir"

  log_file="${log_dir}/$(date +'%Y%m%d_%H%M%S').log"

  exec > >(tee -a "$log_file") 2>&1
  echo "[LOG] Started: $(date)"
  echo "[LOG] File: $log_file"
}

log_stop() {
  echo "[LOG] Stopped: $(date)"
  exec >&2
}

##################################################
# GPU allocation helper
check_and_set_gpu() {
  local REQ_GPUS=${1:-1}        # number of GPUs required (default: 1)
  local GPUS_TO_CHECK=(${2:-})  # user-specified GPU list, e.g., "0 1 3 5" (default: all GPUs)
  local WAIT_TIME=${3:-60}      # wait time between checks in seconds (default: 60)
  local STABILITY_DELAY=10 

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
  _gpu_is_free() {
    local g="$1"
    # 필요시 grep 패턴 수정
    if nvidia-smi -i "$g" | grep -qE "python"; then
      return 1  # 사용 중
    else
      return 0  # 비어 있음
    fi
  }

  while true; do
    FREE_GPUS=()
    for gpu in "${GPUS_TO_CHECK[@]}"; do
      if _gpu_is_free "$gpu"; then
        FREE_GPUS+=("$gpu")
      fi
    done

    if (( ${#FREE_GPUS[@]} >= REQ_GPUS )); then
      TARGET_GPUS=("${FREE_GPUS[@]:0:$REQ_GPUS}")
      echo "[GPU Manager] Candidates: ${TARGET_GPUS[*]} found free. Verifying in ${STABILITY_DELAY}s..."
      sleep "$STABILITY_DELAY"

      local STABLE_GPUS=()
      for gpu in "${TARGET_GPUS[@]}"; do
        if _gpu_is_free "$gpu"; then
          STABLE_GPUS+=("$gpu")
        fi
      done

      if (( ${#STABLE_GPUS[@]} >= REQ_GPUS )); then
        TARGET_GPUS=("${STABLE_GPUS[@]:0:$REQ_GPUS}")
        export CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${TARGET_GPUS[*]}")
        echo "[GPU Manager] Assigned GPU(s): $CUDA_VISIBLE_DEVICES"
        return 0
      else
        echo "[GPU Manager] Candidates not stable. Retrying in ${WAIT_TIME}s..."
        sleep "$WAIT_TIME"
        continue
      fi
    fi

    echo "[GPU Manager] Not enough free GPUs. Retrying in ${WAIT_TIME}s..."
    sleep $WAIT_TIME
  done
}
