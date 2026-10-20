#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_distributed_train.sh --config CONFIG.yaml [--background] [--dry-run] [-- dotlist overrides...]

The YAML owns training configuration, including engine.nnodes. The launcher only
resolves runtime details:
  CUDA_VISIBLE_DEVICES  Select local GPUs; process count is auto-detected.
  PYTHON                Python executable. Default: python.
  CACHE_ROOT            Runtime cache/log root. Default: /tmp/sipai_runtime_cache.
  MASTER_PORT           torchrun rendezvous port. Default: 29500.
  DIST_BACKEND          Training distributed backend. Default: nccl.
  NCCL_IB_DISABLE       Override NCCL RDMA/IB transport. Default: auto
                        (enabled for multi-node hosts with active RDMA devices,
                        disabled otherwise).
  LOG_DIR               Training log directory. Default: CACHE_ROOT/logs.
  LOG_FILE              Per-node training log file. Default: LOG_DIR/sipai_train_<timestamp>_rank<node>.log.
  NODE_RANK             Explicit node rank. GROUP_RANK, MACHINE_RANK, and WORKER_RANK are aliases.
  RANK                  Scheduler rank. If WORLD_SIZE equals engine.nnodes it is treated as NODE_RANK;
                        otherwise global process rank requires LOCAL_WORLD_SIZE or WORLD_SIZE_IS_TOTAL_PROCESSES=1.
  MASTER_ADDR           Rendezvous host. CHIEF_IP and MASTER_IP are aliases.
  POD_NAMESPACE         Optional Kubernetes namespace used to expand service DNS candidates.
  SIPAI_OMP_NUM_THREADS CPU math threads per training process. Default: 1.
  WANDB_API_KEY         Optional W&B API key for cloud metric upload.

For nnodes > 1, run the same command on every allocated node. Schedulers such as
Slurm should provide node identity. Bare SSH launches must set NODE_RANK and
MASTER_ADDR in the environment.

Examples:
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/run_distributed_train.sh \
    --config configs/simate-beta_mem20_ae_80k.yaml

  bash tools/run_distributed_train.sh \
    --config configs/simate-beta_mem20_ae_80k.yaml \
    -- train_cfg.max_steps=1 train_cfg.save_final_checkpoint=false
USAGE
}

CONFIG=""
BACKGROUND=0
DRY_RUN="${DRY_RUN:-0}"
EXTRA_OPTS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="${2:?--config requires a path}"
      shift 2
      ;;
    --background)
      BACKGROUND=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OPTS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "${CONFIG}" ] || [ ! -f "${CONFIG}" ]; then
  echo "--config must point to an existing YAML file." >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/sipai_runtime_cache}"
MASTER_PORT="${MASTER_PORT:-29500}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

NNODES="$("${PYTHON}" - "${CONFIG}" <<'PY'
import sys
try:
    from sipai.config import load_sipai_config_file
except ModuleNotFoundError:
    print("SiPAI and OmegaConf are required; run from the repository environment or set PYTHON=/path/to/python.", file=sys.stderr)
    raise SystemExit(2)

cfg = load_sipai_config_file(sys.argv[1])
print((cfg.get("engine", {}) or {}).get("nnodes", 1))
PY
)"
if [ "${NNODES}" -lt 1 ]; then
  echo "engine.nnodes must be >= 1." >&2
  exit 2
fi

if [ -n "${NPROC_PER_NODE:-}" ] && [ "${NPROC_PER_NODE}" != "auto" ]; then
  :
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NPROC_PER_NODE="$(printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NPROC_PER_NODE="$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ' )"
else
  NPROC_PER_NODE=1
fi
[ -n "${NPROC_PER_NODE}" ] && [ "${NPROC_PER_NODE}" != "0" ] || NPROC_PER_NODE=1
case "${NPROC_PER_NODE}" in
  ''|*[!0-9]*|0) echo "NPROC_PER_NODE must be a positive integer, got ${NPROC_PER_NODE}." >&2; exit 2 ;;
esac

if [ "${NNODES}" = "1" ]; then
  NODE_RANK=0
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
else
  NODE_RANK="${NODE_RANK:-${GROUP_RANK:-${MACHINE_RANK:-${WORKER_RANK:-${SLURM_NODEID:-${OMPI_COMM_WORLD_RANK:-}}}}}}"
  if [ -z "${NODE_RANK}" ] && [ -n "${RANK:-}" ]; then
    case "${RANK}" in
      ''|*[!0-9]*) echo "RANK must be a non-negative integer, got ${RANK}." >&2; exit 2 ;;
    esac
    if [ -n "${WORLD_SIZE:-}" ]; then
      case "${WORLD_SIZE}" in
        ''|*[!0-9]*|0) echo "WORLD_SIZE must be a positive integer, got ${WORLD_SIZE}." >&2; exit 2 ;;
      esac
      if [ "${WORLD_SIZE}" = "${NNODES}" ]; then
        NODE_RANK="${RANK}"
      elif [ "${WORLD_SIZE}" = "$((NNODES * NPROC_PER_NODE))" ]; then
        RANK_DIVISOR="${NPROC_PER_NODE}"
      fi
    fi
    if [ -z "${NODE_RANK}" ] && [ -z "${RANK_DIVISOR:-}" ] && [ -n "${LOCAL_WORLD_SIZE:-}" ]; then
      RANK_DIVISOR="${LOCAL_WORLD_SIZE}"
    elif [ -z "${NODE_RANK}" ] && [ -z "${RANK_DIVISOR:-}" ] && [ "${WORLD_SIZE_IS_TOTAL_PROCESSES:-0}" = "1" ]; then
      RANK_DIVISOR="${NPROC_PER_NODE}"
    elif [ -z "${NODE_RANK}" ] && [ -z "${RANK_DIVISOR:-}" ] && [ "${RANK}" -lt "${NNODES}" ]; then
      NODE_RANK="${RANK}"
    elif [ -z "${NODE_RANK}" ] && [ -z "${RANK_DIVISOR:-}" ]; then
      echo "RANK=${RANK} cannot be mapped to NODE_RANK for nnodes=${NNODES}; set NODE_RANK, LOCAL_WORLD_SIZE, or WORLD_SIZE_IS_TOTAL_PROCESSES=1." >&2
      exit 2
    fi
    if [ -z "${NODE_RANK}" ]; then
      case "${RANK_DIVISOR}" in
        ''|*[!0-9]*|0) echo "Rank divisor must be a positive integer, got ${RANK_DIVISOR}." >&2; exit 2 ;;
      esac
      NODE_RANK="$((RANK / RANK_DIVISOR))"
    fi
  fi
  MASTER_ADDR="${CHIEF_IP:-${MASTER_IP:-${MASTER_ADDR:-}}}"
  if [ -z "${MASTER_ADDR}" ]; then
    if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
      MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
    else
      MASTER_ADDR="${SLURM_LAUNCH_NODE_IPADDR:-}"
    fi
  fi
  if [ -z "${NODE_RANK}" ] || [ -z "${MASTER_ADDR}" ]; then
    echo "nnodes=${NNODES} requires scheduler-provided rank/master environment variables." >&2
    exit 2
  fi
  case "${NODE_RANK}" in
    ''|*[!0-9]*) echo "NODE_RANK must be a non-negative integer, got ${NODE_RANK}." >&2; exit 2 ;;
  esac
  if [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "NODE_RANK=${NODE_RANK} must be smaller than nnodes=${NNODES}." >&2
    exit 2
  fi
fi

if [ "${NNODES}" != "1" ]; then
  MASTER_ADDR_CANDIDATES="${MASTER_ADDR}"
  case "${MASTER_ADDR}" in
    *.*) ;;
    *)
      POD_NAMESPACE="${POD_NAMESPACE:-${NAMESPACE:-${K8S_NAMESPACE:-}}}"
      if [ -n "${POD_NAMESPACE}" ]; then
        MASTER_ADDR_CANDIDATES="${MASTER_ADDR} ${MASTER_ADDR}.${POD_NAMESPACE}.svc ${MASTER_ADDR}.${POD_NAMESPACE}.svc.cluster.local"
      fi
      ;;
  esac
  if [ "${NODE_RANK}" = "0" ] && [ -n "${HOSTNAME:-}" ]; then
    MASTER_ADDR_CANDIDATES="${MASTER_ADDR_CANDIDATES} ${HOSTNAME}"
  fi
  case "${HOSTNAME:-}" in
    *-[0-9][0-9][0-9][0-9])
      MASTER_ADDR_CANDIDATES="${MASTER_ADDR_CANDIDATES} $(printf '%s' "${HOSTNAME}" | sed 's/-[0-9][0-9][0-9][0-9]$/-0000/')"
      ;;
  esac
  export MASTER_ADDR_CANDIDATES MASTER_PORT
  MASTER_ADDR="$("${PYTHON}" - <<'PY'
import os
import socket

port = int(os.environ["MASTER_PORT"])
for host in dict.fromkeys(os.environ["MASTER_ADDR_CANDIDATES"].split()):
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        continue
    print(host)
    break
else:
    raise SystemExit("No MASTER_ADDR candidate could be resolved: " + os.environ["MASTER_ADDR_CANDIDATES"])
PY
)"
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${CACHE_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sipai_train_${RUN_STAMP}_rank${NODE_RANK}.log}"
TORCHELASTIC_LOG_DIR="${TORCHELASTIC_LOG_DIR:-${CACHE_ROOT}/torchelastic}"
mkdir -p "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/inductor" "${LOG_DIR}" "${TORCHELASTIC_LOG_DIR}"

export PYTHONPATH="${PYTHONPATH:-.}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/inductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
export OMP_NUM_THREADS="${SIPAI_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIPAI_OMP_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${SIPAI_OMP_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${SIPAI_OMP_NUM_THREADS:-${OMP_NUM_THREADS}}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

has_active_rdma_device() {
  local hca state_file
  for hca in /sys/class/infiniband/*; do
    [ -e "${hca}" ] || continue
    for state_file in "${hca}"/ports/*/state; do
      [ -r "${state_file}" ] || continue
      if grep -q "ACTIVE" "${state_file}" 2>/dev/null; then
        return 0
      fi
    done
  done
  return 1
}

if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  for iface in bond0 eth0 eth1 ens3 ens4 enp1s0; do
    [ -d "/sys/class/net/${iface}" ] && export NCCL_SOCKET_IFNAME="${iface}" && break
  done
fi
if [ -z "${NCCL_IB_DISABLE:-}" ]; then
  if [ "${NNODES}" != "1" ] && has_active_rdma_device; then
    export NCCL_IB_DISABLE=0
  else
    export NCCL_IB_DISABLE=1
  fi
else
  export NCCL_IB_DISABLE
fi
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TRAIN_ARGS=(--config "${CONFIG}" --launcher torchrun --dist-backend "${DIST_BACKEND}")
[ "${#EXTRA_OPTS[@]}" -eq 0 ] || TRAIN_ARGS+=(--opts "${EXTRA_OPTS[@]}")

CMD=(
  "${PYTHON}" -m torch.distributed.run
  --nnodes="${NNODES}"
  --node-rank="${NODE_RANK}"
  --nproc-per-node="${NPROC_PER_NODE}"
  --master-addr="${MASTER_ADDR}"
  --master-port="${MASTER_PORT}"
  --log-dir "${TORCHELASTIC_LOG_DIR}"
  tools/train.py
  "${TRAIN_ARGS[@]}"
)

{
  echo "Config: ${CONFIG}"
  echo "Log: ${LOG_FILE}"
  echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} DIST_BACKEND=${DIST_BACKEND}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
  if [ -n "${WANDB_API_KEY}" ]; then
    echo "WANDB_API_KEY=<set>"
  else
    echo "WANDB_API_KEY=<unset>"
  fi
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
} | tee "${LOG_FILE}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "Dry run complete."
elif [ "${BACKGROUND}" = "1" ]; then
  nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  echo "Started SiPAI distributed training in background."
  echo "PID: $!"
  echo "Follow log: tail -f ${LOG_FILE}"
else
  if "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    :
  else
    statuses=("${PIPESTATUS[@]}")
    if [ "${statuses[0]}" -ne 0 ]; then
      exit "${statuses[0]}"
    fi
    exit "${statuses[1]}"
  fi
fi
