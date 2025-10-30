#!/usr/bin/env bash
# Day 13 — Automation wrapper for the baseline model
# Usage examples:
#   ./scripts/run_baseline.sh
#   ./scripts/run_baseline.sh -i data/feature_engineered.csv -s 42,1337 -k 5
#   ./scripts/run_baseline.sh -i data/feature_engineered.csv -l label_proxy

set -Eeuo pipefail

IN_CSV="data/feature_engineered.csv"
OUT_ROOT="runs"
SEEDS="42"           # comma-separated, e.g., "42,1337,2025"
K="5"                # CV folds
PYTHON="python3"
LABEL_COL=""         # optional label col (empty means let trainer create proxy)
REQS="requirements.txt"

show_help() {
  cat <<EOF
run_baseline.sh — automate training of the Day-12 baseline

Options:
  -i <path>   Input CSV (default: ${IN_CSV})
  -o <dir>    Root output directory (default: ${OUT_ROOT})
  -s <seeds>  Comma-separated seeds (default: ${SEEDS})
  -k <int>    Stratified K-folds for CV (default: ${K})
  -p <py>     Python interpreter (default: ${PYTHON})
  -l <col>    Label column name (optional; if omitted trainer may build proxy)
  -r <file>   Requirements file to install if needed (default: ${REQS})
  -h          Show this help and exit

It expects scripts/train_logreg.py to accept:
  --in_csv PATH --out_dir DIR --seed INT --cv INT [--label_col NAME]

EOF
}

while getopts ":i:o:s:k:p:l:r:h" opt; do
  case $opt in
    i) IN_CSV="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    s) SEEDS="$OPTARG" ;;
    k) K="$OPTARG" ;;
    p) PYTHON="$OPTARG" ;;
    l) LABEL_COL="$OPTARG" ;;
    r) REQS="$OPTARG" ;;
    h) show_help; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; show_help; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
  esac
done

# --- Environment bootstrap ----------------------------------------------------
if [[ -d ".venv" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON="python"  # use venv python
fi

if [[ -f "$REQS" ]]; then
  # Best-effort install; don't hard fail if offline
  $PYTHON -m pip --version >/dev/null 2>&1 || true
  $PYTHON -m pip install -q -r "$REQS" || echo "[warn] pip install skipped/failed"
fi

# --- Prepare run directory ----------------------------------------------------
ts="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${OUT_ROOT}/${ts}"
mkdir -p "${RUN_DIR}"

# Reproducibility snapshot
if command -v git >/dev/null 2>&1; then
  git rev-parse HEAD 2>/dev/null | tr -d '\n' > "${RUN_DIR}/git_commit.txt" || true
  git diff --no-color > "${RUN_DIR}/git_diff.patch" || true
fi
$PYTHON -m pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>/dev/null || true
cp -n scripts/day12_train_logreg.py "${RUN_DIR}/_snapshot_train_logreg.py" 2>/dev/null || true

# --- Validate inputs ----------------------------------------------------------
[[ -f "${IN_CSV}" ]] || { echo "[error] Missing input CSV: ${IN_CSV}"; exit 3; }
[[ -f "scripts/day12_train_logreg.py" ]] || { echo "[error] Missing scripts/day12_train_logreg.py"; exit 3; }

echo "[info] Starting baseline run @ ${ts}"
echo "[info] Input CSV     : ${IN_CSV}"
echo "[info] Output root   : ${OUT_ROOT}"
echo "[info] Seeds         : ${SEEDS}"
echo "[info] K-folds       : ${K}"
echo "[info] Label column  : ${LABEL_COL:-<auto/proxy>}"

# --- Execute for each seed ----------------------------------------------------
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for seed in "${SEED_ARR[@]}"; do
  SEED_DIR="${RUN_DIR}/seed_${seed}"
  mkdir -p "${SEED_DIR}"

  CMD=( "$PYTHON" "scripts/day12_train_logreg.py"
        --in_csv "$IN_CSV"
        --out_dir "$SEED_DIR"
        --seed "$seed"
        --cv "$K" )
  if [[ -n "${LABEL_COL}" ]]; then
    CMD+=( --label_col "$LABEL_COL" )
  fi

  echo "[info] Running seed ${seed}: ${CMD[*]}"
  # Capture both streams to a log while still printing to console
  { "${CMD[@]}" 2>&1 | tee "${SEED_DIR}/train.log"; } || {
    echo "[warn] Seed ${seed} failed; continuing"
  }
done | tee "${RUN_DIR}/run.log"

echo
echo "[done] All runs finished."
echo "[path] ${RUN_DIR}"
echo "[next] Inspect metrics under each seed dir (metrics.json), and figures/."
