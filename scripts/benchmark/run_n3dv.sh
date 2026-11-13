#!/usr/bin/env bash
# FreeTimeGS++ (no UFM) N3DV benchmark. Safe to Ctrl-C / `docker stop ftgspp-n3dv` at any time and rerun:
# finished stages are skipped, an unfinished stage restarts from scratch (its partial output is removed).
#   bash scripts/benchmark/run_n3dv.sh [RUN_ROOT] [scene ...]
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:-/media/tsaichenghan/2TBSSD/FreeTimeGSPlusPlus/benchmark_20261113}"; shift || true
SCENES="${*:-coffee_martini cook_spinach cut_roasted_beef flame_salmon_1 flame_steak sear_steak}"
RAW=/media/tsaichenghan/2TBSSD/datasets
case "$RUN" in /media/tsaichenghan/2TBSSD/FreeTimeGSPlusPlus/benchmark_*) ;; *) echo "use a benchmark_* dir" >&2; exit 1;; esac
mkdir -p "$RUN/work" "$RUN/output" "$RUN/cache"
CODE="$RUN/code"
if [ ! -d "$CODE" ]; then   # freeze + hash the exact code used
  mkdir -p "$CODE"; rsync -a --exclude .git --exclude __pycache__ "$REPO"/{ftgspp,configs,scripts,pyproject.toml} "$CODE/"
  (cd "$CODE" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$RUN/code_sha256.txt"
  git -C "$REPO" rev-parse HEAD > "$RUN/code_git_head.txt"; git -C "$REPO" diff > "$RUN/code_git_diff.patch"
else
  (cd "$CODE" && sha256sum -c --quiet "$RUN/code_sha256.txt") || { echo "frozen code modified" >&2; exit 1; }
fi
for S in $SCENES; do
  O="$RUN/output/$S"
  [ -f "$O/DONE.json" ] && { echo "SKIP $S"; continue; }
  AVAIL=$(df -BG --output=avail "$RUN" | tail -1 | tr -dc 0-9); [ "$AVAIL" -ge 80 ] || { echo "less than 80G free" >&2; exit 1; }
  echo "=== $S $(date -Is)"
  docker run --rm --name ftgspp-n3dv --gpus all --shm-size 8g --memory 22g --memory-swap 22g --user "$(id -u):$(id -g)" \
    -e FTGS_NUM_WORKERS=4 \
    -e HOME=/tmp -e XDG_CACHE_HOME="$RUN/cache" -e TORCH_HOME="$RUN/cache/torch" -e HF_HOME="$RUN/cache/hf" \
    -e PYTHONUNBUFFERED=1 -w "$RUN/work" \
    -v "$CODE:/opt/ftgspp:ro" -v "$RAW:$RAW:ro" -v "$RUN:$RUN" asdggg/freetimegspp:latest \
    bash -c "python /opt/ftgspp/scripts/benchmark/run_scene.py /opt/ftgspp/configs/benchmark_n3dv/ftgspp_noufm/$S.toml $O \
             && python /opt/ftgspp/scripts/benchmark/eval_unified.py $O $RAW/$S" 2>&1 | tee -a "$O.log"
  [ -f "$O/unified_metrics.json" ] || { echo "scene $S failed" >&2; exit 1; }
  date -Is > "$O/DONE.json"
  rm -rf "$RUN/work/_extracted/$S" "$RUN/work/_memmap/$S" "$RUN/work/_points/$S"   # only this run's intermediates
done
echo "ALL DONE $(date -Is)"
