#!/usr/bin/env bash
# Upload TriWorldBench data + weights to tos://wangsl/triworldbench/
#
# Run on the SOURCE machine (the one with all the local files).
# Reference: docs/REMOTE_MACHINE_SYNC.md section 4
#
# Usage:
#   ./scripts/sync_upload_to_tos.sh             # upload everything
#   ./scripts/sync_upload_to_tos.sh skip-env    # skip the 11 GB uv env
#   ./scripts/sync_upload_to_tos.sh skip-concat # skip the 30 MB concat_split
#
# Each upload is run in the background with a per-target log file under
# logs/tosupload/. Re-running the script while uploads are in progress is
# safe but will start a new upload alongside the old one.

set -euo pipefail

TOS=tos://wangsl/triworldbench
SRC=/disk/worldmodel/wangshilong
WORLDARENA=$SRC/WorldArena/video_quality
LOGDIR=$SRC/logs/tosupload
mkdir -p "$LOGDIR"

skip_env=0
skip_concat=0
for arg in "$@"; do
    case "$arg" in
        skip-env)    skip_env=1 ;;
        skip-concat) skip_concat=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

start_upload() {
    local desc="$1"; shift
    local src="$1"; shift
    local dst="$1"; shift
    local logfile="$LOGDIR/${desc}.log"
    echo "[start] $desc :: $src -> $dst"
    nohup tosutil cp -r -j 8 "$src" "$dst" > "$logfile" 2>&1 &
    echo "  pid=$! log=$logfile"
}

# 1. test_data (956 MB)
start_upload test_data \
    "$SRC/TriWorldBench/test_data" \
    "$TOS/test_data"

# 2. WorldArena/models (41 GB)
start_upload WorldArena_models \
    "$WORLDARENA/models" \
    "$TOS/WorldArena_models"

# 3. JEDi/pretrained_models (9.9 GB: vith16 + ssv2-probe)
start_upload JEDi_pretrained_models \
    "$WORLDARENA/JEDi/pretrained_models" \
    "$TOS/JEDi_pretrained_models"

# 4. third_party/checkpoints (76 MB: Tartan + VFIMamba.pkl symlink)
start_upload WorldArena_third_party_checkpoints \
    "$WORLDARENA/WorldArena/third_party/checkpoints" \
    "$TOS/WorldArena_third_party_checkpoints"

# 5. resnet34 (84 MB)
start_upload resnet34 \
    /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth \
    $TOS/resnet34-b627a593.pth

# 6. VFIMamba.pkl real file (253 MB; breaks symlink chain)
start_upload VFIMamba \
    /disk/worldmodel/public_data/WorldScore/worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl \
    $TOS/VFIMamba.pkl

# 7. (optional) a2v concat_split outputs (30 MB)
if [[ $skip_concat -eq 0 ]]; then
    start_upload a2v_concat_split \
        "$SRC/a2v_v4_khl/concat_split" \
        "$TOS/a2v_concat_split"
else
    echo "[skip] a2v_concat_split (per skip-concat)"
fi

# 8. (optional) uv env tar (11 GB; slow)
if [[ $skip_env -eq 0 ]]; then
    ENV_TAR=/tmp/triworldbench_env.tar.gz
    if [[ ! -f "$ENV_TAR" ]]; then
        echo "[tarring] triworldbench uv env -> $ENV_TAR"
        tar czf "$ENV_TAR" -C /disk/worldmodel/uv/envs triworldbench
    fi
    start_upload triworldbench_env "$ENV_TAR" "$TOS/triworldbench_env.tar.gz"
else
    echo "[skip] triworldbench_env (per skip-env)"
fi

echo ""
echo "All uploads launched. To check progress:"
echo "  ps -ef | grep 'tosutil cp' | grep -v grep"
echo "  ls -lh $LOGDIR/"
echo "  tail -1 $LOGDIR/<name>.log  (look for % progress)"