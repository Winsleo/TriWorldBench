#!/usr/bin/env bash
# Download TriWorldBench data + weights from tos://wangsl/triworldbench/
#
# Run on the REMOTE machine (the one that does not yet have TriWorldBench).
# Reference: docs/REMOTE_MACHINE_SYNC.md section 2
#
# Assumes:
#   - tosutil is in PATH
#   - /disk/worldmodel/wangshilong is writable
#   - LONG_REPO_URL env var is set to the git repo URL
#     (or pass --repo <url>)
#
# Usage:
#   LONG_REPO_URL=https://your.git/host/repo.git ./scripts/sync_download_from_tos.sh
#   ./scripts/sync_download_from_tos.sh --repo https://your.git/host/repo.git
#
# What it does:
#   1. git clone the code
#   2. tosutil cp all data + weights into the right absolute paths
#   3. recreate the weights/ symlinks (git clone leaves them empty)
#   4. (optional) restore the uv env from a tar uploaded with sync_upload_to_tos.sh
#
# This script does NOT install system packages or rebuild the env from pip.
# See docs/REMOTE_MACHINE_SYNC.md section 2.3 and 2.4 for those.

set -euo pipefail

REPO_URL="${LONG_REPO_URL:-}"
SKIP_ENV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_URL="$2"; shift 2 ;;
        --skip-env) SKIP_ENV=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *)
            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$REPO_URL" ]]; then
    echo "ERROR: repo URL not set. Use --repo <url> or LONG_REPO_URL env var." >&2
    exit 1
fi

TOS=tos://wangsl/triworldbench
ROOT=/disk/worldmodel/wangshilong
WB=$ROOT/WorldArena/video_quality
TARGET=$ROOT/TriWorldBench

# ---------- 1. Code ----------
echo "=== [1/5] git clone ==="
mkdir -p "$ROOT"
if [[ -d "$TARGET" ]]; then
    echo "  $TARGET already exists; assuming already cloned. Pulling latest."
    (cd "$TARGET" && git pull --ff-only || true)
else
    git clone "$REPO_URL" "$TARGET"
fi
cd "$TARGET"
echo "  current HEAD: $(git log --oneline -1)"

# ---------- 2. Data + weights ----------
echo ""
echo "=== [2/5] download data + weights from TOS ==="
echo "  (this can take 10-20 minutes for ~50 GB at typical rates)"

# 2.1 test_data
echo "  - test_data (956 MB)"
tosutil cp -r -j 8 "$TOS/test_data" "$TARGET/test_data"

# 2.2 WorldArena/models (41 GB)
echo "  - WorldArena/models (41 GB)"
mkdir -p "$ROOT/WorldArena/video_quality"
tosutil cp -r -j 8 "$TOS/WorldArena_models/models" "$WB/models"

# 2.3 JEDi/pretrained_models (9.9 GB)
echo "  - JEDi/pretrained_models (9.9 GB)"
tosutil cp -r -j 8 "$TOS/JEDi_pretrained_models/pretrained_models" "$WB/JEDi/pretrained_models"

# 2.4 third_party/checkpoints (76 MB)
echo "  - third_party/checkpoints (76 MB)"
tosutil cp -r -j 8 "$TOS/WorldArena_third_party_checkpoints/checkpoints" "$WB/WorldArena/third_party/checkpoints"

# 2.5 resnet34 (84 MB)
echo "  - resnet34 (84 MB)"
mkdir -p /disk/worldmodel/public_model/torch_hub/hub/checkpoints
tosutil cp -j 8 "$TOS/resnet34-b627a593.pth" /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth

# 2.6 VFIMamba.pkl (253 MB)
echo "  - VFIMamba.pkl (253 MB)"
mkdir -p /disk/worldmodel/public_data/WorldScore/worldscore/benchmark/metrics/checkpoints
tosutil cp -j 8 "$TOS/VFIMamba.pkl" /disk/worldmodel/public_data/WorldScore/worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl

# 2.7 (optional) a2v concat_split outputs (30 MB)
if tosutil ls "$TOS/a2v_concat_split/" >/dev/null 2>&1; then
    echo "  - a2v_concat_split (30 MB)"
    tosutil cp -r -j 8 "$TOS/a2v_concat_split" "$ROOT/wangsl_sync/a2v_concat_split" \
        || echo "  (skip a2v_concat_split)"
fi

# ---------- 3. uv env (optional) ----------
if [[ $SKIP_ENV -eq 0 ]] && tosutil ls "$TOS/triworldbench_env.tar.gz" >/dev/null 2>&1; then
    echo ""
    echo "=== [3/5] restore uv env from tar ==="
    mkdir -p /disk/worldmodel/uv/envs
    cd /disk/worldmodel/uv/envs
    if [[ ! -d triworldbench ]]; then
        tosutil cp -j 8 "$TOS/triworldbench_env.tar.gz" .
        tar xzf triworldbench_env.tar.gz
        echo "  uv env restored to /disk/worldmodel/uv/envs/triworldbench"
    else
        echo "  /disk/worldmodel/uv/envs/triworldbench already exists; skipping tar extract"
    fi
else
    echo ""
    echo "=== [3/5] skip uv env (rebuild via pip per docs/REMOTE_MACHINE_SYNC.md section 2.4) ==="
fi

# ---------- 4. Recreate weights/ symlinks ----------
echo ""
echo "=== [4/5] recreate weights/ symlinks ==="
cd "$TARGET"
mkdir -p weights

# pointing into WorldArena/models/
ln -sfn "$WB/models/aesthetic_model"       weights/aesthetic_model
ln -sfn "$WB/models/clip_model"           weights/clip_model
ln -sfn "$WB/models/clip-vit-base-patch16" weights/clip-vit-base-patch16
ln -sfn "$WB/models/depth-anything"       weights/depth-anything
ln -sfn "$WB/models/dino_model"           weights/dino_model
ln -sfn "$WB/models/pyiqa_model"          weights/pyiqa_model
ln -sfn "$WB/models/qwenvl3"              weights/qwenvl3
ln -sfn "$WB/models/raft_model"           weights/raft_model
ln -sfn "$WB/models/sam"                  weights/sam3

# subdirs with symlinks for portability
mkdir -p weights/_hf_downloads/videogenevalkit-checkpoints/worldscore
ln -sfn "$WB/WorldArena/third_party/checkpoints/Tartan-C-T-TSKH-spring540x960-M.safetensors" \
       weights/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth

mkdir -p weights/raft_model/RAFT
ln -sfn "$WB/models/raft_model/models" weights/raft_model/RAFT/models

mkdir -p weights/torch_home/hub/checkpoints
ln -sfn /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth \
       weights/torch_home/hub/checkpoints/resnet34-b627a593.pth

mkdir -p weights/VFIMamba weights/vjepa
ln -sfn "$WB/WorldArena/third_party/checkpoints/VFIMamba.pkl" weights/VFIMamba/VFIMamba.pkl
ln -sfn "$WB/JEDi/pretrained_models/vith16.pth.tar"          weights/vjepa/vith16.pth.tar
ln -sfn "$WB/JEDi/pretrained_models/ssv2-probe.pth.tar"      weights/vjepa/ssv2-probe.pth.tar

# Verify
echo "  verifying symlinks ..."
broken=0
for s in weights/* weights/*/*; do
    [ -L "$s" ] || continue
    if [ ! -e "$s" ]; then
        echo "    BROKEN: $s -> $(readlink "$s")"
        broken=$((broken+1))
    fi
done
if [[ $broken -gt 0 ]]; then
    echo "  WARNING: $broken broken symlink(s). Check that data + weights were downloaded."
else
    echo "  all symlinks resolve OK"
fi

# ---------- 5. Summary ----------
echo ""
echo "=== [5/5] summary ==="
echo "  code:   $TARGET"
echo "  weights: $TARGET/weights/  (symlinks -> $WB and public_model)"
echo ""
echo "Next steps:"
echo "  - install system packages: which ffprobe || sudo apt install -y ffmpeg"
echo "  - if uv env was not uploaded, rebuild per docs/REMOTE_MACHINE_SYNC.md section 2.4"
echo "  - smoke-test:"
echo "      PY=/disk/worldmodel/uv/envs/triworldbench/bin/python"
echo "      \$PY run_eval.py --config config/config_no_gt.yaml \\"
echo "          --gen-root <gen_dir> --gt-root <gt_dir> --output-dir <out_dir>"