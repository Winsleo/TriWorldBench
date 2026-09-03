#!/usr/bin/env bash
# 快速创建 TriWorldBench 评测环境 (uv, 不用 conda)
# 用法: bash scripts/setup_a2v_env.sh [ENV_PATH]
#   ENV_PATH 默认 /disk/worldmodel/uv/envs/triworldbench

set -euo pipefail

ENV_PATH="${1:-/disk/worldmodel/uv/envs/triworldbench}"
PYTHON_VERSION="3.10.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[setup] Creating uv env at $ENV_PATH (Python $PYTHON_VERSION)"
if [ -d "$ENV_PATH" ]; then
    echo "[setup] env already exists, skipping creation"
else
    uv venv "$ENV_PATH" --python "$PYTHON_VERSION" --seed
fi

PY="$ENV_PATH/bin/python"

echo "[setup] Installing requirements.txt (torch + transformers + ...)"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$PROJECT_ROOT/requirements.txt"

echo "[setup] Installing requirements_no_deps.txt (no-deps packages)"
"$PY" -m pip install --no-deps --no-build-isolation -r "$PROJECT_ROOT/requirements_no_deps.txt"

echo "[setup] Smoke test: config loading"
"$PY" -c "
import yaml; from pathlib import Path
cfg = yaml.safe_load(Path('$PROJECT_ROOT/config/config.yaml').read_text())
print('  eval_input:', cfg.get('eval_input'))
print('  gt_dataset:', cfg.get('gt_dataset'))
print('  preprocess.target:', cfg.get('preprocess', {}).get('target_width'), 'x', cfg.get('preprocess', {}).get('target_height'))
print('OK')
"

echo ""
echo "[setup] Done! To use:"
echo "  source $ENV_PATH/bin/activate"
echo "  cd $PROJECT_ROOT"
echo "  ./run_preprocess.sh <method> --dry-run   # verify pipeline"
echo "  ./run_eval.sh --dry-run                  # verify eval"
echo ""
echo "[setup] Weights needed (see DOWNLOAD_LINKS.md):"
echo "  weights/qwenvl3               (Qwen3-VL-8B-Instruct, ~16GB)"
echo "  weights/clip-vit-base-patch16  (CLIP)"
echo "  weights/depth-anything        (Depth-Anything-V2-Small)"
echo "  weights/sam3                  (SAM3)"
echo "  weights/sam/                  (SAM/SAM2)"
echo "  weights/clip_model/           (ViT-B-32, ViT-L-14)"
echo "  weights/vjepa/                (V-JEPA)"
echo ""
echo "[setup] Val dataset needed (see README.md):"
echo "  huggingface-cli download TriWorldBench/Dataset --include 'val_dataset/*' --local-dir $PROJECT_ROOT/"
