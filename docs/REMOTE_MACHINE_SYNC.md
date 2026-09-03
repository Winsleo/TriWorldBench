# TriWorldBench Remote-Machine Sync Guide

> **Purpose**: Replicate the full local-eval setup of TriWorldBench to another machine
> via the `tos://wangsl/` TOS bucket. Code is already in the long git repo; data,
> model weights, and Python env are moved via `tosutil cp`.
>
> **Last verified**: 2026-09-03 (after uploading 46GB+10GB to tos://wangsl/triworldbench/)

---

## 0. What must be on the remote machine

| Item | Size | Where it lives after sync | Notes |
|---|---|---|---|
| Code | 0 (git) | `/disk/worldmodel/wangshilong/TriWorldBench/` | `git clone` from long repo |
| test_data | 956 MB | `TriWorldBench/test_data/` | val_dataset + STATE + vlm_data_base |
| WorldArena models | 41 GB | `/disk/worldmodel/wangshilong/WorldArena/video_quality/models/` | clip, dino, sam3, qwenvl3, ... |
| JEDi pretrained | 9.9 GB | `WorldArena/video_quality/JEDi/pretrained_models/` | vith16.pth.tar (9.7G) + ssv2-probe.pth.tar |
| third_party checkpoints | 76 MB | `WorldArena/video_quality/WorldArena/third_party/checkpoints/` | Tartan-C-T + VFIMamba.pkl symlink |
| resnet34 | 84 MB | `/disk/worldmodel/public_model/torch_hub/hub/checkpoints/` | single file |
| VFIMamba.pkl (real) | 253 MB | `/disk/worldmodel/public_data/WorldScore/.../checkpoints/` | breaks symlink chain |
| Python uv env (optional) | 11 GB | `/disk/worldmodel/uv/envs/triworldbench/` | tar + upload; otherwise rebuild via pip |
| System ffmpeg/ffprobe | 0 | apt install | required by submission_check.py |

---

## 1. What's already in the TOS bucket

`tos://wangsl/triworldbench/` (verified 2026-09-03):

```
test_data/                                   956 MB
WorldArena_models/models/                   41 GB
JEDi_pretrained_models/pretrained_models/   9.9 GB  (extra "pretrained_models/" layer)
WorldArena_third_party_checkpoints/checkpoints/   76 MB  (extra "checkpoints/" layer)
resnet34-b627a593.pth                       84 MB
VFIMamba.pkl                                253 MB
```

Note the `pretrained_models/` and `checkpoints/` extra layers -- they come from the
source path having that final directory. The download commands below map to the
correct final location on the remote machine.

---

## 2. Remote-machine setup (the canonical recipe)

Run on the **remote machine**. Assumes `/disk/worldmodel/wangshilong/` is writable
and that `tosutil` is in PATH.

### 2.1 Code (from long git repo)

```bash
mkdir -p /disk/worldmodel/wangshilong
cd /disk/worldmodel/wangshilong
git clone <LONG_REPO_URL> TriWorldBench
cd TriWorldBench
git log --oneline -3   # confirm latest commit (B2 no-GT patch + SEA-RAFT fix)
```

### 2.2 Download data + weights from TOS

```bash
TOS=tos://wangsl/triworldbench
ROOT=/disk/worldmodel/wangshilong
WB=/disk/worldmodel/wangshilong/WorldArena/video_quality

# 2.2.1 test_data (956 MB)
tosutil cp -r -j 8 \
    $TOS/test_data \
    $ROOT/TriWorldBench/test_data

# 2.2.2 WorldArena/models/ (41 GB)
mkdir -p /disk/worldmodel/wangshilong/WorldArena/video_quality
tosutil cp -r -j 8 \
    $TOS/WorldArena_models/models \
    $WB/models

# 2.2.3 JEDi/pretrained_models/ (9.9 GB)
tosutil cp -r -j 8 \
    $TOS/JEDi_pretrained_models/pretrained_models \
    $WB/JEDi/pretrained_models

# 2.2.4 WorldArena/third_party/checkpoints/ (76 MB)
tosutil cp -r -j 8 \
    $TOS/WorldArena_third_party_checkpoints/checkpoints \
    $WB/WorldArena/third_party/checkpoints

# 2.2.5 resnet34 (84 MB)
mkdir -p /disk/worldmodel/public_model/torch_hub/hub/checkpoints
tosutil cp -j 8 \
    $TOS/resnet34-b627a593.pth \
    /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth

# 2.2.6 VFIMamba.pkl real file (253 MB)
mkdir -p /disk/worldmodel/public_data/WorldScore/worldscore/benchmark/metrics/checkpoints
tosutil cp -j 8 \
    $TOS/VFIMamba.pkl \
    /disk/worldmodel/public_data/WorldScore/worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl
```

### 2.3 System packages

```bash
which ffprobe || sudo apt install -y ffmpeg
```

### 2.4 Python env (two options)

#### Option A: tar-uploaded uv env (recommended; ~11 GB)

If the source machine uploaded the env, restore it:

```bash
mkdir -p /disk/worldmodel/uv/envs
cd /disk/worldmodel/uv/envs
# (you uploaded triworldbench_env.tar.gz to TOS earlier)
tosutil cp tos://wangsl/triworldbench/triworldbench_env.tar.gz .
tar xzf triworldbench_env.tar.gz   # produces envs/triworldbench/...
```

#### Option B: rebuild from requirements

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
uv venv /disk/worldmodel/uv/envs/triworldbench --python 3.10
$PY -m pip install --upgrade pip wheel setuptools

cd $ROOT/TriWorldBench
$PY -m pip install -r requirements.txt
# critical no-deps overrides (some packages conflict on deps)
$PY -m pip install --no-deps imageio imageio-ffmpeg decord ftfy regex
```

Expected installs (versions known-good on the source machine):

```
torch 2.10.0+cu*        transformers 5.13.0.dev0
pyiqa 0.1.14.1          bitsandbytes 0.49.0
opencv-python 4.11.0.86 opencv-python-headless 4.11.0.86
av 13.1.0               decord 0.6.0
h5py 3.16.0             timm 1.0.24
accelerate 1.1.0        easydict 1.13
pillow 11.3.0           safetensors 0.8.0
scipy 1.15.3            numpy 1.26.0
omegaconf 2.3.0         ftfy 6.1.1
regex 2026.5.9          protobuf 7.35.1
```

### 2.5 Recreate the `weights/` symlinks (if `git clone` lost them)

`git clone` does NOT preserve the symlinks in `TriWorldBench/weights/` -- those
were added locally. After `git clone`, the directory is empty. Recreate:

```bash
cd $ROOT/TriWorldBench
rm -rf weights/*

# pointing into WorldArena/models/
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/aesthetic_model        weights/aesthetic_model
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/clip_model              weights/clip_model
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/clip-vit-base-patch16  weights/clip-vit-base-patch16
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/depth-anything         weights/depth-anything
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/dino_model              weights/dino_model
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/pyiqa_model            weights/pyiqa_model
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/qwenvl3                weights/qwenvl3
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/raft_model             weights/raft_model
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/sam                    weights/sam3

# real subdirs (recreate via symlinks for portability)
mkdir -p weights/_hf_downloads weights/torch_home weights/VFIMamba weights/vjepa
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/WorldArena/third_party/checkpoints/Tartan-C-T-TSKH-spring540x960-M.safetensors \
      weights/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth
mkdir -p weights/_hf_downloads/videogenevalkit-checkpoints/worldscore
mv weights/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth \
   weights/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth 2>/dev/null || true
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/models/raft_model/models \
      weights/raft_model/RAFT/models
ln -s /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth \
      weights/torch_home/hub/checkpoints/resnet34-b627a593.pth
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/JEDi/pretrained_models/vith16.pth.tar \
      weights/vjepa/vith16.pth.tar
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar \
      weights/vjepa/ssv2-probe.pth.tar
ln -s /disk/worldmodel/wangshilong/WorldArena/video_quality/WorldArena/third_party/checkpoints/VFIMamba.pkl \
      weights/VFIMamba/VFIMamba.pkl
```

To verify all targets exist:
```bash
for s in weights/*; do
  [ -L "$s" ] || continue
  target=$(readlink "$s")
  [ -e "$s" ] && echo "OK  $s -> $target" || echo "BAD $s -> $target (missing)"
done
```

---

## 3. Run local eval on the remote machine

### 3.1 With pre-generated mp4 (no inference needed)

```bash
# (optional) pull a2v Path A concat_split outputs (30 MB)
mkdir -p /disk/worldmodel/wangsl_sync
tosutil cp -r -j 8 \
    tos://wangsl/triworldbench/a2v_concat_split \
    /disk/worldmodel/wangsl_sync/a2v_concat_split

# stage into submission format
cd /disk/worldmodel/wangshilong/a2v_v4_khl
./scripts/infer_for_submission.sh \
    --method a2v_path_a_baseline \
    --n-eps 20 \
    --output-dir /disk/worldmodel/wangsl_sync/submission/a2v_v4_test

# build fake-GT (only needed for STATE/GT-required metrics; can skip for GT-free only)
cd /disk/worldmodel/wangshilong/TriWorldBench
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
$PY scripts/build_fake_gt.py \
    --gen-root /disk/worldmodel/wangsl_sync/submission/a2v_v4_test \
    --gt-root /disk/worldmodel/wangshilong/TriWorldBench/test_data/gt_dataset_fake \
    --frame-count 462

# run eval (no-GT config -- 9 GT-free metrics)
cd /disk/worldmodel/wangshilong/TriWorldBench
$PY run_eval.py \
    --config config/config_no_gt.yaml \
    --gen-root /disk/worldmodel/wangsl_sync/submission/a2v_v4_test \
    --gt-root ./test_data/gt_dataset_fake \
    --output-dir ./output/a2v_remote_run
cat ./output/a2v_remote_run/summary.csv
```

### 3.2 With full GT eval (20 GT-required + 9 GT-free metrics)

Use `config/config.yaml` instead of `config_no_gt.yaml` and point `--gt-root`
to the real `test_data/gt_dataset` (already downloaded in step 2.2.1).

---

## 4. (For the source machine) Re-upload commands

If something needs to be re-uploaded, run on the **source machine**:

```bash
TOS=tos://wangsl/triworldbench

# test_data
tosutil cp -r -j 8 \
    /disk/worldmodel/wangshilong/TriWorldBench/test_data \
    $TOS/test_data

# WorldArena/models/
tosutil cp -r -j 8 \
    /disk/worldmodel/wangshilong/WorldArena/video_quality/models \
    $TOS/WorldArena_models

# JEDi/pretrained_models/
tosutil cp -r -j 8 \
    /disk/worldmodel/wangshilong/WorldArena/video_quality/JEDi/pretrained_models \
    $TOS/JEDi_pretrained_models

# third_party/checkpoints/
tosutil cp -r -j 8 \
    /disk/worldmodel/wangshilong/WorldArena/video_quality/WorldArena/third_party/checkpoints \
    $TOS/WorldArena_third_party_checkpoints

# resnet34
tosutil cp -j 8 \
    /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth \
    $TOS/resnet34-b627a593.pth

# VFIMamba.pkl (real file, not symlink chain)
tosutil cp -j 8 \
    /disk/worldmodel/public_data/WorldScore/worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl \
    $TOS/VFIMamba.pkl

# uv env (optional, 11 GB)
tar czf /tmp/triworldbench_env.tar.gz -C /disk/worldmodel/uv/envs triworldbench
tosutil cp -j 8 /tmp/triworldbench_env.tar.gz $TOS/triworldbench_env.tar.gz

# a2v concat_split outputs (optional, 30 MB)
tosutil cp -r -j 8 \
    /disk/worldmodel/wangshilong/a2v_v4_khl/concat_split \
    $TOS/a2v_concat_split
```

---

## 5. Gotchas

1. **Symlinks lost in `git clone`** -- see section 2.5. The repo's `weights/` is
   empty after a clean clone; recreate manually.

2. **Symlink chain in `weights/VFIMamba/`** -- points to
   `WorldArena/third_party/checkpoints/VFIMamba.pkl` which itself is a symlink
   to `public_data/...`. Section 2.2.6 uploads the real file to break the chain
   so it works on a fresh machine.

3. **`third_party/checkpoints/VFIMamba.pkl`** -- uploaded as a symlink (not the
   target). The download target path on the remote side is
   `WorldArena/video_quality/WorldArena/third_party/checkpoints/VFIMamba.pkl`,
   but if you then move the real file in step 2.2.6 the symlink still works
   (it points to a file that exists at the path it points to, since we set
   that path to the real file location too).

4. **GPU** -- local eval mostly runs on GPU (CLIP/DINO/qwenvl3/...). CPU-only is
   technically possible for a subset of metrics but is 10-100x slower.

5. **Disk space** -- `WorldArena/models/` is 41 GB, JEDi is 9.9 GB, plus test
   data 1 GB, uv env 11 GB. Plan for ~70 GB minimum.

6. **The `B2 no-GT` patch** -- committed 2026-09-03; git log on the remote
   should show commits like `feat(eval): add build_fake_gt` and
   `feat(eval): support no-GT mode via skip_auto_dependencies flag`. Without
   these, eval on the test_dataset (no real GT) fails.

7. **SEA-RAFT safetensors fix** -- committed 2026-09-03
   (`fix(sea-raft): support safetensors checkpoint format in load_t`).
   Required if you run the flow-metric side of the eval.

---

## 6. Verification checklist

On the remote machine, after running sections 2.1 - 2.5:

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench

# 1. Code is current
git log --oneline | head -3

# 2. test_data exists and is non-empty
ls test_data/gt_dataset | wc -l   # 100 episode dirs expected

# 3. Weights resolve
for s in weights/*; do
  [ -L "$s" ] && [ -e "$s" ] && echo "OK  $s" || echo "BAD $s"
done

# 4. Env imports
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
$PY -c "import torch, h5py, imageio.v3, transformers, opencv; print('env OK')"

# 5. ffprobe available
which ffprobe

# 6. Dry-run eval (B2 no-GT, 2 episodes, ~5 min)
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
mkdir -p /tmp/eval_smoke/gen/episode1 /tmp/eval_smoke/gen/episode2
$PY -c "
import imageio.v3 as iio, numpy as np
from pathlib import Path
for n in (1,2):
    for v in ('head','left','right'):
        Path(f'/tmp/eval_smoke/gen/episode{n}/{v}.mp4').parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(f'/tmp/eval_smoke/gen/episode{n}/{v}.mp4',
                    (np.random.rand(121,240,320,3)*255).astype(np.uint8),
                    fps=30, codec='libx264')
print('mock 2 ep OK')
"
$PY scripts/build_fake_gt.py \
    --gen-root /tmp/eval_smoke/gen \
    --gt-root /tmp/eval_smoke/gt_fake \
    --frame-count 462
$PY run_eval.py \
    --config config/config_no_gt.yaml \
    --gen-root /tmp/eval_smoke/gen \
    --gt-root /tmp/eval_smoke/gt_fake \
    --output-dir /tmp/eval_smoke/out
cat /tmp/eval_smoke/out/summary.csv
```

If all six pass, the remote machine is ready for full-scale local evaluation.