# TriWorldBench 本地评测指南(Agent 执行版)

> **目标读者**:接到"在 X 集群上打通 TriWorldBench 本地评测"任务的任意 agent  
> **前置假设**:能访问 `/disk/worldmodel/` 大盘、能联网到 hf-mirror、有 slurm(A100-80G 推荐)、已装 `uv`  
> **执行时长**:新机首装约 30-60 min(其中 95% 在 pip 装 torch/xformers),加 val_dataset 下载 5 min  
> **产出**:`output/<method>_<ts>/final_results/summary_metrics.csv`(TWB-Score,19 metric)

---

## ⚠️ 关键区分:本地评测 vs 榜单提交(必读)

| 维度 | 本地评测(val_dataset) | 榜单提交(test_dataset) |
|---|---|---|
| **Episode 数** | 100 | **500** |
| **GT 视频** | ✅ 有(`test_data/val/gt_dataset/episodeN/{view}/frames/`) | ❌ **无**, 不可本地看 metric |
| **First frame** | `gt_dataset/episodeN/{view}/frames/frame_000.jpg` | `first_frame/episodeN_{view}.jpg` |
| **指令文件** | `STATE/episodeN.json` | `instructions/episodeN.json` |
| **轨迹文件** | 无 | `data/episodeN.hdf5`(帧数从此读) |
| **mp4 分辨率** | 任意(preprocess 重采样到 320×240) | **必须 320×240**(submission_check.py 报 error) |
| **mp4 帧数** | 任意(preprocess match_gt) | **必须 match HDF5 帧数** |
| **FPS** | 任意 | 任意(推荐 30) |
| **评测执行** | 本地 `run_eval.sh`,立即看 metric | 官方服务器跑,我们看不到具体 metric |
| **得分** | TWB-Score 0-1(本仓库内分数) | TWB-Score × 100 上榜(榜单 60+) |

**简而言之**:
- 本文档主要讲**本地评测**(`val_dataset` 100 ep,自带 GT,可调参)
- **提交**走 `SUBMISSION_README.md` 的 500 ep 流程(详见 §11 + 仓库根 `SUBMISSION_README.md`)

⚠️ **不要把本地 2 ep / 20 ep TWB-Score 当成榜单分数**,只是内部 baseline 锚点。

---

## 0. 一句话总结

跑通以下端到端 pipeline:

```
v4_random10k_qhd_wrist 数据
  → multiview_prepare.py → dataset_mv
  → train_v4_mv.sbatch → ckpts_mv/exp1/step-NNN.safetensors
  → a2v.infer_a2v → 256×960 mp4
  → a2v_to_triworld.py → input/<method>/episodeN/{head,left,right}.mp4
  → ./run_preprocess.sh <method> → generate/<method>_<ts>/
  → ./run_eval.sh → output/<method>_<ts>/final_results/summary_metrics.csv
```

---

## 1. 路径约定(全本仓库固定,不要改)

```
/disk/worldmodel/wangshilong/
├── DiffSynth-Studio/                        # A2V 仓库 (branch: a2v)
├── TriWorldBench/                           # ← 本文档所在仓库
│   ├── docs/LOCAL_EVAL_GUIDE.md             # ← 你正在读的
│   ├── docs/A2V_SETUP.md                    # A2V 适配初版(可参考)
│   ├── docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md  # 早期执行计划
│   ├── scripts/setup_a2v_env.sh             # uv env 一键脚本
│   ├── config/config.yaml                   # 评测配置
│   ├── input/<method>/                      # ← bridge 写入
│   ├── generate/<method>_<ts>/              # preprocess 输出
│   ├── output/<method>_<ts>/                # eval 输出(summary_metrics.csv)
│   ├── test_data/                           # ← val_dataset 解压位置
│   └── weights/                             # ← 13 个 symlink
└── a2v_v4_khl/                              # A2V 三视角工作目录
    ├── multiview_prepare.py
    ├── train_v4_mv.sbatch
    ├── infer_multiview.py
    ├── a2v_to_triworld.py                   # ← A2V→TriWorldBench bridge
    └── dataset_mv/                          # 20 samples × 121 frames
```

---

## 2. 一次性环境搭建(每台新机跑一次,后续跳过)

### 2.1 Python 环境(uv + Python 3.10)

```bash
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python

# 已有 env 跳过,没有则:
uv venv /disk/worldmodel/uv/envs/triworldbench --python 3.10.0 --seed

# 装主依赖(torch 2.10+CUDA、transformers git commit、xformers dev1120 等,~30-60 min)
cd /disk/worldmodel/wangshilong/TriWorldBench
nohup bash scripts/setup_a2v_env.sh > /tmp/triworld_setup.log 2>&1 &
# 跟踪进度: tail -f /tmp/triworld_setup.log | grep -E "Successfully|Downloading"
```

### 2.2 补 no_deps 三包(setup 脚本在这一步会失败,需手动)

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
$PY -m pip install --no-deps --no-build-isolation -e ./external/jepa
$PY -m pip install --no-deps --no-build-isolation \
  ./external/Triworld/video_quality/JEDi/dist/videojedi-0.1.3-py3-none-any.whl
$PY -m pip install --no-deps --no-build-isolation pyiqa==0.1.14.1
```

### 2.3 系统依赖(确保 PATH 里)

```bash
which ffmpeg ffprobe   # 两个都要在,缺则 sudo apt install ffmpeg
nvidia-smi             # 至少 1 张 ≥24GB 显存的卡;VLM metric 推荐 32GB+
```

### 2.4 Smoke 验证(全部跑过才能进 §3)

```bash
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
cd /disk/worldmodel/wangshilong/TriWorldBench
$PY scripts/smoke_imports.py            # 必须输出 [smoke] imports ok
$PY -c "import torch, transformers, qwen_vl_utils, clip, pyiqa, sam3, open_clip, cv2, safetensors; print('all ok')"
```

---

## 3. 一次性数据 + Weights 准备(每台新机跑一次)

### 3.1 val_dataset(100 episodes,~950MB,**新下载**)

```bash
# 用 A2V env 装好了 huggingface_hub
PY_A2V=/disk/worldmodel/uv/envs/a2v/bin/python
export HF_ENDPOINT=https://hf-mirror.com        # ← 国内镜像必设!

# 后台下载
nohup $PY_A2V -c "
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import hf_hub_download
hf_hub_download('TriWorldBench/Dataset', repo_type='dataset',
                filename='val_dataset.tar.gz',
                local_dir='/disk/worldmodel/wangshilong/TriWorldBench/test_data')
" > /tmp/val_download.log 2>&1 &

# 跟踪
tail -f /tmp/val_download.log

# 下载完成后解压 + 清理
cd /disk/worldmodel/wangshilong/TriWorldBench/test_data
tar -xzf val_dataset.tar.gz
rm -f val_dataset.tar.gz
rm -rf .cache

# 验证结构
ls test_data/                 # 应有:data_readme.md gt_dataset STATE test_dataset vlm_data_base
ls gt_dataset | wc -l         # 应为 100
ls STATE | wc -l              # 应为 100
ls gt_dataset/episode1/head/frames/ | wc -l  # 应为 ~143(每视角帧数)
```

**为什么不用 WorldArena 的 val_dataset**:`/disk/worldmodel/public_data/WorldArena/val_dataset/` 只有 hdf5 轨迹+first_frame+instructions,**没有 head/left/right 帧**。必须用 HF 上 `TriWorldBench/Dataset` 提供的完整 val_dataset。

### 3.2 Weights 全量 symlink(13 个路径,**全部复用**)

**核心原则**:WorldArena 和 public_model 已下好所有评测权重,**TriWorldBench 不需要重下任何 model weights**。

```bash
WEIGHTS=/disk/worldmodel/wangshilong/TriWorldBench/weights
WA=/disk/worldmodel/wangshilong/WorldArena/video_quality
mkdir -p "$WEIGHTS"/{raft_model/RAFT,_hf_downloads/videogenevalkit-checkpoints/worldscore,VFIMamba,sam,vjepa,torch_home/hub/checkpoints}

# VLM / CLIP / 基础视觉
ln -sfn "$WA/models/qwenvl3/Qwen3-VL-8B-Instruct"            "$WEIGHTS/qwenvl3"
ln -sfn "$WA/models/clip-vit-base-patch16"                    "$WEIGHTS/clip-vit-base-patch16"
ln -sfn "$WA/models/clip_model"                                "$WEIGHTS/clip_model"
ln -sfn "$WA/models/depth_anything/Depth-Anything-V2-Small-hf" "$WEIGHTS/depth-anything"
ln -sfn "$WA/models/aesthetic_model"                           "$WEIGHTS/aesthetic_model"
ln -sfn "$WA/models/pyiqa_model"                              "$WEIGHTS/pyiqa_model"

# DINO(WorldArena 是 facebookresearch_dino_main,TriWorldBench 要 minimal → symlink)
ln -sfn "$WA/models/dino_model" "$WEIGHTS/dino_model"
ln -sfn "$WA/models/dino_model/facebookresearch_dino_main" "$WEIGHTS/dino_model/facebookresearch_dino_minimal"

# RAFT(WorldArena 是 models/raft_model/models/,TriWorldBench 要 RAFT/models/)
ln -sfn "$WA/models/raft_model/models" "$WEIGHTS/raft_model/RAFT/models"

# V-JEPA(WorldArena 在 JEDi/pretrained_models/)
ln -sfn "$WA/JEDi/pretrained_models/vith16.pth.tar"     "$WEIGHTS/vjepa/vith16.pth.tar"
ln -sfn "$WA/JEDi/pretrained_models/ssv2-probe.pth.tar" "$WEIGHTS/vjepa/ssv2-probe.pth.tar"

# 第三方 motion / 光流(WorldArena 在 third_party/checkpoints/)
ln -sfn "$WA/WorldArena/third_party/checkpoints/Tartan-C-T-TSKH-spring540x960-M.safetensors" \
  "$WEIGHTS/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth"
ln -sfn "$WA/WorldArena/third_party/checkpoints/VFIMamba.pkl" "$WEIGHTS/VFIMamba/VFIMamba.pkl"

# SAM3(GripperDetector 要 sam3.pt + bpe_simple_vocab_16e6.txt.gz)
ln -sfn "$WA/models/sam" "$WEIGHTS/sam3"

# ResNet34(public_model 已缓存)
ln -sfn /disk/worldmodel/public_model/torch_hub/hub/checkpoints/resnet34-b627a593.pth \
  "$WEIGHTS/torch_home/hub/checkpoints/resnet34-b627a593.pth"
```

### 3.3 Weights 验证(全过才能进 §4)

```bash
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
cd /disk/worldmodel/wangshilong/TriWorldBench
$PY -c "
from pathlib import Path
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/lib')
from eval_common import load_config, resolve_path
cfg = load_config(Path('config/config.yaml'))
W = resolve_path(cfg['weights_root'])
for name in ['qwenvl3','clip-vit-base-patch16','depth-anything','clip_model',
             'aesthetic_model','pyiqa_model','dino_model',
             'raft_model/RAFT/models','vjepa','sam3','VFIMamba',
             'torch_home/hub/checkpoints',
             '_hf_downloads/videogenevalkit-checkpoints/worldscore']:
    p = W / name
    print(f'  [{\"OK\" if p.exists() else \"MISS\"}] {name}')
"
```

### 3.4 Config 改 python 解释器(只改一次)

```bash
# config/config.yaml 里 python 字段改成绝对路径
sed -i 's|^python: python$|python: /disk/worldmodel/uv/envs/triworldbench/bin/python|' \
  /disk/worldmodel/wangshilong/TriWorldBench/config/config.yaml
grep "^python:" /disk/worldmodel/wangshilong/TriWorldBench/config/config.yaml
```

### 3.5 关键权重加载验证

```bash
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
cd /disk/worldmodel/wangshilong/TriWorldBench
$PY -c "
import clip, torch
from safetensors import safe_open
W='/disk/worldmodel/wangshilong/TriWorldBench/weights'
m,_ = clip.load(f'{W}/clip_model/ViT-B-32.pt', device='cpu'); print('CLIP ok')
torch.load(f'{W}/raft_model/RAFT/models/raft-things.pth', map_location='cpu', weights_only=False); print('RAFT ok')
torch.load(f'{W}/dino_model/dino_vitbase16_pretrain.pth', map_location='cpu', weights_only=False); print('DINO ok')
with safe_open(f'{W}/_hf_downloads/videogenevalkit-checkpoints/worldscore/Tartan-C-T-TSKH-spring540x960-M.pth', framework='pt') as f:
    print('Tartan-C:', len(list(f.keys())), 'tensors')
" 2>&1 | grep -v "pkg_resources is deprecated"
# 期望输出:CLIP ok / RAFT ok / DINO ok / Tartan-C: 452 tensors
```

---

## 4. A2V 三视角数据(每次评测跑一次)

### 4.1 数据准备(`multiview_prepare.py`)

```bash
PY_A2V=/disk/worldmodel/uv/envs/a2v/bin/python
$PY_A2V /disk/worldmodel/wangshilong/a2v_v4_khl/multiview_prepare.py \
  --root /disk/worldmodel/ziyang/projects/data_generation/khl_workspace/Data_output/v4_random10k_qhd_wrist \
  --task adjust_bottle \
  --episodes 53,68,70,231,240,247,273,297,486,506,582,602,658,670,730,779,783,804,820,962 \
  --tra 1 \
  --output_dir /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv \
  --per_view_height 256 --per_view_width 320 --num_frames 121
# 视角顺序固定 [head | left_wrist | right_wrist],与 TriWorldBench 一致
```

### 4.2 A2V 三视角训练(sbatch,~1-2h)

```bash
sbatch /disk/worldmodel/wangshilong/a2v_v4_khl/train_v4_mv.sbatch
# 监控:tail -f /disk/worldmodel/wangshilong/a2v_v4_khl/logs/<jobid>.out
# 产物:ckpts_mv/exp1/step-{500,800,...}.safetensors
# 单卡 ~55-65GB 显存,exclude 114-118
```

### 4.3 A2V 推理 → 拼接 mp4(~30 min,单卡)

```bash
PY_A2V=/disk/worldmodel/uv/envs/a2v/bin/python
CKPT=/disk/worldmodel/wangshilong/a2v_v4_khl/ckpts_mv/exp1/step-800.safetensors
DATASET=/disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv
OUT=/disk/worldmodel/wangshilong/a2v_v4_khl/concat_out
mkdir -p "$OUT"

for row in $(seq 0 19); do
  $PY_A2V -m a2v.infer_a2v --base_spec wan2.2-ti2v-5b \
    --dataset "$DATASET" --lora "$CKPT" \
    --row $row --num_frames 121 --height 256 --width 960 \
    --control real --output "$OUT/episode${row}.mp4"
done
# 产物:concat_out/episode0.mp4 ... episode19.mp4,每段 256×960 横拼
```

### 4.4 Bridge 切 + 命名 → TriWorldBench input

```bash
PY_A2V=/disk/worldmodel/uv/envs/a2v/bin/python
$PY_A2V /disk/worldmodel/wangshilong/a2v_v4_khl/a2v_to_triworld.py \
  --concat_infer_dir /disk/worldmodel/wangshilong/a2v_v4_khl/concat_out \
  --method a2v_wan22_ti2v_5b_mv3 \
  --output_dir /disk/worldmodel/wangshilong/TriWorldBench/input/a2v_wan22_ti2v_5b_mv3

# 验证
ls /disk/worldmodel/wangshilong/TriWorldBench/input/a2v_wan22_ti2v_5b_mv3/ | head -3
ls /disk/worldmodel/wangshilong/TriWorldBench/input/a2v_wan22_ti2v_5b_mv3/episode1/
# 期望:head.mp4 left.mp4 right.mp4
```

**重要**:
- A2V sample 名 `adjust_bottle__v4__episodeXXX__tra1__mv3` → 数字映射到 `episode{N}`,N 按 list 顺序
- **与 val_dataset 的 episode 顺序不一一对应**,先用自有 20 sample 做 dry-run 验证 pipeline 通,再考虑全 val 评测

---

## 5. TriWorldBench 评测(每次跑一次)

### 5.1 Preprocess(把 input mp4 → 帧树,~10 min,纯 CPU)

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
chmod +x run_eval.sh run_preprocess.sh

# 接口验证
./run_preprocess.sh a2v_wan22_ti2v_5b_mv3 --dry-run --limit 2

# 真跑
./run_preprocess.sh a2v_wan22_ti2v_5b_mv3
# 产物:generate/a2v_wan22_ti2v_5b_mv3_<input_ts>_<run_ts>/
#       含 episode*/{head,left,right}/frames/frame_*.jpg (320×240)
#       和 generate/<method>_<ts>_latest.path 软链指针
```

### 5.2 Edit config.yaml(`eval_input` 改为你刚生成的)

```bash
# 生成完 preprocess 后,latest 指针会自动指向最新 batch
# 直接让 config 指向 _latest 指针(已有默认)
cat /disk/worldmodel/wangshilong/TriWorldBench/config/config.yaml | grep eval_input
# eval_input: generate/<your_method>_latest
# ⚠️ 替换 <your_method> 为真实 method 名(同 §4.4)
sed -i 's|<your_method>|a2v_wan22_ti2v_5b_mv3|g' /disk/worldmodel/wangshilong/TriWorldBench/config/config.yaml
grep "eval_input:" /disk/worldmodel/wangshilong/TriWorldBench/config/config.yaml
```

### 5.3 Run Eval(1-8 卡,~1-2h)

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench

# Dry-run(不真跑,只列 metric)
./run_eval.sh --dry-run
# 期望:列出 19 个 metric 全 OK,无 import 错

# 真跑(单卡亦可,会自动分片)
./run_eval.sh
# 产物:output/a2v_wan22_ti2v_5b_mv3_<input_ts>_<run_ts>/
#       ├── final_results/episode_metrics.csv
#       └── final_results/summary_metrics.csv    ← TWB-Score 在这
```

### 5.4 验证最终产物

```bash
SUMMARY=$(find /disk/worldmodel/wangshilong/TriWorldBench/output \
  -name summary_metrics.csv -path '*final_results*' | tail -1)
echo "Found: $SUMMARY"
cat "$SUMMARY"
# 期望:19 个 metric 列 + TWB-Score 列
```

---

## 6. 已知坑(踩过就避开)

| 坑 | 解决 |
|---|---|
| **`HF_ENDPOINT` 没设** | 国内下载必设 `export HF_ENDPOINT=https://hf-mirror.com`,默认 huggingface.co 不可达 |
| **`requirements_no_deps.txt` 装不上退出** | 手动按 §2.2 三行 pip install(jepa + videojedi wheel + pyiqa) |
| **`python: python` 默认** | config.yaml 改成绝对路径 `/disk/worldmodel/uv/envs/triworldbench/bin/python`,见 §3.4 |
| **GPU 满** | `sinfo -p gpu -N -o "%N %T %G"` 看空闲;`--exclude=114-118`(`118` 是 shaoy VLM 服务) |
| **`xformers==0.0.35.dev1120` 装不上** | 影响不大,直接注释 requirements.txt 那行,其它 metric 仍可跑 |
| **`background_consistency` OOM** | config.yaml 里 `metric_parallel_per_gpu.background_consistency: 1`(38GB 显存峰),其它 metric 默认 |
| **val_dataset 与 A2V sample 不对应** | 先用 §4.4 自有 20 sample dry-run 验证 pipeline 通,再考虑全 val |
| **`photometric_smoothness` 缺 ResNet-34** | 必须预放 `weights/torch_home/hub/checkpoints/resnet34-b627a593.pth`(§3.2 已 symlink);计算节点无网会失败 = -10 分 |
| **`setuptools>=81`** | 已自动降级到 80.10.2;若 import 报 pkg_resources 错,`pip install 'setuptools<81'` |
| **`metric 17/18` CSV 显示是惩罚后分数** | 原始基础分落在 `logs/metric_17_*.log` / `logs/metric_18_*.log`,**别只看 CSV 就下结论** |
| **`action_following` 不在 19 个 metric 里** | TriWorldBench 已剔除(WorldArena 时代曾用),不要误用 |

---

## 7. Definition of Done(每步产物的验证标准)

| Step | DoD |
|---|---|
| 2.4 smoke | `[smoke] imports ok` + `all ok` |
| 3.1 val_dataset | `test_data/val/gt_dataset/` 100 个 episode + `STATE/` 100 个 `.json` |
| 3.3 weights verify | 13 个路径全 `[OK]` |
| 3.5 weight load | CLIP/RAFT/DINO/Tartan 加载无 exception |
| 4.1 dataset_mv | `dataset_mv/metadata.jsonl` 20 行 + 20 个 sample 目录 |
| 4.2 train | ckpt 文件 + tensorboard loss < 0.1 |
| 4.4 bridge | `input/<method>/episode*/{head,left,right}.mp4` 各 ≥1 |
| 5.1 preprocess | `generate/<method>_<ts>/manifest.json` 存在 + episode.json + frames/ |
| 5.3 eval | `output/<method>_<ts>/final_results/summary_metrics.csv` 存在 + 19 个 metric 列 |
| 5.4 final | summary_metrics.csv 可读 + TWB-Score 计算完成 |

---

## 8. 跨机迁移复现清单(checklist)

新机器按以下顺序跑(每条都是独立命令,可分多次执行):

- [ ] `PY=/disk/worldmodel/uv/envs/triworldbench/bin/python` 已有?没有就 `uv venv ... --python 3.10.0 --seed`
- [ ] `which ffmpeg ffprobe` 有输出
- [ ] `nvidia-smi` 有 ≥24GB 卡
- [ ] §2.1 `setup_a2v_env.sh` 跑完(~30-60 min)
- [ ] §2.2 三行手动 pip 装 no_deps
- [ ] §2.4 `smoke_imports.py` 通过
- [ ] §3.1 `val_dataset.tar.gz` 下载+解压完
- [ ] §3.2 weights symlink 13 行执行完
- [ ] §3.3 weights verify 全 OK
- [ ] §3.4 config.yaml python 改成绝对路径
- [ ] §3.5 关键权重加载验证通过
- [ ] §4 A2V 训练+推理完(自有 A2V 流程)
- [ ] §4.4 bridge 生成 input/<method>
- [ ] §5.1 preprocess 跑通,生成 generate/<method>_<ts>/
- [ ] §5.2 config.yaml eval_input 指向 _latest
- [ ] §5.3 eval 跑完,生成 output/<method>_<ts>/final_results/summary_metrics.csv
- [ ] §5.4 验证 summary_metrics.csv 有 TWB-Score

---

## 9. 关联文档

- `docs/A2V_SETUP.md` — A2V 适配初版,包含 A2V 训练细节
- `docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md` — 早期执行计划(本文是它的精简可执行版)
- `docs/METRICS_AND_IMPROVEMENT.md` — 19 metric 逐个拆解 + 提升策略
- `docs/A2V_LOCAL_EVAL_ISSUES_TODO.md` — 已知 bug + 未解决 issue(给接手的 agent)
- `DOWNLOAD_LINKS.md` — 完整 model 权重清单(本文档已直接复用,无需再读)
- `SUBMISSION_README.md` — **榜单提交流程**(500 ep, 320×240, 走 submission_check.py)
- `README.md` — TriWorldBench 官方 overview(看 metric 定义、TWB-Score 算法)

---

## 10. 一行命令:端到端 quick check(不真跑 metric)

环境搭好 + 数据齐了后,用这套命令在登录节点做完整 pipeline 烟雾测试:

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python

# 0. env ok
$PY scripts/smoke_imports.py

# 1. config ok
$PY -c "import sys; sys.path.insert(0,'scripts/lib'); from eval_common import load_config; from pathlib import Path; cfg=load_config(Path('config/config.yaml')); print('eval_input:', cfg['eval_input'])"

# 2. preprocess dry-run(用 input/ 下任意 method 即可)
mkdir -p input/smoke/episode1
for v in head left right; do
  ffmpeg -y -f lavfi -i color=c=black:s=320x240:r=15:d=2 -c:v libx264 input/smoke/episode1/${v}.mp4 2>/dev/null
done
./run_preprocess.sh smoke --limit 1
ls generate/smoke_*/episode1/head/frames/ | wc -l   # 应为 ~30(2秒×15fps)

# 3. eval dry-run
./run_eval.sh --dry-run   # 应列 19 个 metric 全 OK

# 4. 清理 smoke
rm -rf input/smoke generate/smoke_*
```

期望全部 OK → §4-§5 端到端流程已打通,可直接接 A2V 推理跑真评测。

---

## 11. 榜单提交工作流(从本地 eval 到 leaderboard)

> **本节补充 `SUBMISSION_README.md` 的执行细节**(仓库根那份是官方标准,本文不重复复制)。

### 11.1 何时提交

- ❌ **本地 eval TWB-Score < 0.50** — 不要提交,丢人
- 🟡 **0.50 - 0.55** — 可以提交试水,但预期榜单分数不会太好
- ✅ **≥ 0.55** — 准备好提交了
- ❓ **怎么估计榜单分?** — 本地 eval 只能给你一个**内部 baseline**,不是榜单保证。官方 val_dataset 和 test_dataset 是不同分布(虽然都是 RoboTwin2.0)

### 11.2 模型必须输出的内容(官方硬性规格)

```
<your_method>/               # 任意英文/数字名,提交前用 submission_check.py 验
├── episode1/
│   ├── head.mp4      # 320×240, match HDF5 帧数,30 FPS
│   ├── left.mp4
│   └── right.mp4
├── episode2/  ...(同上)
├── ...
└── episode500/...  (同上)
```

**硬性规格(不满足会报错)**:

| 维度 | 要求 | 失败后果 |
|---|---|---|
| episode 编号 | 必须 episode1..episode500,与 test_dataset 对齐 | checker ERROR,不能打包 |
| 每 ep 文件 | 必须是 `head.mp4` + `left.mp4` + `right.mp4` 三个文件 | checker ERROR |
| 分辨率 | **320×240**(与 `first_frame/episodeN_*.jpg` 一致) | checker ERROR(ffprobe 验证) |
| 帧数 | **match `data/episodeN.hdf5` 的 trajectory/action 长度**(从 `joint_action/vector` 第一维读) | 大幅差 WARN,小幅差 WARNING,推荐完全 match |
| FPS | 任意,推荐 30 | 不影响 metric |
| 编码 | MP4(H.264 标准即可) | checker 检查编码可读性 |

**软性建议**:
- 文件名不要带空格、中文
- 整个目录大小预估:500 ep × 3 view × ~500KB/ep ≈ ~750MB,网络传输压力不大

### 11.3 提交前数据准备(本地有 GT 时复用 bridge,无 GT 时直接生成)

**A2V 当前情况**:dataset_mv 只有 20 个 adjust_bottle 样本,**远远不够覆盖 500 ep 的 50 task**。

**短期提交方案**(只交 adjust_bottle 类的 8 个 ep,质量为先):
```bash
# 1. 选出 val_dataset 里所有 adjust_bottle episode(目前只有 ep1+ep2 两个)
#    如果 test_dataset 里 adjust_bottle 类有 8 个,挑出来
PY=/disk/worldmodel/uv/envs/a2v/bin/python
$PY -c "
import h5py, json
for i in range(1, 501):
    inst_path = f'/path/to/test_dataset/instructions/episode{i}.json'
    with open(inst_path) as f: data = json.load(f)
    if 'bottle' in data.get('task', '').lower():
        print(i, data['task'])
" > /tmp/adjust_bottle_eps.txt

# 2. 对这些 ep 跑 A2V 推理(同 §4.3 但要按 HDF5 帧数输出)
#    ⚠️ 当前 121 帧是固定超参,需要按每个 ep 的 HDF5 帧数动态调整
#    这部分代码改动工作: ~半天

# 3. Bridge 切三视角 → 320×240 输出(同 §4.4)
$PY_A2V /disk/worldmodel/wangshilong/a2v_v4_khl/a2v_to_triworld.py \
  --concat_infer_dir ... --method ... --output_dir <your_method>

# 4. 复制 test_dataset/first_frame/ 给的 first_frame 当作 GT 缺失时的参考
#    ⚠️ 但实际 model 是从 first_frame 出发生成的,所以不需要 GT
```

**完整提交方案**(扩 dataset_mv 到 50 task):
```bash
# 1. 扩 dataset_mv:从 v4_random10k_qhd_wrist 抽 50 task × 10 ep = 500 sample
# 2. 重训:数据集变大,可能要 5000+ step(Path A → 25h+ GPU)
# 3. 推理 500 ep × ~2min = ~17h 单卡
# 4. Bridge + submission_check.py
# 总工作量:~3-5 天
```

### 11.4 submission_check.py 验证

```bash
# 下载 submission_check.py(从 GitHub TriWorldBench/TriWorldBench/submission_check.py)
# 验证并打包
python submission_check.py /path/to/<your_method> --output /path/to/submission_output

# 输出:
#   /path/to/submission_output/<your_method>.zip
#   /path/to/submission_output/<your_method>.sha256.txt
```

checker 状态:
- `PASS` — 自动打包,直接发
- `WARNING` — 询问是否继续,可放行(常见:帧数差几帧)
- `ERROR` — 不打包,必须修

### 11.5 邮件提交

发到 `triworld_submit@foxmail.com`,邮件含三样:
1. `<your_method>.zip`(可挂百度网盘/Google Drive,必须能下载)
2. `<your_method>.sha256.txt`(checker 生成)
3. `README.txt`(手写):
   ```
   Model name:
   Contact email:
   Affiliation / Organization / Team Name:
   Project / Code repository:
   Brief introduction: (≤100 words)
   ```

### 11.7 本地 eval 与榜单分数的关系(重要警告)

| 本地 eval | 预期榜单 | 备注 |
|---|---|---|
| 0.45 | ~榜单 60+? | ❓ 不知道。本地只用 1 个 task subset |
| 0.50(ep1+ep2 baseline) | ? | 2 ep 太小,统计意义弱 |
| 跑全 100 val ep | 较可信 | 需要 dataset_mv 扩到 50 task |
| 跑完 500 test ep 提交后 | 真实榜单分 | 唯一可比数字 |

**⚠️ 不要拿本地 0.50 的 TWB-Score × 100 当 50 分发宣传**。榜单分只有官方跑完才算数。

---

## 12. 推荐阅读顺序(新接手 agent)

1. **本文档 §0 + §11**(本地与提交区分)
2. `README.md` — TriWorldBench metric 定义、TWB-Score 算法
3. `SUBMISSION_README.md` — 提交细节权威源
4. `docs/A2V_LOCAL_EVAL_ISSUES_TODO.md` — 已知 bug + 未解决 issue
5. `memory/a2v-patha-first-twb-score.md` — 首次 eval + ep12 baseline 真实数据
6. `memory/triworldbench-local-eval-setup.md` — 环境/weights 基建