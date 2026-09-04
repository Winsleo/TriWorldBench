# A2V Path A: TriWorldBench 评测对接文档

> **用途**：把 A2V 三视角拼接世界模型接入 TriWorldBench 本地评测  
> **适配的 benchmark**：[TriWorldBench](https://github.com/TriWorldBench/TriWorldBench) — 首个三视角具身世界模型基准，评测 head + left_wrist + right_wrist 三视角一致性  
> **最后一次更新**：2026-08-31

---

## 目录

1. [端到端 pipeline 总览](#1-端到端-pipeline-总览)
2. [环境搭建](#2-环境搭建)
3. [数据准备](#3-数据准备)
4. [A2V 三视角训练](#4-a2v-三视角训练)
5. [推理 + 切分 → TriWorldBench 输入格式](#5-推理--切分--triworldbench-输入格式)
6. [TriWorldBench 评测流程](#6-triworldbench-评测流程)
7. [跨机迁移复现](#7-跨机迁移复现)
8. [已知限制](#8-已知限制)

---

## 1. 端到端 pipeline 总览

```
v4_random10k_qhd_wrist (RoboTwin 数据)
    │  multiview_prepare.py (渲染 3 视角 trajectory map + 横向拼接)
    ▼
dataset_mv/ (256×960, 121帧, ~3.4GB for 20 samples)
    │  a2v.train.sh wan2.2-ti2v-5b (DiT + VACE, NPROC=1 GPU)
    ▼
ckpts_mv/exp1/step-NNNN.safetensors
    │  a2v.infer_a2v (推理生成 H×3W 拼接 mp4)
    ▼
infer_multiview.py split (切 3 段 → head.mp4, left.mp4, right.mp4)
    │  a2v_to_triworld.py bridge (episode 命名规范化)
    ▼
TriWorldBench/input/<method>/episodeN/{head,left,right}.mp4
    │  ./run_preprocess.sh <method>
    ▼
generate/<method>_<input_ts>_latest/
    │  ./run_eval.sh
    ▼
output/<method>_.../final_results/summary_metrics.csv (TWB-Score)
```

---

## 2. 环境搭建

### 2.1 Python 环境 (uv, 不用 conda)

```bash
# 创建 uv env (Python 3.10)
uv venv /disk/worldmodel/uv/envs/triworldbench --python 3.10.0 --seed
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python

# 安装 TriWorldBench 依赖
cd /disk/worldmodel/wangshilong/TriWorldBench
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt
$PY -m pip install --no-deps -r requirements_no_deps.txt
```

> **注意**：requirements.txt 包含 torch 2.10 + CUDA 版本，集群需确认 CUDA 兼容。如果只想 dry-run 验证接口，只需 `pip install pyyaml pillow`。

### 2.2 系统依赖

- `ffmpeg` / `ffprobe`（安装在 PATH 里）
- CUDA-capable GPU (评测最高峰值 ~23GB/单卡)

### 2.3 模型权重

详见 `DOWNLOAD_LINKS.md`。按 metrics 目录放置到 `./weights/`：

```bash
# Qwen3-VL-8B-Instruct (VLM 评测, ~16GB)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir weights/qwenvl3

# 其他 weights 按 DOWNLOAD_LINKS.md 下载
```

---

## 3. 数据准备

### 3.1 A2V 原始数据 (v4_random10k_qhd_wrist)

```
/disk/worldmodel/ziyang/projects/data_generation/khl_workspace/Data_output/v4_random10k_qhd_wrist/
    └── <task>/
        └── <episodeXXX>/
            ├── .seed_state.json
            ├── tra1/
            │   ├── data/episodeXXX.hdf5   ← actions + 多相机
            │   ├── video/episodeXXX.mp4   ← 主视角 (head, 1920×1440)
            │   ├── video/episodeXXX_left.mp4  ← 左腕
            │   ├── video/episodeXXX_right.mp4 ← 右腕
            │   └── run_log.json
            └── tra2..tra10/
```

**HDF5 内部结构**：

| 路径 | shape | 说明 |
|---|---|---|
| `endpose/left_endpose` | (T, 7) | [x,y,z,qx,qy,qz,qw] (TCP-frame) |
| `endpose/right_endpose` | (T, 7) | 同上 |
| `endpose/left_gripper` | (T,) | gripper [0,1] |
| `endpose/right_gripper` | (T,) | 同上 |
| `observation/head_camera/intrinsic_cv` | (3, 3) | OpenCV K |
| `observation/head_camera/extrinsic_cv` | (T, 3, 4) | w2c (OpenCV) |
| `observation/{left,right}_camera/...` | 同上 | 左右腕相机 |

**meta 读取规则**：`<tra>/json/metadata.json` 存在则读 `task_definition.instruction_en`；否则 fallback 到同 episode 其他 tra 的 metadata.json，再 fallback 到 `<episode>/json/metadata.json`。

### 3.2 multiview_prepare.py 输出 (A2V 训练数据集)

```bash
python /disk/worldmodel/wangshilong/a2v_v4_khl/multiview_prepare.py \
  --root <v4_random10k_qhd_wrist ROOT> \
  --task <task> --episodes <list> --tra 1 \
  --output_dir /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv \
  --per_view_height 256 --per_view_width 320 --num_frames 121
```

输出目录结构：

```
dataset_mv/
├── metadata.jsonl                          (每行一个 sample, 标准 DiffSynth JSONL)
├── <task>__v4__episodeXXX__tra1__mv3/
│   ├── video/00000..00120.png              (256×960, 3视角拼接)
│   ├── vace_video/00000..00120.png         (256×960, 3 trajectory map 拼接)
│   └── reference/00000.png                (256×960, 首帧)
└── ...
```

- **拼接顺序**: [head | left | right]（从左到右）
- **每视角分辨率**: 256×320（Wan2.2 VAE 16× → 32×40 latent）
- **总分辨率**: 256×960 = 32×60 latent，合法
- **Trajectory map 渲染**: 每视角用各自相机参数（intrinsic + extrinsic）独立投影，然后横向拼接

---

## 4. A2V 三视角训练

### 4.1 训练脚本

```bash
sbatch /disk/worldmodel/wangshilong/a2v_v4_khl/train_v4_mv.sbatch
```

关键参数 (均为 train_v4_mv.sbatch 里的 env 变量)：

| 参数 | 值 | 说明 |
|---|---|---|
| `DATASET` | `/disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv` | multiview_prepare.py 输出 |
| `HEIGHT` / `WIDTH` | `256` / `960` | 3 视角拼接总宽 |
| `FRAMES` | `121` | 4n+1 变长 cap |
| `REPEAT` / `EPOCHS` | `4` / `10` | 20样本 × 4 × 10 = 800 步 |
| `LR` / `LR_SCHEDULE` | `1e-5` / `cosine` | cosine warmup=20 |
| `OPTIMIZER` | `adamw_offload` | 5B 模型 offload ~55-65GB/卡 |
| `LOGGER` | `tensorboard` | slurm 无 wandb API key |
| `--exclude` | `…-118` | shaoy VLM 服务常驻 OOM |
| `OUT` | `ckpts_mv/exp1` | checkpoint 输出 |

### 4.2 预期产物

```
ckpts_mv/exp1/
├── step-20.safetensors
├── step-40.safetensors
├── ... (每 20 步保存一个)
├── step-800.safetensors        ← 最终 checkpoint
└── tensorboard_log/
```

### 4.3 GPU 资源需求

| 项目 | 需求 |
|---|---|
| 单卡 A100-80GB | ~55-65GB 显存（adamw_offload） |
| 磁盘 | 3.4GB 数据 + 5GB ckpt = ~10GB |
| 时间 | ~2h（800 步，单卡） |

---

## 5. 推理 + 切分 → TriWorldBench 输入格式

### 5.1 单样本推理 (a2v.infer_a2v)

```bash
cd /disk/worldmodel/wangshilong/DiffSynth-Studio
export A2V_MODELS_DIR=/disk/worldmodel/public_model
PY=/disk/worldmodel/uv/envs/a2v/bin/python
CKPT=/disk/worldmodel/wangshilong/a2v_v4_khl/ckpts_mv/exp1/step-800.safetensors

# 跑每行推理 (control=real = 正常轨迹控制; control=none = 空白控制)
$PY -m a2v.infer_a2v \
  --base_spec wan2.2-ti2v-5b \
  --dataset /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv \
  --lora "$CKPT" \
  --row 0 --num_frames 121 --height 256 --width 960 \
  --control real --output /tmp/gen_real_row0.mp4
```

### 5.2 批量推理 (可选批量跑)

```bash
# 多行批量
for row in $(seq 0 19); do
  $PY -m a2v.infer_a2v --base_spec wan2.2-ti2v-5b \
    --dataset /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv \
    --lora "$CKPT" \
    --row $row --num_frames 121 --height 256 --width 960 \
    --control real --output /tmp/gen_real_row${row}.mp4
done
```

### 5.3 切分 3 视角 (infer_multiview.py)

```bash
PY=/disk/worldmodel/uv/envs/a2v/bin/python

$PY /disk/worldmodel/wangshilong/a2v_v4_khl/infer_multiview.py split \
  --input /tmp/gen_real_row0.mp4 \
  --output_dir /tmp/gen_real_row0_split \
  --views head,left,right

# 输出:
# /tmp/gen_real_row0_split/head.mp4
# /tmp/gen_real_row0_split/left.mp4
# /tmp/gen_real_row0_split/right.mp4
```

### 5.4 桥接 → TriWorldBench 输入 (a2v_to_triworld.py)

```bash
PY=/disk/worldmodel/uv/envs/a2v/bin/python
METHOD=a2v_wan22_ti2v_5b_mv3

# 方式 A: 如果已经 split 好 (epN/{head,left,right}.mp4)
$PY /disk/worldmodel/wangshilong/a2v_v4_khl/a2v_to_triworld.py \
  --infer_dir /tmp/infer_out_split \
  --method $METHOD \
  --output_dir /disk/worldmodel/wangshilong/TriWorldBench/input/$METHOD

# 方式 B: 如果是拼接 mp4 (*.mp4)
$PY /disk/worldmodel/wangshilong/a2v_v4_khl/a2v_to_triworld.py \
  --concat_infer_dir /tmp/gen_concat \
  --method $METHOD \
  --output_dir /disk/worldmodel/wangshilong/TriWorldBench/input/$METHOD
```

输出：

```
TriWorldBench/input/a2v_wan22_ti2v_5b_mv3/
├── episode1/
│   ├── head.mp4
│   ├── left.mp4
│   └── right.mp4
├── episode2/
│   ├── ...
```

---

## 6. TriWorldBench 评测流程

### 6.1 下载 Val Dataset (100 episodes)

```bash
# Hugging Face: TriWorldBench/Dataset
huggingface-cli download TriWorldBench/Dataset \
  --include "val_dataset/*" \
  --local-dir /disk/worldmodel/wangshilong/TriWorldBench/
```

解压后结构：

```
/test_data/val/gt_dataset/   (GT 三视角参考帧)
/test_data/val/STATE/        (episode 状态标注, JSON)
/test_data/val/vlm_data_base/ (VLM 少样本参考 DB)
```

### 6.2 Preprocess

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
source /disk/worldmodel/uv/envs/triworldbench/bin/activate

# config/config.yaml 里把:
#   eval_input: generate/a2v_wan22_ti2v_5b_mv3_latest
#   python: /disk/worldmodel/uv/envs/triworldbench/bin/python

./run_preprocess.sh a2v_wan22_ti2v_5b_mv3
# dry-run: ./run_preprocess.sh a2v_wan22_ti2v_5b_mv3 --dry-run
```

输出：`generate/a2v_wan22_ti2v_5b_mv3_<timestamp>/`（resampled + resized frames）

### 6.3 Run Evaluation

```bash
./run_eval.sh
# dry-run: ./run_eval.sh --dry-run
# config: CONFIG=config/config.yaml ./run_eval.sh
```

最终结果：

```
output/<method>_<input_ts>_<run_ts>/final_results/
├── episode_metrics.csv    (per-episode per-metric 分数)
└── summary_metrics.csv    (聚合成 TWB-Score)
```

### 6.4 GPU 资源 (评测)

| 项 | 需求 |
|---|---|
| GPU | 1-8 × A100(40GB+)；峰值 ~23GB/单卡 |
| 最慢 metric | VLM consistency（需 Qwen3-VL-8B ~16GB） |
| 100 ep 完整评测 | ~2h (8卡) / ~4h (4卡) |

---

## 7. 跨机迁移复现

### 7.1 必备资源清单

| 资源 | 大小 | 来源 |
|---|---|---|
| **DiffSynth-Studio** (branch: a2v) | ~10GB | git clone + checkout a2v |
| **TriWorldBench** | ~500MB | git clone (GitHub) |
| **A2V 模型权重** (Wan2.2-TI2V-5B) | ~20GB | HF: `Wan-AI/Wan2.2-TI2V-5B` |
| **TriWorldBench 权重** (VLM + metrics) | ~30GB | HF: 详见 DOWNLOAD_LINKS.md |
| **Val Dataset** (100 ep) | ~5GB | HF: `TriWorldBench/Dataset` |
| **v4_random10k_qhd_wrist** 数据 | ~100GB+ | 项目内部数据 |
| **uv envs** (a2v + triworldbench) | ~15GB | pip install |

### 7.2 路径约定

| 变量 | 用途 | 当前值 |
|---|---|---|
| `A2V_MODELS_DIR` | A2V 模型根 | `/disk/worldmodel/public_model` |
| `PY_A2V` | A2V Python | `/disk/worldmodel/uv/envs/a2v/bin/python` |
| `PY_TRI` | TriWorldBench Python | `/disk/worldmodel/uv/envs/triworldbench/bin/python` |
| `DIFFSYNTH_ROOT` | DiffSynth-Studio 仓库根 | `/disk/worldmodel/wangshilong/DiffSynth-Studio` |
| `TRIWORLD_ROOT` | TriWorldBench 仓库根 | `/disk/worldmodel/wangshilong/TriWorldBench` |
| `WORK_ROOT` | A2V 多视角工作目录 | `/disk/worldmodel/wangshilong/a2v_v4_khl/` |
| `CLUSTER_EXCLUDE_NODES` | 排除的 slurm 节点 | `ZJYKP-A100x8-INTEL-114-118` (shaoy VLM) |

### 7.3 迁移步骤

```bash
# 1. 克隆仓库
git clone <DiffSynth-Studio repo> && cd DiffSynth-Studio && git checkout a2v
git clone https://github.com/TriWorldBench/TriWorldBench.git

# 2. 创建 uv envs
uv venv /path/to/envs/a2v --python 3.11 --seed
uv venv /path/to/envs/triworldbench --python 3.10.0 --seed
# 装依赖 (见 §2.1)

# 3. 下载模型权重 (A2V + TriWorldBench)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan-AI/Wan2.2-TI2V-5B
huggingface-cli download TriWorldBench/Dataset --local-dir TriWorldBench/

# 4. 数据准备
python multiview_prepare.py --root <v4_random10k ROOT> --task <task> \
  --episodes <list> --tra 1 --output_dir <output>

# 5. 训练 (等 GPU)
sbatch train_v4_mv.sbatch

# 6. 推理 + 切分 + bridge
python a2v_to_triworld.py --concat_infer_dir <concat_out> \
  --method a2v_wan22_ti2v_5b_mv3 \
  --output_dir TriWorldBench/input/a2v_wan22_ti2v_5b_mv3

# 7. 评测
cd TriWorldBench && ./run_preprocess.sh a2v_wan22_ti2v_5b_mv3
./run_eval.sh
```

### 7.4 集群注意事项

- **节点排除**: `--exclude=ZJYKP-A100x8-INTEL-114-118`（shaoy VLM 服务常驻，slurm idle 但 OOM）
- **wandb**: slurm 无 API key → 训练时 `WANDB_MODE=offline`，评测不需要 wandb
- **torchhub cache**: `TORCH_HOME=/disk/worldmodel/public_model/torch_hub` 或对无网络节点预缓存 ResNet-34

---

## 8. 已知限制

### Path A (横向拼接)

| 限制 | 原因 | 影响 |
|---|---|---|
| **无跨视角注意力** | DiT 处理 256×960 单视频，不知道哪个区域对应哪个视角 | Tri-View Consistency 得分预期低于专门多视角架构 |
| **trajectory map 语义弱** | 每视角 trajectory map 独立投影后拼接，DiT 无法区分跨视角位置关系 | head 看到抓完时 wrist 可能纹丝不动 |
| **后续扩展路径** | Path B: 原生多 VACE + cross-view attention (2-3 周) | 能上 TriWorldBench 榜单 |

### 当前阻塞

- **GPU 资源不足**: job 2678 (单卡训练) PD 等待，2676 占满 32 卡
- **评测 weights 下载**: 需要联网 + ~30GB，集群某些节点无外网

---

## 9. 文件索引

| 文件 | 用途 |
|---|---|
| `v4_adapter.py` | 单视角 adapter（非 TriWorldBench 路径） |
| `multiview_prepare.py` | 三视角拼接数据集生成 |
| `train_v4_mv.sbatch` | 多视角训练 sbatch |
| `infer_multiview.py` | 推理后切 3 段独立 mp4 |
| `a2v_to_triworld.py` | Bridge: A2V 切分 → TriWorldBench input/ 格式 |
| `train_v4.sbatch` | 单视角训练 sbatch（对比基线） |
| `v4_adapter.py` | 单视角 adapter |
