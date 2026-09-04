# TriWorldBench 端到端打通 — Agent 执行规划

> **目标**：把 A2V 三视角世界模型接到 TriWorldBench 本地评测，产出 TWB-Score  
> **当前状态**：代码 + bridge + dry-run 都已就位，缺数据/weights/GPU  
> **给执行 agent**：按本文 checklist 逐步推进，每步有明确产物

---

## 0. 一句话目标

跑通以下端到端 pipeline 并拿到 `output/<method>/final_results/summary_metrics.csv`：

```
v4_random10k_qhd_wrist 数据
  → multiview_prepare.py 生成 dataset_mv
  → train_v4_mv.sbatch 训练出 ckpts_mv/exp1/step-XXX.safetensors
  → a2v.infer_a2v 生成 256×960 mp4
  → infer_multiview.py split 切成 {head,left,right}.mp4
  → a2v_to_triworld.py bridge 成 input/<method>/episodeN/
  → ./run_preprocess.sh <method>
  → ./run_eval.sh
  → summary_metrics.csv (TWB-Score)
```

---

## 1. 关键路径与环境

### 1.1 路径约定（已固定，不要改）

```
/disk/worldmodel/wangshilong/
├── DiffSynth-Studio/                        # A2V 仓库 (branch: a2v)
├── TriWorldBench/                           # TriWorldBench 仓库
│   ├── docs/A2V_SETUP.md                    # 完整复现文档
│   ├── docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md  # ← 本文档
│   ├── scripts/setup_a2v_env.sh             # uv env 一键 setup
│   ├── config/config.yaml                   # 评测配置
│   ├── input/<method>/                      # ← bridge 写入位置
│   ├── generate/<method>_latest/            # preprocess 输出
│   ├── output/<method>_<ts>/final_results/  # eval 输出
│   └── weights/                             # 待下载的 metric weights
└── a2v_v4_khl/                              # A2V 多视角工作目录
    ├── multiview_prepare.py                 # 三视角数据集生成
    ├── train_v4_mv.sbatch                   # 多视角训练 (SAVE_STEPS=500)
    ├── infer_multiview.py                   # 推理后切 3 段
    ├── a2v_to_triworld.py                   # A2V → TriWorldBench 桥接
    ├── dataset_mv/                          # 已生成 (20 样本, 3.4GB)
    ├── ckpts_mv/exp1/                       # 训练输出 (空, 等 GPU)
    └── logs/                                # slurm 日志
```

### 1.2 Python 环境

```bash
# A2V (训练/推理)
PY_A2V=/disk/worldmodel/uv/envs/a2v/bin/python
# 已装 torch 2.5.1+cu121 + h5py + imageio + bitsandbytes + diffsynth

# TriWorldBench (评测)
PY_TRI=/disk/worldmodel/uv/envs/triworldbench/bin/python
# 当前只装了 pyyaml + pillow, 缺 torch 2.10 + transformers + Qwen3-VL + ...
```

### 1.3 模型根

```bash
export A2V_MODELS_DIR=/disk/worldmodel/public_model
# 内含 Wan-AI/Wan2.2-TI2V-5B/{diffusion_pytorch_model-*.safetensors, Wan2.2_VAE.pth, ...}
```

### 1.4 Slurm 集群

```bash
# 分区: gpu (8×A100-80GB/节点)
# 必须排除: --exclude=ZJYKP-A100x8-INTEL-114-118 (shaoy VLM 服务常驻)
# 单卡够训练 (1×A100-80GB, ~55-65GB 显存)
```

---

## 2. 执行 Checklist (按顺序)

### Phase 1: 数据 + Weights 准备 (离线,可并行)

#### Step 1.1: 下载 val_dataset

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
huggingface-cli download TriWorldBench/Dataset \
  --include "val_dataset/*" \
  --local-dir /disk/worldmodel/wangshilong/TriWorldBench/

# 验证: ls test_data/val/gt_dataset/ | head -3  # 应有 episode*/ 子目录
# 验证: ls test_data/val/STATE/ | head -3       # 应有 episode*.json
```

**产物**: `test_data/val/gt_dataset/` + `test_data/val/STATE/` + `test_data/val/vlm_data_base/`

#### Step 1.2: 下载 metric weights (~30GB)

按 `DOWNLOAD_LINKS.md` 下载到 `/disk/worldmodel/wangshilong/TriWorldBench/weights/`。

**优先级 (按必要性)**:
| 权重 | 大小 | 必要性 |
|---|---|---|
| `weights/qwenvl3/` (Qwen3-VL-8B-Instruct) | ~16GB | 必须 (VLM 评测) |
| `weights/torch_home/hub/checkpoints/resnet34-b627a593.pth` | 100MB | 必须 (photometric_smoothness) |
| `weights/clip-vit-base-patch16/` | 600MB | 必须 (semantic_alignment) |
| `weights/clip_model/ViT-B-32.pt` + `ViT-L-14.pt` | 1.5GB | 必须 (multiple metrics) |
| `weights/dino_model/dino_vitbase16_pretrain.pth` | 330MB | 必须 |
| `weights/sam3/` (SAM3) | ~2GB | 必须 (trajectory_accuracy) |
| `weights/sam/`, `weights/vjepa/`, etc. | ~10GB | 必须 |

**一条命令全下 (示例)**:
```bash
export WEIGHTS=/disk/worldmodel/wangshilong/TriWorldBench/weights
mkdir -p "$WEIGHTS"
cd "$WEIGHTS"

# Qwen3-VL
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir qwenvl3
# CLIP
huggingface-cli download openai/clip-vit-base-patch16 --local-dir clip-vit-base-patch16
# Depth-Anything
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf --local-dir depth-anything
# SAM3
huggingface-cli download 1038lab/sam3 --local-dir sam3
# ResNet34
mkdir -p torch_home/hub/checkpoints
wget -O torch_home/hub/checkpoints/resnet34-b627a593.pth \
  https://download.pytorch.org/models/resnet34-b627a593.pth
# VBench/WorldScore weights (from HF dataset)
# ... 按 DOWNLOAD_LINKS.md 列表逐个下
```

#### Step 1.3: 装 TriWorldBench 完整依赖

```bash
bash /disk/worldmodel/wangshilong/TriWorldBench/scripts/setup_a2v_env.sh
# 或手动:
/disk/worldmodel/uv/envs/triworldbench/bin/pip install -r \
  /disk/worldmodel/wangshilong/TriWorldBench/requirements.txt
```

**注意事项**:
- `transformers` 从 git commit 安装(为 Qwen3-VL)
- `setuptools<81` 必要(TensorBoard pkg_resources)
- `--no-deps` 装 `requirements_no_deps.txt` 里的 pyiqa / jepa / 等

---

### Phase 2: 训练 (需要 GPU)

#### Step 2.1: 检查数据集已就绪

```bash
ls /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv/
# 应有 20 个 sample 目录 + metadata.jsonl
wc -l /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv/metadata.jsonl
# 应为 20
```

如果还没生成,跑:
```bash
PY=/disk/worldmodel/uv/envs/a2v/bin/python
$PY /disk/worldmodel/wangshilong/a2v_v4_khl/multiview_prepare.py \
  --root /disk/worldmodel/ziyang/projects/data_generation/khl_workspace/Data_output/v4_random10k_qhd_wrist \
  --task adjust_bottle \
  --episodes 53,68,70,231,240,247,273,297,486,506,582,602,658,670,730,779,783,804,820,962 \
  --tra 1 \
  --output_dir /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv \
  --per_view_height 256 --per_view_width 320 --num_frames 121
```

#### Step 2.2: 提交训练

```bash
sbatch /disk/worldmodel/wangshilong/a2v_v4_khl/train_v4_mv.sbatch
```

监控:
```bash
squeue -u sjtuadmin | grep a2v
tail -f /disk/worldmodel/wangshilong/a2v_v4_khl/logs/<jobid>.out
```

**预期产物**:
```
ckpts_mv/exp1/step-500.safetensors    # SAVE_STEPS=500 (新)
ckpts_mv/exp1/step-800.safetensors    # final (auto save)
ckpts_mv/exp1/tensorboard_log/
```

#### Step 2.3: 验证训练成功

```bash
PY=/disk/worldmodel/uv/envs/a2v/bin/python
$PY -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
files = glob.glob('/disk/worldmodel/wangshilong/a2v_v4_khl/ckpts_mv/exp1/tensorboard_log/events.out.tfevents.*')
ea = EventAccumulator(files[-1]); ea.Reload()
loss = ea.Scalars('loss')
print(f'loss: {len(loss)} pts, first={loss[0].value:.4g}, last={loss[-1].value:.4g}')
"
# 预期: last loss 显著下降 (< 0.1)
```

---

### Phase 3: 推理 + 切分 + Bridge

#### Step 3.1: 推理 (需要 GPU)

```bash
cd /disk/worldmodel/wangshilong/DiffSynth-Studio
export A2V_MODELS_DIR=/disk/worldmodel/public_model
PY=/disk/worldmodel/uv/envs/a2v/bin/python
CKPT=/disk/worldmodel/wangshilong/a2v_v4_khl/ckpts_mv/exp1/step-800.safetensors
DATASET=/disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv
OUT_DIR=/disk/worldmodel/wangshilong/a2v_v4_khl/concat_out
mkdir -p "$OUT_DIR"

# 单行 sanity (先跑 row 0 验证)
$PY -m a2v.infer_a2v \
  --base_spec wan2.2-ti2v-5b \
  --dataset "$DATASET" \
  --lora "$CKPT" \
  --row 0 --num_frames 121 --height 256 --width 960 \
  --control real --output "$OUT_DIR/episode53.mp4"

# 批量
for row in $(seq 0 19); do
  $PY -m a2v.infer_a2v --base_spec wan2.2-ti2v-5b \
    --dataset "$DATASET" \
    --lora "$CKPT" \
    --row $row --num_frames 121 --height 256 --width 960 \
    --control real --output "$OUT_DIR/episode${row}.mp4"
done
```

#### Step 3.2: Bridge 切分 + 命名

```bash
PY=/disk/worldmodel/uv/envs/a2v/bin/python
METHOD=a2v_wan22_ti2v_5b_mv3

# 输入是拼接 mp4,脚本自动 split + 转 TriWorldBench 格式
$PY /disk/worldmodel/wangshilong/a2v_v4_khl/a2v_to_triworld.py \
  --concat_infer_dir /disk/worldmodel/wangshilong/a2v_v4_khl/concat_out \
  --method $METHOD \
  --output_dir /disk/worldmodel/wangshilong/TriWorldBench/input/$METHOD

# 验证
ls /disk/worldmodel/wangshilong/TriWorldBench/input/$METHOD/ | head -3
ls /disk/worldmodel/wangshilong/TriWorldBench/input/$METHOD/episode1/  # 应有 head.mp4 left.mp4 right.mp4
```

**注意**: A2V 推理时 dataset_mv 里 sample 名字是 `adjust_bottle__v4__episodeXXX__tra1__mv3`,bridge 脚本提取 `episodeXXX` 部分,转成 `episode{N}`,注意 N 是按 list 顺序而不是数字顺序。如果 val_dataset 用纯 episode1..N,可能不对应,需要手动对齐。

**对齐策略**:bridge 脚本里可加 `--episode_mapping` 参数;或手动 cp。

---

### Phase 4: TriWorldBench 评测

#### Step 4.1: Preprocess

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
export PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
export CONFIG=config/config.yaml

# 先 dry-run 验证接口
./run_preprocess.sh a2v_wan22_ti2v_5b_mv3 --dry-run --limit 2

# 实际跑
./run_preprocess.sh a2v_wan22_ti2v_5b_mv3
```

**输出**: `generate/a2v_wan22_ti2v_5b_mv3_<timestamp>/`

#### Step 4.2: 配置 config.yaml

编辑 `config/config.yaml`:
```yaml
eval_input: generate/a2v_wan22_ti2v_5b_mv3_latest  # 使用 _latest 指针
metrics: all
output_dir: output
parallel_per_gpu: 4
metric_parallel_per_gpu:
  background_consistency: 1  # 38GB VRAM 需求
  semantic_alignment: 1
  subject_consistency: 2
  trajectory_accuracy: 4
gpu_list: "0,1,2,3,4,5,6,7"
python: /disk/worldmodel/uv/envs/triworldbench/bin/python
weights_root: /disk/worldmodel/wangshilong/TriWorldBench/weights
gt_dataset: /disk/worldmodel/wangshilong/TriWorldBench/test_data/val/gt_dataset
STATE: /disk/worldmodel/wangshilong/TriWorldBench/test_data/val/STATE
vlm_reference_db: /disk/worldmodel/wangshilong/TriWorldBench/test_data/val/vlm_data_base
```

#### Step 4.3: Run Evaluation

```bash
./run_eval.sh --dry-run    # 验证
./run_eval.sh              # 真跑
```

**输出**:
```
output/a2v_wan22_ti2v_5b_mv3_<input_ts>_<run_ts>/
└── final_results/
    ├── episode_metrics.csv
    └── summary_metrics.csv    # ← 目标产物
```

---

### Phase 5: 验证产物

```bash
cat /disk/worldmodel/wangshilong/TriWorldBench/output/a2v_wan22_ti2v_5b_mv3_*/final_results/summary_metrics.csv
# 应有 19 个 metric 列 + TWB-Score 列
```

---

## 3. 已知坑(务必看)

### 3.1 数据对齐(关键!)

A2V 训练的 sample 命名: `adjust_bottle__v4__episode53__tra1__mv3`
TriWorldBench val_dataset 命名: `episode1`, `episode2`, ... (扁平,按 RoboTwin 任务顺序)

**这两个不直接对应**。解决:
- **方案 A**: 用自己的 test split 评测(不跑 val_dataset),改 TriWorldBench 配置指向自己的 gt_dataset
- **方案 B**: 让 bridge 脚本支持 `--episode_mapping` 把 A2V name → TriWorldBench episodeN

**推荐**: 先在自己生成的 20 sample 上做 dry-run,验证 pipeline 通,再考虑全 val 评测。

### 3.2 GPU 资源

- 训练:单卡 1×A100-80GB ~55-65GB,需要 1-2h
- 推理:单卡 ~30 分钟(20 样本)
- 评测:1-8 卡,~1-2h

**当前集群 GPU 紧张**:
- 2676 (configur) 占 112-115 全部 32 卡
- q35-car 占 111
- 116/117 drained
- 118 我们 exclude

**建议**:用 `sinfo -p gpu -N -o "%N %T %G"` 看空闲节点,可能等几小时到 1 天。

### 3.3 wandb 不可用

slurm 没 wandb API key → 必须 `WANDB_MODE=offline` 或 `LOGGER=tensorboard`

### 3.4 TriWorldBench 评测环境依赖大

- `torch==2.10.0`(不是 2.5.1)
- `transformers` 从 git commit
- `xformers==0.0.35.dev1120`(可能装不上)
- 多个 `--no-deps` 包

**应对**:如果 requirements.txt 装失败,优先保证 VLM 评测能跑,其它 metric 可后续补。

---

## 4. Definition of Done

每一步的产物标准:

| Step | Done 标志 |
|---|---|
| 1.1 val_dataset 下载 | `test_data/val/gt_dataset/` + `test_data/val/STATE/` 存在,有 ≥100 个 episode |
| 1.2 weights 下载 | `weights/` 下所有 metric 权重存在;`smoke_imports.py` 通过 |
| 1.3 依赖装好 | `python -c "import torch, transformers, qwen_vl_utils"` 不报错 |
| 2.1 数据集 | `dataset_mv/metadata.jsonl` 20 行 + 20 个 sample 目录 |
| 2.2 训练 | job 状态 R,日志 `[train]` 行正常,loss 在 tensorboard 下降 |
| 2.3 训练验证 | `ckpts_mv/exp1/step-*.safetensors` 存在,最终 loss < 0.1 |
| 3.1 推理 | 20 个 256×960 mp4 产出,帧数=121 |
| 3.2 bridge | `TriWorldBench/input/<method>/episode*/{head,left,right}.mp4` 各 ≥1 |
| 4.1 preprocess | `generate/<method>_*` 目录有 frame tree |
| 4.2 config | `config.yaml` 指向正确路径 |
| 4.3 eval | `output/<method>*/final_results/summary_metrics.csv` 存在,有 ≥10 个 metric 列 |
| 5 验证 | summary_metrics.csv 可读,TWB-Score 计算完成 |

---

## 5. 时间估算

| 阶段 | 时间 | 阻塞 |
|---|---|---|
| 1.1 val_dataset 下 | 10-30 min | 网络 |
| 1.2 weights 下 | 30-120 min | 网络 |
| 1.3 依赖装 | 20-60 min | 网络 + 编译 |
| 2.1 数据准备 | 5 min | 已完成 |
| 2.2 训练 | 1-2 h | **GPU** |
| 2.3 验证 | 1 min | — |
| 3.1 推理 | 30 min | **GPU** |
| 3.2 bridge | 1 min | — |
| 4.1 preprocess | 10 min | weights |
| 4.3 eval | 1-2 h | GPU + weights |
| **总计** | **3-6 小时**(等 GPU 时间不计) | |

---

## 6. 失败应对

| 失败 | 应对 |
|---|---|
| wandb 错 | `WANDB_MODE=offline LOGGER=tensorboard` |
| GPU 满 | `sinfo` 看空闲,排队等 |
| weights 下载失败 | 单独下,看 HF proxy / 镜像 |
| xformers 装不上 | 跳过(PATH env 注释掉对应行) |
| val_dataset 与 A2V sample 命名不匹配 | 用 A2V 自己的 20 样本做测试,不跑全 val100 |
| TriWorldBench 评测脚本 import 失败 | 看 `logs/smoke_imports.log`,逐个装缺的包 |
| TriWorldBench GPU 不够 | 降 `parallel_per_gpu` 到 1,跑慢一点 |

---

## 7. 关键文件路径速查

```bash
# A2V 仓库
DIFFSYNTH=/disk/worldmodel/wangshilong/DiffSynth-Studio

# A2V 多视角工作目录
WORK=/disk/worldmodel/wangshilong/a2v_v4_khl

# 数据集
DATASET_MV=$WORK/dataset_mv          # 三视角拼接 (256×960, 20 样本)
DATASET_MV_SMOKE=$WORK/dataset_mv_smoke  # 5 样本冒烟版

# 模型权重
A2V_MODELS=/disk/worldmodel/public_model   # A2V 模型根
TWB_WEIGHTS=/disk/worldmodel/wangshilong/TriWorldBench/weights  # TriWorldBench 评测权重

# Python 环境
PY_A2V=/disk/worldmodel/uv/envs/a2v/bin/python
PY_TRI=/disk/worldmodel/uv/envs/triworldbench/bin/python

# 文档
DOC_SETUP=/disk/worldmodel/wangshilong/TriWorldBench/docs/A2V_SETUP.md
DOC_PLAN=/disk/worldmodel/wangshilong/TriWorldBench/docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md
```

---

## 8. 已有交付清单

```bash
$WORK/
├── v4_adapter.py              # 单视角 adapter (本次未用)
├── multiview_prepare.py       # 三视角数据集生成 ✓
├── train_v4.sbatch            # 单视角训练 (SAVE_STEPS=500)
├── train_v4_mv.sbatch         # 多视角训练 (SAVE_STEPS=500) ✓
├── infer_multiview.py         # 切 3 段 mp4 ✓
├── a2v_to_triworld.py         # Bridge: A2V → TriWorldBench ✓
├── dataset/                   # 单视角数据集 (1.2GB)
├── dataset_mv/                # 三视角数据集 (3.4GB) ✓
├── dataset_mv_smoke/          # 5 样本冒烟 (912MB)
├── work/                      # adapter 输出
├── ckpts/exp1/                # 单视角 ckpt (step-500, step-800) ✓
└── logs/                      # slurm 日志
```

TriWorldBench 仓库交付:
```
/disk/worldmodel/wangshilong/TriWorldBench/
├── docs/A2V_SETUP.md                      # 完整复现文档 ✓
├── docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md    # ← 本文档 ✓
└── scripts/setup_a2v_env.sh               # uv env 一键 setup ✓
```

---

## 9. 开始执行

**第一步**:检查环境,缺什么补什么:
```bash
# 1. 检查 val_dataset
ls /disk/worldmodel/wangshilong/TriWorldBench/test_data/ 2>&1

# 2. 检查 weights
ls /disk/worldmodel/wangshilong/TriWorldBench/weights/ 2>&1

# 3. 检查 triworldbench 依赖
$PY_TRI -c "import torch; print(torch.__version__)" 2>&1

# 4. 检查 GPU
sinfo -p gpu -N -o "%N %T %G"

# 5. 检查 A2V 数据集
ls /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv/metadata.jsonl
```

按 §2 checklist 推进,遇问题查 §6 失败应对,需深入参考 §1 文档 + §3 坑 + §4 DoD。
