# A2V 训练经验汇总(给其他 agent 借鉴)

> **作者**:wangshilong + ziyang 训练经验汇总
> **最后更新**:2026-09-03
> **目的**:让任何接手 A2V 训练的 agent 能立刻知道"分辨率/帧数 → 资源"映射、关键参数、踩过的坑

---

## 1. 经验来源

| 来源 | 路径 | 用途 |
|---|---|---|
| **ziyang 主仓库** | `/disk/worldmodel/ziyang/projects/a2v/a2v_khl_v2/` | 主线生产训练(2-4 节点) |
| **ziyang 老分支** | `/disk/worldmodel/ziyang/a2v_khl_v2/` | 早期 2 节点 rf-direct 训练 |
| **wangshilong Path A** | `/disk/worldmodel/wangshilong/a2v_v4_khl/` | 单卡 sanity + 256×960 baseline |
| **官方 adapter** | `/disk/worldmodel/wangshilong/DiffSynth-Studio/a2v/` | 模型定义 + 训练入口 |

**ziyang 的关键 sbatch**:
- `train_v3w_rf_direct.sbatch` — 2 节点 × 8 GPU,**on-the-fly 编码**,random+clean 4:1 混合,~2.7h
- `train_v3w_v3bgr_4node.sbatch` — 4 节点 × 8 GPU,全量 unique1999 + cache,~12h
- `cache_train_v3w_rf_15k.sbatch` — 预 cache 路线
- `encode_v3w_rf_15k_4n.sbatch` — 多节点并行 cache 编码

---

## 2. 训练参数参考表(ziyang 实际生产配置)

| 项 | ziyang v3w_rf_direct | ziyang v3bgr_4node | wangshilong Path A | **推荐(640×480)** |
|---|---|---|---|---|
| base model | `wan2.2-ti2v-5b-khl` | 同左 | `wan2.2-ti2v-5b` | `wan2.2-ti2v-5b` |
| 分辨率(H×W) | **224×960**(3 视角拼)| 224×960 | 256×960(3 视角拼)| **480×640**(单视角)|
| 帧数 | 121 | 121 | 121 | **121** |
| 数据集 | rf_data/metadata_mix_4to1.jsonl | cache_v3w_v3bgr_unique1999 | dataset_mv(20 ep)| dataset_hires(待建)|
| 节点 × GPU | 2 × 8 = 16 | 4 × 8 = 32 | 1 × 1 = 1 | **2 × 8 = 16** |
| 优化器 | Adam8bit | Adam8bit | adamw_offload | **Adam8bit** |
| 精度 | bf16 | bf16 | bf16 | bf16 |
| 梯度策略 | grad_ckpt(use_gradient_checkpointing) | 同左 | grad_ckpt + offload | **grad_ckpt** |
| LR | 1e-5 constant | 1e-5 constant | 1e-5 cosine + warmup20 | 1e-5 constant |
| 训练目标 | VACE only | VACE only | VACE only | **VACE only** |
| dataset_repeat | 1 | 1 | 4 | 20(小数据集需要多 repeat)|
| num_epochs | 999(max ~1 epoch 实际)| 999 | 10 | 999 |
| save_steps | 100 | 500 | 500 | 100 |
| log_every | 50 | — | — | 50 |
| 批大小 / GPU | 1 | 1 | 1 | **1**(空间已极限)|
| 单步时间 | ~5s | ~5s | ~3s(256×960) | **~7s**(640×480)|
| 总样本数 | 31k | 1999 | 20×4=80 | 200×20=4000 |
| 总耗时 | ~2.7h | ~12h | ~1h | **~4-6h** |

---

## 3. 分辨率 → 资源映射表

### 关键计算式

```
spatial tokens = (H/16) × (W/16)        # VAE 16× 空间压缩
temporal tokens = num_frames             # 时序 self-attention
latent dim = 48                          # VAE z_dim for Wan
```

### 各分辨率 token 数 + 显存估算

| 分辨率 | H/16 × W/16 | tokens | 占 320×240 比例 | 单卡 VRAM(估)| 单步时间 |
|---|---:|---:|---:|---:|---:|
| **256×320** | 16×20 | 320 | 1.07× | ~40 GB | ~3s |
| **320×240** | 20×15 | 300 | 1.00×(基线)| ~38 GB | ~3s |
| **256×960**(3 视角)| 16×60 | 960 | 3.2× | ~45 GB | ~3s |
| **224×960**(ziyang)| 14×60 | 840 | 2.8× | ~45 GB | ~5s |
| **480×640**(目标)| 30×40 | 1200 | 4.0× | **~60 GB** | **~7s** |
| **480×960** | 30×60 | 1800 | 6.0× | ~75 GB | ~10s |
| **720×1280** | 45×80 | 3600 | 12× | OOM 单卡 | 不可行 |

### 单 GPU VRAM 公式

```
params(bf16)        = 5B × 2 bytes     = 10 GB
gradients(bf16)     = 5B × 2 bytes     = 10 GB  
adam8bit_states     = 5B × 1 byte × 2  = 5-7 GB
activations(peak)   ≈ tokens × 4096 × 4 bytes × DiT_blocks / sqrt(grad_ckpt_factor)
                    ≈ tokens × 1.5 GB     (经验值,带 grad_ckpt)

total ≈ 25 + tokens × 1.5 GB  (大致线性关系)
```

**A100-80GB 安全边界**:total ≤ 70 GB(留10GB 余量给系统/cudnn)

---

## 4. 节点配置推荐矩阵

| 场景 | 节点 × GPU | 适用分辨率 | 适用数据集 | 耗时 |
|---|---|---|---|---|
| **Sanity 验证** | 1 × 1 = 1 | ≤ 320×240 | 10-20 ep | ~1h |
| **小规模**(20-100 ep)| 1 × 4 = 4 | ≤ 256×960 | 50-100 ep | ~2-4h |
| **中等生产**(200-1000 ep)| 2 × 8 = 16 | **480×640**(目标)| 200 ep | **~4-6h** |
| **大规模全量**(10k+ ep)| 4 × 8 = 32 | ≤ 480×640 | 1000+ ep | ~12-24h |
| **超清** | 4 × 8 = 32 + grad_ckpt | 720×1280 | 小 | 多天 |

---

## 5. Cache vs On-the-fly 决策

| 模式 | 适用场景 | cache 大小(200 ep × 121 f × 480×640) |
|---|---|---|
| **On-the-fly**(ziyang rf_direct)| 大数据集(>10k,cache 不可行)| 不需要 cache,每个 step 重新编码 |
| **Pre-cache + train**(推荐 200 ep 级别)| 小-中等数据集 | 200 × 121 × 1(view) × 1200 × 48 × 4 byte = **16.7 GB** |
| **3 视角拼接**| 视角拼接方案 | 200 × 121 × 3(view) × ... = **50 GB** |

**推荐**:200 ep 用 pre-cache(可控、可重试),10000 ep 用 on-the-fly。

---

## 6. 必看踩过的坑

| 坑 | 现象 | 修复 | 记录位置 |
|---|---|---|---|
| **`HF_ENDPOINT` 未设** | HF 下不动 | 设 `https://hf-mirror.com` | `setup_a2v_env.sh` |
| **`WANDB_MODE=online`** 多机卡死 | wandb preflight 联网挂死,零进度 | `WANDB_MODE=offline + srun --unbuffered` | memory `diffsynth-multinode-wandb-hang` |
| **`shaoy VLM 服务` 占用 118** | 任务被 OOM kill | `sbatch --exclude=ZJYKP-A100x8-INTEL-118` | 所有 sbatch 必加 |
| **VACE init 不存在** | 训 VACE 失败 | 5B base 自带 VACE,无需额外 init | ziyang `VACE_INIT` 环境变量可选 |
| **数据格式不对**(v4_random10k 不是 robotwin_adapter 期望)| `prepare.py` 找不到文件 | 写自定义 adapter | `a2v_v4_khl/v4_adapter.py` |
| **grad_ckpt_offload** 比单纯 grad_ckpt 慢但省显存 | 训练慢 30% | 5B + 480×640 单卡必须开 offload | memory `embodied-world-r1-eval-infra` |
| **photometric_smoothness ResNet34** 计算节点无网下不动 | eval 失败 | 预放到 `weights/torch_home/hub/checkpoints/` | `triworldbench-local-eval-setup` |
| **action_following 已废弃** | 用错 metric | 用 TriWorldBench 19 metric | ziyang + TWB 都不在 19 |

---

## 7. 480×640 × 121 帧 推荐起手 sbatch 模板

```bash
#!/bin/bash
# A2V training at 480x640x121 frames, 5B base, 2 nodes x 8 GPU = 16 GPU
# Cache required (200 ep cache = ~17 GB)
#SBATCH --job-name=a2v-hires
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --exclude=ZJYKP-A100x8-INTEL-118
#SBATCH --output=/disk/worldmodel/wangshilong/a2v_v4_hires/logs/%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export A2V_MODELS_DIR=/disk/worldmodel/public_model
export PY=/disk/worldmodel/uv/envs/a2v/bin/python
export WANDB_MODE=offline                    # 防止多机卡死
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# 不要 export WANDB_API_KEY(会让 online preflight 触发)

ROOT=/disk/worldmodel/wangshilong/a2v_v4_hires
DATASET=$ROOT/dataset_hires
CACHE=$ROOT/cache_hires
OUT=$ROOT/ckpts_hires/exp1
mkdir -p "$ROOT/logs" "$CACHE" "$OUT"

NUM_NODES=2
NUM_PROCESSES=$(( NUM_NODES * 8 ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29721

cd /disk/worldmodel/wangshilong/DiffSynth-Studio

# 步骤 1: 单节点 cache 编码(200 ep × 121f × 480×640)
if [ ! -d "$CACHE" ] || [ "$(ls -A $CACHE 2>/dev/null)" = "" ]; then
    echo "[$(date)] Cache missing — encoding..."
    $PY -m a2v.data.cache_encode \
        --manifest $DATASET/manifest.jsonl \
        --output_dir $CACHE \
        --height 480 --width 640 --num_frames 121
fi

# 步骤 2: 多机训练
srun --ntasks="$NUM_NODES" --ntasks-per-node=1 --unbuffered bash -c "
  echo \"[rank \$SLURM_PROCID] \$(hostname)\"
  $PY -m accelerate.commands.launch \
    --num_machines '$NUM_NODES' \
    --num_processes '$NUM_PROCESSES' \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    --mixed_precision bf16 \
    a2v.train_a2v \
    --base_spec wan2.2-ti2v-5b \
    --cache_train \
    --dataset_base_path '$CACHE' \
    --data_file_keys 'video,vace_video' \
    --extra_inputs 'vace_video,input_image' \
    --height 480 --width 640 --num_frames 121 \
    --trainable_models vace \
    --customized_optimizer bitsandbytes.optim.Adam8bit \
    --use_gradient_checkpointing \
    --learning_rate 1e-5 \
    --lr_schedule constant \
    --dataset_repeat 20 \
    --num_epochs 999 \
    --save_steps 100 \
    --output_path '$OUT' \
    --log_every 50 \
    --log_grad_norm 1
"
```

### 起手前 sanity check(必做)

```bash
# 1. 单卡能跑 + 显存够
PY=/disk/worldmodel/uv/envs/a2v/bin/python
A2V_MODELS_DIR=/disk/worldmodel/public_model \
$PY -m a2v.check_load --base_spec wan2.2-ti2v-5b \
    --height 480 --width 640 --num_frames 121
# 看 DiT 层数/VAE z=48/造 vace 形状

# 2. 跑 1 step 看实际显存占用
HEIGHT=480 WIDTH=640 FRAMES=121 \
REPEAT=1 EPOCHS=1 SAVE_STEPS=1 \
LR=1e-5 DATASET=$DATASET OUT=$ROOT/ckpts_hires/sanity \
OPTIMIZER=adamw_offload LOGGER=tensorboard \
bash a2v/train.sh wan2.2-ti2v-5b
# nvidia-smi 看峰值 VRAM,确认 ≤ 70GB
```

### 监控 checklist

- [ ] 训练启动后 `squeue -u $USER` 看 job 状态
- [ ] 第一条 `[train]` 日志行 → 确认 shape 正确
- [ ] 每 100 step 的 loss 应单调降
- [ ] `tensorboard --logdir=ckpts_hires/exp1` → 看 loss 曲线
- [ ] 每 1000 step 检查 ckpt 是否能 resume(load test)

---

## 8. 训练完 → eval 流程

```
ckpts_hires/exp1/step-N.safetensors
  ↓ 推理 batch(480×640×121f × 200 ep ≈ 30 min)
input/<method>/episode{N}/{head,left,right}.mp4
  ↓ preprocess
generate/<method>_<ts>/
  ↓ run_eval (config_no_gt.yaml 或 config.yaml 真 GT)
output/<method>_<ts>_<run_ts>/final_results/{episode,summary}_metrics.csv
  ↓ 读 TWB-Score
cat summary_metrics.csv
```

推理和 eval 详见 `docs/A2V_LOCAL_EVAL_GUIDE.md` 和 `docs/A2V_TRAINING_GUIDE.md`(本文件)。

---

## 9. 与 Embodied-World-R1 的对比

| 项 | Embodied-World-R1 | A2V |
|---|---|---|
| 底座 | Wan-T2V | Wan2.2-TI2V-5B |
| 训练目标 | FlowGRPO | VACE |
| 数据规模 | 50k+ RL rollout | 200-1000 SFT |
| 评测 | EWMScore(11-15 metric)| TWB-Score(19 metric)|
| 节点 | 4 节点 32 GPU | 1-4 节点 8-32 GPU |
| 训练时间 | 数小时-数天 | 2-12 小时 |
| 显存峰值 | ~75 GB | ~60 GB |

**结论**:A2V 训练**比 Embodied-World-R1 简单**(单目标 VACE,无 RL),资源需求中等。

---

## 10. 相关文档

- `docs/A2V_SETUP.md` — A2V 环境搭建
- `docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md` — 端到端执行计划
- `docs/A2V_LOCAL_EVAL_GUIDE.md` — 评测端到端
- `docs/A2V_PATH_B_MULTIVACE_PLAN.md` — Path B 架构升级
- memory `triworldbench-local-eval-setup` — 评测基建
- memory `diffsynth-multinode-wandb-hang` — 多机 wandb 卡死坑
- memory `embodied-world-r1-eval-infra` — 相关 RL 训练经验

---

## 11. FAQ

**Q: 640×480 比 320×240 提升多少?**
A:空间 token 多 4×,运动细节 + 主体一致性应该有 +5-10% 视觉质量,但 TWB-Score 整体可能不变(因为 TWB 主要测的是跨视角一致 + 任务遵循,跟分辨率关系弱)

**Q: 一定要 2 节点吗?单卡能跑 480×640?**
A:**能跑但 batch=1,200 ep × REPEAT=20 = 4000 步,单卡 ~ 8-10 小时**。生产建议 ≥2 节点。

**Q: cache 编码需要多久?**
A:200 ep × 121 f × 480×640 编码单卡约 1.5-2 小时。多机并行(4n)可降到 30 分钟。

**Q: 不训 VACE,训全模型可以吗?**
A:可以但没必要。VACE 是新增的可训练组件,5B 主干已经训好。训全模型需要 10× 显存 + 数据。

**Q: 5B vs 14B 选择?**
A:5B 训练稳定,显存可控。14B 训不动(单节点都 OOM)。ziyang 也只用 5B。

---

## 12. 帧数选择决策(2026-09-03)

> **当前决策**:**保持 num_frames=121 固定**(本项目所有训练统一),未来视情况升级。

### 12.1 为什么是 121 帧?

| 维度 | 121 的合理性 |
|---|---|
| **物理意义** | 121 = 120 帧动作 + 1 帧首帧 = **4 秒 × 30 fps** |
| **Wan2.2 默认** | DiffSynth 官方示例、HF model card、ziyang 全部用 121 |
| **VAE latent T** | (121-1)//4 + 1 = **31**,position encoding 友好 |
| **动作长度匹配** | RoboTwin 原子动作 4-6 秒,121 覆盖大部分 |
| **单卡适配** | 480×640 单卡 60GB(ziyang 实测,留 20GB 余量) |
| **单步时间** | ~7s/step(480×640 单卡) |

### 12.2 备选帧数对比

| 帧数 | 时间 | latent T | 单步时间(480×640)| 评价 |
|---|---|---|---|---|
| 81 | 2.7s | 21 | ~5s | 太短,动作截断 |
| **121** | **4s** | **31** | **~7s** | **✅ 黄金标准** |
| 161 | 5.3s | 41 | ~9s | 较长任务 |
| 241 | 8s | 61 | ~13s | 复杂长序列,接近显存极限 |
| 481 | 16s | 121 | ~25s | OOM 临界 |

### 12.3 为什么不训练可变帧数?

| # | 问题 | 影响 |
|---|---|---|
| 1 | **VAE 时间维 latent shape 必须固定** | patch embedding conv kernel 期望固定 latent shape |
| 2 | **DiT 内 self-attention 时间维** | T 变则 attention 矩阵重算,无法 batch |
| 3 | **位置编码** | RoPE 索引 `[:f]`,f 变了"学到位置分布"变了 |
| 4 | **batch 内不同 n_frames** | DDP 要求同 shape,不同 n 只能 sequential,失 batch 优势 |
| 5 | **数据准备复杂** | 每个 sample 要对齐自己的 n_frames |
| 6 | **推理选哪个模型** | n_test=141 用哪个 ckpt?逻辑混乱 |

**行业惯例**:SVD、CogVideoX、HunyuanVideo、Wan2.1/2.2 全部用**固定帧数**。

### 12.4 三种未来路径(目前选 A)

| 方案 | 工作量 | 推理复杂度 | test_dataset 138-462 帧覆盖 |
|---|---|---|---|
| **A. 单 LoRA @ 121**(当前) | 1× | 简单(传 n_test,模型自动 round) | 训过 121,外推到其他,质量降 |
| **B. 多 LoRA @ 81/121/161/241** | 4× 训 | 中(n_test→nearest) | 全覆盖,每个 ep 选 nearest |
| **C. 可变帧单 LoRA** | 0.5-1×(架构改) | 简单 | 训练一个支持81-241,架构小改 |

### 12.5 何时升级到 B/C?

**触发条件**(任一满足才考虑升级):
- ❌ eval 显示长 ep(>300 帧)的 `flow_score` 明显低于短 ep(<150 帧)
- ❌ eval 显示长 ep 的 `dynamic_state_alignment` 出现明显 motion artifact
- ❌ test_dataset 提交后,submission_check.py 对长 ep 报错(虽然大概率只会 warning)

**升级路径**(已规划,未执行):

```
Step 1 (当前): A — 单 LoRA @ 121,接受 ≤3 帧 warning
Step 2 (若 Step 1 长 ep 质量差): 加训 161 / 241 帧 LoRA → 多 LoRA 推理
Step 3 (若 Step 2 仍不够): 考虑 C — 改 Wan 模型支持可变帧(成本极高,慎)
```

### 12.6 训练侧 vs 推理侧帧数差异

| 训练侧 num_frames | 推理侧 num_frames | 行为 |
|---|---|---|
| 121 (固定) | 121 | 直接 ✅ |
| 121 (固定) | 80-160(差 ≤40 帧)| 外推比 0.5-1.3×,**可接受** |
| 121 (固定) | 161-241(差 ≤120 帧)| 外推比 1.3-2.0×,**有 artifact 风险** |
| 121 (固定) | > 300 | 外推比 > 2.5×,**明显掉质** |

**自动化规则**(在 `prepare_submission.py` 里实现):

```python
if n_test <= 130:        use 121 LoRA         # 直接出
elif n_test <= 200:        use 121 LoRA + post-interp   # 外推 1.6× 以内
elif n_test <= 280:        # 触发升级到 B 的预警
   # 训 161/241 LoRA 后,改用 nearest LoRA
else:   # 长 ep
   # 同上,选 241 LoRA + post-interp
```

### 12.7 当前提交规格总结

| 项 | 当前 | 备选 | 何时切换 |
|---|---|---|---|
| **训练 num_frames** | **121** | 161 / 241(多 LoRA)| 长 ep 质量差时 |
| **训练分辨率** | **480×640** | 720×1280 | 单卡 OOM(暂不考虑多节点) |
| **批大小 / GPU** | **1** | 2 | VRAM 允许时 |
| **优化器** | **Adam8bit** | adamw_offload | Adam8bit 装不上时 |

---

## 13. 扩数据计划(2026-09-03)

> **目标**:dataset_mv 20 ep × 1 task → 200 ep × 50 task(覆盖全 50 task)

### 13.1 当前数据状态

```
/disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv/
├── adjust_bottle__v4__episode{53,68,...}__tra1__mv3/  (20 个)
└── metadata.jsonl  (20 行)
```

**问题**:
- ❌ 只 1 个 task(adjust_bottle)
- ❌ eval TWB val_dataset 时 18/20 ep task 不对应
- ❌ 训练数据多样性不够 → 学不到任务间共性

### 13.2 扩数据策略

| 项 | 数值 |
|---|---|
| **任务数** | 50(覆盖 v4_random10k_qhd_wrist 全部 task) |
| **每任务 ep 数** | 4(原任务取前 4 个 episode) |
| **总 ep 数** | **200** |
| **每 ep 帧数** | 121(对齐 num_frames=121) |
| **视角顺序** | [head \| left | right](与 TriWorldBench 一致) |
| **总存储** | ~50 GB(每 ep ≈ 250 MB) |

### 13.3 扩数据 Pipeline(端到端)

```bash
# Step 1: 多任务批量准备数据(基于现有 multivview_prepare.py,加 task 循环)
PY=/disk/worldmodel/uv/envs/a2v/bin/python
$PY /disk/worldmodel/wangshilong/a2v_v4_khl/multiview_prepare_all_tasks.py \
  --root /disk/worldmodel/ziyang/projects/data_generation/khl_workspace/Data_output/v4_random10k_qhd_wrist \
  --episodes_per_task 4 \
  --output_dir /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv_50t \
  --per_view_height 256 --per_view_width 320 --num_frames 121
# 耗时:~30-60 min(50 task × 4 ep,每 ep ~10s)
# 产物:dataset_mv_50t/ + metadata_50t.jsonl(200 行)

# Step 2: Pre-cache 编码(VAE 预编码 latent,加速后续训练)
PY=/disk/worldmodel/uv/envs/a2v/bin/python
A2V_MODELS_DIR=/disk/worldmodel/public_model \
$PY -m a2v.data.cache_encode \
  --manifest /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv_50t/manifest.jsonl \
  --output_dir /disk/worldmodel/wangshilong/a2v_v4_khl/cache_mv_50t_480x640 \
  --height 480 --width 640 --num_frames 121
# 耗时:单卡 1.5-2 h,多卡 30 min
# 产物:cache_mv_50t_480x640/(~17 GB latent + 元数据)

# Step 3: 单卡 480×640 训(zhang §7 sbatch 模板)
bash a2v_v4_hires/train_hires_480x640.sbatch
# 耗时:~8-10 h(200 ep × REPEAT=20 = 4000 步,单卡 7s/step)
# 产物:ckpts_hires/exp1/step-{100,200,...,4000}.safetensors

# Step 4: 推理 + 提交准备
bash a2v_v4_hires/batch_infer_hires.sh  # 200 ep,30 min
python a2v_v4_hires/prepare_submission.py  # resize 320×240 + frame count 对齐
python submission_check.py /path/to/<method>  # 验证 + 打包
```

### 13.4 各步骤时间表

| 步骤 | 单卡 | 2 节点 | 多机并行 |
|---|---|---|---|
| Step 1 数据准备 | 30-60 min | — | — |
| Step 2 cache 编码 | 1.5-2 h | 45 min | **30 min** (4n) |
| Step 3 训 480×640 | 8-10 h | 2-3 h | — |
| Step 4 推理+提交 | 30 min | 15 min | — |
| **总计** | **~12 h** | **~4 h** | — |

### 13.5 关键决策点

- **数据**:dataset_mv_50t 与现有 dataset_mv **不冲突**(新路径)
- **训练**:与 ziyang §7 模板完全兼容,改 `--dataset_base_path` 即可
- **推理**:沿用现有 batch_infer.sh,改 method 名 + 路径
- **评估**:沿用现有 run_eval.sh + config

### 13.6 风险与缓解

| 风险 | 缓解 |
|---|---|
| 50 task 数据量超出 RoboTwin 仿真覆盖(50 task × 4 ep 都存在)| v4_random10k 实际就是 50 task × ~20 ep,**完全够** |
| cache 编码 OOM(单卡跑 480×640 cache) | 用 `--tiled` 编码,或 batch_size=1 |
| 训时 OOM(数据比单 ep 复杂) | ziyang 已验证 200 ep 单卡 OK |
| 长 ep test 推理掉质 | **当前不做长 ep 提交**(扩数据优先,长 ep 走升级路径 B)|