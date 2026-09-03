# A2V 后续 3 项改进规划(学 ziyang + 评测对齐)

> **触发**: 对比 ziyang 的 TriWorldBench 工作(`/disk/worldmodel/ziyang/projects/a2v/a2v_khl_v2/`)  
> **目标**: 把 wangshilong Path A 推到与 ziyang v1/v2/rf 可比的水平 + 拉远指标  
> **时间估计**: 1 周(串行执行)

---

## 0. 关键澄清:GT-Free Metric 有几个?

**ziyang 跑了 5 个**(不全 GT-free,他也缺一些权重)。**真正 GT-free 的有 9 个**:

### 19 个 metric 的依赖分析

| 依赖类型 | 数量 | Metric 列表 |
|---|---:|---|
| **完全 GT-free**(只读生成视频本身) | **9** | Instruction_Following, Interaction_Quality, Perspectivity, background_consistency, flow_score, photometric_smoothness, subject_consistency, aesthetic_quality, image_quality |
| **需真 GT 帧** | **5** | jepa_similarity, psnr, semantic_alignment, ssim, VQA |
| **需真 STATE**(RoboTwin 动作相位) | **2** | dynamic_state_alignment, trajectory_accuracy |
| **需 qa_val 标注 + 真 GT** | **1** | VQA(已在 GT 类) |
| **State-based VLM judge**(ziyang 说需 STATE) | **2-3** | vlm_consistency01/02/03(我们的 19 跑通了 → 不一定需 STATE,需 debug) |

**ziyang 跑 5 个**(0/1/2/5/7)的原因 = **部分权重缺失 + 部分需真 STATE**:
- ✅ 有:Qwen3-VL-8B, RAFT, SEA-RAFT Tartan-C, ResNet34
- ❌ 缺(离线下不到):CLIP ViT-B-32.pt (jit), DINO 权重, CLIP ViT-L-14, aesthetic head, MUSIQ
- ❌ 缺(没 STATE):dynamic_state_alignment, trajectory_accuracy
- ❌ 缺(需真 GT):jepa, psnr, ssim, semantic_alignment, VQA

### 我们已有的 GT-free metric 能力

**好消息**: `docs/LOCAL_EVAL_GUIDE.md` §3.2 已经把 WorldArena 的所有 weights symlink 到了我们 TriWorldBench/weights/,所以**我们有全部 9 个 GT-free metric 的权重**:

| Metric | 需要的权重 | 我们的状态 |
|---|---|---|
| Instruction_Following | Qwen3-VL-8B | ✅ weights/qwenvl3 |
| Interaction_Quality | (派生自 VLM) | ✅ |
| Perspectivity | (派生自 VLM) | ✅ |
| background_consistency | RAFT | ✅ weights/raft_model/RAFT/models/raft-things.pth |
| flow_score | RAFT | ✅(同) |
| photometric_smoothness | SEA-RAFT Tartan-C + ResNet34 | ✅ weights/_hf.../worldscore/Tartan-C-T-TSKH + torch_home |
| subject_consistency | DINO ViT-B/16 | ✅ weights/dino_model/dino_vitbase16_pretrain.pth |
| aesthetic_quality | CLIP-L + aesthetic head | ✅ weights/clip_model/ViT-L-14 + weights/aesthetic_model/emb_reader |
| image_quality | MUSIQ pyiqa | ✅ weights/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth |

**结论**: **Task 2 可以扩展到 9 个 GT-free metric**(ziyang 5 个 + 我们补 background_consistency / subject_consistency / aesthetic_quality / image_quality),不再受限于权重。

---

## 1. Task A:Random+Clean Mix 重新训练

### 目标

学 ziyang 的 winning 配方(`a2v_rf_s7000`, random+clean 4:1 mix),保持我们的 Path A 架构(真三视角拼接)。预期:
- ziyang rf 5-metric partial = 0.4005
- 我们当前 Path A 5-metric partial ≈ ?(要 Task 2 才算得)
- 目标: Path A + rf mix → 9-GT-free partial TWB-Score 超过 0.50

### 步骤

#### Phase A1: 扩展 dataset_mv 到 random+clean mix (1 天)

**当前**:
- `dataset_mv/` 20 sample × 121 帧,只 random 域(`adjust_bottle` task)
- 总 ~3.4GB

**目标**:
- 50 task × 4 episode × 1 tra = **200 sample × 121 帧**
- 每个 task 选 **2 random + 2 clean episode**(rf mix 4:1 → 改成 1:1)
- 总 ~35GB(200 sample × 121 帧 × 双流 PNG)

**实现**:
1. 看 `multiview_prepare.py` 是否支持多 task
   - 当前 `--task` 是单 task,需要扩展支持 `--task all` 或 `--task_list`
2. 或新写 `multiview_prepare_mix.py`:
   ```bash
   PY=/disk/worldmodel/uv/envs/a2v/bin/python
   $PY /disk/worldmodel/wangshilong/a2v_v4_khl/multiview_prepare_mix.py \
     --root /disk/worldmodel/ziyang/projects/data_generation/khl_workspace/Data_output/v4_random10k_qhd_wrist \
     --output_dir /disk/worldmodel/wangshilong/a2v_v4_khl/dataset_mv_rf \
     --episodes_per_task 4 --random_ratio 0.5 \
     --per_view_height 256 --per_view_width 320 --num_frames 121
   ```
3. 验证:`ls dataset_mv_rf | wc -3` 应 ≥ 150(50 task × 3-4 ep)

#### Phase A2: 训练 sbatch 升级到 4 节点(半天)

**当前** `train_v4_mv.sbatch` 是 1 节点 1 卡,200 sample × 4 × 10 = 8000 步,单卡 ~40h 太久。

**目标** 4 节点 × 8 卡 = 32 卡,分布式:
- 200 sample × 4 × 4 = 3200 步,32 卡 ~4-6h

**实现**(参考 ziyang `train_v3w_rf_direct.sbatch`):
```bash
#SBATCH -J a2v-mv-rf
#SBATCH -p gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH -o /disk/worldmodel/wangshilong/a2v_v4_khl/logs/train_mv_rf_%j.out
#SBATCH --exclude=ZJYKP-A100x8-INTEL-114-118

PY=/disk/worldmodel/uv/envs/a2v/bin/python
ROOT=/disk/worldmodel/wangshilong/a2v_v4_khl
OUT=$ROOT/ckpts_mv_rf/exp1
DATASET=$ROOT/dataset_mv_rf
NUM_NODES=4
NUM_PROCESSES=32
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29721

export PYTHONPATH=/disk/worldmodel/wangshilong/DiffSynth-Studio
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline  # 防止 slurm 卡死,见 memory/diffsynth-multinode-wandb-hang
mkdir -p "$ROOT/logs" "$OUT"

srun --ntasks="$NUM_NODES" --ntasks-per-node=1 bash -c "
  $PY -m accelerate.commands.launch \
    --num_machines $NUM_NODES \
    --num_processes $NUM_PROCESSES \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    a2v.train_a2v \
    --base_spec wan2.2-ti2v-5b \
    --dataset_base_path $DATASET \
    --dataset_metadata_path $DATASET/metadata.jsonl \
    --data_file_keys video,vace_video \
    --extra_inputs vace_video,input_image \
    --height 256 --width 960 --num_frames 121 \
    --trainable_models vace \
    --lr_schedule cosine --lr_warmup_steps 100 --lr_min_ratio 0.0 \
    --learning_rate 1e-5 \
    --dataset_repeat 4 --num_epochs 4 --save_steps 1000 \
    --use_gradient_checkpointing_offload \
    --output_path $OUT \
    --enable_tensorboard_log \
    --log_every 10
"
```

**坑提醒**(memory):
- **不传 wandb env**(slurm 无 key,见 `diffsynth-multinode-wandb-hang`)  
- **不传 WANDB_MODE=online**(会预检联网挂死)  
- **加 `--unbuffered` 给 srun**

#### Phase A3: 推理 + Eval(2-4h)

参考已成熟的 `batch_infer.sh` + `post_infer.sh` + `run_preprocess.sh` + `run_eval.sh`,但 dataset 是新的。

### 验收标准

| 项 | 期望 |
|---|---|
| 训练 loss 单调下降 | tensorboard_log loss < 0.15 |
| 推理 200 episode 成功 | `concat_out/` 200 mp4 |
| 评测 TWB-Score (19-metric full) | > 0.55 |
| 评测 TWB-Score (9-GT-free partial) | > 0.50 |

---

## 2. Task B:9-GT-Free Partial TWB-Score(直接对标 ziyang)

### 目标

跑 **9 个 GT-free metric**(扩展 ziyang 的 5 个),让结果可直接跟 ziyang v1/v2/rf 对比。
- 用 Task A 的训练 ckpt(等 Task A 完成)
- 或先用现有 step-800 跑(快速 sanity)

### 步骤

#### Phase B1: 配置 9-GT-free metric eval(半天)

**TriWorldBench config 改动**(`config/config.yaml`):
```yaml
# 把 metrics 从 "all" 改成 9-GT-free 子集(只跑 GT-free)
metrics: [0, 1, 2, 3, 5, 7, 11, 17, 18]
# 含义:
# 0: Instruction_Following  1: Interaction_Quality  2: Perspectivity
# 3: background_consistency 5: flow_score           7: photometric_smoothness
# 11: subject_consistency    17: aesthetic_quality   18: image_quality
```

**验证 9 个 metric 跑通**(先在 2-ep 干净 baseline 上 sanity):
```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
./run_eval.sh --dry-run  # 列出 9 个 metric 全 OK

# 真跑一次 2-ep baseline
./run_eval.sh  # ~24min
```

#### Phase B2:跑 ziyang 5 metric partial(交叉验证)

把 config 改成 `[0, 1, 2, 5, 7]`(只 ziyang 跑的 5 个),在 20-ep Path A 上跑一次:
- 期望:得到与 ziyang 同样定义的 TWB-Score(partial)
- 对比:
  - 我们 20-ep partial 5-metric(任务不对应)
  - 我们 2-ep partial 5-metric(任务对应)
  - ziyang rf 500 partial 5-metric = 0.4005

#### Phase B3:跑 9-GT-free partial on 现有 step-800

不改任何架构,只改 config:
```yaml
metrics: [0, 1, 2, 3, 5, 7, 11, 17, 18]
```

跑 Path A 20-ep + 2-ep baseline,得 9-GT-free partial TWB-Score。

#### Phase B4:跑 9-GT-free on Task A 新 ckpt

Task A 完成后,自然得到新 ckpt 的 9-GT-free partial TWB-Score。

### 验收标准

| 项 | 期望 |
|---|---|
| 9 metric 全跑通 | summary_metrics.csv 有 9 列非空 |
| 跟 ziyang 5 metric partial 对得上 | 我们 5 metric ⊂ 我们 9 metric,数值子集可比 |
| step-800 9-metric partial | 算 baseline |
| Task A rf-mix 9-metric partial | > step-800 baseline |

---

## 3. Task C:Pseudo-GT 评测(避免 val_dataset 限制)

### 目标

学 ziyang 的 `build_pseudo_gt.py` 思路:**用生成视频自己当 GT**,不依赖 HF val_dataset。

**适用场景**:
- val_dataset 下载失败(集群无外网)
- 想快速评测新 ckpt,不想每次下 val_dataset
- 想跑 ziyang 5 metric 的同等评测(无需 HF)

### 步骤

#### Phase C1:理解 ziyang 的 build_pseudo_gt.py(半天)

参考 `/disk/worldmodel/ziyang/projects/a2v/a2v_khl_v2/twb_data/local_eval/build_pseudo_gt.py`:
- 输入:`<method>/episodeN/{head,left,right}.mp4`
- 输出:`pseudo_gt_pre/episodeN/episodeN.json` + `gt_manifest_<method>.csv`
- 伪 GT 的 instruction 从我们 `dataset_twb/metadata.jsonl` 取
- STATE 伪生成(每集只有 `num_frames` + 空 `phases`)

#### Phase C2:实现我们的 build_pseudo_gt.py(半天)

文件路径:`/disk/worldmodel/wangshilong/TriWorldBench/scripts/build_pseudo_gt.py`  
**复用**:
- `scripts/eval_data_common.py` 的 episode 数据结构
- 我们已有的 `input/<method>/episodeN/` 目录

**接口**:
```python
python /disk/worldmodel/wangshilong/TriWorldBench/scripts/build_pseudo_gt.py \
  --method a2v_wan22_ti2v_5b_mv3 \
  --input_root /disk/worldmodel/wangshilong/TriWorldBench/input/a2v_wan22_ti2v_5b_mv3 \
  --output_root /disk/worldmodel/wangshilong/TriWorldBench/pseudo_gt/a2v_wan22_ti2v_5b_mv3 \
  --num_frames 121 \
  --reference_strategy first_frame  # 或 head_first_3
```

**产出**:
```
pseudo_gt/a2v_wan22_ti2v_5b_mv3/
├── episode1/episode1.json   (instruction, num_frames)
├── episode1/head/frames/frame_00000.jpg..frame_00120.jpg   (从生成视频抽帧)
├── episode1/left/frames/...
├── episode1/right/frames/...
└── gt_manifest.csv  (每行 episode_id, num_frames, task_name)
```

**STATE** 也要生成(空 phase),让 finalize 通过:
```bash
mkdir -p STATE_pseudo/a2v_wan22_ti2v_5b_mv3
for ep in episode1..episode20; do
    echo "{\"episode\":\"$ep\",\"num_frames\":121,\"phases\":[]}" \
        > STATE_pseudo/a2v_wan22_ti2v_5b_mv3/$ep.json
done
```

#### Phase C3:跑 9-GT-free with pseudo-GT(半天)

把 config 改成:
```yaml
# 用 pseudo GT(不需要真 val_dataset)
gt_dataset: pseudo_gt/a2v_wan22_ti2v_5b_mv3
```

跑 9-GT-free metric:
```bash
./run_eval.sh
```

**预期**:
- 大多数 GT-free metric 不依赖 GT 内容,结果应该跟真 GT 几乎一样
- 视觉质量类(subject_consistency / aesthetic / image_quality)因为是 self-comparison,理论上完美,但 metric 实现可能做 cross-frame,需实测
- VLM 类(IF/IQ/Perspectivity)跟真 GT 一样,因为 VLM 只读生成视频

#### Phase C4:验证 pseudo-GT 是否合理(半天)

**对比**:
- 同一个 ckpt,跑两次 eval(真 GT vs pseudo-GT)
- 看 9 个 GT-free metric 的差值 < 0.05 → pseudo-GT 等价

### 验收标准

| 项 | 期望 |
|---|---|
| `build_pseudo_gt.py` 跑通 | 20 episode 全有 pseudo-GT + STATE |
| 9-GT-free pseudo 跟真 GT 差值 < 0.05 | 4 个 metric,2 个分以下 |
| 新机迁移时间 < 5min | 不下 val_dataset |

---

## 4. 三项任务的关系与时间线

```
Week 1:
├─ Day 1: Task A1 扩 dataset_mv (random+clean mix)
├─ Day 2: Task A2 训练 sbatch (4 节点 32 卡)
│         + Task B1 config 9-GT-free
│         + Task C1 阅读 ziyang build_pseudo_gt.py
├─ Day 3: Task B3 跑 9-GT-free on step-800 (baseline)
│         + Task C2 写 build_pseudo_gt.py
├─ Day 4: Task A 训练完成 + 推理 + eval
│         + Task C3 pseudo-GT eval
├─ Day 5: Task B4 跑 9-GT-free on Task A 新 ckpt
│         + Task C4 pseudo vs 真 GT 对比
└─ Day 6-7: 汇总分析 + 文档
```

**关键依赖**:
- Task A 依赖 Task A1 (data)
- Task B 独立(可并行)
- Task C 独立(可并行)

---

## 5. 优先级与工作量

| 任务 | 工作量 | 期望提升 | 推荐度 |
|---|---|---|---|
| **Task B**(9-GT-free eval) | 半天 | 直接可比 ziyang | ⭐⭐⭐⭐⭐(低成本高收益) |
| **Task C**(pseudo-GT) | 1 天 | 简化新机迁移 + 不需 HF | ⭐⭐⭐⭐ |
| **Task A**(rf-mix retrain) | 3-4 天(含等训练) | TWB-Score +0.05-0.10 | ⭐⭐⭐(高成本高收益) |

**建议执行顺序**:**B → C → A**
- B 快速出 baseline 数字
- C 简化新机迁移(对 Task A 也好)
- A 是主要投入

---

## 6. 关键文件改动清单

| 文件 | 任务 | 内容 |
|---|---|---|
| `TriWorldBench/config/config.yaml` | B | `metrics: [0,1,2,3,5,7,11,17,18]` |
| `TriWorldBench/scripts/build_pseudo_gt.py` | C | 新建 |
| `a2v_v4_khl/multiview_prepare_mix.py` | A | 新建(支持多 task + random/clean mix) |
| `a2v_v4_khl/train_v4_mv_rf.sbatch` | A | 新建(4 节点 32 卡) |
| `a2v_v4_khl/dataset_mv_rf/` | A | 新建(200 sample 混合数据集) |

---

## 7. 复现命令速查

### Task B(最快)

```bash
# 9-GT-free baseline(用现有 step-800 + 20-ep data)
cd /disk/worldmodel/wangshilong/TriWorldBench
# 改 config.yaml: metrics: [0, 1, 2, 3, 5, 7, 11, 17, 18]
./run_eval.sh --dry-run  # 验证
./run_eval.sh            # ~50min
cat output/.../summary_metrics.csv
```

### Task C(快速)

```bash
# 写 build_pseudo_gt.py(参考 ziyang)
# 跑 9-GT-free with pseudo GT
./run_eval.sh
```

### Task A(慢,主投入)

```bash
# 1. 扩 data
python multiview_prepare_mix.py --output_dir dataset_mv_rf --episodes_per_task 4

# 2. 训练(等 GPU 空闲)
sbatch train_v4_mv_rf.sbatch

# 3. 推理
bash batch_infer.sh 0 199  # 200 episode

# 4. Bridge + eval
bash post_infer.sh
cd $TWB
./run_preprocess.sh a2v_v3w_rf_mix
ln -sfn a2v_v3w_rf_mix_<ts> a2v_v3w_rf_mix_latest
./run_eval.sh
```

---

## 8. 不做

1. **不重写 multiview_prepare.py**(保留旧版本,新写 mix 版本做兼容)
2. **不改 diffsynth/ 任何原文件**(只在 a2v_v4_khl/ 和 TriWorldBench/scripts/ 加新文件)
3. **不重做 Task A 的 200 sample 数据**(等 GPU 资源够再扩展,目前 20 sample 已够 baseline)

---

## 9. 待办 / 后续

- [ ] Task B 跑完后,看是否需要进一步扩展到 19-metric full eval(Task A 之后做)
- [ ] Task C 的 build_pseudo_gt.py 可以直接复用给 ziyang 的其他 model
- [ ] Task A 完成后,跑 Path B(原生多 VACE)对比,看 Path B 能否突破 0.55

## 10. 关联文档

- `docs/A2V_PATH_A_FIRST_TWB_SCORE.md` — 首次 TWB-Score
- `docs/A2V_LOCAL_EVAL_ISSUES_TODO.md` — 已知 issue + bugfix
- `docs/A2V_PATH_B_MULTIVACE_PLAN.md` — Path B 规划
- `/disk/worldmodel/ziyang/projects/a2v/a2v_khl_v2/twb_data/local_eval/README.md` — ziyang 评测方法