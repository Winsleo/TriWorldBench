# TriWorldBench 本地评测问题清单(待另一个 agent 解决)

> **状态(2026-09-02)**:A2V Path A baseline 首次端到端跑通,TWB-Score=**0.4513**(20 ep 含 task mismatch,详见 `A2V_PATH_A_FIRST_TWB_SCORE.md`)。**ep1+ep2 干净 baseline 已补跑(2026-09-02),TWB-Score=0.4985**,验证了 task mismatch 假设但意外发现 jepa_similarity 与 task mismatch 无关。本文件记录**已发现 + 已规避**的问题,以及**未解决**的 issue,交给后续 agent。

## ⚠️ 本地 vs 提交区分(2026-09-02 修正)

| 维度 | val_dataset(本文档) | test_dataset(提交) |
|---|---|---|
| Episode 数 | 100 | **500** |
| GT 视频 | ✅ 有(可本地看 metric) | ❌ 无(官方服务器跑) |
| 提交邮箱 | — | `triworld_submit@foxmail.com` |
| 提交规格 | — | 320×240 mp4 + match HDF5 帧数 + submission_check.py |

**本地 eval TWB-Score 0.45-0.50 ≠ 榜单 60+**。榜单只有官方跑完才算数。本地分数用来做内部 baseline 对比。

> **关联**:
> - 评测 doc:`docs/LOCAL_EVAL_GUIDE.md`(端到端步骤)
> - 提交 spec:`SUBMISSION_README.md`(500 ep + 320×240)
> - 已知坑:`LOCAL_EVAL_GUIDE.md §6`(general 避坑指南)
> - 实际首次 run:`output/a2v_wan22_ti2v_5b_mv3_20260901_175807_20260901_184936/`(20 ep)
> - 干净 baseline:`output/a2v_ep12_baseline_20260902_142451_20260902_142512/`(2 ep)

---

## 1. 已修(本 session 完成)

### ✅ Bugfix 1:SEA-RAFT `load_ckpt` 不支持 safetensors
- **症状**:`M07 photometric_smoothness` 全 20 episode fail,`invalid load key, '\xa0'`
- **根因**:Tartan-C-T-TSKH-spring540x960-M 文件实际是 `.safetensors` 格式,symlink 改名 `.pth` 但 SEA-RAFT 仍用 `torch.load`(只支持 pickle)
- **修复**:`external/Triworld/video_quality/Triworld/third_party/SEA-RAFT/core/utils/utils.py` 加自动检测
- **遗留**:改的是 `external/` 目录,如果上游 git pull 会被覆盖。建议给 metric 作者提 PR

### ✅ Bugfix 2:`generate/<method>_latest` 是软链不是目录
- **症状**:`prepare_split_views` 报 `FileNotFoundError`
- **根因**:`run_preprocess.sh` 只生成 `<method>_<ts>/` 目录 + `.path` 路径记录文件,**不建 `<method>_latest` symlink**
- **修复**:preprocess 后手动 `cd generate && ln -sfn <method>_<ts> <method>_latest`
- **遗留**:`run_preprocess.sh` 内部应自动建链,建议给 eval 脚本作者提 issue

### ✅ Bugfix 3:torch 2.10 `weights_only=True` 默认
- 包含在 Bugfix 1 改动里(改用 `safetensors.torch.load_file` 或 `weights_only=False`)

---

## 2. 未解决(需另一 agent 接手)

### 🔴 Issue 1:jepa_similarity ≈ 0 (异常)

**症状**:20 个 episode 全部得分 `6e-06`,ep1+ep2 干净 baseline 也是 `2.1e-07`(反而更差)

```
cat output/a2v_wan22_ti2v_5b_mv3_20260901_175807_20260901_184936/final_results/episode_metrics.csv | grep jepa
jepa_similarity,6e-06,6e-06,...

cat output/a2v_ep12_baseline_20260902_142451_20260902_142512/final_results/episode_metrics.csv | grep jepa
jepa_similarity,0,0  # 实际 = 2.1e-07
```

**根因(2026-09-02 已确认,原假设 3 个全错)**:
- ❌ V-JEPA 加载失败 → 日志确认 `<All keys matched successfully>`
- ❌ metric 实现 bug 对比空向量 → 公式是 `exp(-0.4 × MMD×100)`,多项式核 MMD 数学正确
- ❌ input 帧归一化错 → features 成功保存到 train.npy/test.npy
- ✅ **真因: JEDi score = exp(-0.4 × d),20-ep 反推 d=30.0 / 2-ep 反推 d=38.3**,MMD 距离极大
- 🟡 **可能原因 a**: JEDi 用 polynomial-kernel MMD,小样本(2 ep)估计方差大,大样本(20 ep)距离也 30,可能 GT vs 生成视频的 V-JEPA 特征分布确实差异极大(8B VLM/V-JEPA 训练数据可能没出现过这种机器人拼接视频,特征空间被外推到分布尾部)
- 🟡 **可能原因 b**: metric 实现把全部特征混在一个 global pool 里,没区分 head/left/right 视角,可能引入 bias

**debug 路径**(给 agent):
1. 单独跑一个 episode 的 jepa_similarity metric:
   ```bash
   cd /disk/worldmodel/wangshilong/TriWorldBench
   PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
   $PY metrics/runner.py --config config/config.yaml \
     --run-dir output/a2v_wan22_ti2v_5b_mv3_20260901_175807_20260901_184936 \
     --index 6 --name jepa_similarity --skip-prep --limit 1
   ```
2. 检查 `output/.../logs/metric_06_jepa_similarity.log` 看是否有错误
3. **验证 MMD 小样本方差假设**: 跑一个 10+ ep 全 matched task 的 eval,看 jepa 是否恢复到 0.1+ 区间
4. 检查 V-JEPA 权重 symlink:`ls -la weights/vjepa/`,确认 `vith16.pth.tar` 和 `ssv2-probe.pth.tar` 都能读
5. 检查 jepa_similarity metric 实现:`metrics/jepa_similarity.py`,确认是不是 global pool
6. 试着喂一张 GT 帧给 metric 看输出是否非零,验证模型本身
7. **新增**: 手动算 `mmd_poly(real_feats, gen_feats)` 在不同 N 下的方差,确认是否 N=2 时方差爆炸

**预期**:若小样本方差假设成立, 跑 10+ ep matched sample 应恢复到 0.1+;若真因是特征分布外推,则 jepa 可能一直 ≈ 0,需要接受这是 Path A 模型的固有特征

### 🟡 Issue 2:flow_score = 0.12 (偏低)

**症状**:20 个 episode 平均 0.124(预期 baseline 0.3-0.5)

**根因**:A2V 训练只 800 步,5B 模型生成的视频运动幅度偏小(可能 121 帧 50 步 diffusion 不够让模型学会大幅运动)

**修复方向**:
- [短期]跑更多训练步数(5000+ step)再 eval,看 flow_score 是否提升
- [中期]改用 `i2v_fused` first-frame conditioning 时增大 num_inference_steps(50→100)
- [长期]Path B 加 motion prior

**debug 路径**(给 agent):
1. 训练 5000+ 步:`sbatch a2v_v4_khl/train_v4_mv.sbatch` 改 `REPEAT=20 EPOCHS=20`(SAVE_STEPS=1000)
2. 重新 eval 取 step-2000/step-5000 对比 flow_score

### 🟡 Issue 3:任务不对应导致 VLM/VQA 类 metric 偏低

**症状(20 ep baseline)**:Instruction_Following 0.35 / VQA 0.22 / Interaction_Quality 0.29 / trajectory_accuracy 0.08

**ep1+ep2 干净 baseline 验证(2026-09-02 已跑)**:TWB-Score 0.4513 → **0.4985** (+0.047)

| Metric | 20-ep | 2-ep baseline | Δ | 受 task match? |
|---|---|---|---|---|
| Instruction_Following | 0.35 | **0.70** | **+0.35** | ✅ 2x |
| VQA | 0.22 | **0.50** | **+0.28** | ✅ 2.3x |
| vlm_consistency01 | 0.75 | 0.91 | +0.16 | ✅ |
| semantic_alignment | 0.75 | 0.89 | +0.14 | ✅ |
| Interaction_Quality | 0.29 | 0.40 | +0.11 | 🟡 |
| ssim | 0.46 | 0.54 | +0.08 | 🟡 |
| **trajectory_accuracy** | 0.08 | 0.10 | +0.02 | ❌ **独立 bug, 即使 matched 也只 0.10** |
| jepa_similarity | 6e-6 | 2e-7 | 0 | ❌ 不受 task match 影响(见 Issue 1) |
| 其他 11 metric | ≈ | ≈ | ≈ | ❌ 不受影响 |

**结论**:
- ✅ task mismatch 拖累 TWB-Score 约 **0.05**(10% 相对),影响集中在 2 个 task-sensitive metric
- ⚠️ **trajectory_accuracy 即使 task-matched 也只 0.10**,这是独立 bug(GripperDetector / DTW 公式问题,见新 Issue 4)
- ❌ jepa_similarity 不受 task match 影响,见 Issue 1

**修复方向**(给 agent):
- ✅ [短期]只 eval TWB 的 adjust_bottle episode(ep1+ep2),看调整后的 TWB-Score 是否合理 → **已完成,TWB-Score=0.4985**
- [中期]生成 TWB val_dataset 全部 100 episode 的 A2V 推理(目前 dataset_mv 只有 20 sample)  
  → 需要更多 task 的训练数据,扩展 dataset_mv 涵盖 50 task 各 2 episode
- [长期]Path B 配合 prompt-conditioned diffusion,确保任务对应

**ep1+ep2 baseline 复现命令**(完整版见 [[a2v-patha-first-twb-score]]):
```bash
cd /disk/worldmodel/wangshilong/TriWorldBench
mkdir -p input/a2v_ep12_baseline
ln -sfn input/a2v_wan22_ti2v_5b_mv3/episode1 input/a2v_ep12_baseline/episode1
ln -sfn input/a2v_wan22_ti2v_5b_mv3/episode2 input/a2v_ep12_baseline/episode2
./run_preprocess.sh a2v_ep12_baseline
cd generate && ln -sfn a2v_ep12_baseline_<ts> a2v_ep12_baseline_latest && cd ..
# config: eval_input → a2v_ep12_baseline_latest, gpu_list: "4"
./run_eval.sh
```

### 🟢 Issue 4:TWB-Score 整体偏低 vs 榜单

**症状**:Path A 0.4513 vs 榜单 BWM 65.54/WoVR_Plus 65.39

**根因**:Path A 架构根本缺陷——DiT 不知道拼接视频里"左 1/3 = head","中 1/3 = left","右 1/3 = right",无跨视角 attention,Tri-View Consistency 维度天然弱

**修复方向**:Path B 原生多 VACE + 跨视角 attention(规划见 `docs/A2V_PATH_B_MULTIVACE_PLAN.md`,工作 2-3 周,目标 TWB-Score 55-65)

### 🟡 Issue 4 (新增):trajectory_accuracy 即使 task-matched 也只 0.10

**症状**:ep1+ep2 baseline(都是 adjust_bottle,GT 任务对应)trajectory_accuracy = 0.10(20-ep 也是 0.08)。提升 task 对齐对此 metric 几乎无效,说明问题不在数据,在 metric 实现或模型输出。

**根因假设**:
- GripperDetector(SAM3)可能没识别到生成视频里的 gripper,返回空 trajectory → DTW 距离被默认 fill 到最大值
- 也可能是 SAM3 加载的 model 没生效(weight 路径错、推理 device 错)
- 也可能 Trajectory metric 的 GT 轨迹是 head 视角,但生成视频拼接了 3 视角,SAM3 在拼接视频上识别失败

**debug 路径**:
1. 单独跑 trajectory metric 看 log:
   ```bash
   tail -100 /disk/worldmodel/wangshilong/TriWorldBench/output/a2v_ep12_baseline_*/logs/metric_12_trajectory_accuracy.log
   ```
2. 看 `metric_12_trajectory_accuracy` 是否检测到 gripper(`0 detected / 10 frames` 之类)
3. 检查 SAM3 权重 symlink: `ls -la weights/sam3/` 应指向 WorldArena/models/sam
4. 喂一张 GT head 视角帧给 GripperDetector 验证模型工作

**预期**: trajectory_accuracy 应恢复到 0.3-0.6 区间(若 SAM3 检测 OK)

### 🟢 Issue 5:文档外未走过的坑(已知但我们没踩到)

| 坑 | 是否踩到 | 修复建议 |
|---|---|---|
| `HF_ENDPOINT` 未设 | ❌ 没踩到(env 已有) | 首次复现必设 |
| `xformers==0.0.35.dev1120` 装不上 | ❌ 没踩到(env 已有) | requirements.txt 注释掉 |
| `setuptools>=81` import 失败 | ❌ 没踩到(env 已有 80.10.2) | 手动 `pip install 'setuptools<81'` |
| `background_consistency` OOM (38GB/卡) | ❌ 没踩到(config 默认 cap=1) | 保持 `metric_parallel_per_gpu.background_consistency: 1` |
| `photometric_smoothness` ResNet-34 缺 | ❌ 没踩到(symlink 已建) | 首次复现前必须 `ln -sfn` 到 `weights/torch_home/hub/checkpoints/` |
| `metric 17/18` CSV 是惩罚后分数 | ⚠️ 已知,看 raw log | 查 `logs/metric_17_*.log` / `logs/metric_18_*.log` |
| `action_following` 不在 19 metric 里 | ❌ 没踩到 | TriWorldBench 已剔除,别误用 |

---

## 3. 推荐优先级(给接手的 agent)

| 优先级 | 任务 | 工作量 | 预期收益 |
|---|---|---|---|
| ✅ 已完成 | ep1+ep2 baseline (Issue 3 短期) | 1 小时 | TWB-Score=0.4985,验证 task mismatch 假设 |
| 🔴 P0 | **debug jepa_similarity** (Issue 1,真因待定) | 半天 | 修后 TWB-Score 可能 +0.02-0.05 |
| 🔴 P0 | **debug trajectory_accuracy** (新 Issue 4) | 半天 | 修后 TWB-Score 可能 +0.05-0.10 |
| 🟡 P1 | **训练 5000+ 步 + 重新 eval ep1+ep2** (Issue 2) | 1-2 天 | flow_score 可能 +0.1, TWB-Score +0.03-0.05 |
| 🟡 P2 | **跑全 100 ep eval** (Issue 3 中期,需扩 dataset_mv) | 2 天 | 真正可比的 TWB-Score |
| 🟢 P3 | **Path B 实现** (Issue 4 架构) | 2-3 周 | TWB-Score 目标 55-65 |

---

## 4. 给接手 agent 的 must-read 清单

1. `docs/A2V_PATH_A_FIRST_TWB_SCORE.md` — 当前结果与限制
2. `docs/A2V_PATH_B_MULTIVACE_PLAN.md` — 长期改进方向(给探索 agent 用)
3. `docs/LOCAL_EVAL_GUIDE.md` §6 — 已知坑速查
4. `memory/triworldbench-local-eval-setup.md` — 评测基建(环境 + weights + val_dataset)
5. `memory/a2v-patha-first-twb-score.md` — 首次 run 的 19 metric 表 + bugfix
6. `external/Triworld/video_quality/Triworld/third_party/SEA-RAFT/core/utils/utils.py` — 已改的 safetensors 修复(读 diff 理解逻辑)

---

## 5. 快速 reproduce 当前结果

```bash
# 1. 重新生成 20 row 推理(本地 GPU,~30 min)
bash /disk/worldmodel/wangshilong/a2v_v4_khl/batch_infer.sh 0 19

# 2. Bridge → TWB input
bash /disk/worldmodel/wangshilong/a2v_v4_khl/post_infer.sh

# 3. Preprocess
cd /disk/worldmodel/wangshilong/TriWorldBench
./run_preprocess.sh a2v_wan22_ti2v_5b_mv3
cd generate && ln -sfn a2v_wan22_ti2v_5b_mv3_<timestamp> a2v_wan22_ti2v_5b_mv3_latest
cd ..

# 4. Eval(8 卡 ~44 min,需 SEA-RAFT 已修 safetensors)
./run_eval.sh

# 5. 读 TWB-Score
cat output/a2v_wan22_ti2v_5b_mv3_<ts>/final_results/summary_metrics.csv
```

---

## 6. 不要做

1. **不要重做 batch_infer 的 GPU 选择逻辑**(已经验证 GPU 0 可用,但 shaoy 偶尔会塞,优先用 119 本地,slurm GPU 备选)
2. **不要碰 `external/` 之外的上游代码**(diff 都在 `external/Triworld/.../SEA-RAFT/`,其他外部代码可能更新时被覆盖,改了要 PR)
3. **不要把 eval 跑到 100 episode**(除非 train data 扩到 50 task,否则任务不对应会污染分数)
4. **不要在 slurm 上跑 infer**(本地 119 GPU 0 即可,sbatch 排队太慢)