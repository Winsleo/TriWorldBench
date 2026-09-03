# TriWorldBench 19 指标详解 + 提升策略

> **最后更新**:2026-08-31  
> **代码出处**:`/disk/worldmodel/wangshilong/TriWorldBench/metrics/`  
> **TWB-Score 公式**:`scripts/lib/eval_common.py:export_summary_csv`

---

## 总览:TWB-Score 怎么算

```python
# scripts/lib/eval_common.py:export_summary_csv
for metric in metric_names:
    if weights:                                  # episode_weights=true 时按帧数桶加权
        row[metric] = weighted_mean(episodes, weights)
    else:
        row[metric] = mean(episodes)            # 默认等权
row["TWB-Score"] = mean(row[metric])            # ← 19 个 metric 的简单算术平均
```

**含义**:19 个 metric 各自先按 episode 聚合(可选帧数加权),再 19 维算术平均。**没有任何特殊加权**,所以提升策略是**逐个击破**。

| episode 帧数桶 | 权重 |
|---|---|
| < 200 | 0.32 |
| 200-400 | 0.33 |
| > 400 | 0.35 |

---

## 19 指标逐个拆解

### 0. Instruction_Following — 指令遵循

**算法**(VLM 评判,**仅 head 视角**):
```python
# metrics/_common/vlm_judge.py + protocol_0710.py
# Qwen3-VL 看 head 视角视频,按 1-5 打分
score = clamp01(int(vlm_response.score) / 5.0)
# 5="correctly executes"; 4="mostly correct with minor deviations"
# 3="follows intent with execution errors"; 2="major action/object errors"; 1="unrelated"
```

**评判提示词关键点**(`metrics/vlm_prompt/01.txt`):
- 任务文本:`{instruction}`(来自 val_dataset 的 episode.json)
- 仅看 head 视角(主视角)
- 必须识别"机器人臂/末端执行器",**不能误判为人类手**

**提升**:
- ✅ 强化 head 视角的物体识别与动作执行质量
- ✅ 训练数据匹配任务文本(同一 task + instruction 多 shot)
- ❌ 不要把腕视角(close-up,易遮挡)当作 head 评判

---

### 1. Interaction_Quality — 物理交互质量

**算法**(VLM 评判,**三视角聚合**):
```python
# metrics/_common/protocol_0710.py:ingest_vlm + robust_view_aggregate
score_per_view = clamp01(int(vlm_response.score) / 5.0)
# 5="visually convincing interaction physics"; 4="realistic robot-object interaction"
# 3="generally plausible with visible defects"; 2="major physical errors"; 1="impossible contact"
score = robust_view_aggregate(per_view, outlier_config)  # 三视角均值,剔除与 median 偏差 >0.25
```

**评判要点**:接触/穿透/无意义接触、刚体碰撞、夹爪闭合等物理合理性。

**提升**:
- ✅ 渲染时物理一致性(物体不被穿模/不悬浮)
- ✅ 夹爪闭合时机与物体运动同步
- ✅ 失败案例也要"失败得合理"(物体滑落、碰撞),不要瞬间消失

---

### 2. Perspectivity — 3D 视角一致性

**算法**(VLM 评判,三视角聚合):
```python
score = clamp01(int(vlm_response.score) / 5.0)
# 5="fully coherent camera geometry"; 4="stable perspective and depth relationships"
# 3="mostly consistent with minor drift"; 2="unstable 3D structure"; 1="incoherent geometry"
```

**评判要点**:摄像机几何、深度、形状/尺度变化的连贯性。

**提升**:
- ✅ 用相机内外参约束(Intri/Extri),不要"自由学习"3D
- ✅ 同帧多视角几何对得上(腕视角看到的世界 = head 看到的局部)
- ❌ 不要让腕视角突然切换视角/尺度

---

### 3. background_consistency — 背景一致性

**算法**(CLIP ViT-B/32 特征余弦相似度):
```python
# metrics/background_consistency.py:_score_frame_features
# 每帧与首帧、与前一帧做余弦相似度,取较大
sim_pre = max(0, cos(former_feat, frame_feat))
sim_fir = max(0, cos(first_feat, frame_feat))
# 当 dynamic_score <= 0.1213 时(被判定为静止),分还要乘以 dynamic
sim_per_image = sim_per_image * dynamic_score if is_static else sim_per_image
video_score = mean(per_frame_sim)
```

**关键**:CLIP 编码 → 帧间余弦相似度 → 加权聚合;**动起来就只看运动区域,静止要求严苛**。

**提升**:
- ✅ 背景主体保持稳定(桌子、墙、其他物品)
- ✅ 静止阶段(STATE 标 moving_fraction=0)分会被 dynamic 放大,**关键帧不能有抖动/伪影**
- ❌ 背景不能"漂"(随机高频噪声会降低 CLIP 特征相似度)

---

### 4. dynamic_state_alignment — 运动状态对齐(派生)

**算法**(用 dynamic_degree × STATE 标注的 moving_fraction):
```python
# metrics/_common/protocol_0710.py:motion_factor + dynamic_alignment
gen_dynamic = dynamic_degree(video)             # 生成视频实际光流幅度
moving_fraction = state[view].moving_fraction  # 标注阶段应运动的帧比例
threshold = 0.1213                             # 光流阈值

# 期望运动但生成静止: 衰减 exp(under_motion_lambda * (dyn - thres))
# 期望静止但生成运动: 衰减 exp(-unexpected_motion_lambda * dyn)
score = moving_fraction * moving_score + (1 - moving_fraction) * static_score
```

**关键**:生成运动分布要匹配 GT 的动作阶段(approach / grasp / manipulate / release 等)。

**提升**:
- ✅ 阶段对齐:approach 阶段确实接近、grasp 阶段确实闭合夹爪、release 阶段确实松开
- ✅ 动作幅度与 STATE moving_fraction 同分布
- ❌ 不要"全程静态"或"全程狂动"

---

### 5. flow_score — 光流幅度(原始指标)

**算法**(RAFT 双向光流均值):
```python
# metrics/flow_score.py:_pairwise_flow_mean
flow = RAFT(frame_i, frame_{i+1})              # 双向光流
score = mean(|flow|)                           # 像素级光流幅度均值
video_score = mean(per_pair_score)
```

**提升**:
- ✅ 帧间变化要明显但不剧烈(对应 GT 的运动幅度)
- ❌ 全静止 → flow_score ≈ 0(对应 baseline noise)
- ❌ 全乱变 → flow_score 过大(类似"抖动")

**注意**:`flow_score` 是 dynamic_degree 的基础,也会被 dynamic_state_alignment 用作输入。

---

### 6. jepa_similarity — V-JEPA 特征相似度

**算法**(调用 external/JEDi):
```bash
# metrics/jepa_similarity.py 调用外部脚本
python external/Triworld/video_quality/JEDi/main.py \
    --model_dir weights/vjepa ...
# 用 V-JEPA ViT-H/16 编码 gen/GT 视频,计算特征余弦相似度均值
# SSV2 probe 用于动作分类
```

**提升**:
- ✅ 生成视频的语义/动作应与 GT 接近(V-JEPA 学过的视频特征)
- ✅ 动作类别、场景语义匹配
- ❌ 极端偏差动作会扣分

---

### 7. photometric_smoothness — 光度平滑度

**算法**(Tartan-C-TSKH 双向光流 EPE):
```python
# metrics/photometric_smoothness.py:compute_epe + OpticalFlowAverageEndPointErrorMetric
flow_fwd = TartanC(frame_i, frame_{i+1})
flow_bwd = TartanC(frame_{i+1}, frame_i)
warped_x1, warped_y1 = coords + flow_fwd
epe = |warped - coords + flow_bwd| < threshold ? 0 : |...|
# 双向一致性:warped 回去的位置和原坐标的偏差
score = mean(epe)
```

**关键**:**前后向光流一致性**(循环闭合误差),衡量"运动预测是否平滑"。

**提升**:
- ✅ 运动轨迹平滑(物体不会瞬移)
- ✅ ResNet34 用于光度重建(warping loss),**必须有 `weights/torch_home/hub/checkpoints/resnet34`**(已 symlink)
- ❌ 闪烁/突变帧直接拉低(双向不一致)
- **坑**:计算节点无网下不动 resnet34,直接报 -10 分(必须预放)

---

### 8. psnr / 10. ssim — 像素级相似度(GT 对齐)

**算法**(像素级直接比较):
```python
# 标准的 image quality assessment 公式
psnr = 10 * log10(max_val^2 / MSE(gen, gt))
ssim = structural_similarity(gen, gt)          # 含亮度/对比度/结构三组件
# psnr 在 CSV 中是 normalized = (psnr - 0) / 20(因为 score_config.scale_c=20)
```

**提升**:
- ✅ 像素级贴近 GT(颜色、亮度、构图都接近)
- ❌ 这是 GT-dependent 指标,**没有 GT → 无 psnr/ssim**(用 A2V 自有 20 sample 评测时 GT 是 A2V 自己生成的 video,可控)
- ⚠️ TriWorldBench 把这两个作为 GT 对齐指标,**提升空间最小**(本质就是"重建 GT 的能力")

---

### 9. semantic_alignment — 语义对齐(GT 描述)

**算法**(Qwen3-VL 描述 + CLIP 文本余弦):
```python
# metrics/semantic_alignment.py:run_clip_text_similarity
gen_caption = Qwen3VL(gen_video)               # 生成详细描述
gt_caption = Qwen3VL(gt_video)                 # GT 描述(预生成)
score = cos(CLIP_text(gen_caption), CLIP_text(gt_caption))
video_score = mean(per_video_score)
```

**关键**:**用 Qwen3-VL 给 gen/GT 各生成一段文本,再用 CLIP text encoder 算相似度**。

**提升**:
- ✅ 物体的视觉描述与 GT 描述一致(颜色、形状、位置)
- ✅ 动作描述与 GT 描述一致
- ❌ 不要让模型生成"额外物体"或"缺失物体"

---

### 11. subject_consistency — 主体一致性

**算法**(DINO ViT-B/16 自监督特征余弦):
```python
# metrics/subject_consistency.py
sim_pre = max(0, cos(dino(former), dino(frame)))
sim_fir = max(0, cos(dino(first), dino(frame)))
sim_per_image = sim * dynamic_score if is_static else sim  # 静止时加权
video_score = mean(per_frame_sim)
```

**关键**:DINO 自监督特征对物体外观敏感,**背景一致但物体变形就会被扣分**。

**提升**:
- ✅ 同一物体在所有帧里外观稳定(夹爪不变形、机器人臂不变形、物体不闪烁)
- ✅ 物体几何稳定(不被光照/角度扭曲)
- ❌ 夹爪在腕视角里若变形严重,即使"功能正确"也扣分

---

### 12. trajectory_accuracy — 轨迹精度

**算法**(SAM3 夹爪检测 + DTW 距离):
```python
# metrics/trajectory_accuracy.py:eval_traj + NDTW
traj_pred = SAM3_gripper_detect(gen_video)      # SAM3 检测每帧夹爪位置
traj_gt = SAM3_gripper_detect(gt_video)        # GT 夹爪轨迹
dtw = DTW(traj_pred, traj_gt)                   # Dynamic Time Warping 距离
# 取"最远 GT 轨迹索引"(离原点最远的轨迹),作为评估目标
score = 1.0 / (dtw / len(dtw_path))
```

**关键**:从光看 GT,改看**末端执行器轨迹**是否匹配。

**提升**:
- ✅ 末端执行器(夹爪)轨迹与 GT 同步(同步移动、同步停止)
- ✅ SAM3 检测必须能识别到夹爪(`weights/sam3/sam3.pt`,已 symlink)
- ❌ 严重抖手/偏离 GT 轨迹直接低分
- **依赖**:`weights/sam3/` 必须含 `sam3.pt + bpe_simple_vocab_16e6.txt.gz`

---

### 13. vlm_consistency01 — 双视角一致性(few-shot)

**算法**(Qwen3-VL 看 head vs working wrist):
```
提示词:metrics/vlm_prompt/01.txt
- few-shot:加载 test_data/vlm_data_base/{positive,negative}/ 的参考样本
- 评分维度:
  1. state_consistency: head 与 working wrist 是否同一动作状态
  2. object_consistency: head 与 working wrist 是否同一抓取/操作物体
- 起始 5.0,扣分制(图像质量最多 -0.5, 跨视角冲突 -1.5)
score_normalized = parsed.overall_score / 5.0
```

**提升**:
- ✅ head 与腕视角看到的物体/动作一致(夹爪里如果抓住物体,腕视角能看到物体在夹爪中)
- ✅ few-shot 用 negative 库学习"扣分项",确保不犯
- ❌ head 显示夹爪张开 + wrist 显示夹爪闭合 = -1.5

---

### 14. vlm_consistency02 — 双视角一致性(扣分制 4 维度)

**算法**(Qwen3-VL 4 维度扣分):
```
提示词:metrics/vlm_prompt/02.txt
- 左/右腕对应专门模板
- 评分 4 个标准(各扣 0 / 0.5 / 1.0):
  1. 视觉一致性(物体位置/姿态)
  2. 时间一致性(同一时刻)
  3. 物体一致性(抓取对象)
  4. 动作一致性(工作状态)
score_normalized = (5.0 - total_deductions) / 5.0
```

**提升**:
- ✅ 同时优化 4 个维度(每条扣 -0.5 ~ -1.0 都拉低)
- ✅ 重点是腕视角里的"物体识别"和"动作状态"

---

### 15. vlm_consistency03 — 双视角一致性(状态+对象双评分)

**算法**(Qwen3-VL 状态驱动的双评分):
```
提示词:metrics/vlm_prompt/03.txt
- LEFT / RIGHT 各有模板
- 输出 state_consistency_score + object_consistency_score(均 0-5)
- 用 STATE 阶段的 phase_type + description 作为输入参考
- 加权得到 overall score
score_normalized = overall / 5.0
```

**提升**:
- ✅ 模型生成的动作要符合 STATE 描述的阶段
- ✅ state(动作)和 object(物体)双独立评分

---

### 16. VQA — 任务问答准确率

**算法**(Qwen3-VL 答 yes/no 问题):
```python
# metrics/vqa.py:调用 metrics/VQA/run_parallel_eval.sh
# Qwen3-VL 看视频,回答任务相关 yes/no 问题
score = correct_count / total_count            # score_ratio
# 若 view_evaluation 失败(无法判断 working view),分数除以 2
episode_score = score_ratio if view_eval_ok else score_ratio / 2
```

**关键**:Qwen3-VL 当裁判,**回答任务特定问题**(如"是否成功拿起瓶子")。

**提升**:
- ✅ 任务实际成功完成(动作 = 任务要求)
- ✅ 物体状态符合任务预期(瓶子被拿起、门被打开)

---

### 17. aesthetic_quality — 美学质量

**算法**(LAION Aesthetic head + CLIP ViT-L/14):
```python
# metrics/aesthetic_quality.py:laion_aesthetic
# 1. CLIP-L/14 编码每帧
# 2. 训练好的 aesthetic head(线性层)预测 1-10 分
# 3. clip 到 [0, 1]
score = mean(clip(aesthetic_score, 0, 1)) / 10
# 4. CSV 中(若选 QUALITY_PENALTY_METRICS 之一):
score_csv = raw * exp(2.0 * (trajectory - 1)) * exp(2.0 * (vlm_consistency - 1))
# ↑任务一致性差 → 扣分
```

**提升**:
- ✅ 视觉质量高(构图、光影、清晰度)
- ✅ **轨迹精度高**(否则 -2.0 × (1 - trajectory) 倍扣分)
- ✅ **VLM 一致性高**(否则 -2.0 × (1 - vlm_consistency) 倍扣分)

---

### 18. image_quality — 图像技术质量

**算法**(MUSIQ SPAQ):
```python
# metrics/image_quality.py:compute_imaging_quality
# MUSIQ 模型评估每帧技术质量(清晰度、噪声、伪影)
score = mean(MUSIQ(frame)) / 100
# 同样有 quality penalty:
score_csv = raw * exp(2.0 * (trajectory - 1)) * exp(2.0 * (vlm_consistency - 1))
```

**提升**:
- ✅ 图像清晰、低噪声、无 JPEG 块效应
- ✅ 避免模糊帧、过曝
- ✅ **同样依赖 trajectory_accuracy 和 vlm_consistency 不被惩罚**

---

## 🎯 提升策略矩阵(从性价比最高到最低)

| 优先级 | 指标 | 投入产出 | 核心改进 |
|---|---|---|---|
| 🥇 #1 | **trajectory_accuracy** | ★★★★★ | SAM3 检测 + DTW;**末端执行器轨迹对齐 GT** 是质量惩罚的前提,扣 -2 倍 |
| 🥇 #1 | **vlm_consistency 01/02/03** | ★★★★★ | Qwen3-VL 评 head vs wrist;**3 视角一致性**是质量惩罚的前提 |
| 🥈 #2 | **Instruction_Following** | ★★★★ | VLM head 视角评判;**任务文本 + head 视角理解**;训练数据文本多样化 |
| 🥈 #2 | **Interaction_Quality** | ★★★★ | VLM 三视角聚合;**物理交互合理性**(穿透/瞬移/夹爪闭合) |
| 🥉 #3 | **aesthetic_quality** ★raw | ★★★ | CLIP-L/14 + aesthetic head;**视觉质量高**,但因 raw × 惩罚,trajectory/vlm 收益更大 |
| 🥉 #3 | **image_quality** ★raw | ★★★ | MUSIQ;**图像清晰度**,同样 × 惩罚 |
| 4 | **Perspectivity** | ★★★ | VLM 三视角;**3D 几何稳定**(同帧多视角对得上) |
| 4 | **subject_consistency** | ★★★ | DINO;**主体外观稳定**(物体不变形) |
| 4 | **background_consistency** | ★★★ | CLIP-B/32;**背景稳定**,静态段要求严苛 |
| 4 | **dynamic_state_alignment** | ★★★ | 派生;**运动匹配 STATE 阶段** |
| 5 | **flow_score** | ★★ | RAFT;**光流幅度合理**,不能全静也不能狂抖 |
| 5 | **photometric_smoothness** | ★★ | Tartan-C;**前后帧运动平滑** |
| 5 | **semantic_alignment** | ★★ | Qwen-VL 描述 + CLIP;**物体描述与 GT 一致** |
| 6 | **jepa_similarity** | ★ | V-JEPA;**语义级动作对齐** |
| 6 | **VQA** | ★ | VLM 答 yes/no;**任务实际完成** |
| 7 | **psnr / ssim** | ⚠️ | 像素级 GT 对齐;**没 GT 就没分**;用自有 sample 可控 |

---

## 🚀 端到端提升路线图

### Phase 1:基础设施(必做,否则惩罚直接 -2 倍)

1. **trajectory_accuracy 抓稳**:SAM3 检测夹爪 + 真实轨迹对齐训练;失败时该指标 = 0,顺带拉低 aesthetic/image_quality
2. **vlm_consistency 01/02/03 抓稳**:head + wrist 视角看到的世界要"同一个世界";训练时数据增强保证 3 视角几何同步

### Phase 2:任务执行质量

3. **Instruction_Following**:head 视角要清晰展示任务执行;不要在 head 视角里"看不清"
4. **Interaction_Quality**:物理规则(碰撞、夹爪闭合、物体不被穿模)逼真;可加物理引擎仿真
5. **VQA**:完成度(任务成功 vs 失败);生成视频要展示完整任务过程

### Phase 3:视觉一致性

6. **subject_consistency**:DINO 特征稳定 → 主体外观/几何稳定
7. **background_consistency**:CLIP-B/32 → 背景静态段严苛,不能有噪声
8. **Perspectivity**:3 视角几何/尺度/形状连贯

### Phase 4:运动与时间

9. **dynamic_state_alignment**:动作幅度匹配 STATE 阶段(moving_fraction)
10. **flow_score / photometric_smoothness**:光流合理 + 前后帧平滑
11. **jepa_similarity**:语义级动作对齐 V-JEPA 特征

### Phase 5:视觉质量

12. **aesthetic_quality / image_quality**:清晰、构图好、无伪影

### Phase 6:GT 重建(若有 GT)

13. **psnr / ssim / semantic_alignment**:像素级重建 + 语义描述

---

## 💡 关键洞察

1. **质量惩罚连锁反应**:`aesthetic_quality` 和 `image_quality` 在 CSV 里是 `raw × exp(-2×(1-trajectory)) × exp(-2×(1-vlm_consistency))`,即 trajectory/vlm_consistency 拉胯 → 这俩指标直接 ×0.x 起步
2. **CLIP-DINO 静态偏好**:background/subject_consistency 在 moving_fraction=0 时分还要 ×dynamic,**静止段必须真静止**,否则双重扣分
3. **VLM 是天花板**:Qwen3-VL 当评委的 5 个 metric(0 / 1 / 2 / 13 / 14 / 15)占 6 个 metric,占 TWB-Score 1/3 权重,**模型本身的能力上限就是 VLM 评分上限**
4. **DTW 异常敏感**:trajectory_accuracy 用 DTW 而非 MSE,**时序错位(慢半拍/快半拍)直接拉低**
5. **PSNR/SSIM 是鸡肋**:TriWorldBench 强调"tri-view consistency + 任务执行",pixel 级重建只是兜底

---

## 📌 A2V 适配现状(基于 2026-08-31 实际 status)

A2V 当前是**单 DiT 处理宽屏拼接视频**(`multiview_prepare.py` 横向拼接 3 视角),Path A 没有跨视角 attention,所以:

- ❌ tri-view 一致性 几乎拿不到全分(VLM 01/02/03 估计只能 2.5-3.5)
- ✅ 任务/轨迹精度可能拿满分(head 视角在拼接图左侧,任务清晰)
- ❌ PSNR/SSIM 几乎 0(拼接图与 GT 不是同一坐标)
- ✅ aesthetic / image_quality 有 raw(清晰度好),但乘 0.几的惩罚

**Path B**(MULTIVACE,`docs/A2V_PATH_B_MULTIVACE_PLAN.md`)才能真正提 TWB-Score,需要跨视角 attention。

---

## 附录:关键代码位置速查

| 主题 | 文件 | 行/函数 |
|---|---|---|
| TWB-Score 公式 | `scripts/lib/eval_common.py` | `export_summary_csv` |
| Episode 加权桶 | `scripts/lib/eval_common.py` | `FRAME_WEIGHT_BUCKETS` |
| 19 metric 注册 | `scripts/metric_registry.py` | `METRIC_REGISTRY` |
| 协议 0710 聚合 | `metrics/_common/protocol_0710.py` | `compute_episode`, `robust_view_aggregate` |
| 质量惩罚公式 | `metrics/_common/protocol_0710.py` | `include_penalized_metrics` 块 |
| VLM judge 01/02/03 | `metrics/_common/vlm_judge.py`, `vlm_consistency_judge.py` | 各 variant |
| VLM 提示词 | `metrics/vlm_prompt/01.txt`, `02.txt`, `03.txt` | 可编辑文件 |
| DTW 轨迹 | `metrics/trajectory_accuracy.py` | `eval_traj`, `NDTW` |
| 运动状态因子 | `metrics/_common/protocol_0710.py` | `motion_factor`, `dynamic_alignment` |
| 双向 EPE | `metrics/photometric_smoothness.py` | `compute_epe` |
| RAFT 配对 | `metrics/flow_score.py` | `_pairwise_flow_mean` |
| DINO 主体 | `metrics/subject_consistency.py` | `subject_consistency` |
| CLIP 背景 | `metrics/background_consistency.py` | `_score_frame_features` |
| 语义对齐 | `metrics/semantic_alignment.py` | `run_clip_text_similarity` |
| 美学评分 | `metrics/aesthetic_quality.py` | `laion_aesthetic` |
| MUSIQ 技术 | `metrics/image_quality.py` | `technical_quality` |
| SAM3 检测 | `external/Triworld/video_quality/processing/detection_tracking.py` | `GripperDetector.__init__` |
| V-JEPA | `external/Triworld/video_quality/JEDi/` | `app/main.py` |
| STATE 准备 | `scripts/prepare_state_layout.py`, `metrics/_common/segment_episode_phases.py` | |
| Robust 聚合 | `metrics/_common/protocol_0710.py` | `robust_view_aggregate` (MAD outlier filter) |