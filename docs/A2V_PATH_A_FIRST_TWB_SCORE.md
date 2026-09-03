# A2V Path A 首次 TWB-Score 评测结果

> **时间**: 2026-09-01  
> **Method**: a2v_wan22_ti2v_5b_mv3 (Path A 横向拼接 baseline)  
> **Ckpt**: step-800.safetensors (800 步, 单卡, loss 0.07→0.26)  
> **Episode 数**: 20 (adjust_bottle subset)

---

## 结果

### TWB-Score = **0.4513**

```
summary_metrics.csv 路径:
output/a2v_wan22_ti2v_5b_mv3_20260901_175807_20260901_184936/final_results/summary_metrics.csv
```

### 19 metric 排序 (高→低)

| Rank | Metric | Score | 类型 |
|---|---|---|---|
| 1 | vlm_consistency03 | 0.8747 | VLM 视觉一致性 |
| 2 | Perspectivity | 0.8253 | 视角保留 (拼接有效) |
| 3 | subject_consistency | 0.7888 | 主体一致性 |
| 4 | vlm_consistency01 | 0.7508 | VLM 一致性 |
| 5 | semantic_alignment | 0.7489 | 语义对齐 |
| 6 | vlm_consistency02 | 0.7263 | VLM 一致性 |
| 7 | dynamic_state_alignment | 0.5800 | 动态状态 |
| 8 | background_consistency | 0.5446 | 背景一致 |
| 9 | ssim | 0.4565 | 结构相似 |
| 10 | psnr | 0.3616 | 峰值信噪比 |
| 11 | photometric_smoothness | 0.3498 | 光度平滑 |
| 12 | Instruction_Following | 0.3500 | 指令遵循 |
| 13 | image_quality | 0.3151 | 图像质量 |
| 14 | Interaction_Quality | 0.2918 | 交互质量 |
| 15 | VQA | 0.2216 | 视觉问答 |
| 16 | aesthetic_quality | 0.1849 | 美学 |
| 17 | flow_score | 0.1243 | 运动强度 |
| 18 | trajectory_accuracy | 0.0791 | 轨迹精度 |
| 19 | jepa_similarity | 0.000006 | V-JEPA 相似度 (异常) |

---

## 与 TriWorldBench 榜单对比

| Rank | 模型 | 提交方 | TWB-Score | 差距 |
|---|---|---|---|---|
| 1 | BWM | — | 65.54 | -20.41 |
| 2 | WoVR_Plus | CASIA-DRL | 65.39 | -20.26 |
| 3 | PhyxWM | — | 64.26 | -19.13 |
| 4 | BetaBWM | TONGJI SI | 63.62 | -18.49 |
| 5 | Fysiverse-Video | Fysics AI | 62.85 | -17.72 |
| **—** | **A2V Path A** | **本项目** | **45.13** | — |

---

## 已知问题与限制

### 1. 任务不对应 (主要因素)
- A2V 训练数据: 20 个 adjust_bottle episode (sample id 53, 68, 70, 231, 240, 247, 273, 297, 486, 506, 582, 602, 658, 670, 730, 779, 783, 804, 820, 962)
- Eval val_dataset 100 个 episode 跨 50 个 task (只有 2 个是 adjust_bottle)
- 我们把 20 个 A2V sample 映射到 episode1-20(序号对应)
- 结果: 大部分 episode 内容(task)不对应 → Instruction_Following/VQA/Interaction_Quality/trajectory_accuracy 偏低

### 2. jepa_similarity ≈ 0
- 所有 20 episode 得分 6e-6(几乎为 0)
- 怀疑 V-JEPA 模型加载或推理失败,需 debug
- 这不影响 TWB-Score 计算(已包含),但解释时需注意

### 3. flow_score 低
- 平均 0.12 (一般 baseline ~0.3-0.5)
- 800 步训练的 5B 模型生成视频运动幅度偏小(只有 121 帧)

### 4. 拼接架构本身
- 256×960 横向拼接 → DiT 没有视角身份概念
- Tri-View Consistency(三视角一致性)是 TriWorldBench 核心维度,Path A 缺失

---

## Path A → Path B 升级路径

| 维度 | Path A 表现 | Path B 预期提升 |
|---|---|---|
| Tri-View Consistency | 极低(无跨视角 attn) | 中-高(+15) |
| Visual Quality | 中等(0.5-0.8) | 略提升(+2-3) |
| Task Following | 低(任务不对应) | 无变化(同数据) |
| **预期 TWB-Score** | **45.13** | **55-65** |

详见 `docs/A2V_PATH_B_MULTIVACE_PLAN.md`。

---

## 复现命令

```bash
# 1. 推理 20 row
bash /disk/worldmodel/wangshilong/a2v_v4_khl/batch_infer.sh 0 19

# 2. Bridge 到 TWB input
bash /disk/worldmodel/wangshilong/a2v_v4_khl/post_infer.sh

# 3. Preprocess
cd /disk/worldmodel/wangshilong/TriWorldBench
./run_preprocess.sh a2v_wan22_ti2v_5b_mv3

# 4. Eval(需要已修复 SEA-RAFT load_ckpt 支持 safetensors)
./run_eval.sh

# 5. 看 TWB-Score
cat output/<method>_<ts>/final_results/summary_metrics.csv
```

## 评测环境 Bugfix 记录

1. **photometric_smoothness SEA-RAFT load_ckpt 不支持 safetensors**  
   - 修复: `/disk/worldmodel/wangshilong/TriWorldBench/external/Triworld/video_quality/Triworld/third_party/SEA-RAFT/core/utils/utils.py`  
   - 加了 safetensors 自动检测 + 改用 `safetensors.torch.load_file`

2. **`generate/<method>_latest` 软链不存在**  
   - 修复: `cd generate && ln -sfn <method>_<timestamp> <method>_latest`

3. **torch 2.10 weights_only=True 默认 + Tartan-C 是 safetensors**  
   - 修复: 已包含在上述 utils.py 的 load_ckpt 改动里