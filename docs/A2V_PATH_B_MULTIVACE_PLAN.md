# A2V Path B: 原生多视角 VACE 方案探索规划

> **目标**：扩展 A2V 的 VACE 单元支持 N 视角原生输入 + 跨视角 attention，能上 TriWorldBench 榜单  
> **前置**：Path A (`docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md`) 跑通作 baseline + sanity  
> **工作量估算**：2-3 周（1 人全职）  
> **给探索 agent**：本文档为 R&D 探索阶段设计参考，最终架构需根据实验迭代

---

## 0. 为什么需要 Path B

### Path A 的局限

当前已实现的 Path A（横向拼接 3 视角为单视频）作为快速验证 pipeline 是 OK 的，但**架构上有根本缺陷**：

| 缺陷 | 影响 |
|---|---|
| DiT 不知道拼接视频里"左 1/3"对应 head、"中 1/3"对应 left wrist | Tri-View Consistency 核心维度预期严重低于专用多视角模型 |
| 无跨视角 attention | head 抓完时 wrist 可能纹丝不动 |
| trajectory map 拼接后,左右视角的投影位置关系被切碎 | 模型无法学到"头看到抓杯动作时,左腕必然在杯侧"这种结构约束 |
| 拼接视频宽 ×3 → attention 算量 ×9,显存压力 | 即使 patch 注意力,长序列仍是问题 |

### Path B 的承诺

让模型**显式感知视角身份** + **跨视角信息流** + **独立相机投影**,从而通过 TriWorldBench 的核心维度 (Tri-View Consistency)。

### 参考方案 (TriWorldBench 榜单)

| 模型 | 提交方 | TWB-Score (Aug 2026) | 推测架构 |
|---|---|---|---|
| BWM | ? | 65.54 | 大概率自研多视角架构 |
| WoVR_Plus | CASIA-DRL | 65.39 | World-aware View Reconstruction |
| PhyxWM | ? | 64.26 | 物理约束多视角 |
| BetaBWM | TONGJI SI | 63.62 | — |
| Fysiverse-Video | Fysics AI | 62.85 | — |

**任务**：探索这些方案的源码/论文（如能找到），作为 Path B 设计的参考。

---

## 1. 核心设计问题(必须先回答)

### Q1: 多视角如何喂给 VACE?

**当前 VACE 单视角流程** (`diffsynth/pipelines/wan_video.py:649-706`):
```
vace_video: (1, 3, T, H, W)  # 单 3 通道视频
  ↓ VAE encode
vace_video_latents: (1, 2*z_dim, T_lat, H_lat, W_lat)
  ↓ + mask
vace_context: (1, 2*z_dim + P*Q, T_lat, H_lat, W_lat)
  ↓ concat to DiT input
```

**Path B 候选方案**:

#### 方案 B.1: V videos 并行 VAE encode,concat 通道

```
vace_videos: [(1, 3, T, H_i, W_i) for i in 1..N]  # N 个不同分辨率/视角
  ↓ each VAE encode
vace_video_latents: [B, N * 2 * z_dim, T_lat, H_lat, W_lat]  # 通道堆叠
  ↓ + per-view mask
vace_context: [B, N * 2 * z_dim + N * P * Q, T_lat, H_lat, W_lat]
```

**优点**: 改动小,直接扩 vace_in_dim  
**缺点**: 视角间无显式 attention,DiT 仍把它们当独立通道

#### 方案 B.2: V videos + 视角 embedding + 跨视角 attention

```
vace_videos: [(1, 3, T, H_i, W_i) for i in 1..N]
  ↓ each VAE encode → [1, 2*z_dim, T_lat, H_lat_i, W_lat_i]
  ↓ 各自加上视角 embedding (head/left/right)
  ↓ DiT 处理时,在 self-attention 加 cross-view attention layer
```

**优点**: 真正的跨视角信息流  
**缺点**: 需要 DiT 内部架构改动

#### 方案 B.3: 视角-时序分离 attention

```
latents shape: [B, N*T_lat, dim]  # 视角维和时序维 flatten
  ↓ attention: 时序维度 attention + 视角维度 attention (交替或并行)
```

类似 ViT 的 spatial-temporal attention 分离方案。  
**优点**: 算量可控,视角间显式建模  
**缺点**: 需要 DiT 改 attention mask 逻辑

### Q2: 相机参数怎么注入?

**当前 vace_video 不带相机参数**(因为 VACE 输入是 RGB 像素,不需要投影)。  
但 Path B 需要每视角独立相机,因为:
- trajectory map 投影需要 intrinsics/extrinsics
- 多视角几何一致性需要相机位姿

**候选**:
- **Q2.a**: 相机参数编码为 token,加到 DiT 输入(类似 i2v_concat 的 CLIP 注入路径)
- **Q2.b**: 用 PLUCKER 坐标(Plücker embeddings)代替像素坐标,直接喂 vace_video (类似 ABot 9 通道方案)

### Q3: 输出格式?

**当前 VACE 输出单视频**。Path B 输出应是:
- **N 个独立视角视频**?  → eval 时直接给 TriWorldBench
- **1 个共享 latent 再 split**? → 等价 N 视角
- **共享 latent + 视角解码器**? → 更高效但架构复杂

### Q4: First-frame 怎么处理?

当前 ti2v_fused 用 input_image 单张图。Path B 多视角:
- 3 个视角首帧 → 3 张 reference image?
- 拼接 → 同 Path A?
- 跨视角信息流 → 用首帧 attention?

---

## 2. 推荐的探索路径

### Phase 1: 文献调研 (1-3 天)

**关键论文**(优先级降序):
1. WoVR / WoVR_Plus (CASIA-DRL, TWB #2)— 找 arxiv
2. BWM (TWB #1) — 找 arxiv / github
3. PhyxWM — 物理约束多视角
4. CAT4D / 4DiM / 4D-fy / SV4D — 多视角扩散基线
5. ABot-PhysWorld 第 3 通道 + 多视角变体
6. ViewCrafter / ReconFusion / ViVid-1 — 多视角视频生成

**调研产物**:
- 1 页 PDF 总结: 各方案架构对比表
- 重点方案选 1-2 个深入分析(参数量、attention 结构、输入输出)

### Phase 2: 数据格式扩展 (1-2 天)

**目标**: 让 prepare.py 输出多视角原生格式(每个视角独立 PNG + 独立相机参数),而非拼接。

**输入改动**:
```json
{
  "id": "adjust_bottle__v4__episode53__tra1__mv3",
  "videos": {
    "head":  ["path/00000.png", ...],
    "left":  ["path/00000.png", ...],
    "right": ["path/00000.png", ...]
  },
  "vace_videos": {           // 每个视角的 trajectory map (per-view camera)
    "head":  [...],
    "left":  [...],
    "right": [...]
  },
  "vace_reference_images": {  // 每视角首帧
    "head":  "...",
    "left":  "...",
    "right": "..."
  },
  "intrinsics": {             // 每视角 intrinsics
    "head":  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    ...
  },
  "extrinsics": {             // 每视角 extrinsics (c2w)
    "head":  [[[...], ...], ...],  // (T, 4, 4)
    ...
  },
  "prompt": "...",
  "view_order": ["head", "left", "right"]
}
```

**改动点**:
- `multiview_prepare.py`:已生成单目录拼接,可加 `--native_views` 输出原生多视角格式
- `a2v/data/prepare.py`: 单视角版本不动,新增 `--multiview` 模式

### Phase 3: VACE Unit 扩展 (3-5 天)

**目标**: `vace_unit.py` 支持 `vace_videos: list[Tensor]`。

**改动点** (`a2v/vace_unit.py`):

```python
class ParamWanVideoUnit_VACE(WanVideoUnit_VACE):
    def __init__(self, mask_pq: int = 8, num_views: int = 1):
        super().__init__()
        self.mask_pq = int(mask_pq)
        self.num_views = num_views
    
    def process(self, pipe, vace_videos, vace_video_masks, vace_reference_images,
                vace_scale, height, width, num_frames, ...):
        """
        vace_videos: list of (1, 3, T, H_i, W_i) tensor, each view
        vace_reference_images: list of (1, 3, T, H_ref, W_ref) per view (optional)
        """
        if vace_videos is None or len(vace_videos) == 0:
            return {"vace_context": None, "vace_scale": vace_scale}
        
        pipe.load_models_to_device(["vae"])
        
        per_view_latents = []
        per_view_masks = []
        for v_idx, v_video in enumerate(vace_videos):
            v_video = pipe.preprocess_video(v_video)
            v_mask = vace_video_masks[v_idx] if vace_video_masks else torch.ones_like(v_video)
            v_mask = pipe.preprocess_video(v_mask, min_value=0, max_value=1)
            
            inactive = v_video * (1 - v_mask)
            reactive = v_video * v_mask
            inactive_lat = pipe.vae.encode(inactive, ...)
            reactive_lat = pipe.vae.encode(reactive, ...)
            v_latents = torch.concat((inactive_lat, reactive_lat), dim=1)  # [1, 2*z_dim, T, H, W]
            v_mask_lat = rearrange(v_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=self.mask_pq, Q=self.mask_pq)
            v_mask_lat = interpolate(v_mask_lat, ...)
            
            # View embedding (NEW): 加 learnable view embedding 到 channels
            # 简单方案: concat 一个 view_id channel (1 维)
            # 复杂方案: 给 v_latents 加可学习 view-embedding
            v_latents = v_latents + self.view_embeddings[v_idx].view(1, -1, 1, 1, 1)
            
            per_view_latents.append(v_latents)
            per_view_masks.append(v_mask_lat)
        
        # Concatenate all views along channel dim
        vace_video_latents = torch.concat(per_view_latents, dim=1)  # [1, N * 2*z_dim, T, H, W]
        vace_mask_latents = torch.concat(per_view_masks, dim=1)      # [1, N * P*Q, T, H, W]
        
        if vace_reference_images:
            # Similar per-view reference handling
            ...
        
        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
        return {"vace_context": vace_context, "vace_scale": vace_scale}
```

**配套改动**:
- `provision.py`: `create_vace_from_dit` 需适配新 vace_in_dim = N × (2*z_dim + P*Q)
- `base_spec.py`: `WanBaseSpec.vace_in_dim` 改为 `N × (2*z_dim + spatial²)`,Wan2.2 TI2V-5B N=3 → 3 × 352 = 1056
- `train_a2v.py`: 数据集 operator 把多视角 dict 拆成 list of vace_videos

### Phase 4: DiT 跨视角 Attention (5-7 天,最难)

**目标**: DiT 内部 attention 能跨视角共享信息。

**改动点** (`diffsynth/models/wan_video_dit.py`):

**方案 A: Block-diagonal attention mask**
```
shape: [B*T*sum(H_i*W_i)/patch², dim]
按视角分块,每块内 self-attention;不同视角间 attention mask=-inf
```

**方案 B: View-aware attention bias**
```
attn = q @ k.T
attn_bias[view_i, view_j, :] = learnable if i != j else 0
attn = attn + attn_bias
```

**方案 C: 交替 attention**
```
Layer 1: spatial-temporal attn within each view
Layer 2: cross-view attn across views (no time)
```

**方案 D: RoPE-based 视角编码**
```
add view_id to RoPE 3D positional encoding
让 attention 自然区分视角
```

**推荐**: 方案 C (alternating attention),开销可控 + 架构改动小。

### Phase 5: 推理 (2-3 天)

**改动点** (`a2v/infer_a2v.py`):
- 输出 N 个视角独立 mp4(直接给 TriWorldBench)
- 或输出 1 个共享 latent + 视角 split

### Phase 6: 训练 + Eval (1 周)

- 用 Path A 已有的 dataset_mv_smoke (5 样本)做冒烟
- 验证 loss 下降 + 因果门控
- 全量训练(50-100 样本)
- 接入 TriWorldBench 评测(已有 A2V_SETUP.md 流程)

---

## 3. 风险与备选

### 风险 1: 显存爆炸

`vace_in_dim` 从 352 (单) → 1056 (3 视角) → DiT 输入 ×3 → 显存 ×2-3。

**缓解**:
- 降分辨率: 每视角 128×160 → 拼接前单视角显存小
- 切分多卡 DDP
- 用 encoded-cache 减少在线 VAE encode
- 8-bit Adam (`OPTIMIZER=adam8bit`)

### 风险 2: 跨视角 attention 难训练

可能学不到跨视角一致性。

**缓解**:
- 用 Path A baseline 做对比
- 消融:有 vs 无 cross-view attention
- 看 tri-view consistency metric 数值变化

### 风险 3: DiT 改动破坏其他 base_spec

改动 `diffsynth/models/wan_video_dit.py` 会影响所有 VACE-based 训练(包括 Path A 单视角)。

**缓解**:
- 把改动做成 opt-in:`--multiview_attention` flag 启用
- 不改 DiT,在 VACE unit 层加 view embedding(更稳)

### 备选方案: 不改 DiT,只用 view embedding + 通道拼接

如果跨视角 attention 难做,降级方案:
- vace_in_dim = N × (2*z + P*Q) — 通道堆叠
- 加 view-embedding channel — 模型知道哪通道对应哪视角
- 不改 DiT attention — 简单,但仍能学基础视角区分

这个方案能保留大部分 Path A 的代码,改动局限在 vace_unit.py + base_spec.py。

---

## 4. 关键文件改动清单

| 文件 | 改动量 | 优先级 |
|---|---|---|
| `a2v/vace_unit.py` | 大 (核心) | P0 |
| `diffsynth/models/wan_video_dit.py` | 中 (可选) | P1 (跨视角 attn) |
| `a2v/provision.py` | 小 | P0 |
| `a2v/base_spec.py` | 小 | P0 |
| `a2v/data/prepare.py` | 中 (新增多视角模式) | P1 |
| `a2v/multiview_prepare.py` (现有) | 中 | P1 (加 native 输出) |
| `a2v/train_a2v.py` | 中 | P1 |
| `a2v/infer_a2v.py` | 小 | P2 |
| `a2v/data/operators.py` | 小 | P1 |
| `examples/wanvideo/model_training/train.py` | 无需改 | — |

---

## 5. 验证策略

### 单元测试

1. **vace_unit 形状测试**
   ```python
   # 单视角(回归):vace_in_dim=352 ✓
   # 三视角: vace_in_dim=1056 (352*3) ✓
   ```
2. **跨视角 attention mask 形状测试**
3. **VAE encode 数值稳定性测试**

### 集成测试

1. **小数据 overfit**(5 样本,1 个 tra)
   - 验证 loss 单调下降到接近 0
2. **因果门控**(参考 Path A)
   - real vs none vs shuffle 三个 control
   - real 应远好于 none (MAE-GT)
3. **Tri-View Consistency(核心)**
   - 同一 tra 在不同视角下的预测一致性
   - 用 Path A baseline 对比

### TriWorldBench 评测

参考 `docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md` Phase 4-5。

**预期对比**:
| 方案 | TWB-Score (估) | 主要差距 |
|---|---|---|
| Path A (拼接, baseline) | 30-40 | Tri-View Consistency 极低 |
| Path B v1 (无 attn, 仅 view embed) | 40-50 | Tri-View Consistency 弱 |
| Path B v2 (cross-view attn) | 50-60 | 可能接近 BWM/WoVR_Plus |

---

## 6. 时间规划

| 阶段 | 时间 | 产出 |
|---|---|---|
| 文献调研 | 1-3 天 | 1 页对比表 + 1-2 个深入分析 |
| 数据格式扩展 | 1-2 天 | multiview_prepare --native_views 选项 |
| VACE Unit 扩展 | 3-5 天 | 支持 N vace_videos |
| DiT 跨视角 Attn | 5-7 天 | view-aware attention(可选降级) |
| 推理 | 2-3 天 | 输出 N 视角独立 mp4 |
| 训练 + Eval | 1 周 | TriWorldBench score |
| **总计** | **2-3 周** | 上 TriWorldBench 榜单 |

---

## 7. 立即可做 (无 GPU, 无外部资源)

即使没有 GPU 和外部数据,以下工作可以并行推进:

### 7.1 写 vace_unit 扩展版骨架

```bash
# 复制现有 vace_unit.py
cp a2v/vace_unit.py a2v/vace_unit_multiview.py
# 在新文件里加 multi-view 支持(可先 stub,后续实装)
```

### 7.2 设计 base_spec 扩展

```python
# a2v/base_spec.py 新增字段:
@dataclass(frozen=True)
class WanBaseSpec:
    ...
    num_vace_views: int = 1  # NEW: 视角数
    # derived
    @property
    def vace_in_dim(self) -> int:
        per_view = 2 * self.vae_z_dim + self.vae_spatial_factor ** 2
        return per_view * self.num_vace_views
```

### 7.3 文献调研

按 §6 Phase 1 列表,下载/阅读论文,写对比表。

### 7.4 写降级方案(仅通道堆叠,不跨视角 attn)

如果 Path B 全功能版时间紧张,降级方案:
- vace_in_dim = N × (2*z + P*Q)
- view embedding 加到 channels(可选)
- 不改 DiT
- 改动局限: `vace_unit.py` + `base_spec.py`

---

## 8. 路径与文件

```bash
# 已有交付
DIFFSYNTH=/disk/worldmodel/wangshilong/DiffSynth-Studio
TWB=/disk/worldmodel/wangshilong/TriWorldBench
WORK=/disk/worldmodel/wangshilong/a2v_v4_khl

# Path A 数据 (Path B 可复用)
DATASET_MV=$WORK/dataset_mv          # 3视角拼接 (256×960, 20 样本)
TWB_DOC_SETUP=$TWB/docs/A2V_SETUP.md
TWB_DOC_EXEC_PLAN=$TWB/docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md  # Path A exec plan
TWB_DOC_PATH_B=$TWB/docs/A2V_PATH_B_MULTIVACE_PLAN.md         # ← 本文档
```

---

## 9. 给探索 agent 的 TODO

### 必须读
1. `docs/A2V_TRIWORLDBENCH_EXEC_PLAN.md` — Path A baseline
2. `docs/A2V_SETUP.md` — 完整复现文档
3. `a2v/vace_unit.py` — 当前 VACE unit 实现
4. `diffsynth/pipelines/wan_video.py:649-706` — stock VACE 实现
5. `diffsynth/models/wan_video_dit.py` — DiT 实现
6. `a2v/base_spec.py` — 多 base_spec 配置
7. `a2v/A2V_HANDOFF.md` — 项目历史 + 设计哲学
8. `a2v/A2V_HISTORY.md` — 历史经验 + 踩坑

### 必须做
1. **先跑通 Path A baseline**(无 GPU 也行,做 dry-run 验证)
2. **按 §1 核心设计问题逐个回答,选定架构**
3. **按 §6 时间规划推进**
4. **每次改动 vace_unit.py 后跑 `check_parity.py` 确保不破坏单视角**
5. **小数据 overfit 验证(5 样本)**
6. **关键决策写 `A2V_PATH_B_DESIGN.md` 留下记录**

### 不要做
1. **不要改 diffsynth/ 任何原文件**(违反项目约束)
2. **不要破坏 Path A 的 256×960 拼接 dataset**(保持向后兼容)
3. **不要急着全量训练**(先 5 样本 sanity)
4. **不要假设特定 base_spec**(保持 WanBaseSpec 可扩展)

---

## 10. 关键约束(项目规范)

1. **不改 `diffsynth/` 原文件** — 全部 A2V 改动限定在 `a2v/`(SEAM 单元继承方式)
2. **跑 `a2v/check_parity.py` 验证不破坏 Path A baseline**
3. **每次 commit 后跑 `a2v.data.smoke` + `a2v.render.check_projection`**
4. **Wan2.2-TI2V-5B VAE 16× 空间压缩,H/W 必须 32 整除**
5. **vace_in_dim / mask_pq 必须从 spec 派生,不要 hardcode**
6. **multi-view data 走 fallback chain(同 Path A)**
