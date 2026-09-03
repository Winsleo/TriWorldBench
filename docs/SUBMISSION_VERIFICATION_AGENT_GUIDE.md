# TriWorldBench Submission 端到端验证指导(给 agent)

> **目的**: 验证选项 3 — 完整的 submission 流程(下载 test_dataset → 推理 500 ep → 打包 → SHA256 → README.txt)在本地能跑通。
> **状态(2026-09-03)**: 暂未跑过 500 ep 真提交。本指南是给后续 agent 跑端到端验证的 checklist + 复现命令。
> **关联**:
> - `SUBMISSION_README.md` — 官方 spec(500 ep, 320×240, match HDF5 first dim)
> - `docs/A2V_TRAINING_GUIDE.md` — A2V 训练经验
> - `docs/A2V_LOCAL_EVAL_ISSUES_TODO.md` — 本地评测踩坑
> - `submission_check.py` — 官方 validate + 打包 + sha256 工具

## 任务目标

完整跑一次 submission 流程(从 test_dataset 到 .zip + .sha256.txt + README.txt),验证:
1. test_dataset 下载可用(HF 镜像)
2. 模型推理能产出 320×240 mp4
3. 帧数能 match HDF5 first dim(或至少在 tolerance 内)
4. submission_check.py 能 validate + 打包 + sha256
5. README.txt 格式符合官方要求

## 关键约束(必读)

| 项目 | 规格 | 错误会怎样 |
|---|---|---|
| Episode 数 | 500 | 缺 ep → ERROR |
| Resolution | **320×240**(硬规定) | checker 报 ERROR |
| 视角文件 | `head.mp4` / `left.mp4` / `right.mp4`(允许 alias:`left_hand`/`right_hand`/`left-hand`/`right-hand`) | 缺视角 → ERROR |
| 帧数 | match `data/episodeN.hdf5` 的 `joint_action/vector` 第一维 | 差太多 → ERROR,差小 → WARNING |
| FPS | 任意(extract 时不影响 metric,默认 30) | 不影响 |
| 输出目录命名 | `episode1/`, `episode2`, ..., `episode500/` | 严格 1-indexed |
| `<method>.zip` 顶层 | 直接 `episode1/...episode500/`(不要外层目录) | 影响 unzip 后结构 |

submission_check.py 内置 `GT_EPISODE_FRAMES` dict(500 项),可以从源码 `submission_check.py:58+` 拿到。

## 推荐验证策略:分阶段验证

由于 500 ep 推理需要数小时到几天,分阶段验证最稳:

### Phase A:工具链验证(<10 min,不需 GPU)

**目标**: 确认 submission_check.py + ZIP + sha256 + ffprobe 都能工作,即使喂 mock mp4。

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench

# A.1 检查 ffprobe 在 PATH
which ffprobe || sudo apt install -y ffmpeg

# A.2 准备 mock 3 个 ep 的假 mp4(每个 320×240, 30 fps, 121 帧)
PY=/disk/worldmodel/uv/envs/a2v/bin/python
$PY -c "
import imageio.v3 as iio
import imageio
import numpy as np
from pathlib import Path

base = Path('/tmp/submission_mock/a2v_v4_test')
mkdir = base
mkdir.mkdir(parents=True, exist_ok=True)
for ep in range(1, 4):
    epd = mkdir / f'episode{ep}'
    epd.mkdir(exist_ok=True)
    for view in ['head', 'left', 'right']:
        # 320×240, 121 frames, fps=30
        frames = (np.random.rand(121, 240, 320, 3) * 255).astype(np.uint8)
        iio.imwrite(epd / f'{view}.mp4', frames, fps=30, codec='libx264')
print('mock 3 ep OK')
"

# A.3 跑 submission_check.py(应该 PASS)
$PY submission_check.py /tmp/submission_mock/a2v_v4_test --output /tmp/submission_mock/out

# A.4 验证产物
ls -lh /tmp/submission_mock/out/
# 应看到: a2v_v4_test.zip + a2v_v4_test.sha256.txt

# A.5 验证 sha256 一致
sha256sum /tmp/submission_mock/out/a2v_v4_test.zip
cat /tmp/submission_mock/out/a2v_v4_test.sha256.txt  # 应该等于上面那一行

# A.6 验证 zip 内容
unzip -l /tmp/submission_mock/out/a2v_v4_test.zip | head -10
# 顶层应直接有 episode1/episode2/episode3, 每个有 head/left/right.mp4
```

**预期**:`Status: PASS`,生成 `a2v_v4_test.zip` + `a2v_v4_test.sha256.txt`,无 ERROR。

### Phase B:test_dataset 下载验证(30 min)

**目标**: 确认 test_dataset 可下载并解压到正确格式。

```bash
# B.1 设置 HF 镜像(国内)
export HF_ENDPOINT=https://hf-mirror.com

# B.2 用 hf-cli 下载
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python  # 或者 hf 命令
$PY -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='TriWorldBench/Dataset',
    repo_type='dataset',
    allow_patterns=['test_dataset/*'],
    local_dir='/disk/worldmodel/TriWorldBench_data/test_dataset',
)
print('downloaded to:', path)
"

# B.3 验证结构
ls /disk/worldmodel/TriWorldBench_data/test_dataset/
# 应看到: data/ first_frame/ instructions/

ls /disk/worldmodel/TriWorldBench_data/test_dataset/data/ | head -5
ls /disk/worldmodel/TriWorldBench_data/test_dataset/data/ | wc -l  # 应 500
ls /disk/worldmodel/TriWorldBench_data/test_dataset/first_frame/ | head -3
ls /disk/worldmodel/TriWorldBench_data/test_dataset/instructions/ | head -3

# B.4 验证 HDF5 first dim 是 submission_check.py 内的 GT_EPISODE_FRAMES
$PY -c "
import h5py
import json

# 取 submission_check.py 里的 GT dict 校验
import sys
sys.path.insert(0, '/disk/worldmodel/wangshilong/TriWorldBench')
from submission_check import GT_EPISODE_FRAMES

mismatch = 0
for ep_id, expected_n in list(GT_EPISODE_FRAMES.items())[:50]:  # 抽检 50 个
    n = int(ep_id.replace('episode', ''))
    path = f'/disk/worldmodel/TriWorldBench_data/test_dataset/data/{ep_id}.hdf5'
    with h5py.File(path, 'r') as f:
        # 优先读 joint_action/vector, fallback 到第一个 dataset
        if 'joint_action' in f and 'vector' in f['joint_action']:
            n_actual = f['joint_action/vector'].shape[0]
        else:
            first_ds = list(f.values())[0]
            n_actual = first_ds.shape[0]
    if n_actual != expected_n:
        print(f'{ep_id}: HDF5={n_actual} vs GT_dict={expected_n}')
        mismatch += 1
print(f'mismatch: {mismatch}/50')
"
```

**预期**: download 成功,结构正确,GT dict 全部 match(若有 mismatch 说明官方更新过,需重新同步 submission_check.py 里的 dict)。

### Phase C:小规模推理验证(2-5 ep,~30 min-2 h)

**目标**: 用任意模型跑前 2-5 个 episode 验证推理能产出符合规格的 mp4。

**通用推理 wrapper**:`a2v_v4_khl/scripts/infer_for_submission.sh`(2026-09-03 新增)
- 支持多种模型:`a2v_path_a` / `a2v_path_a_baseline` / `ziyang_v3w_rf` / `dummy_random` / `custom`
- 自动 re-encode 到 320×240,121 帧(满足 submission_check.py spec)
- 自动 ffprobe 校验产物

```bash
cd /disk/worldmodel/wangshilong/a2v_v4_khl

# C.1 dummy_random(无 GPU,5 min 验证管道)
./scripts/infer_for_submission.sh \
    --method dummy_random --n-eps 5 \
    --output-dir /tmp/submission_dummy

# C.2 a2v_path_a_baseline(用现有 concat_split,无需 GPU,验证 Path A 模型输出)
./scripts/infer_for_submission.sh \
    --method a2v_path_a_baseline --n-eps 5 \
    --output-dir /tmp/submission_baseline

# C.3 a2v_path_a(跑真 Path A 推理,~25 min/5ep)
./scripts/infer_for_submission.sh \
    --method a2v_path_a --n-eps 5 \
    --output-dir /tmp/submission_a2v \
    --ckpt step-800.safetensors

# C.4 ziyang_v3w_rf(用 ziyang 预生成产物)
./scripts/infer_for_submission.sh \
    --method ziyang_v3w_rf --n-eps 5 \
    --output-dir /tmp/submission_ziyang

# C.5 自定义模型:写自己的脚本,加 --custom-script
./scripts/infer_for_submission.sh \
    --method custom \
    --custom-script /path/to/my_infer.sh \
    --output-dir /tmp/submission_custom
```

**添加新模型**:在 `a2v_v4_khl/scripts/methods/` 新建 `<name>.sh`,在 `infer_for_submission.sh` 的 `METHOD_REGISTRY` 加一行映射。

**注意**:
- `num_frames` 实际应该是 HDF5 first dim(查 GT_EPISODE_FRAMES dict),但当前 Path A 训练固定 121 帧。
- Phase C 主要是验证管道通,帧数偏差(121 vs 实际)由 submission_check.py 的 tolerance 容忍(差 0.5x)。
- 真实 500 ep 提交需要按 GT 帧数生成(每 ep 不同),这是已知 work-around:在 dataset 里循环跑 inference,用 frame_counts dict 控制每 ep 的 num_frames。
- wrapper 默认 target=121 帧,如需不同帧数传 `--target-frames N`

**Phase C 验证**:
```bash
# 跑完看产物
ls /disk/worldmodel/TriWorldBench_data/submission/a2v_v4_test/
# 应有 episode1/, episode2/, ... episode5/

# 验证每 mp4 都是 320×240
for ep in 1 2 3 4 5; do
  for view in head left right; do
    ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,nb_frames,r_frame_rate \
      /disk/worldmodel/TriWorldBench_data/submission/a2v_v4_test/episode${ep}/${view}.mp4
  done
done
```

**预期**: 全部 320×240, nb_frames ≈ 121(可能 ±4 因 VAE 的 time_division_remainder=1)。

### Phase D:submission_check.py 端到端验证(<5 min)

```bash
cd /disk/worldmodel/wangshilong/TriWorldBench

# D.1 跑 checker
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
$PY submission_check.py /disk/worldmodel/TriWorldBench_data/submission/a2v_v4_test \
  --output /tmp/submission_mock/end2end

# D.2 预期 Status: WARNING(因为只有 5 ep,缺 495 ep)
#   然后问 "Continue packaging? (y/N)",输 y 继续
#   或: 提前拷贝 submission_check.py 临时改为不强制 500 ep 检查(不建议改 checker)
# D.3 看产物
ls -lh /tmp/submission_mock/end2end/
cat /tmp/submission_mock/end2end/a2v_v4_test.sha256.txt
```

### Phase E:README.txt 模板(2 min)

按 `SUBMISSION_README.md §5` 格式:

```text
Model name: A2V v4 (WangShilong)
Affiliation / Organization / Team Name: SJTU A2V Lab
Contact email: your_email@example.com
Project / Code repository: https://github.com/your_name/a2v (optional)
Brief introduction: A2V is a tri-view adaptive world model based on Wan2.2-TI2V-5B...
```

放在 `$OUT/README.txt`,发邮件时和 zip + sha256 一起附。

## 端到端全跑(选项 3 终极版,需要 250+ GPU h)

不推荐一次跑完。分批:
1. 1 GPU × 5 ep = 验证管道(~30 min)
2. 1 GPU × 50 ep = 验证批量推理(~5 h)
3. 8 GPU × 500 ep = 全跑(~15-25 h,需 distribute)

## 关键 gotcha(必看)

1. **submission_check.py 内置 500 ep 名单**(submission_check.py:58+ 的 GT_EPISODE_FRAMES dict)。如果缺任何 ep,会报 ERROR 拒打包。Phase C 跑 5 ep 会报这个错,正常,验证管道就好。
2. **帧数 mismatch tolerance** = 0.5(FRAME_ERROR_THRESHOLD)。即 121 帧 vs GT=138: (138-121)/138 ≈ 0.12,在 tolerance 内,只 WARNING。121 vs GT=659: 0.82,ERROR。
3. **ffprobe 必须可用**。`which ffprobe` 应在 PATH,否则 checker 直接 fail。
4. **不允许 view alias 文件混用**:每个 ep 内 3 个 view 文件必须存在(用任一 alias 即可)。Phase C 别只跑 head,不跑 left/right。
5. **output_path 必须存在**:提交前 `mkdir -p $OUT/<method>`。
6. **不要用相对路径做 cwd**:`submission_check.py /abs/path/to/method` 用绝对路径。

## 完成 Definition of Done

- [ ] Phase A mock 3 ep → submission_check.py PASS + zip + sha256 生成
- [ ] Phase B test_dataset 下载 + HDF5 first dim 与 GT dict 匹配
- [ ] Phase C 推理 5 ep → 15 个 mp4 (320×240)
- [ ] Phase D 跑 submission_check.py(预期 WARNING + 确认打包)
- [ ] Phase E README.txt 写好

完成 Phase A-D 后即可对外宣称"submission 管道验证通过"。Phase E 是写 README.txt 的工作,放最后。

## 给接手 agent 的 sanity check 起点

```bash
# 一次性跑 Phase A(无需 GPU,~5 min)
which ffprobe
PY=/disk/worldmodel/uv/envs/triworldbench/bin/python
$PY -c "import imageio.v3 as iio, numpy as np; from pathlib import Path; \
  b = Path('/tmp/submission_mock/a2v_v4_test'); b.mkdir(parents=True, exist_ok=True); \
  [((epd:=b/f'episode{ep}').mkdir(exist_ok=True), \
    [iio.imwrite(epd/f'{v}.mp4', (np.random.rand(121,240,320,3)*255).astype(np.uint8), fps=30, codec='libx264') \
     for v in ['head','left','right']]) for ep in range(1,4)]; print('mock OK')"

cd /disk/worldmodel/wangshilong/TriWorldBench
$PY submission_check.py /tmp/submission_mock/a2v_v4_test --output /tmp/submission_mock/out
ls -lh /tmp/submission_mock/out/
```

如果 Phase A 出错,先 debug(checker 依赖、ffprobe、Python 版本)。Phase A 通过后再做 B/C/D。

## 关联资源

- `submission_check.py` 完整源码在 `TriWorldBench/` 根目录
- `SUBMISSION_README.md` 官方 spec
- HF 数据集 `TriWorldBench/Dataset`
- 榜单 `triworld_submit@foxmail.com`
- 测试数据集 val_dataset (100 ep, 含 GT)与 test_dataset (500 ep, 不含 GT) 区别,详见 `docs/LOCAL_EVAL_GUIDE.md §0`

## 不在范围内(留给后续)

- [ ] 真正的 500 ep 全推理 + 提交邮件(需要大量 GPU 时间)
- [ ] Path B (multi-VACE) 集成后再跑一次提交(预期 TWB-Score 高很多)
- [ ] 自动 batch_infer_test_dataset.sh 写完后整合进 `a2v_v4_khl/`

---

**生成时间**: 2026-09-03
**作者**: Claude (wangshilong session)
**目标读者**: 接手 submission 流程验证的 agent