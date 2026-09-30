# Animation Mesh Pipeline

三阶段流程：动画帧采样 → 变脏拓扑 → GNN 训练用 PLY（per-face feature + label）。

```
角色包（ModelWithAnimationSelected：单角色 或 多角色根）
  → ① 插针导出 clean.fbx（+ skin.npz / pose.npz 骨骼旁路文件）
  → ② 对 clean 写 dirty{强度}.fbx          （不碰骨骼）
  → ③ 调朋友的 mesh_retopo_data_preproc 算 feature/label → dirty*.ply + metadata.json
```

每个阶段都能单独跑（命令见下）；也可以用 `run_pipeline.py` + 一个 JSON 配置一次跑完。

> 动画筛选（~2000 → ~200）在 [`Scripts/Mesh_Clean_To_Dirty/`](../../Mesh_Clean_To_Dirty/)（只放筛选 / 组包脚本，不再维护变脏）。

```
Animation_Mesh_Pipeline/
├── README.md
├── run_pipeline.py                        # 总入口：按配置依次跑 ①②③，可跳过/续跑
├── configs/
│   ├── mixamo.example.json                # 带骨骼角色：①→②→③（含 skin 特征）
│   └── primitive.example.json             # 静态 mesh：跳过 ①，直接 ②→③
├── common/                                # 各阶段共用（不依赖 bpy）
│   ├── naming.py                          # 文件夹命名规则
│   ├── skin_sidecar.py                    # skin.npz / pose.npz 格式
│   └── manifest.py                        # 每阶段的 stageN_manifest.json
├── stage1_sample/
│   └── batch_animation_frame_sampler.py   # ① 采样 clean 变形网格 + 骨骼旁路
├── stage2_dirty/
│   ├── dirty_topology_core.py             # 变脏算法核心
│   ├── generate_dirty_topology.py         # 单 mesh 预览
│   └── batch_apply_dirty.py               # ② 批量变脏
├── stage3_features/
│   ├── build_gnn_dataset.py               # ③ 批量出训练 PLY
│   ├── preproc_adapter.py                 # 唯一接触朋友仓库的地方
│   └── skin_transfer.py                   # clean → dirty 权重重心插值
└── tools/
    └── make_splits.py                     # 按角色划分 train/val/test
```

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| ① | `stage1_sample/batch_animation_frame_sampler.py` | 角色目录 / 多角色根 | `<char>/skin.npz`，`<char>/<anim>_<frame>/{clean.fbx, pose.npz}` |
| ② | `stage2_dirty/batch_apply_dirty.py` | ① 的输出根 | 同目录 `dirty10.fbx` … |
| ③ | `stage3_features/build_gnn_dataset.py` | ① 的输出根（已跑过 ②） | `[split/]<char>/<anim>_<frame>/dirty*.ply` + `metadata.json` |

下面命令里的 `Blender` 指 `/Applications/Blender.app/Contents/MacOS/Blender`，都在本目录下执行。

---

## 一键跑：`run_pipeline.py`

用普通 Python 跑就行（它自己起 Blender 子进程），比如 Blender 自带的 python：

```bash
PY=/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13

$PY run_pipeline.py --config configs/mixamo.example.json --dry_run   # 只打印命令
$PY run_pipeline.py --config configs/mixamo.example.json             # 跑 ①②③
$PY run_pipeline.py --config configs/mixamo.example.json --stages 3 --force   # 只重跑 ③
```

配置格式（相对路径以配置文件所在目录为基准）：

```json
{
  "blender": "/Applications/Blender.app/Contents/MacOS/Blender",
  "work_dir": "../../../../Data/pipeline_runs/mixamo",
  "frames_dir": null,
  "gnn_dir": null,
  "stage1": {"input_dir": "...", "args": {"frame_gap": 20, "max_armatures": 10}},
  "stage2": {"args": {"displace_strengths": [10, 40], "skip_existing": true}},
  "stage3": {"args": {"features": ["position", "normal", "skin_entropy"], "splits": "splits.json"}}
}
```

- `args` 里的键就是各阶段自己的命令行参数：`true` → 只加开关，`false`/`null` → 不加，列表 → 逗号拼接。
- 目录：①输出到 `work_dir/frames`，② 原地写 dirty，③ 输出到 `work_dir/gnn`。`frames_dir` / `gnn_dir` 可覆盖（primitive 配置就是把 `frames_dir` 指到已有的 clean 数据集，并省掉 `stage1`）。
- 省掉某个 `stageN` 段 = 不跑该阶段。③ 默认带 `--python-use-system-env`（需要 scipy），可用 `stage3.blender_args` 覆盖。
- 续跑：`work_dir/run_state.json` 记录每阶段的命令和完成时间。命令没变、上游没重跑、且上次 manifest 为 `ok` 的阶段会被跳过；上游重跑会让下游自动重跑。某阶段失败就停下。
- ② 加 `skip_existing`、③ 本身都按文件续跑；① 每次都重新导出，改了 ① 的参数请换一个 `work_dir`，不要新旧样本混在一起。

---

## 输入约定（多角色 / 单角色）

Stage ① 的 `--input_dir` 读取 **ModelWithAnimationSelected** 布局：

```
ModelWithAnimationSelected/          ← 多角色：传根目录
├── Aj/                              ← 单角色：直接传角色目录
│   ├── Aj.fbx                       # 角色模型（蒙皮网格 + 骨架）
│   └── animations/
│       ├── Back_Flip_To_Uppercut.fbx
│       └── ...
├── Arissa/
│   ├── Arissa.fbx
│   └── animations/...
└── ...
```

- 角色模型：优先 `{目录名}.fbx`，否则用根目录第一个含 MESH 的 FBX
- 动画：只读 `animations/*.fbx`
- 有一级合法角色子目录 → 多角色；目录本身就是角色包 → 单角色

---

## ① 动画帧采样

把动画 Action 赋到角色自己的 armature → 按 `frame_gap` 插针 → 所有 mesh 按名字排序合并成一个对象导出 `clean.fbx`，同时写骨骼旁路文件。

```bash
Blender --background --python stage1_sample/batch_animation_frame_sampler.py -- \
  --input_dir ".../Mixamo_Data/ModelWithAnimationSelected" \
  --output_dir ".../Data/pipeline_runs/mixamo/frames" \
  --max_armatures 50 --frame_gap 20
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input_dir` | `Mixamo_Data/ModelWithAnimationSelected` | 多角色根，或单角色目录 |
| `--output_dir` | `Data/output_animation_frames` | 导出根 |
| `--max_characters` | all | 处理多少个角色；`<0` = 全部 |
| `--character_start` | `0` | 跳过前 N 个角色 |
| `--max_armatures` | `50` | 每个角色采样多少个动画；`<0` = 全部 |
| `--armature_start` | `0` | 每个角色跳过前 N 个动画 |
| `--frame_gap` | `20` | 帧间隔（插针） |
| `--shuffle` | off | 随机抽动画，而不是按文件名 |
| `--seed` | `42` | `--shuffle` 的随机种子 |
| `--export_format` | `fbx` | `fbx` / `obj` / `ply` |

输出：

```
frames/
├── Aj/
│   ├── skin.npz                           # 每角色一份：蒙皮权重 + rest 骨骼
│   ├── Back_Flip_To_Uppercut_1/
│   │   ├── clean.fbx
│   │   └── pose.npz                       # 每帧一份：该帧骨骼位置 + 校验点
│   └── ...
├── stage1_manifest.json
├── run_log.txt
└── error_log.txt
```

---

## ② 变脏拓扑

### 算法

1. **先三角化**
2. **切面 XY 随机位移**（沿顶点法线切平面的局部 u/v）
3. **随机融并**（按比例 collapse 边）
4. **局部病变**（随机中心 → n-ring 细分 + 局部 collapse）

位移量随物体尺寸缩放：

```
magnitude = bbox_diagonal × displace_scale × (displace_strength / 100)
```

### 单 mesh 预览 `generate_dirty_topology.py`

选中 MESH → Run Script → 生成 `_clean`（隐藏）+ `_dirty`。改脚本顶部 `PARAMS`，或：

```bash
Blender --python stage2_dirty/generate_dirty_topology.py -- \
  --displace_strength 50 --merge_edge_ratio 0.08
```

### 批量 `batch_apply_dirty.py`

递归找 `clean.*`，同目录按强度写多档 dirty；在输入根写 `stage2_manifest.json`：

```bash
Blender --background --python stage2_dirty/batch_apply_dirty.py -- \
  --input_dir ".../Data/pipeline_runs/mixamo/frames" \
  --displace_strengths 10,40 --skip_existing
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--displace_strengths` | `10,20,40,70,100` | 每帧导出的位移强度列表（%） |
| `--displace_scale` | `0.002` | 相对 bbox 的基础位移尺度 |
| `--merge_edge_ratio` | `0.05` | 随机融并边比例 |
| `--disease_centers` | `2` | 病变中心数 |
| `--disease_collapse_ratio` | `0.08` | 病变区 collapse 比例 |
| `--no_merge` / `--no_disease` | — | 关掉对应步骤 |
| `--seed` | `42` | 随机种子 |
| `--skip_existing` | off | 已存在的 dirty 文件不重写 |

更细的 knobs 在 `dirty_topology_core.py` 的 `DirtyParams`。

| 目标 | 建议 |
|------|------|
| 轻微脏 | `displace_strength=30~50`, `merge_edge_ratio=0.02`, `disease_centers=1` |
| 中等 | `displace_strength=100`, `merge_edge_ratio=0.05`, `disease_centers=2` |
| 很脏 | `displace_strength=150~200`, `merge_edge_ratio=0.1`, `disease_centers=4` |

---

## ③ GNN 训练数据

对每个样本目录：读 `clean.fbx` 算 label（edge flow、singularity），再对每个 `dirty*.fbx` 算 per-face feature、把 label 从 clean 迁移过来，写 ASCII PLY。feature/label 的计算全部来自朋友的仓库 [`lab/perface-data-preproc-pipeline/mesh_retopo_data_preproc`](../../../../lab/perface-data-preproc-pipeline/mesh_retopo_data_preproc)，本阶段只做批处理、骨骼数据接入和数据划分。

```bash
Blender --background --python-use-system-env \
  --python stage3_features/build_gnn_dataset.py -- \
  --input_dir ".../Data/pipeline_runs/mixamo/frames" \
  --output_dir ".../Data/pipeline_runs/mixamo/gnn" \
  --features "position,normal,principal_directions,curvature,area,aspect_ratio,skin_entropy,joint_distance,bone_axis" \
  --splits configs/splits_mixamo.json
```

`--python-use-system-env` 是必须的：Blender 自带 Python 没有 scipy，要用 `~/.local` 里装的。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input_dir` / `--output_dir` | 必填 | ① 的输出根 / PLY 输出根 |
| `--preproc_root` | `lab/.../mesh_retopo_data_preproc` | 朋友仓库路径 |
| `--features` | 朋友仓库的默认特征 | 逗号分隔；可用名字见其 `src/features.py` 的 `FEATURE_REGISTRY` |
| `--no_label` | off | 只出 feature（推理用） |
| `--splits` | 无 | 划分文件；给了就输出到 `train/` `val/` `test/` 子目录 |
| `--overwrite` | off | 默认已存在的 PLY 会跳过（续跑） |
| `--max_samples` | all | 只处理前 N 个样本（冒烟） |

输出：

```
gnn/
├── metadata.json            # 朋友仓库的 schema：feature 列、label 名 + pipeline 信息
├── stage3_manifest.json
├── stage3_run_log.txt
├── stage3_error_log.txt     # 单个样本失败不会中断，记在这里
└── train/Aj/Back_Flip_To_Uppercut_1/{dirty10.ply, dirty40.ply}
```

注意：label `singularity_prob` 是 0/100 的“非四边形标记”，不是 0~1 概率（朋友仓库的设计）。

### 骨骼特征怎么来的

名字以 `skin_` / `joint_` / `bone_` 开头的特征需要骨骼数据，② 不碰骨骼，所以 ③ 这样补：

- clean 的权重：`skin.npz` 按顶点一一对应（① 导出时顶点顺序已固定，③ 会用 `pose.npz` 里的校验点确认顶点位置对得上）。
- dirty 的权重：dirty 每个顶点投影到 clean 最近的三角形，按重心坐标插值三个角点的权重。manifest 里的 `max_skin_transfer_distance_rel` 记录最大投影距离（相对角色尺寸）。
- 骨骼位置用 `pose.npz` 里该帧的姿态骨骼，而不是 rest pose。

没有 `skin.npz` 的数据（比如 primitive）只能用非骨骼特征。

### 数据划分：`tools/make_splits.py`

按角色（输入根下的一级文件夹）划分，同一个角色的所有帧只会出现在一个集合里：

```bash
$PY tools/make_splits.py --input_dir ".../Mixamo_Data/ModelWithAnimationSelected" \
  --output configs/splits_mixamo.json --test "Castle Guard 02" --val_ratio 0.1 --seed 0
```

输出 `{"train": [...], "val": [...], "test": [...]}`。③ 里不在任何集合中的角色会被跳过，并计入 manifest 的 `unassigned_split`。

---

## 数据约定

| 文件 | 位置 | 内容 |
|------|------|------|
| `skin.npz` | 每角色 | 骨骼名、每顶点 top-8 权重（索引 + 值）、rest 骨骼头尾、角色尺寸、各 mesh 的顶点偏移 |
| `pose.npz` | 每帧 | 该帧骨骼头尾（世界坐标）、帧号、动画名、64 个校验顶点的位置 |
| `stageN_manifest.json` | ①② 在 frames 根，③ 在 gnn 根 | `status`（`ok` / `partial`）、参数、计数、起止时间、pipeline git commit；③ 另记朋友仓库 commit |
| `splits.json` | 自定 | 角色名 → train/val/test |

格式定义在 `common/skin_sidecar.py`，读取不需要 bpy（numpy 即可）。

---

## 推荐跑法

1. **单角色冒烟**：配置里 `stage1.input_dir` 指一个角色包，`max_armatures: 2`，`frame_gap: 60`，`stage3.args.max_samples: 3`
2. **预览 dirty**：Blender 里打开一个 `clean.fbx`，跑 `generate_dirty_topology.py` 调参
3. **划分**：`tools/make_splits.py` 生成 splits，写进 `stage3.args.splits`
4. **正式**：换一个新的 `work_dir`，`run_pipeline.py` 一次跑完
