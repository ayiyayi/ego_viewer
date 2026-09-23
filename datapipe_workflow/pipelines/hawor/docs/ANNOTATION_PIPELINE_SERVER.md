# 数据标注管线 Server

> 版本：v1.0 · 日期：2026-06-02 · 状态：草案
> 项目：HaWoR 第一视角双手运动重建 · 适用分支：`server`

---

## 机器准备

- GPU 机器可按需找 **@朱雪月** 申请。
- 单任务资源基线（来自管线现状）：**1 张 GPU 独占**，峰值显存约 14 GB，权重约 5.6 GB，本地 scratch 盘用于存放抽帧等中间产物（建议 NVMe，按视频长度预留空间）。
- 软件依赖：CUDA 11.7 运行时、Python 3.10、`ffmpeg`（须在 PATH，管线启动会校验）、编译好的 `DROID-SLAM` CUDA 扩展、初始化好的 `Metric3D`。

---

## 管线设计

### 0. 完整 Pipeline（当前实现）

整个系统是**三层嵌套**：服务封装层 → 分段编排层 → 单段核心算法层。

```
HaWoRVideoProcessor.process_video()        ← 第3层：服务封装（GPU池、对象存储、后处理）
  └─ segmented_demo_pipeline.py            ← 第2层：长视频分段编排 + 结果合并
       └─ demo.py  (每个 chunk 跑一次)      ← 第1层：单段 4 阶段核心算法
```

#### 第 1 层：单段核心算法（`demo.py`）

对每个视频段执行 4 个阶段（+ 可选可视化）：

| # | 阶段 | 函数 | 输入 → 输出 | 关键模型 |
|---|---|---|---|---|
| 1 | 检测与跟踪 | `detect_track_video()` | 视频 → 抽帧(30fps) + 手部框/轨迹 | YOLO 手部检测器 `detector.pt` |
| 2 | 手部姿态估计 | `hawor_motion_estimation()` | 帧+轨迹 → 相机系手部参数 + 手部掩码 | 主模型 `hawor.ckpt`（ViT+Transformer） |
| 3 | SLAM 相机轨迹 | `hawor_slam()` | 帧+手部掩码 → 相机轨迹 R/t + 绝对尺度 scale | DROID-SLAM `droid.pth` + Metric3D 深度 |
| 4 | 运动填补 + 坐标转换 | `hawor_infiller()` | 相机系参数+轨迹 → 世界系双手运动 | 填补网络 `infiller.pt`（Transformer） |
| 5 | 可视化（可选） | `run_vis2_on_video` | 世界系手部 → MP4 | aitviewer 离屏渲染 |

- 阶段 4 产出：`pred_trans`（平移）、`pred_rot`（根旋转）、`pred_hand_pose`（45维关节）、`pred_betas`（MANO形状）、`pred_valid`（有效标志）。
- **依赖关系**：阶段 2 的手部掩码喂给阶段 3（让 SLAM 排除手部、只用背景估相机轨迹）；阶段 3 的相机轨迹+尺度喂给阶段 4（把相机系手部对齐到世界系）。`vis_mode=off` 时跳过阶段 5（服务默认关）。

#### 第 2 层：长视频分段编排（`segmented_demo_pipeline.py`）

长视频一次跑完太贵，这层负责切分→逐段跑→合并回单一时间线：

1. **切分** `split_video()` — ffmpeg 按 `segment_seconds`（默认 120s）切成 `chunk_*.mp4`；末段过短则并入前一段（`min_last_segment_seconds`）。
2. **逐段推理** `run_demo_on_chunk()` — 对每个 chunk 调用第 1 层 `demo.py`。
3. **合并**（按累计帧数恢复全局时间轴，`overlap_policy` 处理重叠）：`merge_cam_space()` / `merge_extracted_images()` / `merge_slam()`（按段长加权平均 scale）。
4. **回填** — 把合并结果写到 `merged/` 和原始视频 seq 目录。

> ⚠️ 当前这层**只合并相机系参数和 SLAM**，注释明确 "No world-space alignment is performed"——跨段世界系全局对齐尚未实现。

#### 第 3 层：服务封装（`HaWoRVideoProcessor.process_video()`）

1. **GPU 池** — `gpu_pool.acquire()` 拿卡，设 `CUDA_VISIBLE_DEVICES`，一卡一任务。
2. **跑分段管线** — 子进程调用第 2 层。
3. **拷贝合并产物** — `cam_space/`、`SLAM/`、`extracted_images/` 拷到 `output_dir`。
4. **后处理**（`run_post_steps=True`）：`extract_image_50fps.py` 抽 50fps 帧 + `interpolation.py` 把标注从 30fps 插值到 50fps。
5. **清理** — 删 `_segmented_work` 中间目录。

#### 端到端数据流

```
视频
 └► [切分] chunk_0 … chunk_n
        每段:
        [1 检测跟踪] ─抽帧+手框─► [2 姿态估计] ─相机系参数+掩码─┐
                                                              ├► [3 SLAM] ─轨迹+尺度─► [4 填补/转世界系] ─► 世界系双手运动
        各段结果 ──► [合并 cam_space / SLAM / images 到全局时间轴]
                          └► [50fps 后处理: 抽帧 + 插值上采样]
                                  └► 最终标注产物
                                     (cam_space JSON / SLAM npz / world_space_res*.pth / 可选 vis.mp4)
```

---

### 1. 对象存储读写（petrel-oss / s3mount）

管线的输入视频与输出标注结果都位于**对象存储**上。提供两种接入方式，二选一或混用：

#### 方式 A：petrel-oss SDK（程序内读写，推荐）

参考文档：`http://sdoc.pjlab.org.cn/doc/#/petrel-oss/sdk/SDK安装`

安装与配置：
```bash
pip install petrel-oss-sdk        # 或按内网文档指定的 wheel 安装
```
配置文件 `~/petreloss.conf`（由平台提供 ak/sk 与 endpoint）：
```ini
[DEFAULT]
enable_mc = True

[cluster_name]
host_base = http://<endpoint>
access_key = <ak>
secret_key = <sk>
```

基本用法（封装到管线的存储抽象层 `StorageClient`）：
```python
from petrel_client.client import Client

client = Client(conf_path="~/petreloss.conf")

# 读取一个对象到内存
data = client.get("cluster:s3://bucket/path/video.mp4")

# 写入一个对象
client.put("cluster:s3://bucket/out/result.json", payload_bytes)

# 列举 folder 下所有对象（用于遍历待标注视频）
for entry in client.get_file_iterator("cluster:s3://bucket/videos/"):
    key, size = entry
    ...
```

> 管线内部统一用 `s3://bucket/prefix/...` 风格路径，由 `StorageClient` 屏蔽 petrel 的 `cluster:` 前缀差异。视频/产物较大时，先 `get` 到本地 scratch、处理完再 `put` 回对象存储，避免在内存里搬运大文件。

#### 方式 B：s3mount（把对象存储挂成本地目录）

按 s3mount 使用说明，将 bucket 挂载到本地路径（如 `/mnt/oss/`），管线即可像读本地文件一样读写：
```
s3://bucket/videos/   →  /mnt/oss/videos/
```
- 优点：现有以本地路径为输入的 `HaWoRVideoProcessor.process_video()` **无需改动**，直接传挂载路径。
- 缺点：随机/大量小文件 IO 性能不如本地盘，抽帧等密集 IO 仍建议落到本地 scratch。
- 建议：**输入视频走 s3mount 直读，中间产物落本地 scratch，最终结果写回 s3mount/对象存储。**

> 落地策略：首期用 **方式 B（s3mount）** 最快打通——管线代码几乎零改动；后续对吞吐/可靠性有更高要求时切换到 **方式 A（SDK）** 做精细化读写与重试。

---

### 2. 管线接口

**核心契约：输入「待标注视频 folder 的完整路径」，输出「标注结果的完整路径」。**

路径既可以是对象存储路径（`s3://bucket/...`），也可以是 s3mount 后的本地路径（`/mnt/oss/...`），由 `StorageClient` 统一处理。

#### HTTP 接口

```
POST /v1/annotate
Content-Type: application/json

{
  "input_dir":  "s3://bucket/datasets/egocentric/batch_01/",   // 待标注视频所在 folder
  "output_dir": "s3://bucket/annotations/batch_01/",           // 结果输出 folder（可选，缺省按规则生成）
  "img_focal": 600,            // 可选，默认 600 / 自动估计
  "vis_mode": "off",           // 可选，off / world，默认 off（关渲染省时）
  "overwrite": false           // 可选，结果已存在时是否覆盖
}

→ 202 Accepted
{
  "job_id": "j_abc123",
  "status": "PENDING",
  "input_dir":  "s3://bucket/datasets/egocentric/batch_01/",
  "output_dir": "s3://bucket/annotations/batch_01/"
}
```

```
GET /v1/jobs/{job_id}
→ 200
{
  "job_id": "j_abc123",
  "status": "RUNNING",          // PENDING / RUNNING / SUCCEEDED / FAILED / CANCELED
  "progress": 62,               // 0-100，按 已完成视频数/总数 统计
  "videos_total": 12,
  "videos_done": 7,
  "videos_failed": 0,
  "output_dir": "s3://bucket/annotations/batch_01/"
}
```

```
GET /v1/jobs/{job_id}/result
→ 200 (status == SUCCEEDED)
{
  "output_dir": "s3://bucket/annotations/batch_01/",   // ← 最终交付：标注结果完整路径
  "items": [
    { "video": "video_0.mp4",
      "status": "ok",
      "result_dir": "s3://bucket/annotations/batch_01/video_0/" },
    ...
  ]
}
```

> **最小契约**：调用方只需关心 `input_dir` 进、`output_dir`（`/result` 返回值）出；中间的队列、GPU 调度、分段处理对调用方透明。

#### 行为定义

1. 服务遍历 `input_dir` 下所有视频文件（`.mp4/.mov/.avi`，递归可选）。
2. 每个视频作为一个子任务，分发到 GPU worker，调用现有
   `HaWoRVideoProcessor.process_video(video, out_dir)`。
3. 每个视频的产物写到 `output_dir/<video_stem>/` 下，包含：
   - `cam_space/`（逐帧手部参数 JSON）
   - `SLAM/`（相机轨迹 + scale，NPZ）
   - `world_space_res*.pth`（世界坐标系手部参数）
   - `vis.mp4`（仅 `vis_mode!=off` 时）
4. 全部完成后，`output_dir` 即为「标注结果的完整路径」。

#### 等价 CLI（便于离线批跑 / 联调）
```bash
python -m service.annotate \
  --input_dir  s3://bucket/datasets/egocentric/batch_01/ \
  --output_dir s3://bucket/annotations/batch_01/ \
  --gpu_ids 0,1,2,3
```

#### 调用示例（curl）

启动服务：
```bash
pip install -r service/requirements.txt
HAWOR_GPU_IDS=0,1,2,3 uvicorn service.api:app --host 0.0.0.0 --port 8000
```

1）提交任务（输入视频 folder → 立即返回 job_id）：
```bash
curl -s -X POST http://localhost:8000/v1/annotate \
  -H 'Content-Type: application/json' \
  -d '{
        "input_dir":  "s3://bucket/datasets/egocentric/batch_01/",
        "output_dir": "s3://bucket/annotations/batch_01/",
        "img_focal": 600,
        "vis_mode": "off"
      }'
# → {"job_id":"j_abc123","status":"PENDING","input_dir":"...","output_dir":"..."}
```

2）查询进度：
```bash
curl -s http://localhost:8000/v1/jobs/j_abc123
# → {"job_id":"j_abc123","status":"RUNNING","progress":62,
#    "videos_total":12,"videos_done":7,"videos_failed":0,"output_dir":"..."}
```

3）获取标注结果完整路径（job 完成后）：
```bash
curl -s http://localhost:8000/v1/jobs/j_abc123/result
# → {"output_dir":"s3://bucket/annotations/batch_01/",
#    "items":[{"video":"video_0.mp4","status":"ok",
#              "result_dir":"s3://bucket/annotations/batch_01/video_0/"}, ...]}
```

4）取消任务 / 健康检查：
```bash
curl -s -X POST http://localhost:8000/v1/jobs/j_abc123/cancel
curl -s http://localhost:8000/healthz   # → {"status":"ok","gpu_ids":[0,1,2,3]}
```

5）轮询直到完成（脚本示例）：
```bash
job=$(curl -s -X POST http://localhost:8000/v1/annotate \
  -H 'Content-Type: application/json' \
  -d '{"input_dir":"/mnt/oss/videos/","output_dir":"/mnt/oss/anno/"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
until curl -s http://localhost:8000/v1/jobs/$job | grep -q '"status":"SUCCEEDED"'; do
  curl -s http://localhost:8000/v1/jobs/$job | python -c 'import sys,json;j=json.load(sys.stdin);print(j["status"],j["progress"])'
  sleep 30
done
curl -s http://localhost:8000/v1/jobs/$job/result
```

> 用 s3mount 时，`input_dir`/`output_dir` 直接传挂载后的本地路径（如 `/mnt/oss/...`），无需 petrel 配置。

---

### 3. 数据标注速率

- **当前指标**：处理时长 : 原始视频时长 ≈ **20 : 1（单卡）**。
  即 1 分钟视频单卡约需 20 分钟处理。
- **吞吐估算**：N 卡近似线性加速（一卡一任务），整批吞吐 ≈ `视频总时长 × 20 / N`。
  例：100 分钟素材、4 卡 → ≈ `100×20/4 = 500` 分钟。
- **优化空间（待评估）**：
  - **SLAM 阶段是瓶颈**（DROID-SLAM + Metric3D 逐帧深度），是首要优化对象。
  - 抽帧/IO：中间帧落本地 NVMe scratch，避免直接读写对象存储。
  - 分段并行：长视频按段（默认 100s）拆分到多卡并行，缩短单视频墙钟时间。
  - 关渲染：服务默认 `vis_mode=off`，省去 aitviewer 离屏渲染开销。
  - 推理加速：半精度 / 编译优化 / 批处理姿态估计阶段（需精度回归验证）。
- 速率指标应作为容量规划与超时设置的依据（超时 ≈ `基准 + 20 × 视频时长 × 安全系数`）。

---

## 端到端数据流

```
调用方 ──POST input_dir──► Server ──遍历视频──► 任务队列
                                                   │ 一卡一任务
                              ┌────────────────────┼────────────────────┐
                         GPU worker 0          GPU worker 1   ...   GPU worker N
                              │                                          │
                  ① get/挂载 视频 ◄────────── 对象存储 (petrel-oss / s3mount)
                  ② 本地 scratch 跑 HaWoR 管线（4 阶段）
                  ③ put 产物 ─────────────► output_dir/<video>/
                              │
                              ▼
调用方 ◄──GET /result── output_dir（标注结果完整路径）
```

---

## 落地建议（首期）

1. **存储**：先用 **s3mount** 打通，`input_dir/output_dir` 直接用挂载后的本地路径，`HaWoRVideoProcessor` 零改动；视频抽帧等中间产物强制落本地 scratch。
2. **接口**：`service/annotate.py` 提供 `POST /v1/annotate` 与等价 CLI，内部用 GPU 池（每卡一 worker）批量调度 `process_video`。
3. **可观测**：记录每视频耗时 / 视频时长比，验证 20:1 指标并定位 SLAM 优化收益。
4. **后续**：切换 petrel-oss SDK 做精细读写与重试；接入异步队列（Celery+Redis）做大批量、断点续跑（详见 `docs/SERVICE_DESIGN.md`）。

---

## 容器化部署（Docker）

镜像构建文件见项目根目录 `Dockerfile` / `.dockerignore`。

要点：
- 基础镜像用 **CUDA 11.7 devel**（`DROID-SLAM` 需在构建期 `python setup.py install` 编译 CUDA 扩展）。
- 权重（~3.5GB）和 MANO 模型**不打进镜像**，由 `.dockerignore` 排除、运行时 `-v` 挂载，保持镜像精简。
- 默认 `CMD` 以 FastAPI 服务方式启动。

构建：
```bash
docker build -t hawor-service:1.0 -f Dockerfile .
```

运行（挂载权重 / 数据 / 对象存储 / petrel 配置）：
```bash
docker run --gpus all -p 8000:8000 \
  -e HAWOR_GPU_IDS=0,1,2,3 \
  -v /path/to/weights:/app/weights \
  -v /path/to/_DATA:/app/_DATA \
  -v /mnt/oss:/mnt/oss \                          # s3mount 挂载点(可选)
  -v $HOME/petreloss.conf:/root/petreloss.conf \  # petrel-oss 配置(走 s3:// 时需要)
  hawor-service:1.0
```

| 挂载点 | 内容 | 是否必需 |
|---|---|---|
| `/app/weights` | detector.pt / hawor.ckpt / infiller.pt / droid.pth / metric3d | 必需 |
| `/app/_DATA` | MANO_RIGHT.pkl / MANO_LEFT.pkl | 必需 |
| `/mnt/oss` | s3mount 挂载的对象存储 | 用 s3mount 时 |
| `/root/petreloss.conf` | petrel-oss ak/sk/endpoint | 走 `s3://` 路径时 |

> K8s 部署：将上述挂载替换为 PVC（权重用 ReadOnlyMany）+ 本地 NVMe scratch（emptyDir），`resources.limits: nvidia.com/gpu: N`，详见 `docs/SERVICE_DESIGN.md`。

---

## 待确认问题

- 对象存储集群名 / endpoint / bucket 命名规范（用于 `petreloss.conf` 与路径前缀）。
- `output_dir` 缺省生成规则（如 `input_dir` 同级的 `annotations/` 镜像目录）。
- 是否需要返回逐帧 JSON 的标准化 schema 文档（`cam_space` 当前为旋转矩阵列表）。
- 单视频时长上限与是否强制多卡分段并行。

  一句话概括

  HaWoR 从第一视角视频重建世界坐标系下的双手运动——输入一段 egocentric 视频，输出每一帧左右手的 3D 姿态、形状和在世界空间中的轨迹。

  ---
  原管线的两个入口

  ┌────────────────┬────────────────────────────────────┬──────────────────────────────┐
  │      入口      │                文件                │             角色             │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ 单视频端到端   │ demo.py                            │ 真正的 4 阶段算法管线        │
  ├────────────────┼────────────────────────────────────┼──────────────────────────────┤
  │ 长视频分段编排 │ scripts/segmented_demo_pipeline.py │ 切分 → 逐段调 demo.py → 合并 │
  └────────────────┴────────────────────────────────────┴──────────────────────────────┘

  ▎ 注：scripts/hawor_video_processor.py（GPU 池封装）和我们新建的 service/ 属于服务化层，不算「原管线」。

  ---
  核心算法管线（demo.py）—— 4 个阶段

  视频
   │
   ├─[1] 检测与跟踪  detect_track_video()
   │      ffmpeg 抽帧(30fps) + YOLO 检测手部 + 跨帧跟踪
   │      产出: 抽帧图像、手部框、左右手轨迹
   │
   ├─[2] 手部姿态估计  hawor_motion_estimation()        模型: hawor.ckpt (ViT+Transformer)
   │      估计【相机坐标系】下每帧手部参数 + 渲染手部掩码
   │      产出: cam_space 参数、手部 mask
   │
   ├─[3] SLAM 相机轨迹  hawor_slam()                     DROID-SLAM + Metric3D
   │      用手部 mask 屏蔽手部, 只靠背景估相机轨迹 R/t
   │      Metric3D 估单目深度 → 恢复场景绝对尺度 scale
   │      产出: 相机轨迹 + scale (npz)
   │
   ├─[4] 运动填补 + 坐标转换  hawor_infiller()           模型: infiller.pt (Transformer)
   │      把相机系手部用 SLAM 轨迹对齐到【世界系】
   │      Transformer 填补缺失/遮挡帧
   │      产出: pred_trans / pred_rot / pred_hand_pose / pred_betas / pred_valid
   │
   └─[5] 可视化(可选)  run_vis2_on_video()               aitviewer 离屏渲染 → MP4

  阶段间的关键耦合：
  - 阶段 2 的手部掩码 → 喂给阶段 3，让 SLAM 把手排除掉，只用静态背景估相机运动。
  - 阶段 3 的相机轨迹 + 绝对尺度 → 喂给阶段 4，把相机系的手对齐到世界系。

  ---
  分段编排层（segmented_demo_pipeline.py）
  
  长视频一次跑太贵，所以：
  1. 切分 — ffmpeg 按时长（默认 120s）切成 chunk_*.mp4，末段过短并入前段。
  2. 逐段推理 — 每个 chunk 跑一遍上面的 4 阶段。
  3. 合并 — 按累计帧数恢复全局时间轴，合并各段的 cam_space（手部参数 JSON）、extracted_images（抽帧）、SLAM（轨迹，scale 按段长加权平均）。

  ▎ ⚠️  局限：这层注释明确写了 "No world-space alignment is performed"——跨段之间的世界系全局对齐目前没做，只是把相机系参数和各段 SLAM 拼接起来。

  ---
  最终产物

  ┌──────────────────────┬─────────┬────────────────────────────────────────────────────────┐
  │         产物         │  格式   │                          内容                          │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ cam_space/           │ JSON    │ 逐帧、相机系的手部参数（旋转矩阵等）                   │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ SLAM/...npz          │ NPZ     │ 相机轨迹 R/t + 绝对尺度 scale                          │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ world_space_res*.pth │ PyTorch │ 世界系双手运动（平移/旋转/45维关节/MANO形状/有效标志） │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ extracted_images/    │ JPG     │ 抽帧（含可选 50fps 上采样版本）                        │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ vis_*/...mp4         │ MP4     │ 可选可视化渲染                                         │
  └──────────────────────┴─────────┴────────────────────────────────────────────────────────┘

  ---
  用到的模型/权重（共 ~3.5GB）
  
  ┌────────────────┬────────────────────────────────────────────┬─────────────────────────────────┐
  │      阶段      │                    模型                    │              权重               │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
  │ 检测跟踪       │ YOLO 手部检测器                            │ detector.pt                     │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
  │ 姿态估计       │ HaWoR 主模型（ViT backbone + Transformer） │ hawor.ckpt                      │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
  │ SLAM           │ DROID-SLAM                                 │ droid.pth                       │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
  │ SLAM 深度/尺度 │ Metric3D                                   │ metric_depth_vit_large_800k.pth │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
  │ 运动填补       │ Transformer Infiller                       │ infiller.pt                     │
  ├────────────────┼────────────────────────────────────────────┼─────────────────────────────────┤
  │ 手部模型       │ MANO（左/右手）                            │ MANO_LEFT/RIGHT.pkl             │
  └────────────────┴────────────────────────────────────────────┴─────────────────────────────────┘

  性能特征：必须 GPU，峰值显存 ~14GB；处理耗时 : 视频时长 ≈ 20:1（单卡），其中 SLAM 阶段（DROID-SLAM + Metric3D 逐帧深度）是最大瓶颈。