# HaWoR 推理管线服务化设计文档

> 版本：v1.0 · 日期：2026-06-02 · 状态：草案
> 适用分支：`server`

---

## 1. 背景与目标

HaWoR（CVPR 2025）从第一视角视频重建世界坐标系下的双手运动。当前已有三种入口：

- `demo.py`：单视频端到端 CLI。
- `app.py`：Gradio 演示界面（单 GPU、无队列、无错误恢复）。
- `scripts/hawor_video_processor.py`：**线程安全、带 GPU 池**的 Python API（`HaWoRVideoProcessor`），内部调用 `segmented_demo_pipeline.py` 做长视频分段处理。

本设计的目标：在 **不重写算法管线** 的前提下，把 `HaWoRVideoProcessor` 封装为一个**生产级异步 HTTP 服务**，支持多用户并发提交视频、异步查询进度、下载结果，并具备可扩展、可观测、可恢复的运维能力。

### 非目标
- 不优化模型本身的精度或速度（算法层不动）。
- 不做实时流式推理（管线含 SLAM 全局优化，天然批处理）。
- 不在本期实现多机分布式训练；仅做单机/多机推理横向扩展。

---

## 2. 关键约束（来自管线现状）

| 维度 | 现状 | 对服务的影响 |
|---|---|---|
| 处理模式 | 4 阶段批处理（检测跟踪 → 姿态估计 → SLAM → 填补），SLAM 为瓶颈 | 必须**异步**，单请求秒级到分钟级，不能同步阻塞 HTTP |
| 资源 | 必须 GPU，峰值显存 ~14 GB，权重 ~5.6 GB | 1 个并发任务独占 1 张 GPU；用 GPU 池控制并发 |
| 耗时 | ~60–225 秒 / 100 帧；长视频自动分段（默认 100s/段） | 需任务队列 + 进度反馈；长任务需断点续跑 |
| 进程模型 | `HaWoRVideoProcessor` 通过 `subprocess` 调子脚本，靠 `CUDA_VISIBLE_DEVICES` 绑卡 | 服务进程不直接持有模型，天然进程隔离，崩溃不污染主服务 |
| 输入 | 本地视频文件路径 | 服务需先接收上传/拉取到本地共享存储 |
| 输出 | 目录产物：`cam_space/`(JSON)、`SLAM/`(NPZ)、`extracted_images/`、可选 `world_space_res*.pth`、可选可视化 MP4 | 需对象存储托管 + 下载链接 |

> ⚠️ 注意：当前 `process_video` 是**阻塞**调用（`gpu_pool.acquire()` 会阻塞直到拿到卡），且每任务会 fork 子进程跑完整管线。服务层不能在请求线程里直接调用它。

---

## 3. 总体架构

```
                    ┌──────────────────────────────────────────────┐
                    │                  客户端 / 前端                  │
                    └───────────────┬──────────────────────────────┘
                                    │ HTTPS (REST)
                    ┌───────────────▼──────────────────────────────┐
                    │            API Gateway / FastAPI               │
                    │  - POST /jobs        (提交，立即返回 job_id)    │
                    │  - GET  /jobs/{id}   (查询状态/进度)            │
                    │  - GET  /jobs/{id}/result (拿产物下载链接)      │
                    │  - POST /jobs/{id}/cancel                       │
                    │  - GET  /healthz /metrics                       │
                    └──────┬───────────────────────┬─────────────────┘
                           │ 入队                    │ 读写
                  ┌────────▼─────────┐      ┌────────▼─────────┐
                  │   任务队列         │      │   元数据存储      │
                  │  Redis / Celery   │      │  PostgreSQL      │
                  │  (优先级 + 重试)   │      │  (job 状态/审计)  │
                  └────────┬─────────┘      └──────────────────┘
                           │ 拉取
        ┌──────────────────┼───────────────────────────┐
        │                  │                            │
┌───────▼────────┐ ┌───────▼────────┐          ┌────────▼────────┐
│  GPU Worker 0  │ │  GPU Worker 1  │   ...    │  GPU Worker N   │
│  (1 进程/GPU)   │ │                │          │                 │
│  HaWoRVideo-   │ │  HaWoRVideo-   │          │  HaWoRVideo-    │
│  Processor     │ │  Processor     │          │  Processor      │
└───────┬────────┘ └───────┬────────┘          └────────┬────────┘
        │                  │                            │
        └──────────────────┼────────────────────────────┘
                           │ 读输入 / 写产物
                  ┌────────▼─────────┐
                  │   共享存储         │
                  │  对象存储(S3/MinIO)│
                  │  + 本地 NVMe 缓存  │
                  └──────────────────┘
```

### 组件职责

1. **API 服务（FastAPI）**：无状态，水平可扩展。只负责校验、上传落地、入队、查状态、签发下载链接。**绝不在请求线程跑推理。**
2. **任务队列（Celery + Redis，或 RQ）**：持久化任务、支持优先级、超时、自动重试、取消。
3. **元数据库（PostgreSQL）**：job 全生命周期状态、进度、输入输出指针、错误信息、审计日志。
4. **GPU Worker**：每张 GPU 一个 Celery worker 进程，`--concurrency=1`，进程内持有一个 `HaWoRVideoProcessor(gpu_ids=[本卡])`。从队列取任务 → 下载输入 → `process_video()` → 上传产物 → 回写状态。
5. **共享存储**：对象存储存原始视频与产物；GPU 节点本地 NVMe 做 scratch（`extracted_images/` 等中间帧很占空间）。

---

## 4. 并发与 GPU 调度

两种可选粒度，推荐 **方案 B**：

**方案 A — 单进程内 GPU 池**：一个 worker 进程 `HaWoRVideoProcessor(gpu_ids=[0,1,2,3])` + 线程池提交。
- 优点：直接复用现有 `GpuPool`，改动最小。
- 缺点：单进程持有多卡，OOM 或子进程异常时影响面大；Python GIL 下多线程编排子进程尚可，但难做精细资源隔离。

**方案 B — 每 GPU 一个 worker 进程（推荐）**：
- 启动 N 个 Celery worker，每个 `CUDA_VISIBLE_DEVICES=i`、`--concurrency=1`、`HaWoRVideoProcessor(gpu_ids=[0])`（卡在进程内即 0 号）。
- 队列保证「一卡一任务」，与管线峰值 14GB 显存匹配。
- 优点：故障隔离、弹性伸缩（加卡=加 worker）、调度交给成熟的 Celery。
- 路由：默认共用一个 `hawor` 队列；如需区分长短视频，可建 `hawor.short` / `hawor.long` 两条队列各绑一组卡，避免长任务饿死短任务。

> 显存余量充足（如 A100 80GB）时，可在单卡放 2 个 worker 提升吞吐，但需先用真实负载压测确认峰值不超显存。

---

## 5. 任务生命周期与状态机

```
PENDING ──(worker 取走)──► RUNNING ──(成功)──► SUCCEEDED
   │                          │
   │                          ├─(可恢复错误, 重试<上限)─► PENDING
   │                          └─(失败/超限)──────────────► FAILED
   └─(用户取消)──► CANCELED        RUNNING ─(用户取消)─► CANCELED
```

进度上报：管线分 4 阶段，worker 在每阶段边界回写 `progress`（0–100）与 `stage` 字段（`detect_track` / `motion_est` / `slam` / `infill` / `postproc`）。长视频按「已完成段数 / 总段数」细化。

> 实现提示：现有管线通过 `subprocess` 调子脚本，进度细化需要在 `segmented_demo_pipeline.py` 增加阶段/分段的结构化 stdout（如 `PROGRESS stage=slam seg=2/5`），worker 解析后回写。首期可先只在阶段边界粗粒度上报。

---

## 6. API 设计（REST）

### 6.1 提交任务
```
POST /v1/jobs
Content-Type: multipart/form-data 或 application/json

# 方式一：直接上传
file=<video>, img_focal=600, vis_mode=off, priority=normal

# 方式二：传对象存储 URI（推荐大文件）
{ "input_uri": "s3://bucket/in/v.mp4", "img_focal": 600,
  "vis_mode": "off", "segment_seconds": 100, "run_post_steps": true }

→ 202 Accepted
{ "job_id": "j_abc123", "status": "PENDING", "created_at": "..." }
```

### 6.2 查询状态
```
GET /v1/jobs/{job_id}
→ 200
{ "job_id":"j_abc123", "status":"RUNNING", "stage":"slam",
  "progress":62, "gpu_id":2, "started_at":"...", "logs_url":"..." }
```

### 6.3 获取结果
```
GET /v1/jobs/{job_id}/result
→ 200 (status==SUCCEEDED)
{ "artifacts": {
    "cam_space":   "s3://.../cam_space/",      // 逐帧手部参数 JSON
    "slam":        "s3://.../SLAM/...npz",      // 相机轨迹 + scale
    "world_space": "s3://.../world_space_res_50fps.pth",
    "visualization":"s3://.../vis.mp4"          // vis_mode!=off 时
  },
  "expires_at": "..." }   // 预签名链接有效期
```

### 6.4 其它
- `POST /v1/jobs/{job_id}/cancel` → 撤销排队任务 / 终止运行中子进程。
- `GET /v1/jobs/{job_id}/logs` → 流式返回 `process.log`。
- `GET /healthz`、`GET /metrics`（Prometheus）。

### 参数映射（API → `HaWoRProcessorConfig`）
| API 字段 | Config 字段 | 默认 |
|---|---|---|
| `vis_mode` | `vis_mode` | `off`（服务默认关渲染省时） |
| `segment_seconds` | `segment_seconds` | 100 |
| `run_post_steps` | `run_post_steps`（50fps 上采样+插值） | true |
| `img_focal` | 透传给子脚本 | 600 / 自动估计 |

---

## 7. 数据流与存储

1. **输入落地**：上传或从 `input_uri` 拉取 → GPU 节点本地 scratch（如 `/scratch/hawor/{job_id}/input.mp4`）。
2. **处理**：`processor.process_video(video, out_dir=/scratch/hawor/{job_id}/out)`；`cleanup_intermediate=True` 自动删 `_segmented_work`。
3. **产物上传**：`out_dir` 打包/逐项上传到对象存储 `s3://bucket/jobs/{job_id}/`。
4. **清理**：上传成功后删本地 scratch；设置对象存储生命周期（如 7 天过期）。

存储估算（参考，按视频长度线性增长）：
- `extracted_images/`（30/50fps JPG）最占空间——服务默认**不返回**给用户，可选保留。
- 用户通常只需 `cam_space/`(JSON) + `SLAM/`(NPZ) + `world_space_res*.pth`，体积小。

---

## 8. 可靠性与容错

- **幂等**：`job_id` 作幂等键；worker 取任务前检查产物是否已存在（`process_video(overwrite_output=False)` 会对已存在产物报错，服务层据此跳过/续跑）。
- **重试**：仅对可重试错误（OOM、CUDA 临时故障、节点重启）自动重试，上限 2 次、指数退避；输入损坏/格式错误等用户错误**不重试**，直接 FAILED 带原因。
- **超时**：按视频时长设动态超时（如 `基准 + 系数 × 视频秒数`），超时杀子进程并标记。
- **崩溃恢复**：worker 崩溃后 Celery 重投任务；分段产物落在对象存储，重跑时已完成段可跳过（需 `overwrite_chunks=False`）。
- **OOM 防护**：一卡一任务为基线；监控显存，必要时降并发。

---

## 9. 可观测性

- **指标（Prometheus）**：队列深度、各状态任务数、单任务各阶段耗时、GPU 利用率/显存（DCGM exporter）、失败率、重试率。
- **日志**：结构化 JSON 日志（job_id 贯穿）；管线原始日志保留在 `process.log` 并上传。
- **链路追踪**：每 job 一个 trace，覆盖上传→排队→4 阶段→上传产物。
- **告警**：队列积压、失败率突增、GPU 掉卡、磁盘水位。

---

## 10. 部署形态

### 容器化
- **基础镜像**：CUDA 11.7 runtime + Python 3.10。
- **构建坑点**：`thirdparty/DROID-SLAM` 需 `python setup.py install` 编译 CUDA 扩展，须在镜像构建期完成；`thirdparty/Metric3D` 需初始化。`ffmpeg` 必须在 PATH（`_validate_config` 会校验）。
- **权重**：~5.6 GB（hawor.ckpt / infiller.pt / detector.pt / droid.pth / metric3d）。不打进镜像，用 init-container 或启动时从对象存储/PVC 挂载，避免镜像臃肿。
- **MANO 模型**：`_DATA/` 下的 MANO_RIGHT/LEFT.pkl 需随权重一并提供。

### Kubernetes（推荐）
- **API Deployment**：无状态，HPA 按 QPS/CPU 扩缩。
- **Worker StatefulSet/Deployment**：`resources.limits: nvidia.com/gpu: 1`，每 Pod 一卡；用 node selector/taint 调度到 GPU 节点；挂权重 PVC（ReadOnlyMany）+ 本地 NVMe scratch（emptyDir 或 local PV）。
- **依赖中间件**：Redis、PostgreSQL、对象存储（云 S3 或自建 MinIO）。
- **伸缩**：worker 副本数 = 可用 GPU 数；按队列深度用 KEDA 自动扩缩。

### 单机最小部署（PoC）
docker-compose：API + Redis + Postgres + N 个 worker（各绑一卡）+ MinIO。适合先验证端到端。

---

## 11. 安全

- 鉴权：API Key / OAuth2；按租户隔离 job 与产物（路径含 tenant 前缀）。
- 上传校验：限制大小/格式/时长；用 ffprobe 验证为合法视频再入队。
- 下载：对象存储预签名 URL，短时效。
- 速率限制与配额：每租户并发/日配额，防止打满 GPU。
- 隔离：worker 子进程沙箱化；scratch 按 job 隔离并及时清理。

---

## 12. 实施路线图

| 阶段 | 内容 | 产出 |
|---|---|---|
| **M0 PoC** | FastAPI 包 `HaWoRVideoProcessor`，单卡同步→简单后台线程；本地文件存储 | 能端到端提交/查/下载 |
| **M1 异步化** | 接入 Celery+Redis+Postgres，每 GPU 一 worker，状态机落库 | 多并发、异步、可查进度 |
| **M2 存储与产物** | 对象存储 + 预签名下载 + scratch 清理 + 生命周期 | 产物托管、磁盘可控 |
| **M3 健壮性** | 重试/超时/取消/幂等/断点续段 + 结构化进度上报 | 生产级容错 |
| **M4 可观测+部署** | Prometheus/日志/告警 + K8s 编排 + 压测调并发 | 上线就绪 |

### 首期落地（M0/M1）需要改的代码
1. 新增 `service/` 目录：`api.py`（FastAPI）、`tasks.py`（Celery task 包 `process_video`）、`models.py`（job 表）。
2. `segmented_demo_pipeline.py` 增加结构化进度输出（stdout 标记），供 worker 解析。
3. worker task 内：下载输入 → `HaWoRVideoProcessor(gpu_ids=[0]).process_video(...)` → 上传 `cam_space/SLAM/world_space_res*.pth` → 回写状态。

---

## 13. 开放问题
- 是否需要把可视化 MP4 渲染（`vis_mode=world`，依赖 aitviewer/EGL 离屏渲染）纳入服务？离屏渲染在容器内需配置 EGL，建议设为可选项、默认关闭。
- 长视频上限策略：超过 N 分钟是否强制分段并行到多卡？
- 是否对外暴露逐帧 JSON 的标准化 schema（当前 `cam_space` 为旋转矩阵列表，需文档化字段含义）。

---

## 附：现有可复用资产
- `scripts/hawor_video_processor.py::HaWoRVideoProcessor` — 线程安全、GPU 池、`process_video()` 单入口，**服务化核心**。
- `scripts/segmented_demo_pipeline.py` — 长视频分段、合并时间线。
- `scripts/extract_image_50fps.py` / `scripts/interpolation.py` — 后处理上采样。
- `app.py` — 字段/产物结构参考（Gradio 演示）。
