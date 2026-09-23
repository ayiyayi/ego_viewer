# HaWoR 数据标注服务镜像
#
# 设计要点:
# - 基础镜像: CUDA 11.7 devel (DROID-SLAM 需在构建期编译 CUDA 扩展, 故用 devel 而非 runtime)
# - 权重 (~5.6GB) 和 MANO 模型不打进镜像, 运行时挂载 (见下方 NOTE)
# - 默认以 FastAPI 服务方式启动
#
# 构建:
#   docker build -t hawor-service:1.0 -f Dockerfile .
#
# 运行 (挂载权重 / 数据 / petrel 配置):
#   docker run --gpus all -p 8000:8000 \
#     -e HAWOR_GPU_IDS=0,1,2,3 \
#     -v /path/to/weights:/app/weights \
#     -v /path/to/_DATA:/app/_DATA \
#     -v /mnt/oss:/mnt/oss \                         # s3mount 挂载点(可选)
#     -v $HOME/petreloss.conf:/root/petreloss.conf \ # petrel-oss 配置(可选)
#     hawor-service:1.0

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    PIP_NO_CACHE_DIR=1

# ---- 系统依赖 (ffmpeg 必需; 编译 DROID-SLAM 需要 build-essential/git) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        ffmpeg git ninja-build build-essential \
        libgl1 libglib2.0-0 libegl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- PyTorch (cu117) ----
RUN pip install --upgrade pip && \
    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# ---- HaWoR 依赖 ----
# 先单独拷依赖清单, 利用 Docker 层缓存
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt && \
    pip install pytorch-lightning==2.2.4 --no-deps && \
    pip install lightning-utilities torchmetrics==1.4.0

# ---- 服务额外依赖 (fastapi/uvicorn/pydantic) ----
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install -r /app/service/requirements.txt

# ---- 拷贝项目源码 ----
COPY . /app

# ---- 编译 masked DROID-SLAM 的 CUDA 扩展 (构建期完成) ----
RUN cd thirdparty/DROID-SLAM && python setup.py install

# NOTE: 以下不打进镜像, 运行时通过 -v 挂载:
#   /app/weights          权重 (detector.pt / hawor.ckpt / infiller.pt / droid.pth / metric3d)
#   /app/_DATA            MANO 模型 (MANO_RIGHT.pkl / MANO_LEFT.pkl, 受 MANO license 约束)
#   /root/petreloss.conf  petrel-oss 配置 (走 s3:// 路径时需要)

EXPOSE 8000

# 默认: 启动 HTTP 服务。GPU 列表通过 HAWOR_GPU_IDS 环境变量传入。
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
