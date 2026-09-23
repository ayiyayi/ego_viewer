"""HaWoR 数据标注 HTTP 接口 (FastAPI)。

契约: 输入待标注视频 folder 完整路径 -> 输出标注结果 folder 完整路径。

启动:
    HAWOR_GPU_IDS=0,1,2,3 uvicorn service.api:app --host 0.0.0.0 --port 8000

接口:
    POST /v1/annotate          提交任务, 立即返回 job_id
    GET  /v1/jobs/{job_id}     查询状态/进度
    GET  /v1/jobs/{job_id}/result  获取标注结果完整路径
    POST /v1/jobs/{job_id}/cancel  请求取消
    GET  /healthz
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .jobs import Job, JobStatus, JobStore
from .worker import AnnotationRunner


# ---- 配置(环境变量) -----------------------------------------------------------
def _parse_gpu_ids(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x.strip().isdigit()]


GPU_IDS = _parse_gpu_ids(os.environ.get("HAWOR_GPU_IDS", "0"))
SCRATCH_ROOT = os.environ.get("HAWOR_SCRATCH", "/tmp/hawor_scratch")
PETREL_CONF = os.environ.get("HAWOR_PETREL_CONF") or None


# ---- 请求/响应模型 -----------------------------------------------------------
class AnnotateRequest(BaseModel):
    input_dir: str = Field(..., description="待标注视频所在 folder 的完整路径(s3:// 或本地/s3mount)")
    output_dir: str = Field(..., description="标注结果输出 folder 的完整路径")
    img_focal: float = 600.0
    vis_mode: str = Field("off", description="off | world")
    overwrite: bool = False


class AnnotateResponse(BaseModel):
    job_id: str
    status: str
    input_dir: str
    output_dir: str


# ---- 应用与单例 --------------------------------------------------------------
app = FastAPI(title="HaWoR 数据标注服务", version="1.0")

store = JobStore()
runner = AnnotationRunner(gpu_ids=GPU_IDS, scratch_root=SCRATCH_ROOT, petrel_conf=PETREL_CONF)
# 后台线程池, 把长任务从请求线程剥离; 大小给足以容纳并发 job 数
_bg = ThreadPoolExecutor(max_workers=max(2, len(GPU_IDS) * 2))


@app.post("/v1/annotate", response_model=AnnotateResponse, status_code=202)
def annotate(req: AnnotateRequest) -> AnnotateResponse:
    job = Job(
        input_dir=req.input_dir,
        output_dir=req.output_dir,
        img_focal=req.img_focal,
        vis_mode=req.vis_mode,
        overwrite=req.overwrite,
    )
    store.create(job)
    _bg.submit(runner.run_job, job)  # 异步执行, 不阻塞 HTTP
    return AnnotateResponse(
        job_id=job.job_id,
        status=job.status.value,
        input_dir=job.input_dir,
        output_dir=job.output_dir,
    )


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.public_dict()


@app.get("/v1/jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.SUCCEEDED:
        raise HTTPException(
            status_code=409,
            detail=f"job 未完成 (status={job.status.value}, progress={job.progress})",
        )
    return job.result_dict()


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.cancel_requested = True
    return {"job_id": job_id, "cancel_requested": True}


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "gpu_ids": GPU_IDS}
