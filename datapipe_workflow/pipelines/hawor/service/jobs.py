"""任务模型与内存任务表。

首期用线程安全的内存字典存任务状态(进程内)。生产环境可替换为
PostgreSQL / Redis 而不改 api.py 的调用方式(见 docs/SERVICE_DESIGN.md)。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass
class VideoItem:
    video: str                      # 输入视频完整路径
    status: str = "pending"         # pending / ok / failed
    result_dir: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Job:
    input_dir: str
    output_dir: str
    img_focal: float = 600.0
    vis_mode: str = "off"
    overwrite: bool = False

    job_id: str = field(default_factory=lambda: "j_" + uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.PENDING
    stage: Optional[str] = None
    videos: List[VideoItem] = field(default_factory=list)
    error: Optional[str] = None
    cancel_requested: bool = False

    # ---- 派生指标 ----
    @property
    def videos_total(self) -> int:
        return len(self.videos)

    @property
    def videos_done(self) -> int:
        return sum(1 for v in self.videos if v.status == "ok")

    @property
    def videos_failed(self) -> int:
        return sum(1 for v in self.videos if v.status == "failed")

    @property
    def progress(self) -> int:
        if not self.videos:
            return 0
        return int(100 * (self.videos_done + self.videos_failed) / self.videos_total)

    def public_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "stage": self.stage,
            "progress": self.progress,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "videos_total": self.videos_total,
            "videos_done": self.videos_done,
            "videos_failed": self.videos_failed,
            "error": self.error,
        }

    def result_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "items": [
                {"video": v.video, "status": v.status,
                 "result_dir": v.result_dir, "error": v.error}
                for v in self.videos
            ],
        }


class JobStore:
    """线程安全的内存任务表。"""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())
