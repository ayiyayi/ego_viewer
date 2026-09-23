"""任务执行器: 遍历输入 folder 内视频, 逐个调用 HaWoRVideoProcessor。

调度模型:
- 进程内维护一个有界线程池, 大小 = GPU 数。
- HaWoRVideoProcessor 自带 GpuPool (一卡一任务), 线程池并发提交时
  由 GpuPool 阻塞保证不会超用 GPU。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from scripts.hawor_video_processor import HaWoRVideoProcessor, HaWoRProcessorConfig

from .jobs import Job, JobStatus, JobStore, VideoItem
from .storage import StorageClient, _split_video_name


class AnnotationRunner:
    """承载 GPU 池与存储客户端, 串起单个标注 job 的全流程。"""

    def __init__(
        self,
        gpu_ids: Sequence[int] = (0,),
        scratch_root: str = "/tmp/hawor_scratch",
        petrel_conf: str | None = None,
        vis_mode: str = "off",
    ) -> None:
        self.gpu_ids = list(gpu_ids)
        self.scratch_root = scratch_root
        self.storage = StorageClient(petrel_conf=petrel_conf)
        # 一个 processor 内部持有覆盖所有 GPU 的池; 线程池并发由 GpuPool 限流
        self.processor = HaWoRVideoProcessor(
            gpu_ids=gpu_ids,
            config=HaWoRProcessorConfig(vis_mode=vis_mode),
        )
        os.makedirs(scratch_root, exist_ok=True)

    # ---- 单个 job ------------------------------------------------------------
    def run_job(self, job: Job) -> None:
        """阻塞执行一个 job(通常在后台线程中调用)。"""
        try:
            job.status = JobStatus.RUNNING
            videos = self.storage.list_videos(job.input_dir)
            job.videos = [VideoItem(video=v) for v in videos]
            if not videos:
                job.status = JobStatus.FAILED
                job.error = f"输入目录无视频: {job.input_dir}"
                return

            # 用 GPU 数作为并发度; GpuPool 保证不超用
            max_workers = max(1, len(self.gpu_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self._run_one, job, item) for item in job.videos
                ]
                for f in futures:
                    f.result()  # 异常已在 _run_one 内吞掉并记录到 item

            if job.cancel_requested:
                job.status = JobStatus.CANCELED
            elif job.videos_failed == job.videos_total:
                job.status = JobStatus.FAILED
                job.error = "所有视频处理失败"
            else:
                job.status = JobStatus.SUCCEEDED
        except Exception as e:  # 兜底, 防止后台线程静默崩溃
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {e}"

    # ---- 单个视频 ------------------------------------------------------------
    def _run_one(self, job: Job, item: VideoItem) -> None:
        if job.cancel_requested:
            item.status = "failed"
            item.error = "canceled"
            return

        stem = Path(_split_video_name(item.video)).stem
        with tempfile.TemporaryDirectory(dir=self.scratch_root, prefix=f"{stem}_") as work:
            local_out = os.path.join(work, "out")
            try:
                # 1) 取视频到本地 scratch
                local_video = self.storage.fetch_to_local(
                    item.video, os.path.join(work, _split_video_name(item.video))
                )
                # 2) 本地输出目录
                os.makedirs(local_out, exist_ok=True)

                # 3) 跑 HaWoR 管线(阻塞直到拿到 GPU 并处理完)
                self.processor.process_video(
                    local_video, local_out, overwrite_output=job.overwrite
                )

                # 4) 上传产物到 output_dir/<stem>/
                dst = self.storage.join(job.output_dir, stem)
                self.storage.upload_dir(local_out, dst)

                item.status = "ok"
                item.result_dir = dst
            except Exception as e:
                item.status = "failed"
                item.error = f"{type(e).__name__}: {e}"
                self._preserve_failure_artifacts(job, item, stem, work, local_out)

    def _preserve_failure_artifacts(
        self,
        job: Job,
        item: VideoItem,
        stem: str,
        work: str,
        local_out: str,
    ) -> None:
        """Copy failure logs out of the temporary work dir before it is deleted."""
        try:
            fail_root = os.path.join(work, "failure_artifacts")
            os.makedirs(fail_root, exist_ok=True)
            with open(os.path.join(fail_root, "error.txt"), "w", encoding="utf-8") as f:
                f.write(item.error or "unknown error")
                f.write("\n")
                f.write(f"video: {item.video}\n")

            if os.path.isdir(local_out):
                shutil.copytree(
                    local_out,
                    os.path.join(fail_root, "out"),
                    dirs_exist_ok=True,
                )

            dst = self.storage.join(job.output_dir, "_failed", stem)
            self.storage.upload_dir(fail_root, dst)
            item.result_dir = dst
        except Exception as preserve_error:
            item.error = (
                f"{item.error}; failed to preserve artifacts: "
                f"{type(preserve_error).__name__}: {preserve_error}"
            )
