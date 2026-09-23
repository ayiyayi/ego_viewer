"""等价命令行入口(不起 HTTP 服务, 直接同步跑一个标注 job)。

用法:
    python -m service.annotate \
        --input_dir  s3://bucket/datasets/egocentric/batch_01/ \
        --output_dir s3://bucket/annotations/batch_01/ \
        --gpu_ids 0,1,2,3
"""

from __future__ import annotations

import argparse
import sys

from .jobs import Job, JobStatus
from .worker import AnnotationRunner


def _parse_gpu_ids(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(",", " ").split() if x.strip().isdigit()]


def main() -> None:
    p = argparse.ArgumentParser(description="HaWoR 批量标注 (folder 进 / folder 出)")
    p.add_argument("--input_dir", required=True, help="待标注视频 folder 完整路径")
    p.add_argument("--output_dir", required=True, help="标注结果 folder 完整路径")
    p.add_argument("--gpu_ids", default="0", help='GPU 列表, 如 "0" 或 "0,1,2,3"')
    p.add_argument("--scratch", default="/tmp/hawor_scratch", help="本地中间产物目录")
    p.add_argument("--petrel_conf", default=None, help="petrel-oss 配置文件路径")
    p.add_argument("--vis_mode", default="off", choices=["off", "world"])
    p.add_argument("--img_focal", type=float, default=600.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    runner = AnnotationRunner(
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        scratch_root=args.scratch,
        petrel_conf=args.petrel_conf,
        vis_mode=args.vis_mode,
    )
    job = Job(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_focal=args.img_focal,
        vis_mode=args.vis_mode,
        overwrite=args.overwrite,
    )
    runner.run_job(job)

    print(f"status      : {job.status.value}")
    print(f"videos      : total={job.videos_total} ok={job.videos_done} failed={job.videos_failed}")
    print(f"output_dir  : {job.output_dir}")
    for v in job.videos:
        line = f"  - {v.video} -> {v.status}"
        if v.result_dir:
            line += f" ({v.result_dir})"
        if v.error:
            line += f" [{v.error}]"
        print(line)
    if job.error:
        print(f"error       : {job.error}")

    if job.status != JobStatus.SUCCEEDED:
        sys.exit(1)


if __name__ == "__main__":
    main()
