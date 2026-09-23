"""对象存储抽象层。

统一处理两类路径:
- 对象存储路径: ``s3://bucket/prefix/...``（用 petrel-oss SDK 读写）
- 本地 / s3mount 路径: ``/mnt/oss/...`` 或任意本地目录（直接用 os/shutil）

服务内部只关心 "folder 进 / folder 出"，由本模块屏蔽底层差异:
- ``list_videos``  : 列举输入 folder 下的视频
- ``fetch_to_local``: 把单个视频取到本地 scratch
- ``upload_dir``   : 把本地产物目录上传回输出 folder
- ``exists`` / ``join``: 路径辅助

petrel-oss 用法参考内网文档:
http://sdoc.pjlab.org.cn/doc/#/petrel-oss/sdk/SDK安装
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI")


def is_s3(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("cluster:s3://")


def _split_video_name(key: str) -> str:
    return os.path.basename(key.rstrip("/"))


class StorageClient:
    """根据路径前缀自动选择 petrel-oss 或本地后端。

    设计为惰性初始化 petrel Client: 只有真正访问 s3 路径时才导入并连接,
    这样纯本地 / s3mount 部署无需安装 petrel-oss-sdk。
    """

    def __init__(self, petrel_conf: Optional[str] = None) -> None:
        # petrel 配置文件路径; 默认 ~/petreloss.conf
        self._petrel_conf = petrel_conf or os.path.expanduser("~/petreloss.conf")
        self._client = None  # 惰性创建

    # ---- petrel client (lazy) -------------------------------------------------
    def _petrel(self):
        if self._client is None:
            try:
                from petrel_client.client import Client  # type: ignore
            except ImportError as e:  # pragma: no cover - 取决于运行环境
                raise RuntimeError(
                    "需要访问 s3:// 路径但未安装 petrel-oss-sdk。"
                    "请安装 SDK, 或改用 s3mount 后用本地路径。"
                ) from e
            self._client = Client(conf_path=self._petrel_conf)
        return self._client

    # ---- 路径辅助 -------------------------------------------------------------
    @staticmethod
    def join(base: str, *parts: str) -> str:
        if is_s3(base):
            return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts)
        return os.path.join(base, *parts)

    def exists(self, path: str) -> bool:
        if is_s3(path):
            try:
                return self._petrel().contains(path)
            except Exception:
                # 部分版本无 contains, 退化为 list 判断
                return any(True for _ in self._iter_keys(path))
        return os.path.exists(path)

    # ---- 列举视频 -------------------------------------------------------------
    def _iter_keys(self, prefix: str):
        client = self._petrel()
        # petrel 不同版本可能是 list / get_file_iterator
        if hasattr(client, "get_file_iterator"):
            for item in client.get_file_iterator(prefix):
                # item 可能是 (key, size) 或 key
                yield item[0] if isinstance(item, (tuple, list)) else item
        else:  # pragma: no cover
            base = prefix.rstrip("/") + "/"
            for name in client.list(base):
                if not name.endswith("/"):
                    yield base + name

    def list_videos(self, input_dir: str, recursive: bool = True) -> List[str]:
        """返回 input_dir 下所有视频的完整路径(排序)。"""
        videos: List[str] = []
        if is_s3(input_dir):
            for key in self._iter_keys(input_dir):
                if key.endswith(VIDEO_EXTS):
                    videos.append(key if is_s3(key) else self.join(input_dir, key))
        else:
            root = Path(input_dir)
            it = root.rglob("*") if recursive else root.glob("*")
            for p in it:
                if p.is_file() and p.name.endswith(VIDEO_EXTS):
                    videos.append(str(p))
        videos.sort()
        return videos

    # ---- 取到本地 / 上传 ------------------------------------------------------
    def fetch_to_local(self, src: str, local_path: str) -> str:
        """把单个对象/文件取到本地路径, 返回本地路径。"""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if is_s3(src):
            data = self._petrel().get(src)
            with open(local_path, "wb") as f:
                f.write(data)
        else:
            if os.path.abspath(src) != os.path.abspath(local_path):
                shutil.copy2(src, local_path)
        return local_path

    def upload_dir(self, local_dir: str, dst_dir: str) -> str:
        """递归上传本地目录到目标 folder, 返回目标 folder 路径。"""
        if is_s3(dst_dir):
            client = self._petrel()
            for root, _, files in os.walk(local_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, local_dir)
                    key = self.join(dst_dir, rel.replace(os.sep, "/"))
                    with open(fp, "rb") as f:
                        client.put(key, f.read())
        else:
            os.makedirs(dst_dir, exist_ok=True)
            for root, _, files in os.walk(local_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, local_dir)
                    dst = os.path.join(dst_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(fp, dst)
        return dst_dir
