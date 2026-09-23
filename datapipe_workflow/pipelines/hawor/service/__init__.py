"""HaWoR 数据标注服务包。

模块划分:
- storage.py  : 对象存储抽象（petrel-oss SDK / 本地 & s3mount）
- jobs.py     : 任务模型与内存任务表
- worker.py   : 任务执行器（遍历视频 + 调用 HaWoRVideoProcessor）
- api.py      : FastAPI HTTP 接口
- annotate.py : 等价命令行入口
"""
