# Ego Viewer 交接

可以主要改可视化页面。三个样例的视频、动作标注、手和相机都已经导出，打开页面就能用，不需要再配 HaWoR、标注 API。

## 当前页面

只用 **v7**。页面文件在 `viewer/static/v7/`：

- `index.html` 结构
- `style.css` 样式
- `app.js` 播放、时间轴、手部叠加、World Frame

数据接口在 `viewer/server.py`。改 HTML/CSS/JS 后刷新浏览器即可。改 `server.py` 后需要重启下面的进程。

启动（系统自带的 `python3` 即可，依赖 numpy）：

```bash
cd /data-hyp/ego_viewer/viewer
python3 serve_v7.py
```

浏览器打开 http://127.0.0.1:8777/ 。端口可用环境变量 `V7_PORT` 改，不要占用 8765。

## 现成数据

页面自动读取 `datapipe_workflow/examples/` 下每个带 `input/*.mp4` 的目录。现在有三份，标注和 3D 手都已就绪：

| 目录 | 视频 | 内容 |
|---|---|---|
| `examples/pick_up_box` | `input/93009_1min.mp4` | 客厅，拿放物品 |
| `examples/return_portafilter` | `input/return_portafilter.mp4` | 厨房，咖啡机手柄 |
| `examples/load_drawstring_bag` | `input/file-000.mp4` | 卧室，抽绳袋，36 fps |

每个样例的页面会用到这些文件：

```text
input/*.mp4
output/ego_action_annotation.json
output/keypoints.npz
output/mesh.bin
```

`ego_action_annotation.json` 是一个数组。每段有 `id`、`start_ts`、`end_ts`、`start_frame`、`end_frame`、`scene`、`verb`、`object`、`action`。

World Frame 画的是 `keypoints.npz` 和旁边的 `mesh.bin`，不是原始的 `hands.npz`。`keypoints.npz` 里页面用到的字段是 `joints_world`（2 只手 × 帧 × 21 个关节点）、`cam_pos`、`cam_R`、`width`、`height`、`fps`。没有这两份文件时，World Frame 会停在 “awaiting model output”。

`viewer/samples.json` 里还有一份官方样例，路径在本项目外面。改页面时用上面三个 `examples` 即可。

## 标注是怎么来的

只在需要重新标注新视频时才看这一节。看页面、改样式不用跑这些。

| 内容 | 模型 | 代码 |
|---|---|---|
| 动作分段、动词、物体、句子，以及整段视频的场景 | Gemini API（`pipelines/caption/api_config.json` 里的模型） | `pipelines/caption/atomic_subtask_demo.py` |
| 手部姿态和相机轨迹 | HaWoR（手部重建 + SLAM 相机） | `pipelines/hawor/` |
| 页面上的 21 点骨架和手部网格 | 用 MANO 从 HaWoR 的 `hands.npz` 算出 `keypoints.npz` 和 `mesh.bin` | `viewer/export_keypoints.py` |

统一入口是 `datapipe_workflow/annotate_video.py`。在该目录下：

```bash
python annotate_video.py /path/to/video.mp4 --output examples/my_clip
```

它会调用 API 写动作标注，再跑 HaWoR，并导出网页用的 `keypoints.npz` 和 `mesh.bin`。HaWoR 使用 `env_profiles/default.yaml` 里的 Python：`/data/heyuping/ego_viewer/envs/hawor/bin/python`，GPU 0。抽帧按源视频帧率逐帧进行，不再降到固定 30 fps。

API 密钥已经在 `pipelines/caption/api_config.json`，不要把密钥写进页面或提交到仓库。
