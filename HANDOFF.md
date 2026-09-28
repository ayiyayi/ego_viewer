# Ego Viewer 交接

可以主要改可视化页面。手、相机和动作标注都已经导出，打开页面就能用。

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

页面自动读取 `datapipe_workflow/examples/` 下每个带 `input/*.mp4` 的目录。现在是这六条，都有 3D 手和动作标注：

| 目录 | 视频 | 内容 |
|---|---|---|
| `examples/P03_05_02` | `input/P03_05_02.mp4` | FineBio，约 30 fps，做过遮挡和短空档插值。动作标注参考了人工细粒度标签 |
| `examples/P10_01_01` | `input/P10_01_01.mp4` | 同上 |
| `examples/P20_03_01` | `input/P20_03_01.mp4` | 同上 |
| `examples/P22_02_02` | `input/P22_02_02.mp4` | 同上 |
| `examples/P28_01_01` | `input/P28_01_01.mp4` | 同上 |
| `examples/file-000` | `input/file-000.mp4` | 1920×1080，约 60 fps，只做了遮挡，没有插值。没有人工标签，动作标注只看画面 |

每个样例的页面会用到这些文件：

```text
input/*.mp4
output/ego_action_annotation.json
output/keypoints.npz
output/mesh.bin
```

`ego_action_annotation.json` 是一个数组。每段有 `id`、`start_ts`、`end_ts`、`start_frame`、`end_frame`、`scene`、`verb`、`object`、`action`。`id` 从 1 连续编号。`start_frame` / `end_frame` 是 `round(秒 × fps)`，fps 在同目录的 `provenance.json`。手改时间或删段之后，要重排 `id` 并按这个式子重算帧号。

World Frame 画的是 `keypoints.npz` 和旁边的 `mesh.bin`，不是原始的 `hands.npz`。`keypoints.npz` 里页面用到的字段是 `joints_world`（2 只手 × 帧 × 21 个关节点）、`cam_pos`、`cam_R`、`width`、`height`、`fps`。没有这两份文件时，World Frame 会停在 “awaiting model output”。

`viewer/samples.json` 里还有一份官方样例，路径在本项目外面。改页面时用上面的 `examples` 即可。

## 这次的手部方案

看页面不用跑模型。要重新处理一条视频时，默认路径是：

1. **SAM** 跟踪左右手框。
2. **HaWoR** 只用来估计相机轨迹。1920×1080 用焦距 1000.1，其他分辨率用 600。
3. **WiLoR** 在这条轨迹上估计双手姿态，写成 `keypoints.npz` 和 `mesh.bin`。
4. **后处理**：手腕快于 1.5 m/s 的帧遮掉；相机快于 2.4 m/s 就切断轨迹。只有短于 1/6 秒、两端手腕距离不超过 0.10 m、补帧速度不超过 1.5 m/s、手指姿态相差不超过 0.03 m 的空档才插值。被遮掉的检测不会再补上。阈值按速度和时间写，不按帧数。

入口是 `datapipe_workflow/annotate_video.py`。在该目录下：

```bash
python annotate_video.py /path/to/video.mp4 --hands-only --output examples/my_clip
```

SAM 路径在 `env_profiles/sam3.env`，脚本会自己读。HaWoR / WiLoR 用 `/data/heyuping/ego_viewer/envs/` 里对应的 Python，GPU 0。`--hands hawor` 才改回 HaWoR 自己的手，那条路径不做上面的后处理。

上面五条 `P*` 是按更早的 30 fps 帧规则导出的；`file-000` 只做了遮挡。新跑的视频才会走现在这套速度和时间规则。

## 动作标注

Gemini 把视频切成 20 秒一段，每 0.5 秒一帧做成接触图，再拼成 `ego_action_annotation.json`。接触图、`clips.json`、`segments.json` 跑完就删，样例目录里不留这些中间文件。密钥在 `pipelines/caption/api_config.json`，不要写进页面或提交到仓库。

五条 FineBio 用 `/data-hyp/ego_viewer/tmp/<视频名>.txt` 作参考，例如 `tmp/P03_05_02.txt`。文件前半是粗粒度任务，时钟突然降到 0 附近之后才是细粒度动作。模型只看细粒度行。同一时间左右手各有一条时只留主动作。时间先对齐到 0.5 秒，再检查顺序。参考可以漏原子动作，模型可以按画面补上。`file-000` 没有这份人工文件，所以不加 `--labels`。

只重跑动作、不动手和相机：

```bash
cd /data-hyp/ego_viewer/datapipe_workflow
python annotate_video.py examples/P03_05_02/input/P03_05_02.mp4 \
  --actions-only --labels /data-hyp/ego_viewer/tmp/P03_05_02.txt \
  --output examples/P03_05_02
```

代码在 `pipelines/caption/atomic_subtask_demo.py`，FineBio 参考的整理在 `pipelines/caption/finebio_reference.py`。提示词在 `pipelines/caption/prompts/atomic_subtask_clip.txt`。其中约定：单纯完成插枪头这类操作的按压，并进那个具体动作，不要单独写成 Presses。相邻两段切点上的 action 全文相同就合并成一段。
