# Peter 的手势识别 · Track B

Sam 负责口型、运行时、融合、UI 和语音；Peter 只改本目录。默认使用 MediaPipe 预训练 GestureRecognizer，不另训练模型，也不增加依赖。

这是**四种预定义沟通手势**，不是通用 BSL/ASL 翻译器。模型类别存在不等于现场可靠；所有词表当前保持 `verified: false`，真实试验并同意表达含义后才启用。

## 给 Sam：接入契约

```python
from hackathon.signs.recognizer import SignRecognizer

recognizer = SignRecognizer()  # 一次初始化，不自动下载
try:
    events = recognizer.process(frame_bgr, timestamp_ms)
finally:
    recognizer.close()
```

- `frame_bgr`：外部单摄像头提供的**未镜像** OpenCV BGR 帧，识别器不修改输入、不再开摄像头。
- `timestamp_ms`：同一台机器的 `int(time.perf_counter() * 1000)`，严格递增；返回真实 `hackathon.contracts.SignEvent` 的列表。
- Sam 现有 `_hands` worker 可直接调用，约 15 fps、只保留最新帧。不在 UI/capture 线程推理。
- 原始 top-1 必须是支持类别且达到门槛；不通过 allowlist 把未知手势强塞给支持类别。双手冲突或任一手不明确时不输出。
- 默认至少四次连续观察且持续 250 ms 才发一个最终事件。长于 250 ms 的采样空档中断稳定证据。
- 持续保持同一姿势不重复发。下一次前需放下双手／回到不被识别的中立姿势，连续观察至少 400 ms；直接换手势不绕过去重。
- `score` 是稳定段内最低分类分数，**不是准确率或校准概率**。识别与稳定器默认门槛均为 0.75，与 Sam 的融合一致；低分观察不会先触发锁定、再被融合丢掉。核心流程无网络、无录制、无自动语音。

## 固定标签与建议短语

| 输出 label | MediaPipe 类别 | 动作 | 建议演示短语（不是手语翻译） |
| --- | --- | --- | --- |
| `THUMB_UP` | `Thumb_Up` | 竖拇指 | Yes. |
| `THUMB_DOWN` | `Thumb_Down` | 拇指向下 | No. |
| `OPEN_PALM` | `Open_Palm` | 掌心朝向摄像头 | Please stop. |
| `POINTING_UP` | `Pointing_Up` | 食指向上 | One moment, please. |

不输出 `HELP`、`WATER` 等未训练类别；不能把普通点赞冒充 HELP。动作到短语是演示约定，应先征得使用者同意。详细说明见 `labels.json`；`vocabulary.example.json` 符合 Sam 的 `phrase/matches/verified` 格式，所有项默认不启用。

**集成前必须替换占位映射：**Sam 当前公共词表键为 `HELP/STOP/YES/NO/WATER`，与真实输出没有交集；即使直接把旧项 `verified` 改为 true，也接不到本识别器。经真人测试通过后，由 Sam 更新公共词表，或把本目录示例复制成忽略的本地 JSON，只对通过且同意的项启用，再运行：

```sh
.venv/bin/python -m hackathon.server --sign-only --vocabulary /absolute/path/to/tested-vocabulary.json
```

`vocabulary.example.json` 的项刻意全为 false，直接传入只能检查加载，不能得到实时短语。不要为了让 UI 出结果直接把所有项标为 true。

## 安装模型（只做一次）

从仓库根目录、使用已安装环境：

```sh
.venv/bin/python -m hackathon.signs.assets --download
```

下载 Google 官方、固定版本的 8,373,440 字节模型至 `~/.cache/quietbridge/gesture_recognizer.task`，不写入 Git。SHA-256：`97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`。下载时校验；已有不同文件不会被覆盖。可用 `QUIETBRIDGE_GESTURE_MODEL=/path/to/gesture.task` 或构造参数 `model_path` 指定路径。部署时可复制该缓存文件，准备好后断网可运行。

## 本地检查

```sh
# 轻量逻辑与模拟推理边界测试，不打开摄像头
.venv/bin/python -m pytest hackathon/signs/tests -q

# 独立真人检查：有预览，不说话、不保存；Q退出
.venv/bin/python -m hackathon.signs.demo --camera 0 --seconds 60

# 经同意的本地视频；使用文件帧时间，不把快速回放算实时延迟
.venv/bin/python -m hackathon.signs.demo --video /path/to/consented.mp4 --headless

# 静态图重复帧：只用于验证真实模型加载和接口，不是真人稳定性验证
.venv/bin/python -m hackathon.signs.demo --image /path/to/hand.jpg --headless
```

独立检查器和 Sam 服务器**不要同时运行**，避免抢摄像头。正常合并演示应使用 Sam 的服务器，识别器自身不会打开摄像头。

## 本轮验证记录（2026-11-01）

- 最终合跑既有应用、Sam 整合与本手势模块：167 passed、11 skipped，另有 12 个 unittest subtest 通过。手势区 35 项测试覆盖事件列表接线、门槛对齐、去重、neutral 重新武装、冲突过滤、时间戳、词表示例和模型缺失提示。模型 backend 采用模拟输入，不宣称真人识别成功。
- 11 项跳过来自既有 Windows 专用、主动粘贴及缺少真实视频样片的检查，不是把失败隐藏成成功。最终命令为 `.venv/bin/python -m pytest tests hackathon/tests hackathon/signs/tests -q`。
- Google 官方模型已实际加载。官方静态 `thumbs_down.jpg`、`pointing_up.jpg` 各重复送入十帧，在 `end_ms=300` 各产生一个正确最终事件，保守分数分别约 0.773、0.813，平均处理约 25、23 ms。时间戳为人工构造，所以 300 ms 不是实测真人响应时间。
- 两张官方点赞样图未持续达到 0.75，因此无最终事件；没有为样图降低门槛。尚未验证 `OPEN_PALM`。这些少量样图不能用于计算准确率或决定现场启用词表。
- 在当前进程尝试摄像头：OpenCV 返回 `not authorized to capture video (status 0)`，所以未拿到真人帧。请在 macOS「系统设置 → 隐私与安全性 → 摄像头」允许实际启动 Python 的终端／宿主；不要为过测试绕过权限。该次检查未保存图像或录制。

静态测试图来源：[thumbs_up](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg)、[thumbs_down](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_down.jpg)、[pointing_up](https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg)。这些文件仅在本机临时目录，不随代码提交。

## 现场验收（尚未完成）

1. 每种手势做五次，每次保持约一秒，放下手至少半秒再做；记录正确、漏识别、错误标签。
2. 保持同一手势三秒，只应有一个事件；保持双手冲突不应出事件。
3. 中立姿势 30 秒，记录误触发；换第二个人和光线重复。达不到就保留三个稳定手势，不扩大词表。
4. 人工确认每个动作与短语，再启用对应词表项；完成 Sam 的手势单路、组合一致、冲突确认、断网和连续三次演示。

分类 `process_ms` 只测模型处理，不等于用户从做动作到看到文字的完整延迟。静态样图和合成预测不能证明真人准确率。历史 Lipflow 口型测试也不能证明本模块可用。

## 来源与限制

- [MediaPipe Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python)：预训练类别、VIDEO 模式与时间戳要求。
- [固定版本模型](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)。分发或商用前另行核对第三方模型条款；不要把旧 Lipflow 的 MIT 代码许可当作所有权重许可。
- 原有 Auto-AVSR/LRS3 模型的研究用途限制保持不变。私有录制与个人数据不提交、不上传。
