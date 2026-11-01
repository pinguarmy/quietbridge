# 手势开发区 · Track B

在 `feat/sign-recognizer` 分支工作，代码和测试都放这个目录；公共依赖改动交给 A。

先实现 `recognizer.py`：

```python
from hackathon.contracts import SignEvent


class SignRecognizer:
    def __init__(self): ...
    def process(self, frame_bgr, timestamp_ms: int) -> list[SignEvent]: ...
    def close(self): ...
```

模型只加载一次；没识别到稳定手势返回 `[]`；保持同一姿势不重复发事件，放下手后可以重发。摄像头由 A 的运行时提供，不要在识别器里另开摄像头。

建议文件：`features.py`、`train.py`、`labels.json`、`record_samples.py`、`tests/test_recognizer.py`。按需要创建，不必先搭空架子。

从 4–6 个可区分的手势开始，不够稳就缩到 3 个。实现后在这里补上真实支持的标签、运行命令和测试结果。目前尚未实现识别器，也没有已验证词表。
