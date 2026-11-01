# QuietBridge · 静语桥

**口型与手势，让想说的话被听见。** A local-first lip + gesture communication prototype.

一个摄像头，支持口型、少量手势或两者组合；显示文字，点 **Speak** 用本地语音读出。六小时目标是一个可靠的短演示，不是通用手语翻译器。

## 快速开始

```sh
git clone https://github.com/pinguarmy/quietbridge.git
cd quietbridge
python -m hackathon.server --rehearsal  # Python 3.11/3.12，无需模型的模拟输入演练
```

打开 <http://127.0.0.1:8765>。真实摄像头、模型准备与 Peter 的手势接入见[本地演示指南](hackathon/README.md)。
模拟输入不代表真实识别已验证。Windows 基础应用安装见[原 Lipflow 指南](docs/lipflow-legacy.md)。已安装的开发机无需重跑 setup。

**当前状态：**组合运行时、口型适配、简洁沟通页、独立设置页、确认流程和本地语音已实现；21 项轻量集成测试通过，浏览器模拟输入演练已验证。真实口型准确率与延迟仍待演示 Mac 实测；手势识别器等待 Peter 接入。当前词表是未验证占位项，不进入真实融合。共享类型见 [`hackathon/contracts.py`](hackathon/contracts.py)。

## 两人并行开工

| 谁 | 分支 | 工作范围 |
| --- | --- | --- |
| A · Sam | `feat/lips-ui-fusion` | 口型适配、摄像头、融合、UI、本地语音，以及公共配置 |
| B · Peter | `feat/sign-recognizer` | 仅 `hackathon/signs/`：手势识别、词表、录样和相关测试 |

从 `main` 创建自己的分支，各自提 PR，由 A 合并。Codex Cloud 也用各自分支，避免同时改公共文件。详细但简短的约定见 [AGENTS.md](AGENTS.md)。

```sh
git switch main
git pull --ff-only
git switch -c feat/lips-ui-fusion  # B 改用 feat/sign-recognizer
uv run pytest
```

## 仓库结构

```text
hackathon/             新演示：共享接口、口型适配、融合、UI（逐步实现）
  contracts.py         两人共用的数据类型
  signs/               B 的手势开发区
lipflow/               可复用的口型引擎和原生应用
espnet/                已有模型依赖代码，不做顺手重构
tests/                 现有基础应用测试
scripts/               现有开发工具
docs/                  演示计划和旧版资料
```

## 演示只抓这几件事

- 一个摄像头同时供口型和手势使用，主流程不依赖网络。
- 先做 4–6 个明确的手势，时间不够就保留 3 个可靠的。
- 一致时显示结果；冲突时让用户选择，不编造一句话替用户说。
- 点 Speak 才发声；显示真实延迟，连续跑通三次。
- 可选云端建议最后再加，不阻塞本地流程。

简要时间表见[六小时计划](docs/hackathon.md)。云端可跑轻量测试，真实识别和语音在演示 Mac 上验证。

## 资料与来源

- [原 Lipflow 安装、架构与基准资料](docs/lipflow-legacy.md)保留作参考，其中历史数字不代表本项目的新实测结果。
- [旧版个性化说明](docs/personalization.md)属于基础应用，不是新演示的必做项。
- 项目基于 [amywork777/lipflow](https://github.com/amywork777/lipflow)，原仓库保留为 `upstream`。
- 代码许可见 [LICENSE](LICENSE)；第三方归属与模型限制见 [NOTICE](NOTICE)。LRS3 模型权重按非商业研究用途处理。

仓库为私有。邀请队友：[Settings → Collaborators](https://github.com/pinguarmy/quietbridge/settings/access)。不提交密钥、模型和个人录制。本地 Python 包、命令和数据目录仍叫 Lipflow，避免打断现有安装。
