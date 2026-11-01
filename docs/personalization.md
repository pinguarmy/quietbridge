# Lipflow 极速个性化与唇动输入使用指南 (Personalization Guide)

这个版本保留输入法的按键与光标输入方式，把声音换成摄像头口型输入。先定制词库、录制口型并训练，再用真实样本调优。

这次只需填写一份定制表，然后按 [连续定制流程](session-next.md) 完成后面的步骤。

### 隐私与数据披露 (Privacy & Cloud Disclosures)
- **视频与面部图像**：摄像头视频帧、人脸图像、练习切片与个性化模型参数**100% 完全保存在本地 Mac 上，绝不上载任何视频或面部图像**。
- **云端后处理文本数据**：当配置开启 Gemini 或 Claude 云端整理（`gemini` / `claude` 后端）时，本地唇读模型识别出的候选文本（candidates）、当前应用光标上下文（context）以及定制词表将发送至配置的云端大模型接口进行整理与润色；若使用 `local` 或 `basic` 后端，则全流程完全离线，绝不产生任何网络数据上传。

---

## 快速一次跑通流程（One-Session Flow）

只需五个步骤即可完成定制与体验：

```
[1. 词表与短语定制] ──> [2. 24句练习录制] ──> [3. 本地训练学习] ──> [4. Replay调优验证] ──> [5. 实际输入体验]
```

---

### 第一步：定制您的词库与常用表达 (Customization)

告诉 Lipflow 您经常使用的专有名词、同事姓名、行业术语或口癖。

1. **自动生成本地基础词表与短语**（非覆盖合并）：
   ```bash
   .venv/bin/python -m lipflow.customize --init
   ```
   这会在 `~/Library/Application Support/Lipflow/`（或对应平台用户目录）初始化 `words.txt` 和 `phrases.txt`。

2. **导入您自己的词库或短语**（纯文本安全追加）：
   ```bash
   # 导入常用表达短语（每行一整句，存入 phrases.txt 用于练习和表达召回）
   .venv/bin/python -m lipflow.customize --import-phrases my_phrases.txt

   # 导入专有名词/自定义词表（每行一个词或人名，存入 words.txt 用于唇形候选重排，避免整句污染）
   .venv/bin/python -m lipflow.customize --import-words my_words.txt
   ```
   *您也可以在可视化引导（Onboarding）的 "Teach it your words" 步骤中直接点击 "Import phrases" 或 "Import words" 分别选取文件合入。*

3. **设置首选后端与润色风格**：
   ```bash
   .venv/bin/python -m lipflow.customize --backend gemini --style polish
   ```
   - 默认模型：`gemini-3.8-flash`
   - CPA 接口：`https://cpa.peterhpy.dpdns.org/v1`
   - *说明：开启云端整理会将识别候选与上下文文本发送至上述接口。*

---

### 第二步：录制 24 句个性化唇读句子 (24 Practice Recordings)

在开始前，Lipflow 会从您的自定义短语库与平衡发音句集中生成 24 个练习句：

1. 启动设置引导界面（打开已安装应用，或在终端运行 `onboard` 命令）：
   ```bash
   .venv/bin/python -m lipflow onboard
   # 或直接运行：
   lipflow onboard
   ```
2. 进入 **Mouth each sentence**（练习界面）：
   - 确保面部处于光线明亮处，正对摄像头。
   - **按住右侧 Option 键（Right Option）**，无声自然做口型朗读屏幕上的句子。
   - 念完后**松开按键**，系统会自动保存切片并显示识别反馈。
   - 支持实时回退重录（"Redo last"）或跳过（"Skip sentence"）。

---

### 第三步：本地面部自适应训练 (Train on your face)

录制满 24 句后，系统会自动触发 GPU 本地微调：
- 完全在 Apple Silicon 本地 GPU (MPS) 上执行，无需上传任何数据。
- 实际训练时长取决于您的 Mac 芯片型号（如 M1/M2/M3/M4 系列与 GPU 核心数）及系统当前负载，不作未经测量的固定时长承诺。
- 自动保留 6 句独立验证集：只有微调后的识别准确率确实优于通用模型时，才会应用个性化模型权重。
- 训练完成后提示 Ready，即可开始日常使用。

---

### 第四步：离线回放与后处理调优 (Tuning Replay)

验证不同整理模型（`basic`、`local`、`gemini`）以及不同风格（`faithful` 忠实原文 vs `polish` 豆包式润色）的效果：

```bash
# 1. 忠实模式（计算规整化字错误率 WER、P50/P95 延迟与语义风险）
.venv/bin/python -m lipflow.tune --backend gemini --style faithful --out-md report_faithful.md

# 2. 润色模式（豆包输入体验：去除赘词与口癖、修复唇形歧义、格式化标点，免去误导性 WER，关注语义风险与延迟）
.venv/bin/python -m lipflow.tune --backend gemini --style polish --out-md report_polish.md
```

#### 指标解读：
- **P50 / P95 Latency (Cleanup-Path)**：诚实记录后处理整理路径耗时，不代表全链路按键释放延迟（全链路延迟还包含摄像头取帧、ROI 人脸截取和 VSR 神经网络解码）。
- **Normalized WER**：仅在 `faithful` 风格下且具有已知参考标注时评估；`polish` 风格会进行智能修饰和精简，WER 无法代表真实体验，因此自动隐去。
- **Semantic Risks**：监测是否存在关键否定词（已规范化映射 don't/not 等缩写）、数字混淆或专有名词漂移。
- **Backend & Fallback Tracking**：报告中针对每个样本展示实际生效的后端及降级状态（如 Gemini 遇到网络或超时自动降级至 local 或 basic）。

---

### 第五步：实际输入体验 (Short Real Input Check)

1. 启动常驻输入服务：
   ```bash
   .venv/bin/python -m lipflow
   ```
2. 打开任意文字输入区域（如 Slack、微信、备忘录、终端等）。
3. **按住 Right Option 键**，自然做口型默念任意简短语句（例如：“Sounds good to me”）。
4. **松开按键**：润色后的文字即刻自动打在光标处！
