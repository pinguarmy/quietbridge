# QuietBridge web interface, 2026-11-03

A quiet communication desk. The message is the focal point, with one clear capture action and explicit speech confirmation.

- Product and surfaces: local-first personal communication; main message, settings and personalization pages. Not a general BSL/ASL translator.
- Metaphor and signature: a message sheet beside a viewfinder, with a small capture/review/speak step rail.
- Reference and stack: native accessibility utility clarity; existing HTML, local CSS and vanilla JavaScript. No new UI dependency or remote fonts.
- Palette and type: cool off-white, navy ink, restrained teal actions; system sans-serif controls and generous message text.
- Layout: message before camera in the DOM and visual order at every width. Mouth zoom and technical details use disclosures.
- Storyboard: start camera, build a draft, stop, review or edit, then speak. Errors preserve the message. Personalization explains independent recording sessions in plain language.
- Motion and copy: short state transitions, reduced-motion support, concrete instructions. No claims of measured accuracy.
- Avoid: gradients, blur, decorative illustrations, button walls, fake confidence, training jargon on the daily page.
- Accessibility: visible focus, labelled inputs, 44-pixel touch targets, sufficient contrast, non-color status, live feedback.
- Success: a new user can identify start, finish, edit and speak without diagnostics. Training, validation, final test and explicit activation stay separate.
- Out of scope: new recognition models, general sign-language translation, auto-speaking, remote services or camera permission bypass.

Pass A changes hierarchy. Pass B checks rendered desktop/mobile spacing, keyboard access, disabled/loading states and state transitions. Existing element IDs and API contracts stay intact.

## First-use and everyday communication review, 2026-11-04 late

- Product and surface: the local web app, Settings and Personalize. First successful expression comes before optional model training.
- Metaphor and signature: the existing quiet communication desk, with a short dismissible first-use guide and a direct typed-message route.
- Reference and stack: accessible native utility conventions; existing HTML/CSS/vanilla JavaScript. No remote fonts, new frontend dependency or marketing page.
- Palette, type and layout: preserve off-white/navy/teal and system type. Message remains the focal point. Narrow-screen updates must not move a held or focused capture control.
- Storyboard: understand the active input, type or choose camera, capture, review/edit, explicitly speak, begin another message. Personalize remains optional and returns to the current app.
- Copy and motion: concrete short instructions matched to actual backend and camera state. No decorative motion or accuracy promises.
- Accessibility and privacy: a non-hold capture option, keyboard/focus flow, labelled controls, no auto camera/speech/learning, explicit sensitive-data consent and useful error recovery.
- Avoid: compulsory wizard, diagnostic wall, jargon-led profile creation, fake ready states, dead cross-port links and ratings substituted for user evidence.
- Success: critical journeys have reproducible behavior and no serious unresolved defects. The v1 model scores are historical, not product-quality measurements. Apply review-protocol.md, with independent counterexamples and separate real-user/recognition evidence.

## Paired standard-sign check, 2026-11-04

- Surface and purpose: existing paired recorder, not a new daily-use page. A user can distinguish personal coordinates from a standard-sign video check before opening the camera.
- Signature: one clear capture-type choice followed by a plain consent row, using the existing teal/off-white styling and system type.
- Layout and sequence: choose paired type, sign/negative and intended sentence, consent, capture one short trial, save or discard, then record on a separate occasion for held-out evaluation.
- Copy and privacy: say that face, upper body and hands are saved locally as a 256-square video sequence. No microphone, no cloud or automatic training. Reconfirm each trial.
- Interaction: disabled unavailable model choice; locked controls during recording; overflow offers discard, never a misleading successful partial save. Progress and sessions stay separate from personal gestures.
- Accessibility: native labels and checkboxes, visible focus, 44-pixel actions and live status. No new motion.
- Success and scope: consent reaches the actual capture API, saved video can enter the standard-sign evaluator, and the existing coordinate-only default is unchanged. No model marketplace, license waiver or claim of recognition accuracy.

## Earlier native Lipflow brief


## 产品与界面

产品是用户的无声输入工具。Left Option 留给豆包，Right Option 给 Lipflow。默认 Gemini 经 CPA 自然整理。用户通过问答选项定制，不要求填写外部文件。

界面包括原生设置和个人训练引导，不改 VSR、模型下载、热键状态机和云端接口。

## 视觉方向

隐喻是 Mac 输入面板。唯一标志性细节是清楚的大号 Right ⌥ 键帽。参考原生 macOS 设置的分组与工具条，不复制某个品牌。栈保持 AppKit/PyObjC。

颜色采用雾白 #F5F7F4、石墨 #202923、分隔线 #DCE3DC、苔绿 #356B50。系统 SF 字体，标题 24、行标题 14、辅助文字 12，黑白对比优先。选择控件与按钮使用原生焦点/键盘状态。

设置窗约 720×570，以侧栏分开输入、个人定制、隐私和高级选项。普通输入页突出快捷键、整理方式和摄像头；模型后端、麦克风等不占据首屏。训练页展示真实片段数与训练状态，未训练不伪造准确率。引导保持录制、重录、跳过、训练操作，并说明每次如何按 Right Option。

## 连续操作

问答选择整理方式及领域 → 自动准备词表和常用句 → 摄像头对位 → 24 句口型录制 → 本地训练与保留判断 → 回放调优 → 实际输入。

不增加装饰动画。原生交互立即响应；训练沿用真实进度。禁用玻璃卡片、霓虹、渐变文字、营销主视觉、圆胶囊遍布和技术开关墙。中文文案只讲作用，不承诺未实测准确率。

## 验收

两轮：先布局，再间距、文案与控件微调。实际渲染日常页与训练页截图，检查无截断/重叠；控件持久化与工作线程任务不变；自动测试通过。隐私说明候选文字、近期口述、相关名字可发送云端，视频和音频留本地。

## 不做

不碰豆包配置，不占 Left Option，不重新安装，不更换应用签名，不重做摄像头或硬件兼容检查。
