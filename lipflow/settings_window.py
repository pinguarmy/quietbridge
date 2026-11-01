"""Settings window: light native panel, 720x570, four pages behind a left sidebar.

Pages: 输入 (key, cleanup style, camera), 个人定制 (real clip/word counts and training),
隐私 (what may leave this Mac), 高级 (cleanup backend, whisper mode, data folder).

Text helpers here always set an explicit color: onboarding._text defaults to white for the
HUD, which would vanish on this light background.
"""
from __future__ import annotations

import os
import time

import objc
from AppKit import (
    NSAppearance, NSBackingStoreBuffered, NSButton, NSColor, NSFont, NSFontWeightBold,
    NSFontWeightMedium, NSFontWeightRegular, NSFontWeightSemibold, NSMakeRect, NSPopUpButton,
    NSSwitch, NSTextAlignmentCenter, NSTextField, NSView, NSWindow, NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject

from .hud import _rgb, symbol
from .paths import HOME, PERSONAL_VSR

SW, SH = 720, 570
SIDEBAR_W = 196
CX = 220               # content left edge
CW = SW - CX - 20      # content width (480)


def _hex(h):
    return ((h >> 16 & 255) / 255, (h >> 8 & 255) / 255, (h & 255) / 255)


COLOR_BG = _hex(0xF5F7F4)        # 雾白
COLOR_GRAPHITE = _hex(0x202923)  # 石墨
COLOR_MOSS = _hex(0x356B50)      # 苔绿
COLOR_SEP = _hex(0xDCE3DC)       # 分隔线

# Settings ↔ Cleaner wiring. The keys are what Cleaner("auto")/Options.backend understand.
# English titles stay: tests/test_cleanup_wiring.py pins them, and the actions accept them.
BACKENDS = [("Automatic", "auto"),
            ("Gemini (cloud, text only)", "gemini"),
            ("Local (on-device)", "local"),
            ("Ollama (local server)", "ollama"),
            ("Claude (API key)", "claude"),
            ("Basic (offline rules)", "basic")]
STYLES = [("Faithful (fix mis-read words only)", "faithful"),
          ("Polish (also tidy phrasing)", "polish")]

# Chinese display aliases for the popups; the wiring above never changes.
BACKEND_ALIASES = {"auto": "自动(默认 Gemini)",
                   "gemini": "Gemini(云端,仅文字)",
                   "local": "本地模型(设备上)",
                   "ollama": "Ollama(本地服务)",
                   "claude": "Claude(API 密钥)",
                   "basic": "基础规则(离线)"}
STYLE_ALIASES = {"faithful": "忠实(只修正误读词)",
                 "polish": "自然整理(同时理顺表达)"}
_ALIAS_TO_BACKEND = {v: k for k, v in BACKEND_ALIASES.items()}
_ALIAS_TO_STYLE = {v: k for k, v in STYLE_ALIASES.items()}

PAGES = [("input", "输入", "keyboard"),
         ("personal", "个人定制", "person.crop.circle"),
         ("privacy", "隐私", "lock.shield"),
         ("advanced", "高级", "gearshape")]
PAGE_ALIASES = {"training": "personal"}

DOMAIN_NAMES = {"ai-coding": "AI/编程", "finance": "金融", "education": "教育"}
USE_CASE_NAMES = {"ai-prompts": "AI 指令", "messages": "聊天", "emails": "邮件", "notes": "笔记"}

CLOUD_DISCLOSURE = ("云端整理会把候选文字、近期口述和相关名词发送到所选服务。"
                    "摄像头画面、练习片段和音频始终留在本机。")


def _label(parent, frame, text, size, weight=NSFontWeightRegular, color=None, center=False):
    """Every label in this window gets an explicit color (onboarding._text defaults to white)."""
    t = NSTextField.wrappingLabelWithString_(text)
    t.setFrame_(frame)
    t.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
    t.setTextColor_(color if color is not None else _rgb(COLOR_GRAPHITE))
    t.setSelectable_(False)
    if center:
        t.setAlignment_(NSTextAlignmentCenter)
    parent.addSubview_(t)
    return t


def _sep(parent, y):
    line = NSView.alloc().initWithFrame_(NSMakeRect(CX - SIDEBAR_W, y, CW, 1))
    line.setWantsLayer_(True)
    line.layer().setBackgroundColor_(_rgb(COLOR_SEP).CGColor())
    parent.addSubview_(line)


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                n += 1
    return n


class Settings(NSObject):
    def initWithApp_(self, app):
        self = objc.super(Settings, self).init()
        if self is None:
            return None
        self.app = app
        self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, SW, SH), NSWindowStyleMaskTitled | NSWindowStyleMaskClosable,
            NSBackingStoreBuffered, False)
        self.win.setTitle_("Lipflow 设置")
        self.win.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameAqua"))
        self.win.setOpaque_(True)
        self.win.setBackgroundColor_(_rgb(COLOR_BG))
        self.win.setHasShadow_(True)
        self.win.setReleasedWhenClosed_(False)
        self.win.setContentMinSize_((SW, SH))
        self.win.setContentMaxSize_((SW, SH))
        self.root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, SW, SH))
        self.win.setContentView_(self.root)
        self.current_page = "input"
        self.page = None
        self.nav_buttons = {}
        return self

    @objc.python_method
    def show(self):
        self.build()
        from AppKit import NSApplication, NSFloatingWindowLevel
        self.win.setLevel_(NSFloatingWindowLevel)
        self.win.center()
        self.win.orderFrontRegardless()
        self.win.makeKeyWindow()
        app = NSApplication.sharedApplication()
        app.activate() if hasattr(app, "activate") else app.activateIgnoringOtherApps_(True)

    # -- layout ----------------------------------------------------------------------
    @objc.python_method
    def build(self, page=None):
        """Rebuild the whole window: sidebar plus one page. `page` selects the first page."""
        for sub in list(self.root.subviews()):
            sub.removeFromSuperview()
        self.page = None
        self._build_sidebar()
        self.select_page(page or self.current_page)

    @objc.python_method
    def select_page(self, page):
        page = PAGE_ALIASES.get(page, page)
        if page not in {k for k, _, _ in PAGES}:
            page = "input"
        self.current_page = page
        self._refresh_nav()
        if self.page is not None:
            self.page.removeFromSuperview()
        p = self.page = NSView.alloc().initWithFrame_(NSMakeRect(SIDEBAR_W, 0, SW - SIDEBAR_W, SH))
        self.root.addSubview_(p)
        {"input": self._page_input,
         "personal": self._page_personal,
         "privacy": self._page_privacy,
         "advanced": self._page_advanced}[page](p)

    @objc.python_method
    def _build_sidebar(self):
        side = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, SIDEBAR_W, SH))
        self.root.addSubview_(side)
        edge = NSView.alloc().initWithFrame_(NSMakeRect(SIDEBAR_W - 1, 0, 1, SH))
        edge.setWantsLayer_(True)
        edge.layer().setBackgroundColor_(_rgb(COLOR_SEP).CGColor())
        side.addSubview_(edge)
        _label(side, NSMakeRect(20, SH - 48, SIDEBAR_W - 40, 26), "Lipflow", 20, NSFontWeightBold)
        _label(side, NSMakeRect(20, SH - 68, SIDEBAR_W - 40, 16), "无声输入,按住即说", 11.5,
               color=_rgb(COLOR_GRAPHITE, 0.55))
        self.nav_buttons = {}
        for i, (key, title, icon) in enumerate(PAGES):
            y = SH - 130 - i * 38
            b = NSButton.buttonWithTitle_target_action_("  " + title, self, "choosePage:")
            b.setFrame_(NSMakeRect(12, y, SIDEBAR_W - 24, 32))
            b.setTag_(i)
            b.setBordered_(False)
            b.setWantsLayer_(True)
            b.layer().setCornerRadius_(7)
            b.setImage_(symbol(icon, 13))
            b.setImagePosition_(2)  # NSImageLeft
            b.setAlignment_(0)      # left
            side.addSubview_(b)
            self.nav_buttons[key] = b

    @objc.python_method
    def _refresh_nav(self):
        for key, b in self.nav_buttons.items():
            active = key == self.current_page
            b.layer().setBackgroundColor_(
                _rgb(COLOR_MOSS, 0.13).CGColor() if active else NSColor.clearColor().CGColor())
            b.setFont_(NSFont.systemFontOfSize_weight_(
                13, NSFontWeightSemibold if active else NSFontWeightRegular))
            b.setContentTintColor_(_rgb(COLOR_MOSS) if active else _rgb(COLOR_GRAPHITE, 0.8))

    def choosePage_(self, sender):
        self.select_page(PAGES[sender.tag()][0])

    @objc.python_method
    def _title(self, p, text):
        _label(p, NSMakeRect(CX - SIDEBAR_W, SH - 52, CW, 30), text, 24, NSFontWeightBold)
        return SH - 52

    # -- 输入 -------------------------------------------------------------------------
    @objc.python_method
    def _page_input(self, p):
        s = self.app.settings
        self._title(p, "输入")

        # The one signature detail: a big, plain Right ⌥ keycap.
        cap = NSView.alloc().initWithFrame_(NSMakeRect(24, SH - 196, 148, 96))
        cap.setWantsLayer_(True)
        cap.layer().setCornerRadius_(12)
        cap.layer().setBackgroundColor_(_rgb((1, 1, 1)).CGColor())
        cap.layer().setBorderWidth_(1.5)
        cap.layer().setBorderColor_(_rgb(COLOR_GRAPHITE, 0.35).CGColor())
        base = NSView.alloc().initWithFrame_(NSMakeRect(42, 8, 64, 3))
        base.setWantsLayer_(True)
        base.layer().setCornerRadius_(1.5)
        base.layer().setBackgroundColor_(_rgb(COLOR_GRAPHITE, 0.15).CGColor())
        cap.addSubview_(base)
        _label(cap, NSMakeRect(0, 40, 148, 40), "⌥", 32, NSFontWeightMedium, center=True)
        _label(cap, NSMakeRect(0, 16, 148, 18), "Right Option", 12, NSFontWeightSemibold, center=True)
        p.addSubview_(cap)
        self.keycap = cap
        self.left_option_note = _label(
            p, NSMakeRect(192, SH - 158, 312, 40),
            "按住 Right Option 做口型,松开输入;双击可免按住。", 14, NSFontWeightSemibold)
        _label(p, NSMakeRect(192, SH - 190, 300, 18),
               "Left Option 保留给豆包,互不影响。", 12, color=_rgb(COLOR_GRAPHITE, 0.55))

        from .hotkey import KEYS
        from .camera import list_cameras
        y = SH - 286
        self.key_popup = self._popup(p, y, "按键",
                                     [k.replace("_", " ").title() for k in KEYS],
                                     self.app.opts.key.replace("_", " ").title(), "pickKey:")
        _sep(p, y - 16)
        y -= 66
        style = s.get("cleanup_style", "faithful")
        self.style_popup = self._popup(p, y, "整理方式", list(STYLE_ALIASES.values()),
                                       STYLE_ALIASES[style], "pickStyle:")
        _sep(p, y - 16)
        y -= 66
        cams = [c["name"] for c in list_cameras()]
        cur = s.get("camera", "auto")
        cur_name = next((c["name"] for c in list_cameras() if c["id"] == cur), "自动(内置摄像头)")
        self.cam_popup = self._popup(p, y, "摄像头", ["自动(内置摄像头)"] + cams, cur_name, "pickCam:")
        _label(p, NSMakeRect(CX - SIDEBAR_W, y - 48, CW, 34),
               "整理方式决定识别结果如何修改:忠实只改误读词,自然整理同时理顺表达。",
               11.5, color=_rgb(COLOR_GRAPHITE, 0.55))

    # -- 个人定制 ----------------------------------------------------------------------
    @objc.python_method
    def _page_personal(self, p):
        from .onboarding import saved_clips
        from . import corrections, personal, vocab
        s = self.app.settings
        self._title(p, "个人定制")

        n_clips = len(saved_clips())
        n_words = _count_lines(vocab.PATH)
        n_phrases = _count_lines(personal.PHRASES)
        n_corr = corrections.count()
        y = SH - 96
        self.stats_label = _label(
            p, NSMakeRect(CX - SIDEBAR_W, y, CW, 20),
            f"练习片段 {n_clips} 个 · 词表名词 {n_words} 个 · 常用句 {n_phrases} 句 · 已学纠正 {n_corr} 条",
            13, NSFontWeightMedium)
        y -= 30

        trained = os.path.exists(PERSONAL_VSR)
        t = s.get("training")
        if t:
            status = (f"上次训练({time.strftime('%Y-%m-%d', time.localtime(t['at']))}):"
                      f"留出句正确率 {1 - t['before']:.0%} → {1 - t['after']:.0%}"
                      + (",已保留个人模型。" if t["kept"] else ",未提升,继续用标准模型。"))
        elif trained:
            status = "个人面部模型已启用。"
        else:
            status = "尚未训练,使用标准模型;没有实测数据,不显示准确率。"
        self.model_status = _label(p, NSMakeRect(CX - SIDEBAR_W, y, CW, 36), status, 12.5,
                                   color=_rgb(COLOR_MOSS if trained else COLOR_GRAPHITE,
                                              1.0 if trained else 0.65))
        y -= 46

        domains = [DOMAIN_NAMES.get(d, d) for d in s.get("personalization_domains", [])]
        uses = [USE_CASE_NAMES.get(u, u) for u in s.get("personalization_use_cases", [])]
        if domains or uses:
            line = ""
            if domains:
                line += "领域:" + "、".join(domains)
            if uses:
                line += ("  " if line else "") + "用途:" + "、".join(uses)
            _label(p, NSMakeRect(CX - SIDEBAR_W, y, CW, 20), line, 12,
                   color=_rgb(COLOR_GRAPHITE, 0.65))
            y -= 30

        _label(p, NSMakeRect(CX - SIDEBAR_W, y, CW, 36),
               "每轮录制 24 句新句子,约 5 分钟,基于全部片段在本机重新训练。",
               11.5, color=_rgb(COLOR_GRAPHITE, 0.55))
        y -= 30
        _label(p, NSMakeRect(CX - SIDEBAR_W, y, CW, 20),
               "新增名词或常用句,直接在聊天里告诉 Lipflow,会并入现有词表。",
               11.5, color=_rgb(COLOR_GRAPHITE, 0.55))

        train = NSButton.buttonWithTitle_target_action_("练习并训练", self, "trainMore:")
        train.setFrame_(NSMakeRect(CX - SIDEBAR_W, 40, 150, 32))
        train.setKeyEquivalent_("\r")
        p.addSubview_(train)
        if trained:
            reset = NSButton.buttonWithTitle_target_action_("重置面部模型", self, "resetFace:")
            reset.setFrame_(NSMakeRect(CX - SIDEBAR_W + 162, 40, 140, 32))
            p.addSubview_(reset)

    # -- 隐私 -------------------------------------------------------------------------
    @objc.python_method
    def _page_privacy(self, p):
        s = self.app.settings
        self._title(p, "隐私")
        rows = [("使用当前 App 里的名字", "整理时参考正在输入的 App 中的人名和名词。",
                 s.get("use_context", True), "toggleContext:"),
                ("保留最近 100 个片段", "只用于衡量准确度,片段留在本机。",
                 s.get("save_clips", True), "toggleClips:"),
                ("从我的纠正中学习", "粘贴后改过的词会记进个人词表。",
                 s.get("learn_corrections", True), "toggleLearn:")]
        y = SH - 110
        for i, (label, note, on, action) in enumerate(rows):
            if i:
                _sep(p, y + 34)
            _label(p, NSMakeRect(CX - SIDEBAR_W, y, 320, 20), label, 13, NSFontWeightMedium)
            _label(p, NSMakeRect(CX - SIDEBAR_W, y - 18, 380, 16), note, 11.5,
                   color=_rgb(COLOR_GRAPHITE, 0.55))
            sw = NSSwitch.alloc().initWithFrame_(NSMakeRect(CX - SIDEBAR_W + CW - 50, y - 2, 42, 24))
            sw.setState_(1 if on else 0)
            sw.setTarget_(self)
            sw.setAction_(action)
            p.addSubview_(sw)
            y -= 64
        _sep(p, y + 34)
        _label(p, NSMakeRect(CX - SIDEBAR_W, y - 24, CW, 52), CLOUD_DISCLOSURE, 12,
               color=_rgb(COLOR_GRAPHITE, 0.75))

    # -- 高级 -------------------------------------------------------------------------
    @objc.python_method
    def _page_advanced(self, p):
        s = self.app.settings
        self._title(p, "高级")
        y = SH - 110
        backend = s.get("cleanup_backend", "auto")
        self.backend_popup = self._popup(p, y, "整理后端", list(BACKEND_ALIASES.values()),
                                         BACKEND_ALIASES[backend], "pickBackend:")
        self.disclosure = _label(p, NSMakeRect(CX - SIDEBAR_W, y - 44, CW, 34),
                                 CLOUD_DISCLOSURE, 11.5, color=_rgb(COLOR_GRAPHITE, 0.55))
        self.disclosure.setHidden_(backend not in ("gemini", "claude", "auto"))
        _sep(p, y - 62)
        y -= 118
        _label(p, NSMakeRect(CX - SIDEBAR_W, y, 320, 20), "轻语模式", 13, NSFontWeightMedium)
        _label(p, NSMakeRect(CX - SIDEBAR_W, y - 18, 380, 16), "口型加麦克风音频,首次开启会下载模型。",
               11.5, color=_rgb(COLOR_GRAPHITE, 0.55))
        self.whisper_switch = NSSwitch.alloc().initWithFrame_(
            NSMakeRect(CX - SIDEBAR_W + CW - 50, y - 2, 42, 24))
        self.whisper_switch.setState_(1 if s.get("whisper", False) else 0)
        self.whisper_switch.setTarget_(self)
        self.whisper_switch.setAction_("toggleWhisper:")
        p.addSubview_(self.whisper_switch)

        data = NSButton.buttonWithTitle_target_action_("打开数据文件夹", self, "openData:")
        data.setFrame_(NSMakeRect(CX - SIDEBAR_W, 40, 150, 32))
        p.addSubview_(data)
        _label(p, NSMakeRect(CX - SIDEBAR_W + 162, 48, CW - 162, 18),
               "片段、常用句和你的模型,从不上传。", 11.5, color=_rgb(COLOR_GRAPHITE, 0.55))

    # -- control helpers ----------------------------------------------------------------
    @objc.python_method
    def _popup(self, p, y, label, items, current, action):
        _label(p, NSMakeRect(CX - SIDEBAR_W, y + 5, 240, 20), label, 13, NSFontWeightMedium)
        pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(CX - SIDEBAR_W + CW - 220, y, 220, 30), False)
        pop.addItemsWithTitles_(items)
        pop.selectItemWithTitle_(current)
        pop.setTarget_(self)
        pop.setAction_(action)
        p.addSubview_(pop)
        return pop

    # -- actions ------------------------------------------------------------------------
    @objc.python_method
    def _save(self):
        from .app import save_settings
        save_settings(self.app.settings)

    def close_(self, sender):
        self.win.orderOut_(None)

    def trainMore_(self, sender):
        self.win.orderOut_(None)
        self.app.show_setup(start_at="practice")

    @objc.python_method
    def _confirm_reset(self):
        from AppKit import NSAlert, NSAlertFirstButtonReturn
        alert = NSAlert.alloc().init()
        alert.setMessageText_("重置面部模型?")
        alert.setInformativeText_("将删除个人面部模型并恢复标准模型。练习片段和词表会保留。")
        alert.addButtonWithTitle_("重置")
        alert.addButtonWithTitle_("取消")
        return alert.runModal() == NSAlertFirstButtonReturn

    def resetFace_(self, sender):
        if not self._confirm_reset():
            return
        if os.path.exists(PERSONAL_VSR):
            os.remove(PERSONAL_VSR)
        self.app.settings.pop("training", None)
        self._save()
        self.app.jobs.put(("reload",))
        self.build()

    def pickKey_(self, sender):
        from .hotkey import KEYS
        key = sender.titleOfSelectedItem().lower().replace(" ", "_")
        self.app.opts.key = key
        self.app.ptt.keycode, self.app.ptt.mask = KEYS[key]
        self.app.settings["key"] = key
        self._save()

    def pickCam_(self, sender):
        from .camera import list_cameras
        name = sender.titleOfSelectedItem()
        cam = next((c["id"] for c in list_cameras() if c["name"] == name), "auto")
        self.app.settings["camera"] = cam
        self.app.camera.set_source(cam)
        self._save()

    @objc.python_method
    def _backend_for_title(self, title):
        return dict(BACKENDS).get(title) or _ALIAS_TO_BACKEND[title]

    @objc.python_method
    def _style_for_title(self, title):
        return dict(STYLES).get(title) or _ALIAS_TO_STYLE[title]

    def pickBackend_(self, sender):
        key = self._backend_for_title(sender.titleOfSelectedItem())
        if key == "auto":
            self.app.settings.pop("cleanup_backend", None)
        else:
            self.app.settings["cleanup_backend"] = key
        self._save()
        self.disclosure.setHidden_(key not in ("gemini", "claude", "auto"))
        # the Cleaner is (re)built on the model thread, never here on the UI thread
        self.app.jobs.put(("cleanup", key, self.app.settings.get("cleanup_style")))

    def pickStyle_(self, sender):
        style = self._style_for_title(sender.titleOfSelectedItem())
        self.app.settings["cleanup_style"] = style
        self._save()
        backend = self.app.settings.get("cleanup_backend", "auto")
        self.app.jobs.put(("cleanup", backend, style))

    def toggleContext_(self, sender):
        self.app.settings["use_context"] = bool(sender.state())
        self._save()

    def toggleClips_(self, sender):
        self.app.settings["save_clips"] = bool(sender.state())
        self._save()

    def toggleWhisper_(self, sender):
        on = bool(sender.state())
        self.app.settings["whisper"] = on
        self._save()
        if on and self.app.av_reader is None:
            self.app.jobs.put(("whisper",))  # downloads the model the first time

    def toggleLearn_(self, sender):
        self.app.settings["learn_corrections"] = bool(sender.state())
        self._save()

    def openData_(self, sender):
        os.system(f'open "{HOME}"')
