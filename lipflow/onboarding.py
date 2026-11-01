"""First-run setup: permissions → import words & phrases → mouth ~24 sentences → train on your face.

The practice step uses the normal push-to-talk key (Right Option), so onboarding is also the tutorial.
Clips are saved with their known text under ~/Library/Application Support/Lipflow/clips/onboarding.
Training holds 6 of them out and only keeps the face model if it reads those better than the stock model.
"""
from __future__ import annotations

import glob
import os
import random
import re
import shlex
import subprocess
import threading
import time

import numpy as np
import objc
import Quartz
from AppKit import (
    NSApp, NSBackingStoreBuffered, NSButton, NSColor, NSFont, NSFontWeightBold,
    NSFontWeightMedium, NSFontWeightRegular, NSFontWeightSemibold, NSImageScaleProportionallyUpOrDown,
    NSImageView, NSMakeRect, NSProgressIndicator, NSTextAlignmentCenter, NSTextField, NSView, NSWindow,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskClosable, NSWindowStyleMaskFullSizeContentView, NSWindowStyleMaskTitled,
    NSAttributedString, NSFontAttributeName, NSForegroundColorAttributeName, NSMutableParagraphStyle,
    NSParagraphStyleAttributeName, NSAppearance,
)
from Foundation import NSObject, NSTimer
from PyObjCTools import AppHelper

from .hud import ACCENT, AMBER, GREEN, _glass, _nsimage, _rgb, symbol

from .paths import HOME as DIR
from .practice import CLIPS, N_HELD_OUT, N_SENTENCES, practice_sentences, save_clip, saved_clips  # noqa: F401

WW, WH = 620, 600

# Palette - Native Light Solid / Moss (FRONTEND_BRIEF.md)
# 雾白 #F5F7F4, 石墨 #202923, 分隔线 #DCE3DC, 苔绿 #356B50, 次要文本 #5A695E
COLOR_BG_TINT = (0.961, 0.969, 0.957)     # #F5F7F4
COLOR_GRAPHITE = (0.125, 0.161, 0.137)    # #202923
COLOR_MUTED = (0.353, 0.412, 0.369)       # #5A695E
COLOR_LINE = (0.863, 0.890, 0.863)        # #DCE3DC
COLOR_MOSS = (0.208, 0.420, 0.314)        # #356B50
COLOR_AMBER = (0.78, 0.45, 0.10)          # Readable amber on light bg
COLOR_CARD_BG = (1.0, 1.0, 1.0)           # Pure white card on misty bg


def _text(parent, frame, text, size, weight=NSFontWeightRegular, alpha=0.95, center=False, color=None):
    """Shared _text helper.

    IMPORTANT: default color must stay WHITE for HUD and external callers (like settings_window.py).
    """
    t = NSTextField.wrappingLabelWithString_(text)
    t.setFrame_(frame)
    t.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
    t.setTextColor_(color or _rgb((1, 1, 1), alpha))
    t.setSelectable_(False)
    if center:
        t.setAlignment_(NSTextAlignmentCenter)
    parent.addSubview_(t)
    return t


def _ob_text(parent, frame, text, size, weight=NSFontWeightRegular, alpha=1.0, center=False, color=None):
    """Onboarding-specific text helper defaulting to graphite text for light theme."""
    t = NSTextField.wrappingLabelWithString_(text)
    t.setFrame_(frame)
    t.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
    t.setTextColor_(color or _rgb(COLOR_GRAPHITE, alpha))
    t.setSelectable_(False)
    if center:
        t.setAlignment_(NSTextAlignmentCenter)
    parent.addSubview_(t)
    return t


class GlassWindow(NSWindow):
    """Borderless window with rounded corners and key window opt-in."""

    def canBecomeKeyWindow(self):
        return True

    def canBecomeMainWindow(self):
        return True


class Onboarding(NSObject):
    """Owns the setup window. `app` is the running Lipflow controller."""

    def initWithApp_(self, app):
        self = objc.super(Onboarding, self).init()
        if self is None:
            return None
        self.app = app
        self.sentences = []
        self.i = 0
        self.timer = None
        self.win = GlassWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WW, WH), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        self.win.setTitle_("设置 Lipflow")
        self.win.setMovableByWindowBackground_(True)
        self.win.setHasShadow_(True)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        self.win.setReleasedWhenClosed_(False)
        self.win.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameAqua"))

        # Root view with opaque solid light background and border
        self.root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WW, WH))
        self.root.setWantsLayer_(True)
        self.root.layer().setCornerRadius_(16)
        self.root.layer().setMasksToBounds_(True)
        self.root.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_BG_TINT, 1.0))
        self.root.layer().setBorderWidth_(1.0)
        self.root.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        self.win.setContentView_(self.root)
        self.win.center()

        # Close button in top-right
        close = NSButton.buttonWithTitle_target_action_("", self, "close:")
        close.setBordered_(False)
        close.setImage_(symbol("xmark", 11, NSFontWeightMedium))
        close.setContentTintColor_(_rgb(COLOR_MUTED, 0.8))
        close.setFrame_(NSMakeRect(WW - 38, WH - 38, 26, 26))
        close.setWantsLayer_(True)
        close.layer().setCornerRadius_(13)
        close.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_GRAPHITE, 0.05))
        close.setToolTip_("关闭设置（可随时从菜单栏重新打开）")
        self.root.addSubview_(close)
        self.page = None
        return self

    # -- window --------------------------------------------------------------------
    @objc.python_method
    def show(self):
        if self.page is None:
            self.welcome()
        if not hasattr(self, "_shown"):
            self._shown = True
        self.app.hud.hide()  # drop the "Loading…" pill
        from AppKit import NSApplication, NSFloatingWindowLevel
        app = NSApplication.sharedApplication()
        self.win.setLevel_(NSFloatingWindowLevel)
        self.win.center()
        self.win.orderFrontRegardless()
        self.win.makeKeyWindow()
        if hasattr(app, "activate"):
            app.activate()
        else:
            app.activateIgnoringOtherApps_(True)

    @objc.python_method
    def _new_page(self, icon, title, subtitle):
        if self.timer:
            self.timer.invalidate()
            self.timer = None
        if self.page is not None:
            self.page.removeFromSuperview()
        self.page = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, WW, WH))
        self.root.addSubview_positioned_relativeTo_(self.page, -1, None)  # below close button

        # Top badge
        badge = NSView.alloc().initWithFrame_(NSMakeRect((WW - 56) / 2, WH - 110, 56, 56))
        badge.setWantsLayer_(True)
        badge.layer().setCornerRadius_(14)
        badge.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_MOSS, 0.10))
        badge.layer().setBorderWidth_(1.0)
        badge.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_MOSS, 0.20))
        iv = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, 56, 56))
        iv.setImageScaling_(0)
        iv.setImage_(symbol(icon, 22, NSFontWeightSemibold))
        iv.setContentTintColor_(_rgb(COLOR_MOSS))
        badge.addSubview_(iv)
        self.page.addSubview_(badge)
        self.page_icon = iv

        # Title & subtitle
        self.page_title = _ob_text(self.page, NSMakeRect(40, WH - 156, WW - 80, 32), title, 22, NSFontWeightBold,
                                   center=True)
        self.page_sub = _ob_text(self.page, NSMakeRect(50, WH - 212, WW - 100, 50), subtitle, 13,
                                 color=_rgb(COLOR_MUTED, 0.95), center=True)
        return self.page

    @objc.python_method
    def _button(self, title, action, x, y, w=180, primary=False, h=38, parent=None):
        """Native-styled rectangular button with slight rounding (8px) rather than pills."""
        b = NSButton.buttonWithTitle_target_action_(title, self, action)
        b.setBordered_(False)
        b.setFrame_(NSMakeRect(x, y, w, h))
        b.setWantsLayer_(True)
        b.layer().setCornerRadius_(8)
        if primary:
            b.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_MOSS, 1.0))
            b.layer().setBorderWidth_(0)
            text_color = _rgb((1, 1, 1), 1.0)
        else:
            b.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(1.0, 1.0, 1.0, 0.9))
            b.layer().setBorderWidth_(1.0)
            b.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
            text_color = _rgb(COLOR_GRAPHITE, 0.9)

        para = NSMutableParagraphStyle.alloc().init()
        para.setAlignment_(NSTextAlignmentCenter)
        b.setAttributedTitle_(NSAttributedString.alloc().initWithString_attributes_(title, {
            NSFontAttributeName: NSFont.systemFontOfSize_weight_(13.5, NSFontWeightMedium),
            NSForegroundColorAttributeName: text_color,
            NSParagraphStyleAttributeName: para}))
        if primary:
            b.setKeyEquivalent_("\r")
        (parent or self.page).addSubview_(b)
        return b

    @objc.python_method
    def _steps(self, n):
        """Bottom steps bar."""
        labels = ["系统权限", "定制词库", "口型跟读", "模型训练"]
        w = 110
        x0 = (WW - w * len(labels)) / 2
        for k, lab in enumerate(labels):
            done, cur = k < n, k == n
            if cur:
                c = _rgb(COLOR_MOSS, 1.0)
                weight = NSFontWeightBold
            elif done:
                c = _rgb(COLOR_MOSS, 0.7)
                weight = NSFontWeightMedium
            else:
                c = _rgb(COLOR_MUTED, 0.45)
                weight = NSFontWeightRegular
            _ob_text(self.page, NSMakeRect(x0 + k * w, 24, w, 18), ("✓ " if done else "") + lab, 12,
                     weight, center=True, color=c)

    # -- 1. welcome -----------------------------------------------------------------
    @objc.python_method
    def welcome(self):
        p = self._new_page(
            "mouth",
            "Lipflow 无声输入",
            "按住右侧 Option，自然默念，松开即可直接上屏。\n无需出声，适合在安静办公室、会议与开放环境专注输入。"
        )

        # Central feature card
        card = NSView.alloc().initWithFrame_(NSMakeRect(60, 175, WW - 120, 195))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(12)
        card.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_CARD_BG, 1.0))
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        p.addSubview_(card)

        # Hotkey visual callout inside card
        # Keycap for Right ⌥
        keycap = NSView.alloc().initWithFrame_(NSMakeRect(24, 125, 96, 44))
        keycap.setWantsLayer_(True)
        keycap.layer().setCornerRadius_(8)
        keycap.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_BG_TINT, 1.0))
        keycap.layer().setBorderWidth_(1.2)
        keycap.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        card.addSubview_(keycap)

        _ob_text(keycap, NSMakeRect(0, 8, 96, 26), "Right ⌥", 14, NSFontWeightBold, center=True,
                 color=_rgb(COLOR_MOSS))

        _ob_text(card, NSMakeRect(136, 137, 330, 22), "专属无声热键：右侧 Option", 14, NSFontWeightSemibold)
        _ob_text(card, NSMakeRect(136, 116, 340, 20), "左侧 Option 保留给豆包等工具，两项输入完全独立", 12,
                 color=_rgb(COLOR_MUTED))

        # Divider inside card
        div = NSView.alloc().initWithFrame_(NSMakeRect(20, 102, WW - 160, 1))
        div.setWantsLayer_(True)
        div.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        card.addSubview_(div)

        # Features list
        f1 = "• 隐私与数据：摄像头画面、音频及口型模型完全保留在本地 Mac，绝不上传"
        f2 = "• 智能云端润色：仅在开启整理时，将候选文本与上下文发送至 CPA 模型"
        f3 = "• 定制词库与训练：根据日常领域与常用句训练，耗时约 5 分钟"
        _ob_text(card, NSMakeRect(24, 68, WW - 168, 24), f1, 12, color=_rgb(COLOR_MUTED))
        _ob_text(card, NSMakeRect(24, 42, WW - 168, 24), f2, 12, color=_rgb(COLOR_MUTED))
        _ob_text(card, NSMakeRect(24, 16, WW - 168, 24), f3, 12, color=_rgb(COLOR_MUTED))

        self._button("开始设置", "goPermissions:", (WW - 200) / 2, 95, 200, primary=True)

    # -- 2. permissions -------------------------------------------------------------
    def goPermissions_(self, sender):
        p = self._new_page(
            "lock.shield",
            "系统运行权限",
            "macOS 仅需在初次运行时授权。授权后即可正常捕捉口型与模拟按键上屏。"
        )
        self._steps(0)
        self.perm_rows = []

        card = NSView.alloc().initWithFrame_(NSMakeRect(60, 170, WW - 120, 210))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(12)
        card.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_CARD_BG, 1.0))
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        p.addSubview_(card)

        rows = [
            ("camera.fill", "摄像头", "仅在本地捕捉面部与唇部动作", "askCamera:"),
            ("keyboard", "输入监视", "感知右侧 Option 键按下与抬起", "askInput:"),
            ("text.cursor", "辅助功能", "将识别结果直接粘贴至当前输入光标", "askAccess:")
        ]

        for k, (icon, name, why, action) in enumerate(rows):
            y = 142 - k * 64
            if k > 0:
                div = NSView.alloc().initWithFrame_(NSMakeRect(20, y + 60, WW - 160, 1))
                div.setWantsLayer_(True)
                div.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 0.7))
                card.addSubview_(div)

            iv = NSImageView.alloc().initWithFrame_(NSMakeRect(20, y + 10, 28, 28))
            iv.setImage_(symbol(icon, 18, NSFontWeightMedium))
            iv.setContentTintColor_(_rgb(COLOR_MOSS))
            card.addSubview_(iv)

            _ob_text(card, NSMakeRect(60, y + 20, 240, 20), name, 13.5, NSFontWeightSemibold)
            _ob_text(card, NSMakeRect(60, y + 2, 240, 18), why, 11.5, color=_rgb(COLOR_MUTED))

            status = NSImageView.alloc().initWithFrame_(NSMakeRect(WW - 236, y + 12, 22, 22))
            card.addSubview_(status)
            btn = self._button("前往授权", action, WW - 210, y + 7, 80, h=30, parent=card)
            self.perm_rows.append((status, btn))

        self.perm_next = self._button("继续", "goWords:", (WW - 200) / 2, 95, 200, primary=True)

        self.perm_restart = self._button("已在系统设置中开启？重启 Lipflow", "restartForPerms:",
                                         (WW - 280) / 2, 52, 280, h=30)
        self.perm_restart.setHidden_(True)
        self._asked_perms = False
        self.refreshPerms_(None)
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0, self, "refreshPerms:", None, True)

    @objc.python_method
    def _perm_state(self):
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        return [
            AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo) == 3,
            bool(Quartz.CGPreflightListenEventAccess()),
            bool(Quartz.CGPreflightPostEventAccess())
        ]

    def refreshPerms_(self, timer):
        states = self._perm_state()
        for (status, btn), ok in zip(self.perm_rows, states):
            status.setImage_(symbol("checkmark.circle.fill" if ok else "circle.dashed", 18))
            status.setContentTintColor_(_rgb(COLOR_MOSS) if ok else _rgb(COLOR_MUTED, 0.4))
            btn.setHidden_(ok)
        ready = all(states)
        self.perm_next.setEnabled_(ready)
        # For non-glass Aqua button, use text alpha or background alpha for disabled state
        if ready:
            self.perm_next.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_MOSS, 1.0))
        else:
            self.perm_next.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_MOSS, 0.35))
        self.perm_restart.setHidden_(ready or not self._asked_perms)

    def restartForPerms_(self, sender):
        from AppKit import NSApplication, NSBundle
        app = str(NSBundle.mainBundle().bundlePath())
        if not app.endswith(".app"):
            return
        subprocess.Popen(["/bin/bash", "-c", f"sleep 0.6; open {shlex.quote(app)}"], start_new_session=True)
        NSApplication.sharedApplication().terminate_(None)

    def askCamera_(self, sender):
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        if AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo) == 0:
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, lambda ok: None)
        else:
            os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"')

    def askInput_(self, sender):
        self._asked_perms = True
        if not Quartz.CGRequestListenEventAccess():
            os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"')

    def askAccess_(self, sender):
        self._asked_perms = True
        if not Quartz.CGRequestPostEventAccess():
            os.system('open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"')

    # -- 3. your words (profile + import) --------------------------------------------
    def goWords_(self, sender):
        from .personal import PHRASES, WISPR_DIR
        from .vocab import PATH as WORDS_PATH
        have_phrases = os.path.exists(PHRASES)
        have_words = os.path.exists(WORDS_PATH)

        p = self._new_page(
            "text.book.closed",
            "准备您的输入词库",
            "口型容易产生同音混淆。Lipflow 结合您的领域词库在本地快速重排候选，并在云端进行自然润色。"
        )
        self._steps(1)

        # Card showing configured profile and corpus stats
        card = NSView.alloc().initWithFrame_(NSMakeRect(60, 160, WW - 120, 215))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(12)
        card.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_CARD_BG, 1.0))
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        p.addSubview_(card)

        # Domain badges / active settings
        domains = self.app.settings.get("personalization_domains", ["ai-coding", "finance", "education"])
        domain_names = {
            "ai-coding": "AI / 编程",
            "finance": "金融 / 投研",
            "education": "学校 / 学习"
        }
        domain_str = " · ".join(domain_names.get(d, d) for d in domains)
        _ob_text(card, NSMakeRect(24, 172, 100, 22), "已选定制领域", 12, NSFontWeightBold, color=_rgb(COLOR_MOSS))
        _ob_text(card, NSMakeRect(126, 172, WW - 270, 22), domain_str, 12.5, NSFontWeightMedium)

        # Use cases
        use_cases = self.app.settings.get("personalization_use_cases", ["ai-prompts", "messages", "emails", "notes"])
        case_names = {
            "ai-prompts": "AI 提问",
            "messages": "即时沟通",
            "emails": "邮件往来",
            "notes": "个人笔记"
        }
        case_str = " · ".join(case_names.get(c, c) for c in use_cases)
        _ob_text(card, NSMakeRect(24, 144, 100, 20), "常用输入场景", 12, NSFontWeightBold, color=_rgb(COLOR_MOSS))
        _ob_text(card, NSMakeRect(126, 144, WW - 270, 20), case_str, 12, color=_rgb(COLOR_MUTED))

        # Divider
        div = NSView.alloc().initWithFrame_(NSMakeRect(20, 132, WW - 160, 1))
        div.setWantsLayer_(True)
        div.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        card.addSubview_(div)

        # Corpus count & status (reads from disk or profile defaults if empty)
        n_p = 0
        if have_phrases:
            n_p = sum(1 for _ in open(PHRASES, encoding="utf-8", errors="ignore"))
        n_w = 0
        if have_words:
            n_w = sum(1 for line in open(WORDS_PATH, encoding="utf-8", errors="ignore")
                      if line.strip() and not line.startswith("#"))

        # If zero phrases found (clean test env), compute from domains so user sees corpus
        if n_p == 0 and n_w <= 1:
            from .profile import corpus
            w_list, p_list = corpus(domains)
            n_w, n_p = len(w_list), len(p_list)

        self.words_status = _ob_text(
            card, NSMakeRect(20, 78, WW - 160, 44),
            f"已就绪词库：{n_w} 个专有名词 · {n_p} 条常用句\n预设词库已根据上述领域合并，可直接继续下一步。",
            13, center=True
        )

        # Optional import secondary buttons
        can_wispr = os.path.isdir(WISPR_DIR)
        btn_w = 125
        gap = 10
        if can_wispr:
            total_w = btn_w * 3 + gap * 2
            x_start = (WW - 120 - total_w) / 2
            self._button("导入常用句", "importPhrases:", x_start, 22, btn_w, primary=False, h=32, parent=card)
            self._button("导入专有名词", "importWords:", x_start + btn_w + gap, 22, btn_w, primary=False, h=32, parent=card)
            self._button("导入 Wispr", "importWispr:", x_start + (btn_w + gap) * 2, 22, btn_w, primary=False, h=32, parent=card)
        else:
            total_w = btn_w * 2 + gap
            x_start = (WW - 120 - total_w) / 2
            self._button("导入常用句", "importPhrases:", x_start, 22, btn_w, primary=False, h=32, parent=card)
            self._button("导入专有名词", "importWords:", x_start + btn_w + gap, 22, btn_w, primary=False, h=32, parent=card)

        # Main Continue CTA
        cont_w = 220
        self._button("使用就绪词库继续", "goPractice:", (WW - cont_w) / 2, 92, cont_w, primary=True)

    def importPhrases_(self, sender):
        from AppKit import NSOpenPanel, NSModalResponseOK
        from .customize import import_plain_text
        panel = NSOpenPanel.openPanel()
        panel.setTitle_("选择常用句纯文本文件")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        if panel.runModal() == NSModalResponseOK:
            url = panel.URLs()[0]
            file_path = str(url.path())
            self.words_status.setStringValue_("正在合并导入的句子…")

            def work():
                try:
                    res = import_plain_text(phrases_file=file_path)
                    msg = f"已成功导入句子！新增 {res['phrases_added']} 条常用句（当前总计 {res['phrases_after']} 条）。"
                except Exception as e:
                    msg = f"导入失败：{e}"
                AppHelper.callAfter(self.words_status.setStringValue_, msg)
                AppHelper.callAfter(self.app.cleaner.personal.__init__)

            threading.Thread(target=work, daemon=True).start()

    def importWords_(self, sender):
        from AppKit import NSOpenPanel, NSModalResponseOK
        from .customize import import_plain_text
        panel = NSOpenPanel.openPanel()
        panel.setTitle_("选择专有名词纯文本文件")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        if panel.runModal() == NSModalResponseOK:
            url = panel.URLs()[0]
            file_path = str(url.path())
            self.words_status.setStringValue_("正在合并专有名词…")

            def work():
                try:
                    res = import_plain_text(words_file=file_path)
                    msg = f"已成功导入词汇！新增 {res['words_added']} 个专有名词（当前总计 {res['words_after']} 个）。"
                except Exception as e:
                    msg = f"导入失败：{e}"
                AppHelper.callAfter(self.words_status.setStringValue_, msg)
                AppHelper.callAfter(self.app.cleaner.personal.__init__)

            threading.Thread(target=work, daemon=True).start()

    def importTextFile_(self, sender):
        self.importPhrases_(sender)

    def importWispr_(self, sender):
        from .personal import import_wispr
        self.words_status.setStringValue_("正在读取 Wispr Flow 历史记录…")

        def work():
            try:
                st = import_wispr()
                msg = f"已导入 {st['phrases']:,} 条历史常用句。"
                if st["new_names"]:
                    msg += f" 识别到 {len(st['new_names'])} 个专有名字（可随时在设置中修改）。"
            except Exception as e:
                msg = f"导入失败：{e}"
            AppHelper.callAfter(self.words_status.setStringValue_, msg)
            AppHelper.callAfter(self.app.cleaner.personal.__init__)

        threading.Thread(target=work, daemon=True).start()

    # -- 4. practice -----------------------------------------------------------------
    def goPractice_(self, sender):
        os.makedirs(CLIPS, exist_ok=True)
        done = {c["text"] for c in saved_clips()}
        self.sentences = practice_sentences(N_SENTENCES, exclude=done)
        self.i = 0
        prior = len(done)
        p = self._new_page(
            "quote.bubble",
            "口型录制与跟读练习",
            "按住右侧 Option 键，面向摄像头用平时语速默念英文例句，念完后松开按键。\n练习覆盖常见音素口型，句式保持英文以匹配模型。"
            + (f"（已录制 {prior} 个片段，新录制将追加训练）" if prior else "")
        )
        self._steps(2)

        # Card container for sentence and video preview
        card = NSView.alloc().initWithFrame_(NSMakeRect(60, 115, WW - 120, 270))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(12)
        card.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_CARD_BG, 1.0))
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        p.addSubview_(card)

        # Hotkey badge on top left of card
        hotkey_tag = NSView.alloc().initWithFrame_(NSMakeRect(20, 230, 110, 24))
        hotkey_tag.setWantsLayer_(True)
        hotkey_tag.layer().setCornerRadius_(6)
        hotkey_tag.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_BG_TINT, 1.0))
        hotkey_tag.layer().setBorderWidth_(1.0)
        hotkey_tag.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        _ob_text(hotkey_tag, NSMakeRect(0, 3, 110, 18), "按住 Right ⌥", 11.5, NSFontWeightBold,
                 center=True, color=_rgb(COLOR_MOSS))
        card.addSubview_(hotkey_tag)

        # Progress count on top right of card
        self.count = _ob_text(card, NSMakeRect(WW - 280, 232, 140, 18), "", 12, NSFontWeightBold,
                              center=False, color=_rgb(COLOR_MOSS))

        # Sentence prompt (English practice sentences remain English)
        self.prompt = _ob_text(card, NSMakeRect(20, 155, WW - 160, 68), "", 18, NSFontWeightMedium,
                               center=True, color=_rgb(COLOR_GRAPHITE))

        # Camera frame preview
        self.video = NSImageView.alloc().initWithFrame_(NSMakeRect((WW - 120 - 150) / 2, 42, 150, 100))
        self.video.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.video.setWantsLayer_(True)
        self.video.layer().setCornerRadius_(10)
        self.video.layer().setMasksToBounds_(True)
        self.video.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_GRAPHITE, 0.08))
        self.video.layer().setBorderWidth_(1.0)
        self.video.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        card.addSubview_(self.video)

        # Feedback row under camera
        self.feedback = _ob_text(card, NSMakeRect(20, 12, WW - 160, 22), "准备就绪：按住 Right Option 开始默念",
                                 12, center=True, color=_rgb(COLOR_MUTED))

        # Action buttons
        self._button("重录上一句", "redo:", WW / 2 - 165, 62, 150, h=36)
        self._button("跳过当前句", "skip:", WW / 2 + 15, 62, 150, h=36)

        self.app.camera.ensure_open()
        self.app.onboarding = self
        self._show_sentence()

    @objc.python_method
    def _show_sentence(self):
        if not self.sentences:
            return
        self.count.setStringValue_(f"第 {self.i + 1} 句 / 共 {N_SENTENCES} 句")
        current_text = self.sentences[self.i % len(self.sentences)]
        self.prompt.setStringValue_(current_text)
        self.app.onboarding_text = current_text

    @objc.python_method
    def set_frame(self, video_bgr):
        if self.app.onboarding is self and video_bgr is not None:
            self.video.setImage_(_nsimage(video_bgr))

    @objc.python_method
    def clip_done(self, rec_ok: bool, message: str, rois=None, text=None, raw=None):
        """Called by the app (main thread) after each practice recording."""
        if not rec_ok:
            self.feedback.setTextColor_(_rgb(COLOR_AMBER))
            self.feedback.setStringValue_(message)
            return
        path = save_clip(rois, text, raw)
        self.round_clips = getattr(self, "round_clips", []) + [path]
        self.feedback.setTextColor_(_rgb(COLOR_MOSS))
        self.feedback.setStringValue_(f"已保存。模型初步识别：\"{(raw or '').lower()}\"")
        self.i += 1
        if self.i >= N_SENTENCES:
            self.app.onboarding = None
            return self.goTrain_(None)
        self._show_sentence()

    def redo_(self, sender):
        mine = getattr(self, "round_clips", [])
        if mine and self.i > 0:
            os.remove(mine.pop())
            self.i -= 1
            self.feedback.setStringValue_("已删除上一条录制片段，请重新按住 Right Option 默念。")
            self._show_sentence()

    def skip_(self, sender):
        if self.sentences:
            self.sentences.append(self.sentences.pop(self.i % len(self.sentences)))
            self._show_sentence()

    # -- 5. train ----------------------------------------------------------------------
    def goTrain_(self, sender):
        self.app.onboarding = None
        p = self._new_page(
            "cpu",
            "本地微调面部口型模型",
            "训练在 Mac 本地 GPU 上运行，约需 4-5 分钟。\n训练期间 Lipflow 暂停无声输入，您可以正常进行其他工作。"
        )
        self._steps(3)

        card = NSView.alloc().initWithFrame_(NSMakeRect(60, 160, WW - 120, 205))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(12)
        card.layer().setBackgroundColor_(Quartz.CGColorCreateSRGB(*COLOR_CARD_BG, 1.0))
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(Quartz.CGColorCreateSRGB(*COLOR_LINE, 1.0))
        p.addSubview_(card)

        # Progress bar
        self.progress = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(40, 145, WW - 200, 12))
        self.progress.setIndeterminate_(False)
        self.progress.setMinValue_(0)
        self.progress.setMaxValue_(100)
        card.addSubview_(self.progress)

        self.train_status = _ob_text(card, NSMakeRect(30, 95, WW - 180, 40), "正在初始化训练环境…", 13.5,
                                    center=True, color=_rgb(COLOR_MUTED))
        self.result = _ob_text(card, NSMakeRect(30, 30, WW - 180, 55), "", 14, NSFontWeightSemibold,
                               center=True)

        self.done_btn = self._button("开始使用 Lipflow", "finish:", (WW - 220) / 2, 85, 220, primary=True)
        self.done_btn.setHidden_(True)
        self.app.jobs.put(("train", self))

    @objc.python_method
    def report(self, pct: float, text: str):
        AppHelper.callAfter(self.progress.setDoubleValue_, pct)
        AppHelper.callAfter(self.train_status.setStringValue_, text)

    @objc.python_method
    def finished(self, before: float, after: "float | None", kept: bool, note: str):
        def ui():
            self.progress.setDoubleValue_(100)
            self.page_title.setStringValue_("训练完成，可以开始使用" if (after is None or kept) else "设置完成")
            self.page_sub.setStringValue_(
                "随时在任意应用中按住右侧 Option 默念即可输入。\n可以在菜单栏中随时进行追加练习，录制更多片段可持续增强识别准确度。"
            )
            self.page_icon.setImage_(symbol("checkmark", 24, NSFontWeightBold))
            self.page_icon.setContentTintColor_(_rgb(COLOR_MOSS))
            if after is None:
                self.result.setStringValue_(note)
            else:
                self.result.setTextColor_(_rgb(COLOR_MOSS) if kept else _rgb(COLOR_AMBER))
                self.result.setStringValue_(
                    f"保留验证集实测正确率：{1 - before:.0%} → {1 - after:.0%}"
                    + ("" if kept else "\n本次训练未超过基准模型，已保留原有标准模型。")
                )
            self.train_status.setStringValue_(note if after is not None else "")
            self.done_btn.setHidden_(False)
        AppHelper.callAfter(ui)

    def close_(self, sender):
        self.app.onboarding = None
        self.app.camera.track_always = False
        self.win.orderOut_(None)

    def finish_(self, sender):
        from .app import load_settings, save_settings
        s = load_settings()
        s["onboarded"] = True
        save_settings(s)
        self.app.settings["onboarded"] = True
        self.win.orderOut_(None)
        self.app.camera.track_always = False
        self.app.hud.show("done", "Lipflow 已就绪", "在任意位置按住 Right Option 默念输入", 3.0)
