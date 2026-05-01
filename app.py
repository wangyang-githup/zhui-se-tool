"""
追色 - AI智能配色工具 v2.1
新布局：左侧70%预览 + 右侧30%工具栏

功能：
1. 参考图导入 - 加载目标色调的参考图片
2. 色彩分析 - 提取参考图和原片的色彩特征
3. 智能追色 - Lab空间色彩迁移 + ΔE肤色保护
4. 预设系统 - 8组内置预设一键套用
5. LUT导出 - 导出.cube格式的LUT文件

无OpenCV依赖，纯PIL + NumPy + scipy实现
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from PIL import ImageTk, ImageDraw
import numpy as np
import os
import json
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# 导入自定义模块
from color_engine import ColorTransfer, load_image, save_image
from feature_extractor import FeatureExtractor
from grading_generator import ColorGradingGenerator
from color_renderer import ColorRenderer
from evaluator import Evaluator


class ProgressWindow:
    """带动画的进度窗口"""

    def __init__(self, parent, title: str = "处理中"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("400x120")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()

        x = (self.window.winfo_screenwidth() - 400) // 2
        y = (self.window.winfo_screenheight() - 120) // 2
        self.window.geometry(f"400x120+{x}+{y}")

        self.label = ttk.Label(self.window, text="正在处理...", font=("Arial", 11))
        self.label.pack(pady=(15, 5))

        self.progress = ttk.Progressbar(self.window, mode="indeterminate", length=350)
        self.progress.pack(pady=5)
        self.progress.start(10)

        self.percent_label = ttk.Label(self.window, text="0%", font=("Arial", 10))
        self.percent_label.pack(pady=5)

    def update_text(self, text: str, percent: int = None):
        self.label.configure(text=text)
        if percent is not None:
            self.percent_label.configure(text=f"{percent}%")
        self.window.update()

    def close(self):
        self.progress.stop()
        self.window.destroy()


class ZsTrackerApp:
    """追色工具主应用 - v2.1 新布局"""

    # 配色方案
    BG_DARK    = "#0a0a0b"
    BG_CARD    = "#141416"
    BG_INPUT   = "#1a1a1c"
    BORDER     = "#2a2a2e"
    TEXT_MAIN  = "#ffffff"
    TEXT_MUTE  = "#888888"
    TEXT_HINT  = "#555555"
    ACCENT     = "#3b82f6"
    ORANGE     = "#f59e0b"
    GREEN      = "#22c55e"
    RED        = "#ef4444"
    BTN_BG     = "#3b82f6"
    BTN_HOVER  = "#2563eb"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("追色 v2.1 - AI智能配色工具")
        self.root.configure(bg=self.BG_DARK)
        self.root.geometry("1400x820")
        self.root.minsize(1200, 750)

        # 加载预设
        self._load_presets()

        # 状态变量
        self.reference_path: Optional[str] = None
        self.reference_img: Optional[np.ndarray] = None
        self.target_imgs: List[Tuple[str, np.ndarray]] = []
        self.current_result: Optional[np.ndarray] = None
        self.current_index = 0
        self.output_dir = os.path.expanduser("~/Desktop")
        self.selected_preset: Optional[Dict] = None

        # 专业模块
        self.feature_extractor = FeatureExtractor()
        self.grading_generator = ColorGradingGenerator()
        self.color_renderer = ColorRenderer()
        self.ref_features = None

        # 调色参数
        self.skin_protection = tk.BooleanVar(value=True)
        self.skin_strength = tk.DoubleVar(value=0.80)
        self.tone_strength = tk.DoubleVar(value=0.8)
        self.color_strength = tk.DoubleVar(value=0.9)
        self.lut_size = tk.IntVar(value=33)
        self.mode = tk.StringVar(value="quick")
        self.auto_preview = tk.BooleanVar(value=False)

        # 指标变量
        self.metric_delta_e = tk.StringVar(value="—")
        self.metric_skin = tk.StringVar(value="—")
        self.metric_speed = tk.StringVar(value="—")
        self.metric_coverage = tk.StringVar(value="—")

        # 预览图引用（防止GC）
        self.preview_labels = {}

        # 快捷键绑定
        self.root.bind("<Command-Return>", lambda e: self.start_color_transfer())
        self.root.bind("<Control-Return>", lambda e: self.start_color_transfer())

        self._init_ui()

    def _load_presets(self):
        """加载预设配置"""
        preset_path = Path(__file__).parent / "presets.json"
        if preset_path.exists():
            with open(preset_path, encoding="utf-8") as f:
                data = json.load(f)
                self.presets = data.get("presets", [])
                self.preset_categories = data.get("categories", {})
        else:
            self.presets = []
            self.preset_categories = {}

    # ──────────────────────────────────────────
    #  UI 初始化
    # ──────────────────────────────────────────

    def _init_ui(self):
        """初始化UI - 70/30新布局"""
        self._setup_styles()

        # 顶部工具栏
        self._build_header()

        # 主内容区
        content = tk.Frame(self.root, bg=self.BG_DARK)
        content.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # 左侧预览区（70%）
        left = tk.Frame(content, bg=self.BG_DARK)
        left.pack(side="left", fill="both", expand=True)
        self._build_preview_area(left)

        # 右侧工具栏（30%，固定380px）
        right = tk.Frame(content, bg=self.BG_DARK, width=380)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_tools(right)

        # 底部状态栏
        status_bar = tk.Frame(self.root, bg="#111113", height=28)
        status_bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="就绪 — 导入参考图和原图开始追色")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("PingFang SC", 10), bg="#111113", fg="#6c7086",
                 anchor="w", padx=16).pack(fill="x", side="left")

    def _setup_styles(self):
        """配置ttk样式"""
        style = ttk.Style()
        style.configure("Dark.TRadiobutton", background=self.BG_DARK, foreground=self.TEXT_MAIN)
        # 滚动条样式
        style.configure("Dark.Horizontal.TScrollbar", background=self.BORDER)

    def _build_header(self):
        """顶部工具栏"""
        header = tk.Frame(self.root, bg=self.BG_CARD, height=50)
        header.pack(fill="x", padx=12, pady=(12, 0))
        header.pack_propagate(False)

        # Logo
        tk.Label(header, text="追色", font=("PingFang SC", 15, "bold"),
                 bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(side="left", padx=(12, 0))

        # 导航标签
        nav_frame = tk.Frame(header, bg=self.BG_CARD)
        nav_frame.pack(side="left", padx=20)
        for label in ["预设", "批量", "历史"]:
            tk.Label(nav_frame, text=label, font=("PingFang SC", 12),
                     bg=self.BG_CARD, fg=self.TEXT_MUTE, cursor="hand2").pack(side="left", padx=12)

        # 右侧信息
        info_frame = tk.Frame(header, bg=self.BG_CARD)
        info_frame.pack(side="right", padx=12)

        self.img_info_label = tk.Label(info_frame, text="",
                                       font=("PingFang SC", 10), bg=self.BG_CARD, fg=self.TEXT_MUTE)
        self.img_info_label.pack(side="left", padx=12)

        tk.Button(info_frame, text="导出", font=("PingFang SC", 11),
                  bg=self.ACCENT, fg=self.TEXT_MAIN, bd=0, padx=14, pady=5,
                  cursor="hand2", command=self.save_result).pack(side="right")

    def _build_preview_area(self, parent):
        """左侧预览区：三联对比 + 底部指标"""
        preview_container = tk.Frame(parent, bg=self.BG_CARD)
        preview_container.pack(fill="both", expand=True, pady=(0, 10))

        # 三联对比（水平排列）
        panels = [
            ("参考图", "ref", "点击上传参考图", self.load_reference, False),
            ("原图", "src", "点击上传原图", self.load_targets, False),
            ("结果图", "result", "等待追色", None, True),
        ]

        panels_frame = tk.Frame(preview_container, bg=self.BG_CARD)
        panels_frame.pack(fill="both", expand=True, padx=14, pady=14)

        for title, key, placeholder, cmd, is_result in panels:
            self._make_preview_card(panels_frame, title, key, placeholder, cmd, is_result)

        # 底部指标条
        metrics_frame = tk.Frame(parent, bg=self.BG_DARK)
        metrics_frame.pack(fill="x", side="bottom")

        metrics = [
            ("色差 ΔE", self.metric_delta_e, ""),
            ("肤色偏差", self.metric_skin, "安全"),
            ("处理速度", self.metric_speed, "—"),
            ("覆盖度", self.metric_coverage, "—"),
        ]
        for label_text, var, default in metrics:
            card = tk.Frame(metrics_frame, bg=self.BG_CARD)
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            tk.Label(card, text=label_text, font=("PingFang SC", 10),
                     bg=self.BG_CARD, fg=self.TEXT_MUTE).pack(padx=12, pady=(8, 0))
            val_label = tk.Label(card, text=default, font=("PingFang SC", 13, "bold"),
                                 bg=self.BG_CARD, fg=self.TEXT_MAIN)
            val_label.pack(padx=12, pady=(2, 8))
            if var != self.metric_delta_e:
                var.set(default)

    def _make_preview_card(self, parent, title: str, key: str,
                           placeholder: str, cmd, is_result: bool):
        """创建单个预览卡片"""
        card = tk.Frame(parent, bg=self.BG_INPUT, bd=0)
        card.pack(side="left", fill="both", expand=True, padx=6)

        # 标题行
        title_frame = tk.Frame(card, bg=self.BG_INPUT)
        title_frame.pack(fill="x", padx=10, pady=(10, 6))

        tk.Label(title_frame, text=title, font=("PingFang SC", 11),
                 bg=self.BG_INPUT, fg=self.TEXT_MUTE).pack(side="left")

        status_text = "点击上传" if not is_result else "等待追色"
        status_color = self.ACCENT if not is_result else self.ORANGE
        status = tk.Label(title_frame, text=status_text, font=("PingFang SC", 10),
                          bg=self.BG_INPUT, fg=status_color, cursor="hand2" if cmd else "")
        status.pack(side="right")

        # 预览区域（16:9比例容器）
        canvas_h = 280 if not is_result else 280
        canvas = tk.Label(card, bg=self.BG_DARK, cursor="hand2" if cmd else "")
        canvas.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        canvas.config(width=400, height=canvas_h)
        self._make_placeholder(canvas, placeholder)
        self.preview_labels[key] = canvas

        if cmd:
            canvas.bind("<Button-1>", lambda e, c=cmd: c())

        # 结果图底部标签
        if is_result:
            result_tag = tk.Label(card, text="原图", font=("PingFang SC", 9),
                                  bg=self.BG_DARK, fg=self.TEXT_MUTE)
            result_tag.place(relx=0.98, rely=0.98, anchor="se", x=-12, y=-12)
            self.result_tag = result_tag

    def _build_tools(self, parent):
        """右侧工具栏"""
        container = tk.Frame(parent, bg=self.BG_DARK)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # 预设选择
        self._build_preset_section(container)

        # 核心参数
        self._build_param_section(container)

        # 肤色保护
        self._build_skin_section(container)

        # 开始追色（主按钮）
        self._build_start_button(container)

        # 导出选项
        self._build_export_section(container)

    def _build_preset_section(self, parent):
        """预设选择区"""
        frame = tk.Frame(parent, bg=self.BG_CARD)
        frame.pack(fill="x", pady=(0, 8))

        # 标题
        header = tk.Frame(frame, bg=self.BG_CARD)
        header.pack(fill="x", padx=12, pady=(10, 6))
        tk.Label(header, text="预设", font=("PingFang SC", 12, "bold"),
                 bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(side="left")
        tk.Label(header, text="+ 新建", font=("PingFang SC", 10),
                 bg=self.BG_CARD, fg=self.TEXT_MUTE, cursor="hand2").pack(side="right")

        # 预设卡片容器（横向滚动）
        scroll_frame = tk.Frame(frame, bg=self.BG_CARD)
        scroll_frame.pack(fill="x", padx=12, pady=(0, 10))

        self.preset_buttons = []
        self.preset_canvas = tk.Canvas(scroll_frame, bg=self.BG_CARD,
                                        height=56, highlightthickness=0,
                                        xscrollcommand=lambda *a: None)
        self.preset_scroll = tk.Frame(self.preset_canvas, bg=self.BG_CARD)
        self.preset_window = self.preset_canvas.create_window((0, 0),
                                                                window=self.preset_scroll, anchor="w")

        def on_configure(e):
            self.preset_canvas.configure(scrollregion=self.preset_canvas.bbox("all"))

        self.preset_scroll.bind("<Configure>", on_configure)
        self.preset_canvas.pack(fill="x")
        self.preset_canvas.bind("<MouseWheel>", lambda e: self.preset_canvas.xview_scroll(
            int(-1 * (e.delta / 120)), "units"))

        self._render_preset_cards()

    def _render_preset_cards(self):
        """渲染预设卡片"""
        for widget in self.preset_scroll.winfo_children():
            widget.destroy()
        self.preset_buttons = []

        for preset in self.presets:
            colors = preset.get("preview_colors", ["#333", "#444"])
            card = tk.Frame(self.preset_scroll, bg=self.BG_CARD, cursor="hand2", height=50)
            card.pack(side="left", fill="y", padx=(0, 6))

            # 颜色预览块
            preview = tk.Frame(card, bg=colors[0], width=60, height=36)
            preview.pack(pady=(0, 4))
            preview.pack_propagate(False)

            # 预设名称
            name = tk.Label(card, text=preset["name"][:4], font=("PingFang SC", 9),
                           bg=self.BG_CARD, fg=self.TEXT_MUTE)
            name.pack()

            # 绑定点击
            def on_click(p=preset, c=card):
                self._select_preset(p, c)

            for widget in [card, preview, name]:
                widget.bind("<Button-1>", lambda e, pc=preset, cc=card: self._select_preset(pc, cc))

        # 更新滚动区域
        self.preset_scroll.update_idletasks()
        self.preset_canvas.configure(scrollregion=self.preset_canvas.bbox("all"))

    def _select_preset(self, preset: Dict, card: tk.Frame):
        """选择预设"""
        # 重置所有卡片边框
        for btn in self.preset_buttons:
            try:
                btn.config(bg=self.BG_CARD, highlightbackground=self.BG_CARD)
            except:
                pass

        # 高亮选中卡片
        card.config(highlightbackground=self.ACCENT, highlightthickness=2)
        self.selected_preset = preset

        # 应用预设参数到滑块
        params = preset.get("params", {})
        if "exposure" in params:
            self.tone_strength.set(max(0, min(1.5, params.get("exposure", 0) / 2 + 0.8)))
        if "contrast" in params:
            self.color_strength.set(max(0, min(1.5, params.get("contrast", 0) / 50 + 0.9)))

        self.status_var.set(f"已选择预设：{preset['name']}（{preset.get('scene', '')}）")

    def _build_param_section(self, parent):
        """核心参数区"""
        frame = tk.Frame(parent, bg=self.BG_CARD)
        frame.pack(fill="x", pady=(0, 8))

        tk.Label(frame, text="参数", font=("PingFang SC", 12, "bold"),
                 bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(anchor="w", padx=12, pady=(10, 8))

        sliders = [
            ("曝光", self.tone_strength, -1.0, 2.0, "{:+.1f}"),
            ("对比度", self.color_strength, -50, 50, "{:+.0f}"),
            ("饱和度", tk.DoubleVar(value=0), -100, 100, "{:+.0f}"),
            ("色调", tk.DoubleVar(value=0), -100, 100, "{:+.0f}"),
        ]

        self.slider_vars = {}
        for label, var, from_, to, fmt in sliders:
            row = tk.Frame(frame, bg=self.BG_CARD)
            row.pack(fill="x", padx=12, pady=2)

            top_row = tk.Frame(row, bg=self.BG_CARD)
            top_row.pack(fill="x")

            tk.Label(top_row, text=label, font=("PingFang SC", 10),
                     bg=self.BG_CARD, fg=self.TEXT_MUTE).pack(side="left")

            default_val = var.get()
            val_lbl = tk.Label(top_row, text=fmt.format(default_val),
                               font=("PingFang SC", 10, "bold"),
                               bg=self.BG_CARD, fg=self.TEXT_MAIN, width=5)
            val_lbl.pack(side="right")

            s = ttk.Scale(row, from_=from_, to=to, variable=var,
                           orient="horizontal", length=280)
            s.pack(fill="x", pady=(0, 6))

            # 变量引用
            self.slider_vars[label] = (var, val_lbl, fmt)

    def _build_skin_section(self, parent):
        """肤色保护区"""
        frame = tk.Frame(parent, bg=self.BG_CARD)
        frame.pack(fill="x", pady=(0, 8))

        # 标题行
        header = tk.Frame(frame, bg=self.BG_CARD)
        header.pack(fill="x", padx=12, pady=(10, 6))
        tk.Label(header, text="肤色保护", font=("PingFang SC", 12, "bold"),
                 bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(side="left")

        # 开关
        switch = tk.Frame(header, bg=self.BG_CARD, cursor="hand2")
        switch.pack(side="right")

        def toggle_skin():
            val = not self.skin_protection.get()
            self.skin_protection.set(val)
            bg = self.GREEN if val else self.TEXT_MUTE
            knob.place(x=20 if val else 2, rely=0.5, anchor="center")

        bg_track = tk.Frame(switch, bg=self.TEXT_MUTE, width=38, height=22, bd=0)
        bg_track.pack()
        bg_track.pack_propagate(False)
        knob = tk.Frame(bg_track, bg=self.TEXT_MAIN, width=18, height=18, bd=0)
        knob.place(x=2, rely=0.5, anchor="center")
        switch.bind("<Button-1>", lambda e: toggle_skin())

        # 状态卡片
        status_row = tk.Frame(frame, bg=self.BG_CARD)
        status_row.pack(fill="x", padx=12, pady=(0, 8))

        for label_text in ["色差", "状态"]:
            stat = tk.Frame(status_row, bg=self.BG_INPUT)
            stat.pack(side="left", fill="x", expand=True, padx=(0, 4))
            tk.Label(stat, text=self.metric_delta_e.get() if label_text == "色差" else "安全",
                     font=("PingFang SC", 11, "bold"), bg=self.BG_INPUT,
                     fg=self.GREEN).pack(pady=6)
            tk.Label(stat, text=label_text, font=("PingFang SC", 9),
                     bg=self.BG_INPUT, fg=self.TEXT_MUTE).pack(pady=(0, 6))

    def _build_start_button(self, parent):
        """开始追色主按钮"""
        frame = tk.Frame(parent, bg=self.BG_CARD)
        frame.pack(fill="x", pady=(0, 8))

        # 主按钮
        btn = tk.Frame(frame, bg=self.ORANGE, cursor="hand2")
        btn.pack(fill="x", padx=12, pady=12)

        def on_enter(e): btn.config(bg="#d97706")
        def on_leave(e): btn.config(bg=self.ORANGE)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        btn_inner = tk.Frame(btn, bg=self.ORANGE)
        btn_inner.pack(fill="x", ipady=8)
        btn_inner.bind("<Button-1>", lambda e: self.start_color_transfer())

        # 播放图标（三角形）
        icon_canvas = tk.Canvas(btn_inner, width=24, height=24, bg=self.ORANGE, bd=0, highlightthickness=0)
        icon_canvas.pack(side="left", padx=(12, 8))
        icon_canvas.create_polygon(7, 5, 7, 19, 19, 12, fill=self.TEXT_MAIN, outline="")

        tk.Label(btn_inner, text="开始追色", font=("PingFang SC", 13, "bold"),
                 bg=self.ORANGE, fg=self.TEXT_MAIN).pack(side="left")

        # 快捷键提示
        tk.Label(frame, text="快捷键: ⌘ + Enter", font=("PingFang SC", 9),
                 bg=self.BG_CARD, fg=self.TEXT_MUTE).pack(pady=(0, 12))

        # 保存引用
        self.start_btn = btn

    def _build_export_section(self, parent):
        """导出选项"""
        frame = tk.Frame(parent, bg=self.BG_CARD)
        frame.pack(fill="x")

        tk.Label(frame, text="导出", font=("PingFang SC", 12, "bold"),
                 bg=self.BG_CARD, fg=self.TEXT_MAIN).pack(anchor="w", padx=12, pady=(10, 8))

        btn_row = tk.Frame(frame, bg=self.BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=(0, 10))

        for label, cmd, is_primary in [
            ("图片", self.save_result, True),
            ("LUT", self.export_lut, False),
        ]:
            bg = self.ACCENT if is_primary else self.BG_INPUT
            fg = self.TEXT_MAIN if is_primary else self.TEXT_MUTE
            b = tk.Button(btn_row, text=label, command=cmd, bg=bg, fg=fg,
                          font=("PingFang SC", 11), bd=0, padx=10, pady=7,
                          cursor="hand2")
            b.pack(side="left", fill="x", expand=True, padx=(0, 6))
            if is_primary:
                def on_enter(e, b=b): b.config(bg=self.BTN_HOVER)
                def on_leave(e, b=b): b.config(bg=self.ACCENT)
            else:
                def on_enter(e, b=b): b.config(bg=self.BORDER)
                def on_leave(e, b=b): b.config(bg=self.BG_INPUT)
            b.bind("<Enter>", on_enter)
            b.bind("<Leave>", on_leave)

    # ──────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────

    def _make_placeholder(self, label, text):
        """显示占位符"""
        label.config(image="", text=text, fg=self.TEXT_HINT,
                     font=("PingFang SC", 11), compound="center")
        label.image = None

    def _show_preview(self, key: str, arr: np.ndarray, max_h: int = 280):
        """显示预览图"""
        img_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8, mode='RGB')

        w, h = pil_img.size
        target_h = max_h
        scale = target_h / h if h > target_h else 1.0
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        canvas_w = self.preview_labels[key].winfo_width() or 400
        canvas_h = self.preview_labels[key].winfo_height() or max_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), "#0a0a0b")
        x = (canvas_w - pil_img.width) // 2
        y = (canvas_h - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))

        tk_img = ImageTk.PhotoImage(canvas)
        lbl = self.preview_labels[key]
        lbl.config(image=tk_img, text="", compound="center")
        lbl.image = tk_img

    def _show_preview_raw(self, key: str, pil_img: Image.Image):
        """直接显示PIL图片"""
        canvas_w = self.preview_labels[key].winfo_width() or 400
        canvas_h = self.preview_labels[key].winfo_height() or 280

        w, h = pil_img.size
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (canvas_w, canvas_h), "#0a0a0b")
        x = (canvas_w - pil_img.width) // 2
        y = (canvas_h - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))

        tk_img = ImageTk.PhotoImage(canvas)
        lbl = self.preview_labels[key]
        lbl.config(image=tk_img, text="", compound="center")
        lbl.image = tk_img

    # ──────────────────────────────────────────
    #  图片加载
    # ──────────────────────────────────────────

    def load_reference(self):
        """加载参考图"""
        filepath = filedialog.askopenfilename(
            title="选择参考图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if not filepath:
            return
        try:
            img = load_image(filepath)
            self.reference_path = filepath
            self.reference_img = img
            self._show_preview("ref", img)

            if self.mode.get() == "pro":
                self.ref_features = self.feature_extractor.extract(img)
                self.status_var.set(f"✓ 参考图已加载（专业模式）— {os.path.basename(filepath)}")
            else:
                self.status_var.set(f"✓ 参考图已加载（快速模式）— {os.path.basename(filepath)}")

            if self.target_imgs and self.auto_preview.get():
                self._run_transfer_async()

        except Exception as e:
            import traceback
            messagebox.showerror("错误", f"加载失败: {str(e)}\n{traceback.format_exc()}")

    def load_targets(self):
        """加载目标图"""
        filepaths = filedialog.askopenfilenames(
            title="选择原图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if filepaths:
            self._load_batch(filepaths)

    def load_batch(self):
        """批量导入"""
        dirpath = filedialog.askdirectory(title="选择文件夹")
        if dirpath:
            exts = ["jpg", "jpeg", "png", "bmp", "tiff"]
            filepaths = []
            for ext in exts:
                filepaths.extend(Path(dirpath).glob(f"*.{ext}"))
                filepaths.extend(Path(dirpath).glob(f"*.{ext.upper()}"))
            if filepaths:
                self._load_batch([str(f) for f in filepaths])

    def _load_batch(self, filepaths):
        """处理批量文件"""
        self.target_imgs = []
        for fp in filepaths:
            try:
                img = load_image(fp)
                self.target_imgs.append((fp, img))
            except:
                pass

        if self.target_imgs:
            self.current_index = 0
            self._update_target_preview()
            count = len(self.target_imgs)
            # 更新图片信息
            if self.target_imgs:
                _, img = self.target_imgs[self.current_index]
                h, w = img.shape[:2]
                self.img_info_label.config(text=f"图像: {w}×{h}")

            self.status_var.set(f"已加载 {count} 张图片 — 点击「开始追色」")

    def _update_target_preview(self):
        """更新目标图预览"""
        if self.target_imgs:
            _, img = self.target_imgs[self.current_index]
            self._show_preview("src", img)
            if self.img_info_label:
                h, w = img.shape[:2]
                self.img_info_label.config(text=f"图像: {w}×{h}")
        else:
            self._make_placeholder(self.preview_labels.get("src", tk.Label()), "点击上传原图")
            self.img_info_label.config(text="")

    def prev_image(self):
        """上一张"""
        if self.target_imgs and self.current_index > 0:
            self.current_index -= 1
            self._update_target_preview()
            self._make_placeholder(self.preview_labels.get("result", tk.Label()), "等待追色")
            self.current_result = None

    def next_image(self):
        """下一张"""
        if self.target_imgs and self.current_index < len(self.target_imgs) - 1:
            self.current_index += 1
            self._update_target_preview()
            self._make_placeholder(self.preview_labels.get("result", tk.Label()), "等待追色")
            self.current_result = None

    def browse_output(self):
        """选择输出目录"""
        dirpath = filedialog.askdirectory(title="选择输出目录")
        if dirpath:
            self.output_dir = dirpath

    # ──────────────────────────────────────────
    #  追色处理
    # ──────────────────────────────────────────

    def start_color_transfer(self):
        """开始追色"""
        if self.reference_img is None:
            messagebox.showwarning("提示", "请先导入参考图")
            return
        if not self.target_imgs:
            messagebox.showwarning("提示", "请先导入原图")
            return

        self._run_transfer_async()

    def _run_transfer_async(self):
        """异步执行追色"""
        thread = threading.Thread(target=self._color_transfer_thread, daemon=True)
        thread.start()

    def _color_transfer_thread(self):
        """追色线程"""
        import time
        t0 = time.time()

        try:
            progress = ProgressWindow(self.root, "追色处理中")
            _, target_img = self.target_imgs[self.current_index]

            if self.mode.get() == "pro" and self.ref_features is not None:
                # 专业模式
                progress.update_text("分析原图...", 20)
                src_features = self.feature_extractor.extract(target_img)

                progress.update_text("生成追色方案...", 40)
                params = self.grading_generator.generate(self.ref_features, src_features)
                params = params.apply_strength(0.8)

                progress.update_text("渲染调色结果...", 70)
                result = self.color_renderer.render(target_img, params, self.ref_features)
            else:
                # 快速模式
                progress.update_text("分析参考图...", 20)
                transfer = ColorTransfer()
                transfer.analyze(self.reference_img)

                progress.update_text("执行色彩迁移...", 60)
                result = transfer.transfer(
                    target_img,
                    tone_strength=self.tone_strength.get(),
                    color_strength=self.color_strength.get(),
                    skin_protect=self.skin_strength.get(),
                    use_skin_protect=self.skin_protection.get()
                )

            elapsed_ms = int((time.time() - t0) * 1000)
            self.current_result = result

            def update_ui():
                self._show_preview("result", result)
                # 计算评价指标
                try:
                    metrics = Evaluator.compute_metrics(target_img, result)
                    self.metric_delta_e.set(str(metrics["delta_e"]))
                    skin_text = "安全" if metrics["skin_safe"] else "注意"
                    skin_color = self.GREEN if metrics["skin_safe"] else self.ORANGE
                    self.metric_skin.set(skin_text)
                    self.metric_speed.set(f"{elapsed_ms}ms")
                    self.metric_coverage.set(f"{metrics['coverage']}%")
                except Exception:
                    self.metric_delta_e.set("—")
                    self.metric_skin.set("—")
                    self.metric_speed.set(f"{elapsed_ms}ms")
                    self.metric_coverage.set("—")
                # 更新结果图标签
                if hasattr(self, "result_tag"):
                    self.result_tag.config(text="追色完成")
                self.status_var.set(f"✓ 追色完成（{elapsed_ms}ms）")

            self.root.after(0, update_ui)
            progress.close()

        except Exception as e:
            import traceback
            self.root.after(0, lambda: messagebox.showerror("错误",
                f"处理失败: {str(e)}\n{traceback.format_exc()}"))

    # ──────────────────────────────────────────
    #  导出
    # ──────────────────────────────────────────

    def export_lut(self):
        """导出LUT"""
        if self.reference_img is None:
            messagebox.showwarning("提示", "请先导入参考图")
            return

        filepath = filedialog.asksaveasfilename(
            title="保存LUT文件",
            defaultextension=".cube",
            initialdir=self.output_dir,
            initialfile="ColorMatch_LUT",
            filetypes=[("Cube LUT文件", "*.cube"), ("所有文件", "*.*")]
        )

        if filepath:
            thread = threading.Thread(target=self._export_lut_thread, args=(filepath,), daemon=True)
            thread.start()

    def _export_lut_thread(self, filepath: str):
        """LUT导出线程"""
        try:
            progress = ProgressWindow(self.root, "生成LUT")
            transfer = ColorTransfer()
            transfer.analyze(self.reference_img)

            from color_engine import LUTGenerator
            lut_val = self.lut_combo.get() if hasattr(self, "lut_combo") else "33"
            if "17" in lut_val:
                size = 17
            elif "65" in lut_val:
                size = 65
            else:
                size = 33

            generator = LUTGenerator(lut_size=size)
            progress.update_text("生成LUT中...", 50)
            generator.generate(
                transfer,
                tone_strength=self.tone_strength.get(),
                color_strength=self.color_strength.get(),
                output_path=filepath,
                title=Path(filepath).stem
            )

            self.root.after(0, lambda: self.status_var.set(f"✓ LUT已保存: {filepath}"))
            progress.close()
            self.root.after(0, lambda: messagebox.showinfo("成功", f"LUT已导出:\n{filepath}"))

        except Exception as e:
            import traceback
            self.root.after(0, lambda: messagebox.showerror("错误",
                f"导出失败: {str(e)}\n{traceback.format_exc()}"))

    def save_result(self):
        """保存结果"""
        if self.current_result is None:
            messagebox.showwarning("提示", "请先执行追色处理")
            return

        filepath = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            initialdir=self.output_dir,
            initialfile=f"ColorMatched_{self.current_index+1:03d}",
            filetypes=[("JPEG图片", "*.jpg *.jpeg"), ("PNG图片", "*.png"), ("所有文件", "*.*")]
        )

        if filepath:
            try:
                save_image(self.current_result, filepath)
                self.status_var.set(f"✓ 已保存: {filepath}")
                messagebox.showinfo("成功", f"图片已保存:\n{filepath}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def batch_process(self):
        """批量处理"""
        if self.reference_img is None:
            messagebox.showwarning("提示", "请先导入参考图")
            return
        if not self.target_imgs:
            messagebox.showwarning("提示", "请先导入原图")
            return

        batch_dir = os.path.join(self.output_dir, "追色结果")
        os.makedirs(batch_dir, exist_ok=True)

        thread = threading.Thread(target=self._batch_process_thread, args=(batch_dir,), daemon=True)
        thread.start()

    def _batch_process_thread(self, output_dir: str):
        """批量处理线程"""
        try:
            progress = ProgressWindow(self.root, "批量处理中")
            total = len(self.target_imgs)

            transfer = ColorTransfer()
            transfer.analyze(self.reference_img)

            for i, (path, target_img) in enumerate(self.target_imgs):
                progress.update_text(f"处理第 {i+1}/{total} 张...", int((i/total)*100))
                result = transfer.transfer(
                    target_img,
                    tone_strength=self.tone_strength.get(),
                    color_strength=self.color_strength.get(),
                    skin_protect=self.skin_strength.get(),
                    use_skin_protect=self.skin_protection.get()
                )
                output_path = os.path.join(output_dir, f"ColorMatched_{i+1:03d}.jpg")
                save_image(result, output_path)

            progress.update_text("处理完成！", 100)
            self.root.after(0, lambda: self.status_var.set(f"✓ 批量处理完成: {total} 张图片"))
            progress.close()
            self.root.after(0, lambda: messagebox.showinfo("完成",
                f"批量处理完成！\n\n已处理 {total} 张图片\n保存位置: {output_dir}"))

        except Exception as e:
            import traceback
            self.root.after(0, lambda: messagebox.showerror("错误",
                f"处理失败: {str(e)}\n{traceback.format_exc()}"))

    def _reset_params(self):
        """重置参数"""
        self.tone_strength.set(0.8)
        self.color_strength.set(0.9)
        self.skin_protection.set(True)
        self.skin_strength.set(0.80)
        self.status_var.set("参数已重置")


def main():
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.2)
    app = ZsTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
