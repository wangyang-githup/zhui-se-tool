"""
追色工具 — 主界面 (app.py) v2.0
=====================================
CustomTkinter 紫色暗色主题
统一引擎 (ZhuiseEngine) — 简化模式 + 专家模式 + 风格预设

使用方法: python3 app.py
"""

import os
import sys
import queue
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image as PILImage, ImageTk

# 确保当前目录在搜索路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_engine import (
    ZhuiseEngine, load_image, save_image, BUILTIN_PRESETS,
    ColorGradingParams, ToneParams,
)


# ══════════════════════════════════════════════════════════
#  主题配色 (Catppuccin Mocha)
# ══════════════════════════════════════════════════════════

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

C = {
    "bg":        "#1e1e2e",
    "surface":   "#313244",
    "overlay":   "#45475a",
    "text":      "#cdd6f4",
    "subtext":   "#6c7086",
    "accent":    "#cba6f7",
    "accent2":   "#b4befe",
    "green":     "#a6e3a1",
    "red":       "#f38ba8",
    "blue":      "#89b4fa",
    "yellow":    "#f9e2af",
    "peach":     "#fab387",
    "card":      "#242438",
    "border":    "#45475a",
}


# ══════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════

def np_to_ctk_image(arr, max_w=380, max_h=280):
    """float32 RGB [0,1] → CTkImage 预览"""
    u8 = (arr.clip(0, 1) * 255).astype('uint8')
    pil = PILImage.fromarray(u8, mode="RGB")
    w, h = pil.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)


# ══════════════════════════════════════════════════════════
#  主应用
# ══════════════════════════════════════════════════════════

class App(ctk.CTk):
    PREVIEW_W = 380
    PREVIEW_H = 280

    def __init__(self):
        super().__init__()
        self.title("追色工具  ✦  Zhuise Color Matcher")
        self.configure(fg_color=C["bg"])
        self.resizable(True, True)
        self.minsize(1200, 750)
        self.geometry("1350x800")

        # 引擎
        self.engine = ZhuiseEngine()

        # 状态
        self.ref_img_np = None
        self.src_img_np = None
        self.result_np = None
        self.ref_path = ""
        self.src_path = ""
        self._processing = False
        self._lock = threading.Lock()
        self._work_queue = queue.Queue()

        # 构建UI
        self._build_ui()

    # ──────────────────────────────────────────
    #  UI 构建
    # ──────────────────────────────────────────

    def _build_ui(self):
        # ── 顶栏 ──
        header = ctk.CTkFrame(self, fg_color=C["bg"], height=50)
        header.pack(fill="x", padx=20, pady=(14, 4))
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="✦ 追色工具", font=ctk.CTkFont(family="PingFang SC", size=22, weight="bold"),
            text_color=C["accent"]
        ).pack(side="left")

        ctk.CTkLabel(
            header, text="Zhuise Color Matcher  |  Lab 色彩迁移 + 肤色保护 + 专家管线",
            font=ctk.CTkFont(family="PingFang SC", size=11),
            text_color=C["subtext"]
        ).pack(side="left", padx=14)

        # 模式切换
        self._mode_var = ctk.StringVar(value="expert")
        mode_frame = ctk.CTkFrame(header, fg_color="transparent")
        mode_frame.pack(side="right")
        ctk.CTkSegmentedButton(
            mode_frame, values=["简化", "专家"],
            variable=self._mode_var, command=self._on_mode_change,
            font=ctk.CTkFont(family="PingFang SC", size=11),
            selected_color=C["accent"], selected_hover_color=C["accent2"],
            fg_color=C["surface"], unselected_color=C["surface"],
        ).pack()

        # ── 主区域 ──
        main = ctk.CTkFrame(self, fg_color=C["bg"])
        main.pack(fill="both", expand=True, padx=20, pady=4)

        # 左栏：图片预览
        left = ctk.CTkFrame(main, fg_color=C["bg"])
        left.pack(side="left", fill="both", expand=True)
        self._build_preview_panel(left)

        # 右栏：参数控制
        right = ctk.CTkFrame(main, fg_color=C["bg"], width=320)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_control_panel(right)

        # ── 底部状态栏 ──
        self._status_var = ctk.StringVar(value="就绪 — 请导入参考图和原图")
        status_bar = ctk.CTkFrame(self, fg_color="#181825", height=32)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        ctk.CTkLabel(
            status_bar, textvariable=self._status_var,
            font=ctk.CTkFont(family="PingFang SC", size=10),
            text_color=C["subtext"], anchor="w"
        ).pack(fill="x", padx=16, side="left")

    # ── 图片预览区 ──

    def _build_preview_panel(self, parent):
        """三格预览: 参考图 | 原图 | 结果图"""
        row = ctk.CTkFrame(parent, fg_color=C["bg"])
        row.pack(fill="both", expand=True)

        self._preview_labels = {}
        self._preview_cards = {}

        panels = [
            ("参考图  Reference", "ref"),
            ("原图  Source", "src"),
            ("结果图  Result", "result"),
        ]

        for title, key in panels:
            card = ctk.CTkFrame(
                row, fg_color=C["card"], corner_radius=8,
                border_width=1, border_color=C["border"]
            )
            card.pack(side="left", fill="both", expand=True, padx=5, pady=4)
            self._preview_cards[key] = card

            ctk.CTkLabel(
                card, text=title,
                font=ctk.CTkFont(family="PingFang SC", size=11, weight="bold"),
                text_color=C["text"]
            ).pack(pady=(8, 2))

            # 预览区
            preview = ctk.CTkLabel(
                card, text="", fg_color="#0f0f17",
                width=self.PREVIEW_W, height=self.PREVIEW_H,
                font=ctk.CTkFont(family="PingFang SC", size=10),
                text_color=C["overlay"], corner_radius=4
            )
            preview.pack(padx=8, pady=(0, 6), fill="both", expand=True)
            self._preview_labels[key] = preview

            if key == "ref":
                preview.bind("<Button-1>", lambda e: self._load_ref())
                self._set_placeholder(key, "点击导入参考图")
            elif key == "src":
                preview.bind("<Button-1>", lambda e: self._load_src())
                self._set_placeholder(key, "点击导入原图")
            else:
                self._set_placeholder(key, "追色结果将显示在这里")

        # 按钮行
        btn_row = ctk.CTkFrame(parent, fg_color=C["bg"])
        btn_row.pack(fill="x", padx=5, pady=(0, 6))

        ctk.CTkButton(
            btn_row, text="↓ 保存结果图", command=self._save_result,
            fg_color=C["green"], hover_color="#8dd88a", text_color="#1e1e2e",
            font=ctk.CTkFont(family="PingFang SC", size=11, weight="bold"),
            width=140, corner_radius=6
        ).pack(side="right", padx=5)

        ctk.CTkButton(
            btn_row, text="↓ 导出 .cube LUT", command=self._export_lut,
            fg_color=C["blue"], hover_color="#7aa8e8", text_color="#1e1e2e",
            font=ctk.CTkFont(family="PingFang SC", size=11, weight="bold"),
            width=160, corner_radius=6
        ).pack(side="right", padx=5)

        # 评价指标
        self._eval_label = ctk.CTkLabel(
            parent, text="", font=ctk.CTkFont(family="PingFang SC", size=10),
            text_color=C["subtext"], anchor="w"
        )
        self._eval_label.pack(fill="x", padx=5, pady=(0, 4))

    # ── 参数控制区 ──

    def _build_control_panel(self, parent):
        """右侧标签页参数面板"""
        self._tabview = ctk.CTkTabview(parent, fg_color=C["bg"], corner_radius=8,
                                        segmented_button_fg_color=C["surface"],
                                        segmented_button_selected_color=C["accent"],
                                        segmented_button_unselected_color=C["surface"])
        self._tabview.pack(fill="both", expand=True)

        # 三个标签页
        tab_basic = self._tabview.add("基础")
        tab_expert = self._tabview.add("进阶")
        tab_style = self._tabview.add("风格")

        self._build_basic_tab(tab_basic)
        self._build_expert_tab(tab_expert)
        self._build_style_tab(tab_style)

    def _build_basic_tab(self, tab):
        """基础参数: 影调/色调强度 + 肤色保护"""
        self._section(tab, "⚙  追色参数")

        self._tone_var = ctk.DoubleVar(value=0.8)
        self._color_var = ctk.DoubleVar(value=0.9)
        self._slider_row(tab, "影调强度  Tone", self._tone_var, 0, 1.0)
        self._slider_row(tab, "色调强度  Color", self._color_var, 0, 1.0)

        self._section(tab, "👤  肤色保护")
        self._skin_on_var = ctk.BooleanVar(value=True)
        self._skin_var = ctk.DoubleVar(value=0.85)
        ctk.CTkCheckBox(
            tab, text="启用肤色保护", variable=self._skin_on_var,
            font=ctk.CTkFont(family="PingFang SC", size=11),
            text_color=C["text"], fg_color=C["accent"], hover_color=C["accent2"],
            command=self._on_param_change
        ).pack(anchor="w", pady=2)
        self._slider_row(tab, "保护强度  Protect", self._skin_var, 0, 1.0)

        ctk.CTkLabel(
            tab, text="检测肤色区域，匹配色调时保留\n人物皮肤原色（HSV+Lab双空间检测）",
            font=ctk.CTkFont(family="PingFang SC", size=9),
            text_color=C["subtext"], justify="left"
        ).pack(anchor="w", pady=(0, 4))

        # 全局强度
        self._section(tab, "🎚  全局强度")
        self._strength_var = ctk.DoubleVar(value=1.0)
        self._slider_row(tab, "强度  Strength", self._strength_var, 0, 1.0)

        # 操作按钮
        ctk.CTkFrame(tab, fg_color=C["border"], height=1).pack(fill="x", pady=10)
        ctk.CTkButton(
            tab, text="▶  开始追色", command=self._run_transfer,
            fg_color=C["accent"], hover_color=C["accent2"], text_color="#1e1e2e",
            font=ctk.CTkFont(family="PingFang SC", size=14, weight="bold"),
            height=40, corner_radius=8
        ).pack(fill="x", pady=3)

        ctk.CTkButton(
            tab, text="重置参数", command=self._reset_params,
            fg_color=C["surface"], hover_color=C["overlay"], text_color=C["text"],
            font=ctk.CTkFont(family="PingFang SC", size=11),
            corner_radius=6
        ).pack(fill="x", pady=2)

    def _build_expert_tab(self, tab):
        """进阶参数: Lightroom 风格面板"""
        # ── 影调 ──
        self._section(tab, "📐  影调 (基础面板)")
        self._exposure_var = ctk.DoubleVar(value=0.0)
        self._contrast_var = ctk.DoubleVar(value=0.0)
        self._highlights_var = ctk.DoubleVar(value=0.0)
        self._shadows_var = ctk.DoubleVar(value=0.0)
        self._whites_var = ctk.DoubleVar(value=0.0)
        self._blacks_var = ctk.DoubleVar(value=0.0)
        self._slider_row(tab, "曝光  Exposure", self._exposure_var, -3.0, 3.0, fmt="{:+.2f} EV")
        self._slider_row(tab, "对比度  Contrast", self._contrast_var, -100, 100, fmt="{:+.0f}")
        self._slider_row(tab, "高光  Highlights", self._highlights_var, -100, 100, fmt="{:+.0f}")
        self._slider_row(tab, "阴影  Shadows", self._shadows_var, -100, 100, fmt="{:+.0f}")
        self._slider_row(tab, "白色  Whites", self._whites_var, -100, 100, fmt="{:+.0f}")
        self._slider_row(tab, "黑色  Blacks", self._blacks_var, -100, 100, fmt="{:+.0f}")

        # ── 分离色调 ──
        self._section(tab, "🎨  分离色调")
        self._hi_hue_var = ctk.DoubleVar(value=0.0)
        self._hi_sat_var = ctk.DoubleVar(value=0.0)
        self._sh_hue_var = ctk.DoubleVar(value=0.0)
        self._sh_sat_var = ctk.DoubleVar(value=0.0)
        self._slider_row(tab, "高光色相  Hi-Hue", self._hi_hue_var, 0, 360, fmt="{:.0f}°")
        self._slider_row(tab, "高光饱和  Hi-Sat", self._hi_sat_var, 0, 100, fmt="{:.0f}")
        self._slider_row(tab, "阴影色相  Sh-Hue", self._sh_hue_var, 0, 360, fmt="{:.0f}°")
        self._slider_row(tab, "阴影饱和  Sh-Sat", self._sh_sat_var, 0, 100, fmt="{:.0f}")

        # ── 白平衡 ──
        self._section(tab, "🌡  白平衡校准")
        self._temp_var = ctk.DoubleVar(value=0.0)
        self._tint_var = ctk.DoubleVar(value=0.0)
        self._slider_row(tab, "色温  Temp", self._temp_var, -100, 100, fmt="{:+.0f}")
        self._slider_row(tab, "色调  Tint", self._tint_var, -100, 100, fmt="{:+.0f}")

        # ── 自动分析按钮 ──
        ctk.CTkFrame(tab, fg_color=C["border"], height=1).pack(fill="x", pady=8)
        ctk.CTkButton(
            tab, text="🔍 自动分析差异", command=self._auto_analyze,
            fg_color=C["blue"], hover_color="#7aa8e8", text_color="#1e1e2e",
            font=ctk.CTkFont(family="PingFang SC", size=11, weight="bold"),
            corner_radius=6
        ).pack(fill="x", pady=2)

    def _build_style_tab(self, tab):
        """风格预设"""
        self._section(tab, "🎬  内置风格预设")

        preset_info = {
            "日系清新": ("🌿 滨田英明风格：高亮度低对比，微冷偏青白", C["green"]),
            "暗调电影": ("🎬 低调高反差，冷蓝暗调", C["blue"]),
            "复古胶片": ("📷 提黑场复古感，高光偏橙阴影偏绿", C["yellow"]),
            "青橙色调": ("🎭 电影级青橙对比色调", C["peach"]),
            "Kodak Portra": ("🎞 经典人像胶片色调，肤色通透", C["accent"]),
        }

        for name, (desc, color) in preset_info.items():
            frame = ctk.CTkFrame(tab, fg_color=C["card"], corner_radius=6,
                                  border_width=1, border_color=C["border"])
            frame.pack(fill="x", pady=3, padx=4)

            ctk.CTkLabel(
                frame, text=name,
                font=ctk.CTkFont(family="PingFang SC", size=12, weight="bold"),
                text_color=color, anchor="w"
            ).pack(fill="x", padx=10, pady=(8, 0))

            ctk.CTkLabel(
                frame, text=desc,
                font=ctk.CTkFont(family="PingFang SC", size=10),
                text_color=C["subtext"], anchor="w"
            ).pack(fill="x", padx=10, pady=(2, 4))

            ctk.CTkButton(
                frame, text="应用", width=60, height=28,
                command=lambda n=name: self._apply_preset(n),
                fg_color=color, hover_color=C["overlay"], text_color="#1e1e2e",
                font=ctk.CTkFont(family="PingFang SC", size=10, weight="bold"),
                corner_radius=4
            ).pack(anchor="e", padx=10, pady=(0, 8))

        # ── 参考图特征显示 ──
        self._section(tab, "📊  参考图特征")
        self._ref_info_label = ctk.CTkLabel(
            tab, text="未加载参考图",
            font=ctk.CTkFont(family="PingFang SC", size=10),
            text_color=C["subtext"], justify="left", anchor="w"
        )
        self._ref_info_label.pack(fill="x", padx=4, pady=2)

    # ──────────────────────────────────────────
    #  UI 辅助方法
    # ──────────────────────────────────────────

    def _section(self, parent, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=(12, 2))
        ctk.CTkLabel(
            f, text=text,
            font=ctk.CTkFont(family="PingFang SC", size=12, weight="bold"),
            text_color=C["accent"]
        ).pack(anchor="w")

    def _slider_row(self, parent, label, var, from_, to, fmt="{:.0%}"):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=2)

        top = ctk.CTkFrame(f, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(
            top, text=label,
            font=ctk.CTkFont(family="PingFang SC", size=10),
            text_color=C["text"]
        ).pack(side="left")
        val_lbl = ctk.CTkLabel(
            top, text=fmt.format(var.get()),
            font=ctk.CTkFont(family="PingFang SC", size=10, weight="bold"),
            text_color=C["accent"], width=60
        )
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(
            f, from_=from_, to=to, variable=var,
            command=lambda v, l=val_lbl, fm=fmt, va=var: (
                l.configure(text=fm.format(va.get())),
                self._on_param_change()
            ),
            button_color=C["accent"], button_hover_color=C["accent2"],
            progress_color=C["accent"], fg_color=C["surface"],
            height=16
        )
        slider.pack(fill="x", pady=(2, 0))

    def _set_placeholder(self, key, text):
        lbl = self._preview_labels[key]
        lbl.configure(text=text, image=None)
        lbl._ctk_img = None

    def _show_preview(self, key, arr):
        ctk_img = np_to_ctk_image(arr, self.PREVIEW_W, self.PREVIEW_H)
        lbl = self._preview_labels[key]
        lbl.configure(image=ctk_img, text="")
        lbl._ctk_img = ctk_img  # 防止 GC

    def _set_status(self, msg):
        self._status_var.set(msg)

    # ──────────────────────────────────────────
    #  图片加载
    # ──────────────────────────────────────────

    def _load_ref(self):
        path = filedialog.askopenfilename(
            title="选择参考图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self.ref_img_np = load_image(path)
            self.engine.load_reference(self.ref_img_np)
            self._show_preview("ref", self.ref_img_np)
            self.ref_path = os.path.basename(path)
            self._set_status(f"✓ 参考图已加载：{self.ref_path}  |  已分析色彩特征")
            self._update_ref_info()
            if self.src_img_np is not None:
                self._run_transfer_async()
        except Exception as e:
            messagebox.showerror("错误", f"加载参考图失败：{e}")

    def _load_src(self):
        path = filedialog.askopenfilename(
            title="选择原图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            self.src_img_np = load_image(path)
            self._show_preview("src", self.src_img_np)
            self.src_path = os.path.basename(path)
            self._set_status(f"✓ 原图已加载：{self.src_path}")
            if self.ref_img_np is not None:
                self._run_transfer_async()
        except Exception as e:
            messagebox.showerror("错误", f"加载原图失败：{e}")

    def _update_ref_info(self):
        """更新参考图特征显示"""
        feat_dict = self.engine.get_ref_features_dict()
        if not feat_dict:
            return
        lab = feat_dict.get("lab", {})
        skin = feat_dict.get("skin", {})
        tone = feat_dict.get("tone", {})
        info = (
            f"Lab: L={lab.get('mean_L', '?'):.1f} a={lab.get('mean_a', '?'):.1f} b={lab.get('mean_b', '?'):.1f}\n"
            f"影调: p50={tone.get('p50', '?'):.1f} std={tone.get('std', '?'):.1f}\n"
            f"肤色: {'✓ 检测到' if skin.get('detected') else '✗ 未检测到'}"
            f" ({skin.get('mask_ratio', 0):.1%})"
        )
        self._ref_info_label.configure(text=info)

    # ──────────────────────────────────────────
    #  追色执行
    # ──────────────────────────────────────────

    def _on_mode_change(self, value):
        """模式切换回调"""
        self._set_status(f"切换至{'简化' if value == '简化' else '专家'}模式")

    def _on_param_change(self):
        """参数变化回调 (可扩展为自动预览)"""
        pass

    def _get_mode(self):
        return "simple" if self._mode_var.get() == "简化" else "expert"

    def _check_ready(self):
        if self.ref_img_np is None:
            messagebox.showwarning("提示", "请先导入参考图")
            return False
        if self.src_img_np is None:
            messagebox.showwarning("提示", "请先导入原图")
            return False
        return True

    def _run_transfer(self):
        if not self._check_ready():
            return
        self._run_transfer_async()

    def _run_transfer_async(self):
        if self.ref_img_np is None or self.src_img_np is None:
            return
        with self._lock:
            if self._processing:
                return
            self._processing = True

        mode = self._get_mode()
        src_copy = self.src_img_np.copy()

        # 提前读取参数（主线程安全操作 Variable.get）
        if mode == "simple":
            params = {
                "mode": "simple",
                "tone_strength": self._tone_var.get(),
                "color_strength": self._color_var.get(),
                "skin_protect": self._skin_var.get(),
                "use_skin_protect": self._skin_on_var.get(),
                "strength": self._strength_var.get(),
            }
        else:
            grading = ColorGradingParams(
                tone=ToneParams(
                    exposure=self._exposure_var.get(),
                    contrast=self._contrast_var.get(),
                    highlights=self._highlights_var.get(),
                    shadows=self._shadows_var.get(),
                    whites=self._whites_var.get(),
                    blacks=self._blacks_var.get(),
                ),
                strength=self._strength_var.get(),
            )
            from color_engine import SplitToneParams, CalibrationParams
            grading.split_tone = SplitToneParams(
                highlight_hue=self._hi_hue_var.get(),
                highlight_sat=self._hi_sat_var.get(),
                shadow_hue=self._sh_hue_var.get(),
                shadow_sat=self._sh_sat_var.get(),
            )
            grading.calibration = CalibrationParams(
                temp=self._temp_var.get(),
                tint=self._tint_var.get(),
            )
            params = {
                "mode": "expert",
                "grading": grading,
                "strength": self._strength_var.get(),
            }

        self._work_queue.put(params)

        def work():
            self.after(0, lambda: self._set_status("⏳ 追色中..."))
            try:
                p = self._work_queue.get()

                if p["mode"] == "simple":
                    result = self.engine.render(
                        src_copy,
                        mode="simple",
                        tone_strength=p["tone_strength"],
                        color_strength=p["color_strength"],
                        skin_protect=p["skin_protect"],
                        use_skin_protect=p["use_skin_protect"],
                    )
                else:
                    result = self.engine.render(
                        src_copy,
                        mode="expert",
                        strength=p["strength"],
                        params=p["grading"],
                    )

                self.result_np = result
                eval_result = self.engine.evaluate(src_copy, result)

                def update_ui():
                    self._show_preview("result", result)
                    mode_text = "简化" if p["mode"] == "simple" else "专家"
                    self._set_status(f"✓ 追色完成 ({mode_text}模式)")
                    if "summary" in eval_result:
                        self._eval_label.configure(text=f"📊 {eval_result['summary']}")
                    self._reset_processing()

                self.after(0, update_ui)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"追色异常:\n{tb}")
                self.after(0, lambda: self._set_status(f"✗ 追色失败：{e}"))
                self.after(0, lambda: messagebox.showerror("追色失败", f"{e}\n\n{tb}"))
                self.after(0, self._reset_processing)

        threading.Thread(target=work, daemon=True).start()

    def _reset_processing(self):
        with self._lock:
            self._processing = False

    def _auto_analyze(self):
        """自动分析差异并填充专家参数（异步）"""
        if not self._check_ready():
            return

        self._set_status("⏳ 分析中...")

        def work():
            try:
                from color_engine import FeatureExtractor, ColorGradingGenerator
                extractor = FeatureExtractor()
                generator = ColorGradingGenerator()
                ref_feat = self.engine.ref_features
                src_feat = extractor.extract(self.src_img_np)
                params = generator.generate(ref_feat, src_feat, strength=self._strength_var.get())

                def update_ui():
                    self._exposure_var.set(params.tone.exposure)
                    self._contrast_var.set(params.tone.contrast)
                    self._highlights_var.set(params.tone.highlights)
                    self._shadows_var.set(params.tone.shadows)
                    self._whites_var.set(params.tone.whites)
                    self._blacks_var.set(params.tone.blacks)
                    self._hi_hue_var.set(params.split_tone.highlight_hue)
                    self._hi_sat_var.set(params.split_tone.highlight_sat)
                    self._sh_hue_var.set(params.split_tone.shadow_hue)
                    self._sh_sat_var.set(params.split_tone.shadow_sat)
                    self._temp_var.set(params.calibration.temp)
                    self._tint_var.set(params.calibration.tint)
                    self._mode_var.set("专家")
                    reasons = params.reasons
                    reason_text = "\n".join(f"• {v}" for v in reasons.values() if v)
                    self._set_status(f"✓ 自动分析完成，已填充参数\n{reason_text}")
                    self._run_transfer_async()

                self.after(0, update_ui)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("分析失败", f"{e}"))

        threading.Thread(target=work, daemon=True).start()

    def _apply_preset(self, name):
        """应用风格预设"""
        if not self._check_ready():
            return

        preset = BUILTIN_PRESETS.get(name)
        if not preset:
            return

        # 填充简化参数
        self._tone_var.set(preset.tone_strength)
        self._color_var.set(preset.color_strength)
        self._skin_var.set(preset.skin_protect)

        # 如果有专家参数，也填充
        if preset.expert_params:
            ep = preset.expert_params
            self._exposure_var.set(ep.tone.exposure)
            self._contrast_var.set(ep.tone.contrast)
            self._highlights_var.set(ep.tone.highlights)
            self._shadows_var.set(ep.tone.shadows)
            self._whites_var.set(ep.tone.whites)
            self._blacks_var.set(ep.tone.blacks)
            self._hi_hue_var.set(ep.split_tone.highlight_hue)
            self._hi_sat_var.set(ep.split_tone.highlight_sat)
            self._sh_hue_var.set(ep.split_tone.shadow_hue)
            self._sh_sat_var.set(ep.split_tone.shadow_sat)
            self._temp_var.set(ep.calibration.temp)
            self._tint_var.set(ep.calibration.tint)

        self._set_status(f"✓ 已应用预设：{name}")
        self._run_transfer_async()

    # ──────────────────────────────────────────
    #  保存 / 导出
    # ──────────────────────────────────────────

    def _save_result(self):
        if self.result_np is None:
            messagebox.showwarning("提示", "请先执行追色")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果图", defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")]
        )
        if not path:
            return
        try:
            save_image(self.result_np, path)
            self._set_status(f"✓ 结果已保存：{os.path.basename(path)}")
            messagebox.showinfo("保存成功", f"已保存到：\n{path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{e}")

    def _export_lut(self):
        if self.engine.ref_features is None:
            messagebox.showwarning("提示", "请先导入参考图以分析色彩特征")
            return
        path = filedialog.asksaveasfilename(
            title="导出 .cube LUT", defaultextension=".cube",
            filetypes=[("Cube LUT", "*.cube")]
        )
        if not path:
            return

        mode = self._get_mode()
        lut_size = 33  # 默认标准精度

        def work():
            self.after(0, lambda: self._set_status("⏳ 生成 LUT 中..."))
            try:
                title = os.path.splitext(os.path.basename(path))[0]
                self.engine.export_lut(
                    output_path=path, mode=mode,
                    tone_strength=self._tone_var.get(),
                    color_strength=self._color_var.get(),
                    lut_size=lut_size, title=title,
                )
                self.after(0, lambda: self._set_status(f"✓ LUT 已导出：{os.path.basename(path)}"))
                self.after(0, lambda: messagebox.showinfo(
                    "导出成功",
                    f"LUT 文件已导出：\n{path}\n\n"
                    f"可直接导入 Premiere Pro / After Effects / DaVinci Resolve 使用。\n"
                    f"精度：{lut_size}³"
                ))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"✗ LUT 导出失败：{e}"))

        threading.Thread(target=work, daemon=True).start()

    # ──────────────────────────────────────────
    #  其他
    # ──────────────────────────────────────────

    def _reset_params(self):
        self._tone_var.set(0.8)
        self._color_var.set(0.9)
        self._skin_var.set(0.85)
        self._skin_on_var.set(True)
        self._strength_var.set(1.0)
        self._exposure_var.set(0.0)
        self._contrast_var.set(0.0)
        self._highlights_var.set(0.0)
        self._shadows_var.set(0.0)
        self._whites_var.set(0.0)
        self._blacks_var.set(0.0)
        self._hi_hue_var.set(0.0)
        self._hi_sat_var.set(0.0)
        self._sh_hue_var.set(0.0)
        self._sh_sat_var.set(0.0)
        self._temp_var.set(0.0)
        self._tint_var.set(0.0)
        self._set_status("参数已重置")


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
