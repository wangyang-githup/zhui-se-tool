"""
追色工具 — CustomTkinter 现代界面
使用方法: python3 app.py
"""

import os
import sys
import threading
import customtkinter as ctk
from PIL import Image as PILImage

# ── 放在 import ctk 之后，设置全局外观 ──────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")   # 基础蓝色，我们用自定义色

# ── 现在导入其他模块 ──────────────────────────────────────
import numpy as np
from color_engine import ColorTransfer, LUTGenerator, load_image, save_image, rgb_to_lab

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def np_to_ctk(arr: np.ndarray, size: tuple = None) -> ctk.CTkImage:
    """float32 RGB [0,1] → CTkImage（支持自动缩放）"""
    u8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    pil = PILImage.fromarray(u8)
    if size:
        return ctk.CTkImage(pil, size=size)
    w, h = pil.size
    return ctk.CTkImage(pil, size=(w, h))


def fit_size(w: int, h: int, max_w: int, max_h: int) -> tuple:
    """计算适应 max_w×max_h 的缩放尺寸"""
    scale = min(max_w / w, max_h / h, 1.0)
    return int(w * scale), int(h * scale)


# ═══════════════════════════════════════════════════════
#  自定义组件
# ═══════════════════════════════════════════════════════

class ImageCard(ctk.CTkFrame):
    """
    图片预览卡片
    - 点击加载图片（ref/src）
    - 显示缩略图（result）
    """

    def __init__(self, parent, title: str, key: str,
                 placeholder: str, on_click=None):
        super().__init__(
            parent,
            fg_color="#2a2a3e",
            corner_radius=16,
            border_width=0,
        )
        self.key = key
        self.on_click = on_click
        self.img_np = None
        self._displayed = None

        # 标题
        self.title_lbl = ctk.CTkLabel(
            self, text=title,
            font=("PingFang SC", 12, "bold"),
            text_color="#cdd6f4",
        )
        self.title_lbl.pack(pady=(14, 6))

        # 图片区域（点击触发加载）
        self.canvas = ctk.CTkLabel(
            self, text="",
            width=220, height=150,
            fg_color="#14141e",
            corner_radius=12,
            font=("PingFang SC", 11),
            text_color="#6c7086",
            cursor="hand2" if on_click else "",
        )
        self.canvas.pack(padx=12, pady=(0, 8))
        self._set_placeholder(placeholder)
        if on_click:
            self.canvas.bind("<Button-1>", lambda _: on_click())

    def _set_placeholder(self, text: str):
        self.canvas.configure(text=f"\n\n{text}", image=None)
        self.canvas._image = None

    def show(self, arr: np.ndarray):
        """显示 numpy 图片"""
        self.img_np = arr
        w, h = arr.shape[1], arr.shape[0]
        fit_w, fit_h = fit_size(w, h, 220, 150)
        ctk_img = np_to_ctk(arr, size=(fit_w, fit_h))
        self.canvas.configure(text="", image=ctk_img)
        self.canvas._image = ctk_img   # 防止 GC

    def clear(self, placeholder: str = None):
        self.img_np = None
        if placeholder:
            self._set_placeholder(placeholder)


class ParamSlider(ctk.CTkFrame):
    """参数滑块：标签 + 数值 + 滑块，一行"""

    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, fmt: str = "{:.0%}", command=None):
        super().__init__(parent, fg_color="transparent")

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text=label,
                     font=("PingFang SC", 12),
                     text_color="#cdd6f4").pack(side="left")
        self.val_lbl = ctk.CTkLabel(row, text=fmt.format(default),
                                     font=("PingFang SC", 12, "bold"),
                                     text_color="#cba6f7", width=50)
        self.val_lbl.pack(side="right")

        self.var = ctk.DoubleVar(value=default)
        self.fmt = fmt

        slider = ctk.CTkSlider(
            self, from_=from_, to=to,
            variable=self.var,
            progress_color="#cba6f7",
            button_color="#cba6f7",
            button_hover_color="#b48ee8",
            fg_color="#313244",
            command=self._on_change,
        )
        slider.pack(fill="x", pady=(2, 0))
        self._command = command

    def _on_change(self, val):
        self.val_lbl.configure(text=self.fmt.format(val))
        if self._command:
            self._command(val)

    def get(self) -> float:
        return self.var.get()

    def set(self, val: float):
        self.var.set(val)
        self.val_lbl.configure(text=self.fmt.format(val))


# ═══════════════════════════════════════════════════════
#  主应用
# ═══════════════════════════════════════════════════════

class App(ctk.CTk):
    PREVIEW_W = 220
    PREVIEW_H = 150
    BG        = "#1e1e2e"
    CARD      = "#2a2a3e"
    ACCENT    = "#cba6f7"
    TEXT      = "#cdd6f4"
    MUTED     = "#6c7086"
    GREEN     = "#a6e3a1"
    RED       = "#f38ba8"

    def __init__(self):
        super().__init__()
        self.title("追色工具")
        self.geometry("1100 700")
        self.configure(fg_color=self.BG)
        self.resizable(True, True)

        self.engine   = ColorTransfer()
        self.lut_gen  = LUTGenerator(lut_size=33)

        self.ref_img_np   = None
        self.src_img_np   = None
        self.result_np    = None

        self._build_ui()

    # ──────────────────────────────────────────
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── 标题栏 ──────────────────────────────
        header = ctk.CTkFrame(self, fg_color=self.BG, height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        ctk.CTkLabel(
            header, text="✦", font=("PingFang SC", 22),
            text_color=self.ACCENT,
        ).pack(side="left", padx=(18, 4), pady=12)

        ctk.CTkLabel(
            header, text="追色工具", font=("PingFang SC", 20, "bold"),
            text_color=self.TEXT,
        ).pack(side="left", pady=12)

        ctk.CTkLabel(
            header,
            text="Portrait Color Matcher  ·  Lab 色彩迁移 + 肤色保护",
            font=("PingFang SC", 11), text_color=self.MUTED,
        ).pack(side="left", padx=14, pady=12)

        self.status_lbl = ctk.CTkLabel(
            header, text="就绪 — 请先导入参考图和原图",
            font=("PingFang SC", 11), text_color=self.MUTED,
        )
        self.status_lbl.pack(side="right", padx=20, pady=12)

        # ── 主区域 ──────────────────────────────
        main = ctk.CTkFrame(self, fg_color=self.BG)
        main.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.grid_rowconfigure(1, weight=1)

        # 左：图片预览
        left = ctk.CTkFrame(main, fg_color=self.BG)
        left.pack(side="left", fill="both", expand=True)

        # 三格预览
        cards = [
            ("参考图  Reference", "ref",   "点击导入参考图",   self._load_ref),
            ("原图  Source",      "src",   "点击导入原图",     self._load_src),
            ("结果图  Result",    "result","追色结果将显示在这里", None),
        ]
        self.cards = {}
        card_row = ctk.CTkFrame(left, fg_color=self.BG)
        card_row.pack(fill="both", expand=True, pady=(0, 12))
        for title, key, ph, cmd in cards:
            card = ImageCard(card_row, title, key, ph, on_click=cmd)
            card.pack(side="left", fill="both", expand=True, padx=6)
            self.cards[key] = card

        # 按钮行
        btn_row = ctk.CTkFrame(left, fg_color=self.BG, height=44)
        btn_row.pack(fill="x", pady=(0, 0))
        btn_row.pack_propagate(False)

        self._mkbtn(btn_row, "↻ 重置",        self._reset_params,
                    fg_color="#313244", text_color=self.TEXT,
                    w=120).pack(side="left", padx=4)

        self._mkbtn(btn_row, "↓ 保存结果图",   self._save_result,
                    fg_color="#45475a", text_color=self.TEXT,
                    w=130).pack(side="right", padx=4)

        self._mkbtn(btn_row, "⬇ 导出 .cube LUT", self._export_lut,
                    fg_color="#45475a", text_color=self.TEXT,
                    w=160).pack(side="right", padx=4)

        # 右：参数面板
        right = ctk.CTkFrame(main, fg_color=self.CARD, width=270, corner_radius=16)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        right_inner = ctk.CTkScrollableFrame(
            right, fg_color="transparent",
            scrollbar_button_color=self.CARD,
            scrollbar_fg_color="#45475a",
            width=270,
        )
        right_inner.pack(fill="both", expand=True, padx=12, pady=12)

        self._build_controls(right_inner)

        # ── 底部状态条 ──────────────────────────
        footer = ctk.CTkFrame(self, fg_color="#181825", height=28)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_propagate(False)
        ctk.CTkLabel(
            footer, text="支持 JPG · PNG · TIFF · BMP · WebP  |  © 追色工具",
            font=("PingFang SC", 10), text_color="#45475a",
        ).pack(pady=4)

    # ──────────────────────────────────────────
    def _build_controls(self, parent):

        def section(icon: str, title: str) -> ctk.CTkFrame:
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", pady=(16, 6))
            ctk.CTkLabel(f, text=f"{icon}  {title}",
                         font=("PingFang SC", 13, "bold"),
                         text_color=self.ACCENT).pack(anchor="w")
            return f

        def sep():
            ctk.CTkFrame(parent, height=1, fg_color="#313244",
                         corner_radius=1).pack(fill="x", pady=(8, 4))

        # ── 追色参数 ───────────────────────────
        section("⚙", "追色参数")

        self.tone_slider = ParamSlider(
            parent, "影调强度  Tone",
            0, 1, 0.7, "{:.0%}",
            command=lambda v: self._maybe_preview()
        )
        self.tone_slider.pack(fill="x", pady=4, ipady=4)

        self.color_slider = ParamSlider(
            parent, "色调强度  Color",
            0, 1, 0.9, "{:.0%}",
            command=lambda v: self._maybe_preview()
        )
        self.color_slider.pack(fill="x", pady=4, ipady=4)

        sep()

        # ── 肤色保护 ───────────────────────────
        section("👤", "肤色保护")

        self.skin_on = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            parent, text="启用肤色保护",
            variable=self.skin_on,
            onvalue=True, offvalue=False,
            progress_color=self.ACCENT,
            button_color=self.ACCENT,
            button_hover_color="#b48ee8",
            font=("PingFang SC", 12), text_color=self.TEXT,
            command=lambda: self._maybe_preview()
        ).pack(anchor="w", pady=(0, 4))

        self.skin_slider = ParamSlider(
            parent, "保护强度  Protect",
            0, 1, 0.85, "{:.0%}",
            command=lambda v: self._maybe_preview()
        )
        self.skin_slider.pack(fill="x", pady=4, ipady=4)

        ctk.CTkLabel(
            parent, text="肤色区域将保留原始色彩，\n防止追色导致肤色偏黄/青灰",
            font=("PingFang SC", 10), text_color=self.MUTED,
            justify="left", anchor="w",
        ).pack(anchor="w", pady=(0, 4))

        sep()

        # ── LUT 设置 ───────────────────────────
        section("💾", "LUT 设置")

        ctk.CTkLabel(parent, text="LUT 精度",
                     font=("PingFang SC", 11), text_color=self.TEXT,
                     anchor="w").pack(anchor="w")

        self.lut_var = ctk.StringVar(value="33")
        sizes = [("17  轻量", "17"), ("33  标准", "33"), ("65  高精度", "65")]
        for label, val in sizes:
            ctk.CTkRadioButton(
                parent, text=label, variable=self.lut_var, value=val,
                font=("PingFang SC", 11), text_color=self.TEXT,
                fg_color=self.ACCENT, hover_color="#b48ee8",
            ).pack(anchor="w", padx=12, pady=2)

        sep()

        # ── 核心操作按钮 ────────────────────────
        ctk.CTkButton(
            parent, text="▶  开始追色",
            command=self._run_transfer,
            height=48,
            corner_radius=12,
            font=("PingFang SC", 15, "bold"),
            fg_color=self.ACCENT, hover_color="#b48ee8",
            text_color="#1e1e2e",
        ).pack(fill="x", pady=(12, 4), ipady=4)

    def _mkbtn(self, parent, text, cmd, **kw):
        w  = kw.pop("w", None)
        fg = kw.pop("fg_color", "#313244")
        tc = kw.pop("text_color", self.TEXT)
        return ctk.CTkButton(
            parent, text=text, command=cmd,
            height=38, corner_radius=10,
            font=("PingFang SC", 12),
            fg_color=fg, hover_color="#45475a",
            text_color=tc, width=w,
            **kw
        )

    # ──────────────────────────────────────────
    #  事件
    # ──────────────────────────────────────────

    def _load_ref(self):
        path = ctk.filedialog.askopenfilename(
            title="选择参考图",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("所有", "*.*")],
        )
        if not path:
            return
        try:
            self.ref_img_np = load_image(path)
            self.engine.analyze(self.ref_img_np)
            self.cards["ref"].show(self.ref_img_np)
            self._set_status(f"✓ 参考图已加载：{os.path.basename(path)}")
            self._maybe_preview()
        except Exception as e:
            import traceback
            self._set_status(f"✗ 加载失败：{e}")
            print(traceback.format_exc())

    def _load_src(self):
        path = ctk.filedialog.askopenfilename(
            title="选择原图",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"), ("所有", "*.*")],
        )
        if not path:
            return
        try:
            self.src_img_np = load_image(path)
            self.cards["src"].show(self.src_img_np)
            self._set_status(f"✓ 原图已加载：{os.path.basename(path)}")
            self._maybe_preview()
        except Exception as e:
            import traceback
            self._set_status(f"✗ 加载失败：{e}")
            print(traceback.format_exc())

    def _maybe_preview(self):
        if self.ref_img_np is not None and self.src_img_np is not None:
            self._run_transfer()

    def _run_transfer(self):
        if self.ref_img_np is None:
            self._set_status("⚠ 请先导入参考图")
            return
        if self.src_img_np is None:
            self._set_status("⚠ 请先导入原图")
            return
        self._set_status("⏳ 追色中...")
        threading.Thread(target=self._transfer_work, daemon=True).start()

    def _transfer_work(self):
        try:
            result = self.engine.transfer(
                self.src_img_np.copy(),
                tone_strength=self.tone_slider.get(),
                color_strength=self.color_slider.get(),
                skin_protect=self.skin_slider.get(),
                use_skin_protect=self.skin_on.get(),
            )
            self.result_np = result
            skin_tag = "（已启用肤色保护）" if self.skin_on.get() else ""
            msg = f"✓ 追色完成 {skin_tag}"
            self.after(0, lambda: self._set_status(msg))
            self.after(0, lambda: self.cards["result"].show(result))
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.after(0, lambda: self._set_status(f"✗ 追色失败：{e}"))

    def _save_result(self):
        if self.result_np is None:
            self._set_status("⚠ 请先执行追色")
            return
        path = ctk.filedialog.asksaveasfilename(
            title="保存结果图",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")],
        )
        if not path:
            return
        try:
            save_image(self.result_np, path)
            self._set_status(f"✓ 已保存：{os.path.basename(path)}")
        except Exception as e:
            self._set_status(f"✗ 保存失败：{e}")

    def _export_lut(self):
        if self.engine.ref_stats is None:
            self._set_status("⚠ 请先导入参考图")
            return
        path = ctk.filedialog.asksaveasfilename(
            title="导出 .cube LUT",
            defaultextension=".cube",
            filetypes=[("Cube LUT", "*.cube")],
        )
        if not path:
            return
        self._set_status("⏳ 生成 LUT 中...")
        threading.Thread(target=self._lut_work, args=(path,), daemon=True).start()

    def _lut_work(self, path: str):
        try:
            title = os.path.splitext(os.path.basename(path))[0]
            lut_size = int(self.lut_var.get())
            gen = LUTGenerator(lut_size=lut_size)
            gen.generate(
                self.engine,
                tone_strength=self.tone_slider.get(),
                color_strength=self.color_slider.get(),
                output_path=path,
                title=title,
            )
            self.after(0, lambda: self._set_status(
                f"✓ LUT 已导出（{lut_size}³）：{os.path.basename(path)}"
            ))
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.after(0, lambda: self._set_status(f"✗ LUT 导出失败：{e}"))

    def _reset_params(self):
        self.tone_slider.set(0.7)
        self.color_slider.set(0.9)
        self.skin_slider.set(0.85)
        self.skin_on.set(True)
        self._set_status("参数已重置")
        self._maybe_preview()

    def _set_status(self, msg: str):
        self.status_lbl.configure(text=msg)


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
