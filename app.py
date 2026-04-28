"""
追色工具 — 现代 Tkinter 界面（macOS 打包稳定版）
使用方法: python3 app.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image as PILImage
import numpy as np

# 放在 import tkinter 之后，设置 macOS 深色模式兼容
try:
    from tkmacosx import Button  # macOS 风格按钮（可选，有则用无则降级）
    HAS_MACOSX = True
except ImportError:
    HAS_MACOSX = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_engine import ColorTransfer, LUTGenerator, load_image, save_image


# ═══════════════════════════════════════════════════════
#  配色常量
# ═══════════════════════════════════════════════════════

BG       = "#1e1e2e"   # 主背景
CARD     = "#252536"   # 卡片背景
CARD2    = "#2a2a3e"   # 卡片背景2（略浅）
TEXT     = "#cdd6f4"   # 主文字
MUTED    = "#6c7086"   # 次要文字
ACCENT   = "#cba6f7"   # 紫色强调
BTN_BG   = "#313244"   # 按钮背景
BTN_HV   = "#45475a"   # 按钮悬停
BTN_ACC  = ACCENT      # 主按钮背景
GREEN    = "#a6e3a1"   # 成功绿


# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def np_to_photo(arr: np.ndarray) -> tk.PhotoImage:
    """float32 RGB [0,1] → tk.PhotoImage"""
    u8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    pil = PILImage.fromarray(u8, mode='RGB')
    return pil  # 返回 PIL Image，交给 CTkLabel 或 Label 组件显示


def fit_size(w: int, h: int, max_w: int, max_h: int) -> tuple:
    scale = min(max_w / w, max_h / h, 1.0) if w > 0 and h > 0 else 1.0
    return int(w * scale), int(h * scale)


# ═══════════════════════════════════════════════════════
#  图片预览卡片
# ═══════════════════════════════════════════════════════

class ImageCard(tk.Frame):
    def __init__(self, parent, title: str, key: str,
                 placeholder: str, on_click=None):
        super().__init__(parent, bg=CARD2, bd=0)
        self.key = key
        self.on_click = on_click
        self.img_np = None

        # 标题
        tk.Label(self, text=title, bg=CARD2, fg=TEXT,
                 font=("PingFang SC", 11, "bold")).pack(pady=(14, 6))

        # 图片容器（深色背景）
        self.canvas = tk.Label(self, text="", bg="#14141e", width=230,
                              height=160, cursor="hand2")
        self.canvas.pack(padx=10, pady=(0, 8))
        self._set_placeholder(placeholder)

        if on_click:
            self.canvas.bind("<Button-1>", lambda _: on_click())
            # 整个卡片区域可点击
            for w in [self.canvas]:
                pass  # 已在 canvas 绑定

    def _set_placeholder(self, text: str):
        self.canvas.configure(image="", text=f"\n\n{text}", fg=MUTED,
                             font=("PingFang SC", 11), compound="center")

    def show(self, arr: np.ndarray):
        self.img_np = arr
        pil = np_to_photo(arr)
        w, h = pil.size
        fit_w, fit_h = fit_size(w, h, 230, 160)
        # 缩放
        pil = pil.resize((fit_w, fit_h), PILImage.LANCZOS)
        self.photo = tk.PhotoImage(pil)  # 保存在 self 中防止 GC
        self.canvas.configure(image=self.photo, text="", compound="center")


# ═══════════════════════════════════════════════════════
#  参数滑块行
# ═══════════════════════════════════════════════════════

class ParamSlider(tk.Frame):
    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, fmt: str = "{:.0%}", command=None):
        super().__init__(parent, bg=BG)
        self.fmt = fmt
        self._command = command
        self.var = tk.DoubleVar(value=default)

        # 标签行
        row = tk.Frame(self, bg=BG)
        row.pack(fill="x")
        tk.Label(row, text=label, bg=BG, fg=TEXT,
                 font=("PingFang SC", 11)).pack(side="left")
        self.val_lbl = tk.Label(row, text=fmt.format(default),
                                bg=BG, fg=ACCENT,
                                font=("PingFang SC", 11, "bold"), width=50,
                                anchor="e")
        self.val_lbl.pack(side="right")

        # 滑块（ttk 样式）
        style = ttk.Style()
        style.configure("Accent.Horizontal.TScale",
                        background=BG)
        style.configure("Accent.Horizontal.TScale",
                        troughcolor=CARD2,
                        sliderlength=18,
                        thickness=6)
        s = ttk.Scale(self, from_=from_, to=to, variable=self.var,
                      orient="horizontal", style="Accent.Horizontal.TScale",
                      command=self._on_change)
        s.pack(fill="x", pady=(3, 0))

        # 同步 ttk 样式（macOS 深色兼容）
        s.configure(style="Accent.Horizontal.TScale")

    def _on_change(self, val):
        self.val_lbl.configure(text=self.fmt.format(float(val)))
        if self._command:
            self._command(float(val))

    def get(self) -> float:
        return self.var.get()

    def set(self, val: float):
        self.var.set(val)
        self.val_lbl.configure(text=self.fmt.format(val))


# ═══════════════════════════════════════════════════════
#  macOS 风格按钮（兼容深色主题）
# ═══════════════════════════════════════════════════════

def make_btn(parent, text: str, command, **kw) -> tk.Button:
    bg   = kw.pop("bg",   BTN_BG)
    fg   = kw.pop("fg",   TEXT)
    acb  = kw.pop("activebackground", BTN_HV)
    acf  = kw.pop("activeforeground", TEXT)
    hbg  = kw.pop("hoverbackground",  BTN_HV)
    hfg  = kw.pop("hoverforeground",  TEXT)
    h    = kw.pop("height", 34)
    bd   = kw.pop("bd", 0)
    padx = kw.pop("padx", 12)
    pady = kw.pop("pady", 4)
    font = kw.pop("font", ("PingFang SC", 11))
    w    = kw.pop("width", None)
    relief = kw.pop("relief", "flat")
    cursor = kw.pop("cursor", "hand2")

    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg=fg, activebackground=acb,
                    activeforeground=acf, font=font,
                    height=h, width=w, bd=bd, padx=padx, pady=pady,
                    relief=relief, cursor=cursor,
                    highlightthickness=0, **kw)

    def on_enter(e):
        btn.config(bg=hbg, fg=hfg)
    def on_leave(e):
        btn.config(bg=bg, fg=fg)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


# ═══════════════════════════════════════════════════════
#  主题样式配置（ttk 组件深色化）
# ═══════════════════════════════════════════════════════

def configure_styles():
    style = ttk.Style()
    # Checkbutton
    style.configure("TCheckbutton",
                    background=BG, foreground=TEXT,
                    indicatorbackground=CARD2, indicatorforeground=ACCENT)
    style.map("TCheckbutton",
              background=[("active", BG)],
              foreground=[("active", TEXT)])

    # Radiobutton
    style.configure("TRadiobutton",
                    background=BG, foreground=TEXT,
                    indicatorbackground=CARD2, indicatorforeground=ACCENT)
    style.map("TRadiobutton",
              background=[("active", BG)],
              foreground=[("active", TEXT)])

    # Scale
    style.configure("Accent.Horizontal.TScale",
                    background=BG,
                    troughcolor=CARD2,
                    slidercolor=ACCENT,
                    darkcolor=BG,
                    lightcolor=BG)


# ═══════════════════════════════════════════════════════
#  主应用
# ═══════════════════════════════════════════════════════

class App(tk.Tk):
    PW = 230   # 预览宽度
    PH = 160   # 预览高度

    def __init__(self):
        super().__init__()
        configure_styles()

        self.title("追色工具")
        self.configure(bg=BG)
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.resizable(True, True)

        # 引擎
        self.engine  = ColorTransfer()
        self.lut_gen = LUTGenerator(lut_size=33)

        self.ref_img_np  = None
        self.src_img_np  = None
        self.result_np   = None

        self._build_ui()

    def _build_ui(self):
        # ── 标题栏 ──────────────────────────────
        header = tk.Frame(self, bg=BG, height=56)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="✦", font=("PingFang SC", 20),
                 bg=BG, fg=ACCENT).pack(side="left", padx=(20, 4), pady=12)
        tk.Label(header, text="追色工具", font=("PingFang SC", 18, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", pady=12)
        tk.Label(header,
                 text="Portrait Color Matcher  ·  Lab 色彩迁移 + 肤色保护",
                 font=("PingFang SC", 10), bg=BG, fg=MUTED).pack(
                     side="left", padx=14, pady=12)
        self.status_var = tk.StringVar(value="就绪 — 请先导入参考图和原图")
        self.status_lbl = tk.Label(header, textvariable=self.status_var,
                                    font=("PingFang SC", 10), bg=BG, fg=MUTED)
        self.status_lbl.pack(side="right", padx=20, pady=12)

        # ── 主区域（左右分栏） ───────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # 左：图片预览
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        cards_data = [
            ("参考图  Reference", "ref",   "点击导入参考图",    self._load_ref),
            ("原图  Source",      "src",   "点击导入原图",      self._load_src),
            ("结果图  Result",    "result","追色结果将显示在这里", None),
        ]
        self.cards = {}
        card_row = tk.Frame(left, bg=BG)
        card_row.pack(fill="both", expand=True, pady=(0, 10))
        for title, key, ph, cmd in cards_data:
            card = ImageCard(card_row, title, key, ph, on_click=cmd)
            card.pack(side="left", fill="both", expand=True, padx=5)
            self.cards[key] = card

        # 按钮行
        btn_row = tk.Frame(left, bg=BG, height=40)
        btn_row.pack(fill="x", pady=(0, 0))
        btn_row.pack_propagate(False)

        make_btn(btn_row, "↻ 重置",         self._reset_params,
                 width=100).pack(side="left", padx=4)
        make_btn(btn_row, "⬇ 导出 .cube LUT", self._export_lut,
                 bg=BTN_HV, width=140).pack(side="right", padx=4)
        make_btn(btn_row, "↓ 保存结果图",    self._save_result,
                 bg=BTN_HV, width=120).pack(side="right", padx=4)

        # ── 右：参数面板 ─────────────────────────
        right_outer = tk.Frame(main, bg=CARD2, width=272)
        right_outer.pack(side="right", fill="y", padx=(12, 0))
        right_outer.pack_propagate(False)

        canvas = tk.Canvas(right_outer, bg=CARD2, bd=0, highlightthickness=0,
                           width=272)
        scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=canvas.yview)
        self.param_frame = tk.Frame(canvas, bg=CARD2)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.param_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_frame, anchor="nw")

        # 绑定鼠标滚轮滚动
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_controls(self.param_frame)

        # ── 底部 ────────────────────────────────
        footer = tk.Frame(self, bg="#181825", height=26)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer,
                 text="支持 JPG · PNG · TIFF · BMP · WebP  |  © 追色工具",
                 font=("PingFang SC", 9), bg="#181825",
                 fg="#45475a").pack(pady=4)

    def _build_controls(self, parent):
        def section(icon: str, title: str) -> tk.Frame:
            f = tk.Frame(parent, bg=CARD2)
            f.pack(fill="x", pady=(14, 2))
            tk.Label(f, text=f"{icon}  {title}",
                     bg=CARD2, fg=ACCENT,
                     font=("PingFang SC", 12, "bold"),
                     anchor="w").pack(padx=14, pady=(12, 4))
            tk.Frame(parent, height=1, bg=BG).pack(fill="x", padx=14, pady=(0, 4))
            return f

        def note(text: str):
            tk.Label(parent, text=text, bg=CARD2, fg=MUTED,
                     font=("PingFang SC", 9), justify="left",
                     anchor="w").pack(padx=14, anchor="w", pady=(0, 6))

        # ── 追色参数 ───────────────────────────
        section("⚙", "追色参数")
        note("拖动滑块实时预览效果")

        self.tone_slider = ParamSlider(
            parent, "影调强度  Tone", 0, 1, 0.7, "{:.0%}",
            command=lambda v: self._maybe_preview())
        self.tone_slider.pack(fill="x", padx=14, pady=3, ipady=2)

        self.color_slider = ParamSlider(
            parent, "色调强度  Color", 0, 1, 0.9, "{:.0%}",
            command=lambda v: self._maybe_preview())
        self.color_slider.pack(fill="x", padx=14, pady=3, ipady=2)

        # ── 肤色保护 ───────────────────────────
        section("👤", "肤色保护")

        self.skin_on = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(parent, text="启用肤色保护",
                             variable=self.skin_on,
                             onvalue=True, offvalue=False,
                             command=lambda: self._maybe_preview())
        cb.pack(anchor="w", padx=12, pady=(0, 4))

        self.skin_slider = ParamSlider(
            parent, "保护强度  Protect", 0, 1, 0.85, "{:.0%}",
            command=lambda v: self._maybe_preview())
        self.skin_slider.pack(fill="x", padx=14, pady=3, ipady=2)

        note("肤色区域将保留原始色彩\n防止追色导致肤色偏黄/青灰")

        # ── LUT 设置 ───────────────────────────
        section("💾", "LUT 设置")

        self.lut_var = tk.StringVar(value="33")
        for label, val in [("17  轻量", "17"), ("33  标准 ✓", "33"), ("65  高精度", "65")]:
            ttk.Radiobutton(parent, text=label, variable=self.lut_var,
                           value=val.replace(" ✓", ""),
                           style="TRadiobutton").pack(anchor="w", padx=28, pady=1)

        # ── 分隔 ──────────────────────────────
        tk.Frame(parent, height=8, bg=CARD2).pack()
        btn = make_btn(parent, "▶  开始追色", self._run_transfer,
                       bg=BTN_ACC, fg="#1e1e2e",
                       font=("PingFang SC", 14, "bold"),
                       height=46, width=None, padx=14, pady=8,
                       hoverbackground="#b48ee8")
        btn.pack(fill="x", padx=14, pady=(6, 4))

    # ──────────────────────────────────────────
    def _load_ref(self):
        path = filedialog.askopenfilename(
            title="选择参考图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
                       ("所有文件", "*.*")])
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
            traceback.print_exc()

    def _load_src(self):
        path = filedialog.askopenfilename(
            title="选择原图",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
                       ("所有文件", "*.*")])
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
            traceback.print_exc()

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
            tag = "（已启用肤色保护）" if self.skin_on.get() else ""
            self.after(0, lambda: self._set_status(f"✓ 追色完成 {tag}"))
            self.after(0, lambda: self.cards["result"].show(result))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self._set_status(f"✗ 追色失败：{e}"))

    def _save_result(self):
        if self.result_np is None:
            self._set_status("⚠ 请先执行追色")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果图", defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")])
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
        path = filedialog.asksaveasfilename(
            title="导出 .cube LUT", defaultextension=".cube",
            filetypes=[("Cube LUT", "*.cube")])
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
                f"✓ LUT 已导出（{lut_size}³）：{os.path.basename(path)}"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self._set_status(f"✗ LUT 导出失败：{e}"))

    def _reset_params(self):
        self.tone_slider.set(0.7)
        self.color_slider.set(0.9)
        self.skin_slider.set(0.85)
        self.skin_on.set(True)
        self._set_status("参数已重置")
        self._maybe_preview()

    def _set_status(self, msg: str):
        self.status_var.set(msg)


# ═══════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
