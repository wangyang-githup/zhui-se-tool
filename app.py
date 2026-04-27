"""
追色工具 — 主界面 (app.py)
使用方法: python3 app.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# 把当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_engine import ColorTransfer, LUTGenerator, load_image, save_image


# ══════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════

def np_to_pil(arr: np.ndarray) -> Image.Image:
    """float32 RGB [0,1] → PIL Image"""
    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


def pil_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """缩放图片以适应预览框，保持比例"""
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# ══════════════════════════════════════════════════════════
#  主应用
# ══════════════════════════════════════════════════════════

class App(tk.Tk):
    PREVIEW_W = 380
    PREVIEW_H = 280
    PANEL_BG  = "#1e1e2e"
    CARD_BG   = "#2a2a3e"
    TEXT_FG   = "#cdd6f4"
    ACCENT    = "#cba6f7"
    BTN_BG    = "#313244"
    BTN_HOVER = "#45475a"
    GREEN     = "#a6e3a1"
    RED       = "#f38ba8"

    def __init__(self):
        super().__init__()
        self.title("追色工具  ✦  Portrait Color Matcher")
        self.configure(bg=self.PANEL_BG)
        self.resizable(True, True)
        self.minsize(900, 680)

        self.engine = ColorTransfer()
        self.lut_gen = LUTGenerator(lut_size=33)

        self.ref_img_np  = None   # 参考图 float32 RGB
        self.src_img_np  = None   # 原图   float32 RGB
        self.result_np   = None   # 结果图 float32 RGB

        self.ref_path_var = tk.StringVar(value="未选择")
        self.src_path_var = tk.StringVar(value="未选择")
        self.status_var   = tk.StringVar(value="就绪 — 请先导入参考图和原图")

        self._build_ui()

    # ──────────────────────────────────────────
    #  UI 构建
    # ──────────────────────────────────────────

    def _build_ui(self):
        # 标题栏
        header = tk.Frame(self, bg=self.PANEL_BG)
        header.pack(fill="x", padx=20, pady=(18, 6))
        tk.Label(header, text="✦ 追色工具", font=("PingFang SC", 22, "bold"),
                 bg=self.PANEL_BG, fg=self.ACCENT).pack(side="left")
        tk.Label(header, text="Portrait Color Matcher  |  Lab 色彩迁移 + 肤色保护",
                 font=("PingFang SC", 11), bg=self.PANEL_BG, fg="#6c7086").pack(side="left", padx=14)

        # 主区域
        main = tk.Frame(self, bg=self.PANEL_BG)
        main.pack(fill="both", expand=True, padx=20, pady=4)

        # 左栏：图片导入 + 预览
        left = tk.Frame(main, bg=self.PANEL_BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_image_panel(left)

        # 右栏：参数控制
        right = tk.Frame(main, bg=self.PANEL_BG, width=270)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._build_controls(right)

        # 底部状态栏
        status_bar = tk.Frame(self, bg="#181825", height=32)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("PingFang SC", 10), bg="#181825", fg="#6c7086",
                 anchor="w", padx=16).pack(fill="x", side="left")

    def _build_image_panel(self, parent):
        """三格预览区：参考图 | 原图 | 结果图"""
        row = tk.Frame(parent, bg=self.PANEL_BG)
        row.pack(fill="both", expand=True)

        panels = [
            ("参考图  Reference", "ref"),
            ("原图  Source",      "src"),
            ("结果图  Result",    "result"),
        ]
        self.preview_labels = {}
        self.preview_frames = {}

        for title, key in panels:
            card = tk.Frame(row, bg=self.CARD_BG, bd=0, highlightthickness=1,
                            highlightbackground="#313244")
            card.pack(side="left", fill="both", expand=True, padx=5, pady=4)
            self.preview_frames[key] = card

            tk.Label(card, text=title, font=("PingFang SC", 10, "bold"),
                     bg=self.CARD_BG, fg=self.TEXT_FG).pack(pady=(8, 4))

            canvas = tk.Label(card, bg="#0f0f17", width=self.PREVIEW_W,
                              height=self.PREVIEW_H, cursor="hand2")
            canvas.pack(padx=8, pady=(0, 6))
            self.preview_labels[key] = canvas

            if key == "ref":
                canvas.bind("<Button-1>", lambda e: self._load_ref())
                self._make_placeholder(canvas, "点击导入参考图")
            elif key == "src":
                canvas.bind("<Button-1>", lambda e: self._load_src())
                self._make_placeholder(canvas, "点击导入原图")
            else:
                self._make_placeholder(canvas, "追色结果将显示在这里")

        # 原图/结果切换按钮
        btn_row = tk.Frame(parent, bg=self.PANEL_BG)
        btn_row.pack(fill="x", padx=5, pady=(0, 6))
        self._btn(btn_row, "↓ 保存结果图", self._save_result,
                  bg=self.GREEN, fg="#1e1e2e").pack(side="right", padx=5)
        self._btn(btn_row, "↓ 导出 .cube LUT", self._export_lut,
                  bg="#89b4fa", fg="#1e1e2e").pack(side="right", padx=5)

    def _build_controls(self, parent):
        """右侧参数面板"""
        def section(text):
            f = tk.Frame(parent, bg=self.PANEL_BG)
            f.pack(fill="x", pady=(14, 2))
            tk.Label(f, text=text, font=("PingFang SC", 11, "bold"),
                     bg=self.PANEL_BG, fg=self.ACCENT).pack(anchor="w")
            return f

        def slider_row(parent, label, var, from_, to, default, fmt="{:.0%}"):
            f = tk.Frame(parent, bg=self.PANEL_BG)
            f.pack(fill="x", pady=3)
            top = tk.Frame(f, bg=self.PANEL_BG)
            top.pack(fill="x")
            tk.Label(top, text=label, font=("PingFang SC", 10),
                     bg=self.PANEL_BG, fg=self.TEXT_FG).pack(side="left")
            val_lbl = tk.Label(top, text=fmt.format(default),
                               font=("PingFang SC", 10, "bold"),
                               bg=self.PANEL_BG, fg=self.ACCENT, width=5)
            val_lbl.pack(side="right")
            s = ttk.Scale(f, from_=from_, to=to, variable=var,
                          orient="horizontal", length=240)
            s.pack(fill="x")

            def update_label(*_):
                val_lbl.config(text=fmt.format(var.get()))
                if self._auto_preview.get() and self.ref_img_np is not None and self.src_img_np is not None:
                    self._run_transfer_async()

            var.trace_add("write", update_label)
            return s

        # 追色参数
        section("⚙  追色参数")
        self._tone_var   = tk.DoubleVar(value=0.8)
        self._color_var  = tk.DoubleVar(value=0.9)
        slider_row(parent, "影调强度  Tone",  self._tone_var,  0, 1, 0.8)
        slider_row(parent, "色调强度  Color", self._color_var, 0, 1, 0.9)

        # 肤色保护
        section("👤  肤色保护")
        self._skin_protect_on = tk.BooleanVar(value=True)
        self._skin_var = tk.DoubleVar(value=0.85)

        chk_f = tk.Frame(parent, bg=self.PANEL_BG)
        chk_f.pack(fill="x", pady=2)
        tk.Checkbutton(chk_f, text="启用肤色保护", variable=self._skin_protect_on,
                       bg=self.PANEL_BG, fg=self.TEXT_FG, selectcolor=self.CARD_BG,
                       activebackground=self.PANEL_BG, font=("PingFang SC", 10),
                       command=self._on_skin_toggle).pack(anchor="w")
        slider_row(parent, "保护强度  Protect", self._skin_var, 0, 1, 0.85)

        tk.Label(parent,
                 text="检测肤色区域，在匹配\n色调时保留人物皮肤原色",
                 font=("PingFang SC", 9), bg=self.PANEL_BG, fg="#6c7086",
                 justify="left").pack(anchor="w", pady=(0, 4))

        # LUT 参数
        section("💾  LUT 设置")
        lut_f = tk.Frame(parent, bg=self.PANEL_BG)
        lut_f.pack(fill="x", pady=3)
        tk.Label(lut_f, text="精度", font=("PingFang SC", 10),
                 bg=self.PANEL_BG, fg=self.TEXT_FG).pack(side="left")
        self._lut_size_var = tk.StringVar(value="33（标准）")
        combo = ttk.Combobox(lut_f, textvariable=self._lut_size_var, width=14,
                             values=["17（轻量）", "33（标准）", "65（高精度）"],
                             state="readonly")
        combo.pack(side="right")

        # 选项
        section("🔧  选项")
        self._auto_preview = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="滑动时自动预览",
                       variable=self._auto_preview,
                       bg=self.PANEL_BG, fg=self.TEXT_FG, selectcolor=self.CARD_BG,
                       activebackground=self.PANEL_BG,
                       font=("PingFang SC", 10)).pack(anchor="w", pady=2)

        # 操作按钮
        tk.Frame(parent, bg="#313244", height=1).pack(fill="x", pady=14)
        self._btn(parent, "▶  开始追色", self._run_transfer,
                  bg=self.ACCENT, fg="#1e1e2e",
                  font=("PingFang SC", 13, "bold")).pack(fill="x", pady=3, ipady=6)
        self._btn(parent, "重置参数", self._reset_params).pack(fill="x", pady=2)

    # ──────────────────────────────────────────
    #  辅助 UI 方法
    # ──────────────────────────────────────────

    def _btn(self, parent, text, cmd, bg=None, fg=None, font=None, **kw):
        bg = bg or self.BTN_BG
        fg = fg or self.TEXT_FG
        font = font or ("PingFang SC", 10)
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      font=font, relief="flat", cursor="hand2",
                      activebackground=self.BTN_HOVER, activeforeground=fg,
                      bd=0, padx=10, pady=4, **kw)
        b.bind("<Enter>", lambda e: b.config(bg=self.BTN_HOVER) if bg == self.BTN_BG else None)
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _make_placeholder(self, label, text):
        label.config(image="", text=text, fg="#45475a",
                     font=("PingFang SC", 10), compound="center")
        label.image = None

    def _show_preview(self, key: str, arr: np.ndarray):
        img = pil_fit(np_to_pil(arr), self.PREVIEW_W, self.PREVIEW_H)
        tk_img = ImageTk.PhotoImage(img)
        lbl = self.preview_labels[key]
        lbl.config(image=tk_img, text="", compound="center")
        lbl.image = tk_img   # 防止 GC 回收
        lbl.update_idletasks()

    def _set_status(self, msg: str, color: str = None):
        self.status_var.set(msg)
        if color:
            # 找状态栏 label 并改颜色（简化处理）
            pass

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
            self.engine.analyze(self.ref_img_np)
            self._show_preview("ref", self.ref_img_np)
            self.ref_path_var.set(os.path.basename(path))
            self._set_status(f"✓ 参考图已加载：{os.path.basename(path)}  |  已分析色彩特征")
            # 如果原图也已加载，自动预览
            if self.src_img_np is not None and self._auto_preview.get():
                self._run_transfer_async()
        except Exception as e:
            import traceback
            messagebox.showerror("错误", f"加载参考图失败：{e}\n{traceback.format_exc()}")

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
            self.src_path_var.set(os.path.basename(path))
            self._set_status(f"✓ 原图已加载：{os.path.basename(path)}")
            # 如果参考图也已加载，自动预览
            if self.ref_img_np is not None and self._auto_preview.get():
                self._run_transfer_async()
        except Exception as e:
            import traceback
            messagebox.showerror("错误", f"加载原图失败：{e}\n{traceback.format_exc()}")

    # ──────────────────────────────────────────
    #  追色执行
    # ──────────────────────────────────────────

    def _check_ready(self) -> bool:
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
        """在子线程中执行追色，避免冻结 UI"""
        # 提前在主线程读取参数值（Tkinter 变量不能跨线程访问）
        tone    = self._tone_var.get()
        color   = self._color_var.get()
        skin    = self._skin_var.get()
        use_sk  = self._skin_protect_on.get()
        src_img = self.src_img_np.copy()

        def work():
            self.after(0, lambda: self._set_status("⏳ 追色中..."))
            try:
                result = self.engine.transfer(
                    src_img,
                    tone_strength=tone,
                    color_strength=color,
                    skin_protect=skin,
                    use_skin_protect=use_sk,
                )
                self.result_np = result
                skin_info = "（已启用肤色保护）" if use_sk else ""
                status_msg = f"✓ 追色完成{skin_info}  影调 {tone:.0%}  色调 {color:.0%}"

                def update_ui():
                    self._show_preview("result", result)
                    self._set_status(status_msg)

                self.after(0, update_ui)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print("追色异常:\n", tb)
                self.after(0, lambda: self._set_status(f"✗ 追色失败：{e}"))
                self.after(0, lambda: messagebox.showerror("追色失败", f"{e}\n\n{tb}"))

        threading.Thread(target=work, daemon=True).start()

    # ──────────────────────────────────────────
    #  保存 / 导出
    # ──────────────────────────────────────────

    def _save_result(self):
        if self.result_np is None:
            messagebox.showwarning("提示", "请先执行追色")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果图",
            defaultextension=".jpg",
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
        if self.engine.ref_stats is None:
            messagebox.showwarning("提示", "请先导入参考图以分析色彩特征")
            return
        path = filedialog.asksaveasfilename(
            title="导出 .cube LUT",
            defaultextension=".cube",
            filetypes=[("Cube LUT", "*.cube")]
        )
        if not path:
            return

        # 解析 LUT 精度选项
        size_map = {"17（轻量）": 17, "33（标准）": 33, "65（高精度）": 65}
        lut_size = size_map.get(self._lut_size_var.get(), 33)
        self.lut_gen = LUTGenerator(lut_size=lut_size)

        def work():
            self._set_status("⏳ 生成 LUT 中，请稍候...")
            try:
                title = os.path.splitext(os.path.basename(path))[0]
                self.lut_gen.generate(
                    self.engine,
                    tone_strength=self._tone_var.get(),
                    color_strength=self._color_var.get(),
                    output_path=path,
                    title=title,
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

    def _on_skin_toggle(self):
        if self.src_img_np is not None and self.ref_img_np is not None:
            if self._auto_preview.get():
                self._run_transfer_async()

    def _reset_params(self):
        self._tone_var.set(0.8)
        self._color_var.set(0.9)
        self._skin_var.set(0.85)
        self._skin_protect_on.set(True)
        self._set_status("参数已重置")


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
