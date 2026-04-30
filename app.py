"""
追色 - AI智能配色工具 v2.0
基于完整色彩工程的专业追色系统

功能：
1. 参考图导入 - 加载目标色调的参考图片
2. 色彩分析 - 提取参考图和原片的色彩特征
3. 智能追色 - Lab空间色彩迁移 + ΔE肤色保护
4. LUT导出 - 导出.cube格式的LUT文件
5. 批量处理 - 支持多图批量调色

无OpenCV依赖，纯PIL + NumPy + scipy实现
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from PIL import ImageTk
import numpy as np
import os
import threading
from pathlib import Path
from typing import Optional, Tuple, List

# 导入自定义模块
from color_engine import ColorTransfer, load_image, save_image
from feature_extractor import FeatureExtractor
from grading_generator import ColorGradingGenerator
from color_renderer import ColorRenderer


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
    """追色工具主应用"""

    # 配色方案
    PREVIEW_W = 320
    PREVIEW_H = 240
    PANEL_BG  = "#1e1e2e"
    CARD_BG   = "#2a2a3e"
    TEXT_FG   = "#cdd6f4"
    ACCENT    = "#cba6f7"
    BTN_BG    = "#313244"
    BTN_HOVER = "#45475a"
    GREEN     = "#a6e3a1"
    RED       = "#f38ba8"
    BLUE      = "#89b4fa"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("追色 v2.0 - AI智能配色工具")
        self.root.configure(bg=self.PANEL_BG)
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        # 状态变量
        self.reference_path: Optional[str] = None
        self.reference_img: Optional[np.ndarray] = None
        self.target_imgs: List[Tuple[str, np.ndarray]] = []
        self.current_result: Optional[np.ndarray] = None
        self.current_index = 0
        self.output_dir = os.path.expanduser("~/Desktop")

        # 专业模块
        self.feature_extractor = FeatureExtractor()
        self.grading_generator = ColorGradingGenerator()
        self.color_renderer = ColorRenderer()
        self.ref_features = None

        # 调色参数
        self.skin_protection = tk.BooleanVar(value=True)
        self.skin_strength = tk.DoubleVar(value=0.85)
        self.tone_strength = tk.DoubleVar(value=0.8)
        self.color_strength = tk.DoubleVar(value=0.9)
        self.lut_size = tk.IntVar(value=33)
        self.mode = tk.StringVar(value="quick")  # quick=快速模式, pro=专业模式
        self.auto_preview = tk.BooleanVar(value=False)  # 默认关闭自动预览

        # 预览图引用（防止GC）
        self.preview_labels = {}

        self._init_ui()

    def _init_ui(self):
        """初始化UI - 参考第一版布局"""

        # 顶部标题栏
        header = tk.Frame(self.root, bg=self.PANEL_BG)
        header.pack(fill="x", padx=20, pady=(18, 6))

        tk.Label(header, text="✦ 追色工具", font=("PingFang SC", 22, "bold"),
                 bg=self.PANEL_BG, fg=self.ACCENT).pack(side="left")
        tk.Label(header, text="Portrait Color Matcher  |  Lab 色彩迁移 + 肤色保护",
                 font=("PingFang SC", 11), bg=self.PANEL_BG, fg="#6c7086").pack(side="left", padx=14)

        # 模式切换
        tk.Label(header, text="模式:", font=("PingFang SC", 11),
                 bg=self.PANEL_BG, fg=self.TEXT_FG).pack(side="right", padx=(0, 5))
        ttk.Radiobutton(header, text="快速追色", variable=self.mode, value="quick",
                        style="Dark.TRadiobutton").pack(side="right", padx=2)
        ttk.Radiobutton(header, text="专业模式", variable=self.mode, value="pro",
                        style="Dark.TRadiobutton").pack(side="right", padx=2)

        # 主区域
        main = tk.Frame(self.root, bg=self.PANEL_BG)
        main.pack(fill="both", expand=True, padx=20, pady=4)

        # 右栏：控制面板（固定宽度）- 必须先pack，否则expand会把左侧撑满
        right = tk.Frame(main, bg=self.PANEL_BG, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_controls(right)

        # 左栏：图片区域 - 后pack且expand=True填充剩余空间
        left = tk.Frame(main, bg=self.PANEL_BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_image_panel(left)

        # 底部状态栏
        status_bar = tk.Frame(self.root, bg="#181825", height=32)
        status_bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="就绪 — 请导入参考图和目标图")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("PingFang SC", 10), bg="#181825", fg="#6c7086",
                 anchor="w", padx=16).pack(fill="x", side="left")

        self._setup_styles()

    def _build_image_panel(self, parent):
        """三格预览区：参考图 | 目标图 | 结果图"""
        row = tk.Frame(parent, bg=self.PANEL_BG)
        row.pack(fill="both", expand=True)

        panels = [
            ("参考图  Reference", "ref", "点击导入参考图", self.load_reference),
            ("目标图  Source",    "src", "点击导入目标图", self.load_targets),
            ("结果图  Result",    "result", "追色结果将显示在这里", None),
        ]

        for title, key, placeholder, cmd in panels:
            card = tk.Frame(row, bg=self.CARD_BG, bd=0, highlightthickness=1,
                            highlightbackground="#313244")
            card.pack(side="left", fill="both", expand=True, padx=5, pady=4)

            tk.Label(card, text=title, font=("PingFang SC", 10, "bold"),
                     bg=self.CARD_BG, fg=self.TEXT_FG).pack(pady=(8, 4))

            # 预览区域
            canvas = tk.Label(card, bg="#0f0f17", width=self.PREVIEW_W,
                              height=self.PREVIEW_H, cursor="hand2")
            canvas.pack(padx=8, pady=(0, 6))
            self._make_placeholder(canvas, placeholder)
            self.preview_labels[key] = canvas

            # 绑定点击事件
            if cmd:
                canvas.bind("<Button-1>", lambda e, c=cmd: c())

        # 底部按钮行
        btn_row = tk.Frame(parent, bg=self.PANEL_BG)
        btn_row.pack(fill="x", padx=5, pady=(0, 6))

        # 图片导航（多图时）
        self.nav_frame = tk.Frame(btn_row, bg=self.PANEL_BG)
        self.nav_frame.pack(side="left")
        self._btn(self.nav_frame, "◀ 上一张", self.prev_image,
                  bg=self.BTN_BG, width=8).pack(side="left", padx=2)
        self._btn(self.nav_frame, "下一张 ▶", self.next_image,
                  bg=self.BTN_BG, width=8).pack(side="left", padx=2)
        self.img_counter = tk.Label(self.nav_frame, text="未加载",
                                    font=("PingFang SC", 10), bg=self.PANEL_BG, fg="#6c7086")
        self.img_counter.pack(side="left", padx=8)

        # 批量导入
        self._btn(btn_row, "📁 批量导入", self.load_batch,
                  bg=self.BTN_BG, width=10).pack(side="right", padx=2)

        # 操作按钮
        self._btn(btn_row, "↓ 保存结果图", self.save_result,
                  bg=self.GREEN, fg="#1e1e2e", width=10).pack(side="right", padx=2)
        self._btn(btn_row, "↓ 导出 .cube LUT", self.export_lut,
                  bg=self.BLUE, fg="#1e1e2e", width=12).pack(side="right", padx=2)

    def _build_controls(self, parent):
        """右侧参数面板"""
        def section(text):
            f = tk.Frame(parent, bg=self.PANEL_BG)
            f.pack(fill="x", pady=(14, 2))
            tk.Label(f, text=text, font=("PingFang SC", 11, "bold"),
                     bg=self.PANEL_BG, fg=self.ACCENT).pack(anchor="w")
            return f

        def slider_row(label, var, from_, to, default, fmt="{:.0%}"):
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
                          orient="horizontal", length=280)
            s.pack(fill="x")

            def update_label(*_):
                val_lbl.config(text=fmt.format(var.get()))
                if self.auto_preview.get() and self.reference_img is not None and self.target_imgs:
                    self._run_transfer_async()

            var.trace_add("write", update_label)
            return s

        # 追色参数
        section("⚙  追色参数")
        slider_row("影调强度  Tone",  self.tone_strength,  0, 1.5, 0.8)
        slider_row("色调强度  Color", self.color_strength, 0, 1.5, 0.9)

        # 肤色保护
        section("👤  肤色保护")
        chk_f = tk.Frame(parent, bg=self.PANEL_BG)
        chk_f.pack(fill="x", pady=2)
        tk.Checkbutton(chk_f, text="启用肤色保护", variable=self.skin_protection,
                       bg=self.PANEL_BG, fg=self.TEXT_FG, selectcolor=self.CARD_BG,
                       activebackground=self.PANEL_BG, font=("PingFang SC", 10)).pack(anchor="w")
        slider_row("保护强度  Protect", self.skin_strength, 0, 1, 0.85)

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
        self.lut_combo = ttk.Combobox(lut_f, textvariable=tk.StringVar(value="33（标准）"),
                                       width=14,
                                       values=["17（轻量）", "33（标准）", "65（高精度）"],
                                       state="readonly")
        self.lut_combo.pack(side="right")

        # 选项
        section("🔧  选项")
        tk.Checkbutton(parent, text="滑动时自动预览",
                       variable=self.auto_preview,
                       bg=self.PANEL_BG, fg=self.TEXT_FG, selectcolor=self.CARD_BG,
                       activebackground=self.PANEL_BG,
                       font=("PingFang SC", 10)).pack(anchor="w", pady=2)

        # 输出目录
        section("📂  输出目录")
        output_f = tk.Frame(parent, bg=self.PANEL_BG)
        output_f.pack(fill="x", pady=3)
        self.output_entry = ttk.Entry(output_f, textvariable=tk.StringVar(value=self.output_dir))
        self.output_entry.pack(fill="x")
        self._btn(output_f, "浏览...", self.browse_output,
                  bg=self.BTN_BG, width=8).pack(pady=3)

        # 操作按钮
        tk.Frame(parent, bg="#313244", height=1).pack(fill="x", pady=14)
        self._btn(parent, "▶  开始追色", self.start_color_transfer,
                  bg=self.ACCENT, fg="#1e1e2e",
                  font=("PingFang SC", 13, "bold")).pack(fill="x", pady=3, ipady=6)
        self._btn(parent, "📁 批量处理", self.batch_process,
                  bg=self.GREEN, fg="#1e1e2e",
                  font=("PingFang SC", 11, "bold")).pack(fill="x", pady=3, ipady=4)
        self._btn(parent, "重置参数", self._reset_params).pack(fill="x", pady=2)

    def _setup_styles(self):
        """配置样式"""
        style = ttk.Style()
        style.configure("Dark.TRadiobutton", background=self.PANEL_BG, foreground=self.TEXT_FG)

    def _btn(self, parent, text, cmd, bg=None, fg=None, font=None, width=None, **kw):
        """创建按钮"""
        bg = bg or self.BTN_BG
        fg = fg or self.TEXT_FG
        font = font or ("PingFang SC", 10)
        opts = {"relief": "flat", "cursor": "hand2",
                "activebackground": self.BTN_HOVER, "activeforeground": fg, "bd": 0, "padx": 10, "pady": 4}
        if width:
            opts["width"] = width
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, font=font, **opts)
        b.bind("<Enter>", lambda e: b.config(bg=self.BTN_HOVER) if bg == self.BTN_BG else None)
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _make_placeholder(self, label, text):
        """显示占位符"""
        label.config(image="", text=text, fg="#45475a",
                     font=("PingFang SC", 10), compound="center")
        label.image = None

    def _show_preview(self, key: str, arr: np.ndarray):
        """显示预览图"""
        img_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8, mode='RGB')

        # 缩放适应
        w, h = pil_img.size
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h, 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        # 居中放置
        canvas = Image.new("RGB", (self.PREVIEW_W, self.PREVIEW_H), "#0f0f17")
        x = (self.PREVIEW_W - pil_img.width) // 2
        y = (self.PREVIEW_H - pil_img.height) // 2
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

            # 专业模式：提取参考图特征
            if self.mode.get() == "pro":
                self.ref_features = self.feature_extractor.extract(img)
                self.status_var.set(f"✓ 参考图已加载（专业模式）— {os.path.basename(filepath)}")
            else:
                self.status_var.set(f"✓ 参考图已加载（快速模式）— {os.path.basename(filepath)}")

            # 自动预览
            if self.target_imgs and self.auto_preview.get():
                self._run_transfer_async()

        except Exception as e:
            import traceback
            messagebox.showerror("错误", f"加载失败: {str(e)}\n{traceback.format_exc()}")

    def load_targets(self):
        """加载目标图"""
        filepaths = filedialog.askopenfilenames(
            title="选择目标图",
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
            self.status_var.set(f"已加载 {count} 张图片 — 点击开始追色")

    def _update_target_preview(self):
        """更新目标图预览"""
        if self.target_imgs:
            _, img = self.target_imgs[self.current_index]
            self._show_preview("src", img)
            self.img_counter.config(text=f"第 {self.current_index + 1} / {len(self.target_imgs)} 张")
            self.nav_frame.pack(side="left")
        else:
            self._make_placeholder(self.preview_labels.get("src", tk.Label()), "点击导入目标图")
            self.img_counter.config(text="未加载")

    def prev_image(self):
        """上一张"""
        if self.target_imgs and self.current_index > 0:
            self.current_index -= 1
            self._update_target_preview()
            self._make_placeholder(self.preview_labels.get("result", tk.Label()), "追色结果将显示在这里")
            self.current_result = None

    def next_image(self):
        """下一张"""
        if self.target_imgs and self.current_index < len(self.target_imgs) - 1:
            self.current_index += 1
            self._update_target_preview()
            self._make_placeholder(self.preview_labels.get("result", tk.Label()), "追色结果将显示在这里")
            self.current_result = None

    def browse_output(self):
        """选择输出目录"""
        dirpath = filedialog.askdirectory(title="选择输出目录")
        if dirpath:
            self.output_dir = dirpath
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, dirpath)

    # ──────────────────────────────────────────
    #  追色处理
    # ──────────────────────────────────────────

    def start_color_transfer(self):
        """开始追色"""
        if self.reference_img is None:
            messagebox.showwarning("提示", "请先导入参考图")
            return
        if not self.target_imgs:
            messagebox.showwarning("提示", "请先导入目标图")
            return

        self._run_transfer_async()

    def _run_transfer_async(self):
        """异步执行追色"""
        thread = threading.Thread(target=self._color_transfer_thread, daemon=True)
        thread.start()

    def _color_transfer_thread(self):
        """追色线程"""
        try:
            progress = ProgressWindow(self.root, "追色处理中")
            _, target_img = self.target_imgs[self.current_index]

            if self.mode.get() == "pro" and self.ref_features is not None:
                # 专业模式
                progress.update_text("分析目标图...", 20)
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

            self.current_result = result

            def update_ui():
                self._show_preview("result", result)
                self.status_var.set("✓ 追色完成！")

            self.root.after(0, update_ui)
            progress.close()

        except Exception as e:
            import traceback
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理失败: {str(e)}\n{traceback.format_exc()}"))

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
            lut_val = self.lut_combo.get()
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
            self.root.after(0, lambda: messagebox.showerror("错误", f"导出失败: {str(e)}\n{traceback.format_exc()}"))

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
            messagebox.showwarning("提示", "请先导入目标图")
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
            self.root.after(0, lambda: messagebox.showinfo("完成", f"批量处理完成！\n\n已处理 {total} 张图片\n保存位置: {output_dir}"))

        except Exception as e:
            import traceback
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理失败: {str(e)}\n{traceback.format_exc()}"))

    def _reset_params(self):
        """重置参数"""
        self.tone_strength.set(0.8)
        self.color_strength.set(0.9)
        self.skin_protection.set(True)
        self.skin_strength.set(0.85)
        self.status_var.set("参数已重置")


def main():
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.2)
    app = ZsTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
