import threading
import queue
import io
import sys
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from video_to_pdf_phash import extract_frames_to_pdf, parse_crop


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try:
            # Prefer Per-Monitor V2 when available (Win10+)
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return
        except Exception:
            pass
        try:
            # Fallback to Per-Monitor (Win8.1+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            return
        except Exception:
            pass
        try:
            # Legacy system-wide awareness
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


class StreamToQueue(io.TextIOBase):
    def __init__(self, line_queue: "queue.Queue[str]") -> None:
        self._queue = line_queue

    def write(self, s: str) -> int:
        if s:
            self._queue.put(s)
        return len(s)

    def flush(self) -> None:
        return None


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("video2pdf GUI")
        # Adjust Tk scaling to current DPI for crisp rendering
        try:
            dpi = float(self.winfo_fpixels('1i'))  # pixels per inch
            self.tk.call('tk', 'scaling', dpi / 72.0)
        except Exception:
            pass
        self.geometry("760x680")
        self._build_ui()

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self.after(100, self._drain_log)

    def _build_ui(self) -> None:
        pad = 8
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        # File inputs
        file_frame = ttk.LabelFrame(main, text="输入/输出")
        file_frame.pack(fill=tk.X, padx=0, pady=(0, pad))

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.outdir_var = tk.StringVar()

        self._row(file_frame, "输入视频", self.input_var, browse=lambda: self._browse_file(self.input_var, [
            ("Video files", "*.mp4;*.mov;*.mkv;*.avi;*.flv;*.webm"), ("All files", "*.*")
        ]))
        self._row(file_frame, "输出PDF(可空)", self.output_var, browse=lambda: self._save_file(self.output_var, (
            ("PDF", "*.pdf"), ("All files", "*.*")
        )))
        self._row(file_frame, "帧输出目录(可空)", self.outdir_var, browse=lambda: self._choose_dir(self.outdir_var))

        # Parameters
        params = ttk.LabelFrame(main, text="参数")
        params.pack(fill=tk.X, padx=0, pady=(0, pad))

        self.sample_var = tk.StringVar(value="0.5")
        self.threshold_var = tk.StringVar(value="10")
        self.crop_var = tk.StringVar(value="")
        self.scale_width_var = tk.StringVar(value="")
        self.max_pages_var = tk.StringVar(value="")
        self.a4_var = tk.BooleanVar(value=False)

        self._row(params, "采样秒数", self.sample_var)
        self._row(params, "阈值(1-63)", self.threshold_var)
        self._row(params, "裁剪 x,y,w,h(可空)", self.crop_var)
        self._row(params, "统一宽度(像素,可空)", self.scale_width_var)
        self._row(params, "最大页数(可空)", self.max_pages_var)

        a4_row = ttk.Frame(params)
        a4_row.pack(fill=tk.X, pady=(4, 0))
        a4_cb = ttk.Checkbutton(a4_row, text="A4 排版", variable=self.a4_var)
        a4_cb.pack(side=tk.LEFT)

        # Auto-trim
        trim = ttk.LabelFrame(main, text="自动去白边")
        trim.pack(fill=tk.X, padx=0, pady=(0, pad))
        self.auto_trim_var = tk.BooleanVar(value=True)
        self.auto_trim_ratio_var = tk.StringVar(value="0.98")
        self.auto_trim_pad_var = tk.StringVar(value="6")
        self.auto_trim_sides_var = tk.StringVar(value="tb")

        trim_top = ttk.Frame(trim)
        trim_top.pack(fill=tk.X)
        ttk.Checkbutton(trim_top, text="启用自动去白边", variable=self.auto_trim_var).pack(side=tk.LEFT)

        grid = ttk.Frame(trim)
        grid.pack(fill=tk.X, pady=(4, 0))
        self._grid_row(grid, "比例阈值(0-1)", self.auto_trim_ratio_var, 0)
        self._grid_row(grid, "边界余量(px)", self.auto_trim_pad_var, 1)

        side_row = ttk.Frame(trim)
        side_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(side_row, text="方向").pack(side=tk.LEFT)
        side = ttk.Combobox(side_row, textvariable=self.auto_trim_sides_var, values=("tb", "all"), state="readonly", width=8)
        side.pack(side=tk.LEFT, padx=(8, 0))

        # Auto-crop
        ac = ttk.LabelFrame(main, text="自动裁剪(PPT区域)")
        ac.pack(fill=tk.X, padx=0, pady=(0, pad))
        self.auto_crop_var = tk.BooleanVar(value=True)
        self.auto_crop_pad_var = tk.StringVar(value="6")
        self.auto_crop_min_area_var = tk.StringVar(value="0.05")

        ac_top = ttk.Frame(ac)
        ac_top.pack(fill=tk.X)
        ttk.Checkbutton(ac_top, text="启用自动裁剪", variable=self.auto_crop_var).pack(side=tk.LEFT)

        ac_grid = ttk.Frame(ac)
        ac_grid.pack(fill=tk.X, pady=(4, 0))
        self._grid_row(ac_grid, "外扩(px)", self.auto_crop_pad_var, 0)
        self._grid_row(ac_grid, "最小面积比例", self.auto_crop_min_area_var, 1)

        # Action and log
        action = ttk.Frame(main)
        action.pack(fill=tk.X, pady=(0, pad))
        self.run_btn = ttk.Button(action, text="开始运行", command=self._on_run)
        self.run_btn.pack(side=tk.LEFT)

        self.log = tk.Text(main, height=18, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        ttk.Scrollbar(self.log, command=self.log.yview)
        self.log.configure(state=tk.NORMAL)

    def _row(self, parent: tk.Widget, label: str, var: tk.StringVar, browse=None) -> None:
        r = ttk.Frame(parent)
        r.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r, text=label, width=20).pack(side=tk.LEFT)
        e = ttk.Entry(r, textvariable=var)
        e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if browse is not None:
            ttk.Button(r, text="浏览...", command=browse).pack(side=tk.LEFT, padx=(8, 0))

    def _grid_row(self, parent: tk.Widget, label: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, width=20).grid(row=row, column=0, sticky=tk.W, pady=2)
        e = ttk.Entry(parent, textvariable=var, width=16)
        e.grid(row=row, column=1, sticky=tk.W, pady=2)

    def _browse_file(self, var: tk.StringVar, types) -> None:
        path = filedialog.askopenfilename(title="选择视频", filetypes=types)
        if path:
            var.set(path)

    def _save_file(self, var: tk.StringVar, types) -> None:
        path = filedialog.asksaveasfilename(title="选择输出PDF", defaultextension=".pdf", filetypes=types)
        if path:
            var.set(path)

    def _choose_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            var.set(path)

    def _append_log(self, text: str) -> None:
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _drain_log(self) -> None:
        try:
            while True:
                chunk = self._log_queue.get_nowait()
                self._append_log(chunk)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _on_run(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("运行中", "请等待当前任务完成")
            return

        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("缺少输入", "请先选择输入视频")
            return

        self.log.delete("1.0", tk.END)
        self.run_btn.configure(state=tk.DISABLED)

        def work() -> None:
            stdout = StreamToQueue(self._log_queue)
            stderr = StreamToQueue(self._log_queue)
            try:
                input_p = Path(input_path)
                output_pdf_str = self.output_var.get().strip()
                output_pdf = Path(output_pdf_str) if output_pdf_str else input_p.with_suffix('.pdf')
                out_dir_str = self.outdir_var.get().strip()
                out_dir = Path(out_dir_str) if out_dir_str else input_p.parent / 'slides_phash'

                # Parse numerics with defaults
                sample_seconds = float(self.sample_var.get() or 0.5)
                threshold = int(self.threshold_var.get() or 10)
                crop = parse_crop(self.crop_var.get().strip() or None)
                scale_width = int(self.scale_width_var.get()) if (self.scale_width_var.get().strip()) else None
                max_pages = int(self.max_pages_var.get()) if (self.max_pages_var.get().strip()) else None
                a4 = bool(self.a4_var.get())

                auto_trim = bool(self.auto_trim_var.get())
                auto_trim_ratio = float(self.auto_trim_ratio_var.get() or 0.98)
                auto_trim_pad = int(self.auto_trim_pad_var.get() or 6)
                auto_trim_sides = self.auto_trim_sides_var.get() or 'tb'

                auto_crop = bool(self.auto_crop_var.get())
                auto_crop_pad = int(self.auto_crop_pad_var.get() or 6)
                auto_crop_min_area_ratio = float(self.auto_crop_min_area_var.get() or 0.05)

                from contextlib import redirect_stdout, redirect_stderr
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    extract_frames_to_pdf(
                        video_path=input_p,
                        output_pdf=output_pdf,
                        output_dir=out_dir,
                        sample_seconds=sample_seconds,
                        threshold=threshold,
                        crop_region=crop,
                        scale_width=scale_width,
                        a4=a4,
                        max_pages=max_pages,
                        auto_trim=auto_trim,
                        auto_trim_ratio=auto_trim_ratio,
                        auto_trim_pad=auto_trim_pad,
                        auto_trim_sides=auto_trim_sides,
                        auto_crop=auto_crop,
                        auto_crop_pad=auto_crop_pad,
                        auto_crop_min_area_ratio=auto_crop_min_area_ratio,
                    )
                self._log_queue.put("\n完成.\n")
            except Exception as exc:
                self._log_queue.put(f"\n错误: {exc}\n")
                messagebox.showerror("运行失败", str(exc))
            finally:
                self.run_btn.configure(state=tk.NORMAL)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()


if __name__ == "__main__":
    _enable_windows_dpi_awareness()
    App().mainloop()


