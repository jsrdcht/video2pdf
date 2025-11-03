## video2pdf: 从教学视频提取PPT为PDF

将视频中的PPT切换页自动抽取为图片，并合成为 PDF（每页一张）。

### 功能
- pHash 相似度去重，保留真正的“翻页”帧
- 自动裁剪（Auto-crop）：从首帧检测 PPT 所在白色/浅色区域
- 自动去白边（Auto-trim）：去除每页顶部/底部或四周白色留白
- 可选 A4 版式排版

### 依赖
- Python 3.8+
- OpenCV, Pillow, NumPy（见 `requirements.txt`）

Windows（conda 环境 `pytorch`）：
```
conda run -n pytorch pip install -r requirements.txt
```

### 一键运行（Windows）
```
.\\video2pdf.bat -i "E:\\video2pdf\\input.mp4" --auto-crop --auto-trim
```

#### 常用参数
- `-i, --input`：输入视频
- `-o, --output`：输出 PDF（默认与视频同名）
- `--out-dir`：抽取的图片输出目录（默认 `slides_phash`）
- `--sample-seconds`：采样间隔秒数，数值越小越密集（默认 0.5）
- `--threshold`：pHash 汉明距离阈值，越大保留越少（默认 10，常用 8–16）
- `--auto-crop`：自动从首帧检测 PPT 区域
  - `--auto-crop-pad`：在检测框外扩像素余量（默认 6）
  - `--auto-crop-min-area-ratio`：最小区域面积占比阈值（默认 0.05）
- `--crop x,y,w,h`：手动裁剪，覆盖 `--auto-crop`
- `--auto-trim`：自动去白边（默认仅上下）
  - `--auto-trim-sides tb|all`：仅上下或四周（默认 tb）
  - `--auto-trim-ratio`：将每行/列白色占比≥该值视为留白（默认 0.98）
  - `--auto-trim-pad`：在裁切边界处各留出像素余量（默认 6）
- `--a4`：将图片居中放到 A4 纸张（300DPI）

#### 示例
1) 自动裁剪 + 自动去白边，生成标准 PDF：
```
.\\video2pdf.bat -i "E:\\video2pdf\\input.mp4" --auto-crop --auto-trim -o "E:\\video2pdf\\slides.pdf"
```

2) 指定裁剪框并去四周白边：
```
.\\video2pdf.bat -i "E:\\video2pdf\\input.mp4" --crop 9,187,1415,795 --auto-trim --auto-trim-sides all
```

3) 更密集抽帧并减少误报：
```
.\\video2pdf.bat -i "E:\\video2pdf\\input.mp4" --sample-seconds 0.3 --threshold 12 --auto-crop --auto-trim
```

### 说明
- 自动裁剪仅在首个被保留的帧上估计一次，并应用到后续帧，保证页面统一。
- 自动去白边同样基于首个保留帧计算统一边界，保证版式一致。
- 若视频含人物窗口/水印遮挡，建议配合手动 `--crop` 精确圈定 PPT 区域。

### 常见问题
- 结果页数太多：增大 `--threshold` 或增大 `--sample-seconds`。
- 漏页：减小 `--threshold` 或减小 `--sample-seconds`。
- 仍有留白：开启 `--auto-trim-sides all` 或微调 `--auto-trim-ratio`。


### 打包为可执行文件

#### Windows 本地打包（生成可点击的 exe）
1. 双击或在命令行运行：
   ```
   .\build_windows.bat
   ```
2. 产物在 `dist\video2pdf\` 目录下（“单目录 onedir” 方式，更稳定、启动更快）。
   - 运行示例：
     ```
     dist\video2pdf\video2pdf.exe -i "E:\\video2pdf\\input.mp4" --auto-crop --auto-trim
     ```
   - 若想做“单文件 onefile”可执行（启动稍慢，体积更大），可直接尝试：
     ```
     .venv\Scripts\pyinstaller --onefile video_to_pdf_phash.py
     ```

> 提示：这是一个命令行程序，双击 exe 若未传参会一闪而过。建议在命令行使用，或把常用参数写到你的 `video2pdf.bat` 中。

#### macOS 本地打包
在 macOS 上运行（不能在 Windows 上交叉编译 mac 版本）：
```
bash build_macos.sh
```
产物在 `dist/video2pdf/` 目录。
运行示例：
```
./dist/video2pdf/video2pdf -i input.mp4 --auto-crop --auto-trim
```

#### 一次性跨平台产物（CI）
项目内已提供 GitHub Actions 工作流：`.github/workflows/build.yml`。
- 在 GitHub 仓库中手动触发（Workflow Dispatch），或推送 `v*` 标签即可自动在 Windows 和 macOS Runner 上分别构建并上传产物。
- 无需自备 mac 机器，即可拿到两个平台的构建结果（Artifacts）。

#### 注意事项
- 不能在 Windows 上直接生成 macOS 可执行文件；请在 macOS 本地或使用 CI 构建。
- OpenCV/NumPy 体积较大，打包后的目录较大属正常；首次启动可能略慢。
- 如需隐藏控制台，可在 spec 中把 `console=True` 改为 `False`，但由于本工具会输出处理日志，隐藏控制台不利于排障。
