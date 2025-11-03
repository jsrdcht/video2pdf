# PyInstaller spec for video_to_pdf_phash.py
# Build (Windows): .venv\Scripts\pyinstaller.exe e:\video2pdf\video_to_pdf_phash.spec
# Build (macOS):    .venv/bin/pyinstaller e:/video2pdf/video_to_pdf_phash.spec

# NOTE: Onefile is convenient but large and slower to start with heavy deps (cv2, numpy).
# You can switch to onedir by setting onefile=False below.

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
# Be conservative; cv2/PIL hooks generally cover these, but allow extension when needed
hiddenimports += collect_submodules('cv2')


a = Analysis(
    ['video_to_pdf_phash.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='video2pdf',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,   # set to False to hide console (not recommended for this CLI tool)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='video2pdf'
)

