@echo off
setlocal enableextensions

REM Build Windows executable for video_to_pdf_phash.py using PyInstaller
REM Output will be under .\dist\video2pdf\ (onedir) or .\dist\video2pdf.exe (onefile if adjusted in spec)

set REPO_DIR=%~dp0
pushd "%REPO_DIR%"

REM Resolve Python
set PY_CMD=py -3
%PY_CMD% -V >nul 2>&1 || set PY_CMD=python
%PY_CMD% -V >nul 2>&1 || (
  echo Python not found. Please install Python 3 and add it to PATH.
  goto :error_pause
)

if not exist .venv (
  echo Creating venv...
  %PY_CMD% -m venv .venv || goto :error_pause
)

call .venv\Scripts\activate.bat || goto :error_pause
python -m pip install --upgrade pip wheel || goto :error_pause
if exist requirements.txt (
  python -m pip install -r requirements.txt || goto :error_pause
) else (
  echo requirements.txt not found, continuing...
)
python -m pip install pyinstaller || goto :error_pause

REM Clean previous build
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build GUI (windowed)
pyinstaller --clean "%REPO_DIR%video_to_pdf_gui.spec" || goto :error_pause

echo.
echo Build finished. Output in %REPO_DIR%dist\
echo.
pause
popd
exit /b 0

:error
echo Build failed with error %errorlevel%.
popd
exit /b %errorlevel%

:error_pause
echo.
echo Build failed with error %errorlevel%.
echo Please read the error above. Press any key to close.
pause
popd
exit /b %errorlevel%

