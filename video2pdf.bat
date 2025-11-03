@echo off
setlocal EnableDelayedExpansion

REM Windows runner for video_to_pdf_phash.py using conda env 'pytorch'
REM Usage: video2pdf.bat -i <video_path> [options]

set CONDA_ENV=pytorch
set SCRIPT=%~dp0video_to_pdf_phash.py

REM If input path contains non-ASCII, avoid PowerShell parsing: call via python -c
REM But here we directly pass args to python through conda run

if not exist "%SCRIPT%" (
  echo Script not found: %SCRIPT%
  exit /b 2
)

REM Prefer conda run to avoid activate issues
conda run -n %CONDA_ENV% python "%SCRIPT%" %*
set EXITCODE=%ERRORLEVEL%
exit /b %EXITCODE%
