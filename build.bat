@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem =========================
rem Config
rem =========================
set "APP_NAME=CameraRecorderGUI"
set "PYTHON=python"
set "ICON=app.ico"
set "MAIN=motion-clip-recorder.py"
set "FFMPEG_SRC=ffmpeg.exe"
set "LOG=build_log.txt"

> "%LOG%" echo [START] %DATE% %TIME%
>>"%LOG%" echo Project: %APP_NAME%
>>"%LOG%" echo.

goto :main

rem =========================
rem Helper
rem =========================
:run
echo.
echo ===== RUN: %* =====
>>"%LOG%" echo ===== RUN: %* =====
call %* >>"%LOG%" 2>&1
if errorlevel 1 goto :err
exit /b 0

:tail
echo.
echo === Tail of %LOG% (last 60 lines) ===
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 60"
echo.
pause
exit /b

:err
echo.
echo #############################################
echo [BUILD FAILED] See %LOG% for details.
echo #############################################
>>"%LOG%" echo [BUILD FAILED]
goto :tail

rem =========================
rem Main
rem =========================
:main
where %PYTHON% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  >>"%LOG%" echo [ERROR] Python not found in PATH.
  goto :tail
)

if not exist "%MAIN%" (
  echo [ERROR] Can't find "%MAIN%" in this folder.
  >>"%LOG%" echo [ERROR] Missing %MAIN%
  goto :tail
)

rem ---- SAFE CLEANUP (don’t fail if not present) ----
if exist build (
  call :run rmdir /s /q build
) else (
  echo [INFO] build\ not found, skipping. & >>"%LOG%" echo [INFO] no build dir
)
if exist dist (
  call :run rmdir /s /q dist
) else (
  echo [INFO] dist\ not found, skipping. & >>"%LOG%" echo [INFO] no dist dir
)
if exist "%APP_NAME%.spec" (
  call :run del /q "%APP_NAME%.spec"
) else (
  echo [INFO] %APP_NAME%.spec not found, skipping. & >>"%LOG%" echo [INFO] no spec
)

rem ---- VENV & DEPS ----
call :run %PYTHON% -m venv .venv
call :run .venv\Scripts\activate
call :run python -m pip install --upgrade pip
call :run pip install opencv-python Pillow sounddevice numpy pyinstaller

rem ---- Portable import test ----
call :run python -c "import cv2, PIL.Image, PIL.ImageTk, sounddevice, numpy; print('Imports OK')"

rem ---- Build exe ----
set "ICON_ARG="
if exist "%ICON%" (
  set "ICON_ARG=--icon "%ICON%""
) else (
  echo [WARN] %ICON% not found, building without icon.
  >>"%LOG%" echo [WARN] Icon missing.
)

call :run pyinstaller --clean --noconfirm --noconsole --onefile %ICON_ARG% --name "%APP_NAME%" ^
  --hidden-import=PIL._tkinter_finder "%MAIN%"

rem ---- Bundle ffmpeg (optional) ----
if exist "%FFMPEG_SRC%" (
  echo.
  echo [+] Copying ffmpeg.exe...
  >>"%LOG%" echo [+] Copying ffmpeg.exe...
  call :run copy /Y "%FFMPEG_SRC%" "dist\ffmpeg.exe"
) else (
  echo [INFO] ffmpeg.exe not found at %FFMPEG_SRC% (optional).
  >>"%LOG%" echo [INFO] No ffmpeg at %FFMPEG_SRC%
)

echo.
echo [SUCCESS] Build finished. EXE: dist\%APP_NAME%.exe
>>"%LOG%" echo [SUCCESS] Build finished.
goto :tail
