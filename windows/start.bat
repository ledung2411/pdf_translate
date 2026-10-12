@echo off
chcp 65001 >nul
title Dich PDF khoa hoc Anh - Viet
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Chua cai dat. Hay chay windows\install.bat truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.launcher %*
if errorlevel 1 pause
