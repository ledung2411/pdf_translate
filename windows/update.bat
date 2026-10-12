@echo off
chcp 65001 >nul
title Cap nhat - Dich PDF khoa hoc
cd /d "%~dp0.."
where git >nul 2>nul
if %errorlevel%==0 (
  if exist ".git" (
    git pull
  ) else (
    echo Thu muc nay khong phai ban git - hay tai ban zip moi nhat va giai nen de len.
  )
) else (
  echo Khong co git - hay tai ban zip moi nhat va giai nen de len thu muc nay.
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-ml.txt
echo Xong.
pause
