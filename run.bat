@echo off
REM Khởi động web tool tại http://localhost:8000 (Windows)
cd /d "%~dp0"
if "%PORT%"=="" set PORT=8000
python -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
