@echo off
chcp 65001 >nul
title Cai dat - Dich PDF khoa hoc
REM Double-click to install. Arguments are passed to install.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo.
  echo Cai dat chua thanh cong. Xem file install.log trong thu muc du an.
  pause
)
