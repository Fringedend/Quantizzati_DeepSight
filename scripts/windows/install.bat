@echo off
title Installatore DeepSight
echo.
echo ==========================================================
echo   Avvio dell'installatore tramite PowerShell...
echo ==========================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
