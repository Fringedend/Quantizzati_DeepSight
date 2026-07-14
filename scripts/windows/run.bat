@echo off
title DeepSight
echo.
echo ==========================================================
echo   Avvio dell'applicazione tramite PowerShell...
echo ==========================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
echo.
echo Se l'applicazione non si e' avviata, controlla i messaggi sopra.
pause
