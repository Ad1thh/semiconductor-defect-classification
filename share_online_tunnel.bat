@echo off
title Cloudflare Public Tunnel - Semiconductor Defect Classification
cd /d "%~dp0"
echo ========================================================
echo   STARTING SECURE CLOUDFLARE PUBLIC TUNNEL
echo ========================================================
echo.
echo Ensuring local inference server is accessible on port 8000...
echo.
cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
