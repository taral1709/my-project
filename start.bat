@echo off
title CinePredict.ai - Starting...
color 0A

echo.
echo  ========================================
echo   CinePredict.ai - Starting Servers
echo  ========================================
echo.

echo  [1/2] Starting Backend (FastAPI on port 8000)...
start "CinePredict Backend" cmd /k "cd /d %~dp0backend && python main.py"

echo  [2/2] Starting Frontend (Next.js on port 3000)...
start "CinePredict Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo  ========================================
echo   Both servers are starting up...
echo   Wait ~10 seconds, then open:
echo   http://localhost:3000
echo  ========================================
echo.

timeout /t 10 /nobreak > nul

echo  Opening website in browser...
start "" "http://localhost:3000"

echo.
echo  Done! You can close this window.
pause
