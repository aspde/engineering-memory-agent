@echo off
cd /d G:\Projects\ema
call .venv\Scripts\activate.bat

echo.
echo ============================================
echo   EMA Backend Server
echo ============================================
echo.

echo Stopping all Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting on http://localhost:8000
echo.
python -m uvicorn backend.main:app --port 8000 --host 0.0.0.0 --reload
