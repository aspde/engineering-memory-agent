@echo off
cd /d G:\Projects\ema\frontend

echo.
echo ============================================
echo   EMA Frontend
echo ============================================
echo.

:: Check if node_modules exists
if not exist "node_modules\" (
    echo Installing dependencies...
    call npm install
    echo.
)

echo Starting EMA frontend on http://localhost:5173
echo Backend proxy: http://localhost:8000
echo.
call npm run dev
