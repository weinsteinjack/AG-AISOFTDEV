@echo off
echo ========================================
echo Starting React Component Preview Server
echo ========================================
echo.
echo Server will start on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
echo Opening browser in 3 seconds...
echo.

timeout /t 3 /nobreak >nul
start http://localhost:8000

python -m http.server 8000

