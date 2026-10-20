@echo off
setlocal
title Cranfield Hybrid IR Launcher
cd /d "%~dp0"

echo ============================================
echo      CRANFIELD HYBRID IR - STARTUP
echo ============================================
echo.
echo Project folder: %CD%
echo.

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Expected: %~dp0.venv\Scripts\python.exe
    echo Make sure this file is saved in the project root:
    echo C:\Users\saumy\cranfield-hybrid-ir
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0frontend_server.py" (
    echo ERROR: frontend_server.py was not found in this folder.
    echo Copy this launcher to the project root beside frontend_server.py.
    echo.
    pause
    exit /b 1
)

echo Starting Python search server in a separate window...
start "Cranfield Hybrid IR Server" cmd /k ""%~dp0.venv\Scripts\python.exe" "%~dp0frontend_server.py""

echo.
echo Waiting for the server to load the dataset and index.
echo This may take a minute or two on the first start.
echo.

set /a attempts=0

:check_server
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if %errorlevel%==0 goto server_ready

set /a attempts+=1
if %attempts% GEQ 90 goto server_timeout

timeout /t 2 /nobreak >nul
goto check_server

:server_ready
echo.
echo SUCCESS: Cranfield search server is responding.
echo Opening http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
echo.
echo Keep the "Cranfield Hybrid IR Server" window open while using the website.
echo To stop the server, press Ctrl+C in that server window.
echo.
pause
exit /b 0

:server_timeout
echo.
echo The server did not respond within 3 minutes.
echo Check the "Cranfield Hybrid IR Server" window for a Python error.
echo Do not close that window until you have copied the error message.
echo.
pause
exit /b 1
