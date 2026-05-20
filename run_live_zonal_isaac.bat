@echo off
setlocal

set ISAAC_PYTHON=C:\Users\kirar\isaac-sim-standalone-5.0.0-windows-x86_64\python.bat
set PROJECT_ROOT=%~dp0

if not exist "%ISAAC_PYTHON%" (
  echo [ERROR] Isaac Python not found: %ISAAC_PYTHON%
  pause
  exit /b 1
)

echo [INFO] Starting zonal live Isaac Sim run...
"%ISAAC_PYTHON%" "%PROJECT_ROOT%experiments\run_zonal_live.py"

echo.
echo [INFO] Script finished.
pause

