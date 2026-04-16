@echo off
setlocal

set ISAAC_PYTHON=C:\isaacsim\python.bat
set PROJECT_ROOT=%~dp0

if not exist "%ISAAC_PYTHON%" (
  echo [ERROR] Isaac Python not found: %ISAAC_PYTHON%
  pause
  exit /b 1
)

echo [INFO] Starting live Isaac Sim run...
"%ISAAC_PYTHON%" "%PROJECT_ROOT%experiments\run_random_walk_live.py"

echo.
echo [INFO] Script finished.
pause

