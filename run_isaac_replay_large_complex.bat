@echo off
setlocal

set ISAAC_PYTHON=C:\Users\kirar\isaac-sim-standalone-5.0.0-windows-x86_64\python.bat
set PROJECT_ROOT=%~dp0

if not exist "%ISAAC_PYTHON%" (
  echo [ERROR] Isaac Python not found: %ISAAC_PYTHON%
  echo Edit ISAAC_PYTHON in this .bat if your install path differs.
  pause
  exit /b 1
)

if "%~1"=="" (
  echo Usage: %~nx0 ^<path-to-coverage-lab-result.json^> [extra args...]
  echo Example ^(cmd.exe^):
  echo   %~nx0 results\lab\ideal_coverage_demo\ml_goal_allocated_seed0.json --geometry-yaml experiments_lab\scenes\large_complex_dynamic.yaml --stride 2
  echo PowerShell: MUST use .\ prefix  Example:
  echo   .\%~nx0 results\lab\ideal_coverage_demo\...
  echo Otherwise PowerShell will not start the script / empty Isaac stage.
  echo GIF: add --compat-renderer --gif-out path\to\stem_isaac.gif --camera topdown --gif-fps 24 --gif-substeps 4 --auto-close
  pause
  exit /b 1
)

REM По умолчанию: только D3D12 ^(без Vulkan^) — см. docs/ISAAC_LARGE_COMPLEX_REPLAY.md
REM   0 = выкл.   1 = D3D12   2 = D3D12+compatibilityMode ^(может чёрный RTX viewport^)
REM Отключить: set COVERAGE_LAB_ISAAC_COMPAT=0
if not defined COVERAGE_LAB_ISAAC_COMPAT set COVERAGE_LAB_ISAAC_COMPAT=1

echo [INFO] Isaac Sim 5 replay (CoverageLab JSON^)...
"%ISAAC_PYTHON%" "%PROJECT_ROOT%experiments\isaac5_replay_coverage_lab.py" %*

echo.
echo [INFO] Script finished.
pause
