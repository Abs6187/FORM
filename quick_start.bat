@echo off
REM ============================================================================
REM Indian Sign Language Recognition - Quick Start (FastAPI Server)
REM ============================================================================
REM This script quickly starts the FastAPI web server with minimal checks
REM ============================================================================

echo.
echo ============================================================================
echo ISL Recognition - Quick Start
echo ============================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Navigate to project directory
cd "%~dp0Indian-Sign-Language-Recognition"

REM Check if app.py exists
if not exist "app.py" (
    echo [ERROR] app.py not found!
    pause
    exit /b 1
)

REM Check model file
if not exist "lstm-model\170-0.83.hdf5" (
    echo [ERROR] Model file not found!
    pause
    exit /b 1
)

echo [INFO] Starting FastAPI server...
echo.
echo Server URL: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

REM Run the server
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server failed to start!
    echo Try running run_project.bat for detailed diagnostics.
    pause
)

exit /b 0
