@echo off
REM ============================================================================
REM Indian Sign Language Recognition - Project Runner
REM ============================================================================
REM This script runs the ISL Recognition project with comprehensive error handling
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo Indian Sign Language Recognition - Project Runner
echo ============================================================================
echo.

REM Set project directory
set "PROJECT_DIR=%~dp0Indian-Sign-Language-Recognition"
set "ROOT_DIR=%~dp0"

REM Color codes for better output (using built-in Windows commands)
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "INFO=[INFO]"
set "WARNING=[WARNING]"

REM ============================================================================
REM STEP 1: Check if Python is installed
REM ============================================================================
echo %INFO% Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %SUCCESS% Python %PYTHON_VERSION% found!
echo.

REM ============================================================================
REM STEP 2: Verify project directory exists
REM ============================================================================
echo %INFO% Checking project directory...
if not exist "%PROJECT_DIR%" (
    echo %ERROR% Project directory not found: %PROJECT_DIR%
    echo Please ensure the Indian-Sign-Language-Recognition folder exists.
    pause
    exit /b 1
)
echo %SUCCESS% Project directory found!
echo.

REM ============================================================================
REM STEP 3: Check if requirements.txt exists
REM ============================================================================
echo %INFO% Checking requirements file...
if not exist "%PROJECT_DIR%\requirements.txt" (
    echo %ERROR% requirements.txt not found in project directory!
    pause
    exit /b 1
)
echo %SUCCESS% Requirements file found!
echo.

REM ============================================================================
REM STEP 4: Check if model file exists
REM ============================================================================
echo %INFO% Checking model file...
if not exist "%PROJECT_DIR%\lstm-model\170-0.83.hdf5" (
    echo %ERROR% Model file not found: lstm-model\170-0.83.hdf5
    echo Please ensure the trained model file exists.
    pause
    exit /b 1
)
echo %SUCCESS% Model file found!
echo.

REM ============================================================================
REM STEP 5: Check/Install dependencies
REM ============================================================================
echo %INFO% Checking Python dependencies...
echo This may take a moment...
echo.

REM Check if critical packages are installed
python -c "import fastapi, tensorflow, mediapipe, cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo %WARNING% Dependencies are missing or incomplete.
    echo.
    choice /C YN /M "Would you like to install dependencies now"
    if !errorlevel! equ 1 (
        echo.
        echo %INFO% Installing dependencies from requirements.txt...
        cd "%PROJECT_DIR%"
        pip install -r requirements.txt
        if !errorlevel! neq 0 (
            echo %ERROR% Failed to install dependencies!
            pause
            exit /b 1
        )
        echo %SUCCESS% Dependencies installed successfully!
        echo.
    ) else (
        echo %ERROR% Cannot run project without dependencies!
        pause
        exit /b 1
    )
) else (
    echo %SUCCESS% All critical dependencies are installed!
    echo.
)

REM ============================================================================
REM STEP 6: Run installation test
REM ============================================================================
echo %INFO% Running installation tests...
cd "%ROOT_DIR%"
if exist "test_installation.py" (
    python test_installation.py
    if !errorlevel! neq 0 (
        echo %WARNING% Installation test failed! Some features may not work properly.
        echo.
        choice /C YN /M "Continue anyway"
        if !errorlevel! equ 2 (
            exit /b 1
        )
    ) else (
        echo %SUCCESS% Installation test passed!
    )
    echo.
)

REM ============================================================================
REM STEP 7: Display menu and run selected mode
REM ============================================================================
:menu
echo ============================================================================
echo Select Running Mode:
echo ============================================================================
echo.
echo 1. Web API Server (FastAPI - Production Mode)
echo 2. Webcam Real-time Detection (Testing Mode)
echo 3. Command-line Video Processing
echo 4. Run Installation Test Only
echo 5. Exit
echo.
choice /C 12345 /N /M "Enter your choice (1-5): "
set CHOICE=%errorlevel%

echo.
echo ============================================================================

if %CHOICE% equ 1 goto run_api
if %CHOICE% equ 2 goto run_webcam
if %CHOICE% equ 3 goto run_cmdline
if %CHOICE% equ 4 goto run_test
if %CHOICE% equ 5 goto end

REM ============================================================================
REM MODE 1: Web API Server
REM ============================================================================
:run_api
echo %INFO% Starting FastAPI Web Server...
echo.
cd "%PROJECT_DIR%"
if not exist "app.py" (
    echo %ERROR% app.py not found!
    pause
    goto end
)

echo Server will start on http://localhost:8000
echo.
echo Endpoints:
echo   - POST /upload-video/  : Upload video for prediction
echo   - GET  /test/          : Health check
echo.
echo Press CTRL+C to stop the server
echo.
python app.py
set APP_ERROR=!errorlevel!

if !APP_ERROR! neq 0 (
    echo.
    echo %ERROR% FastAPI server stopped with error code: !APP_ERROR!
    pause
)
goto end

REM ============================================================================
REM MODE 2: Webcam Real-time Detection
REM ============================================================================
:run_webcam
echo %INFO% Starting Webcam Detection Mode...
echo.
cd "%PROJECT_DIR%"
if not exist "deploy-code.py" (
    echo %ERROR% deploy-code.py not found!
    pause
    goto end
)

echo This will capture 10 videos (5 seconds each) from your webcam.
echo Make sure your webcam is connected and not in use.
echo.
pause
echo.

python deploy-code.py
set WEBCAM_ERROR=!errorlevel!

if !WEBCAM_ERROR! neq 0 (
    echo.
    echo %ERROR% Webcam detection failed with error code: !WEBCAM_ERROR!
    echo.
    echo Common issues:
    echo - Webcam not connected or in use by another application
    echo - Missing dependencies (OpenCV, MediaPipe)
    echo - Insufficient permissions
    pause
) else (
    echo.
    echo %SUCCESS% Webcam detection completed successfully!
    pause
)
goto end

REM ============================================================================
REM MODE 3: Command-line Video Processing
REM ============================================================================
:run_cmdline
echo %INFO% Command-line Video Processing Mode
echo.
cd "%PROJECT_DIR%"
if not exist "run-through-cmd-line.py" (
    echo %ERROR% run-through-cmd-line.py not found!
    pause
    goto end
)

echo Enter the full path to your video file:
echo (Supported formats: .mp4, .avi, .mov)
echo.
set /p VIDEO_PATH="Video path: "

if not exist "%VIDEO_PATH%" (
    echo %ERROR% Video file not found: %VIDEO_PATH%
    pause
    goto end
)

echo.
echo %INFO% Processing video...
echo.
python run-through-cmd-line.py -i "%VIDEO_PATH%"
set PROCESS_ERROR=!errorlevel!

if !PROCESS_ERROR! neq 0 (
    echo.
    echo %ERROR% Video processing failed with error code: !PROCESS_ERROR!
    echo.
    echo Common issues:
    echo - Unsupported video format
    echo - Corrupted video file
    echo - Video too short or too long
    pause
) else (
    echo.
    echo %SUCCESS% Video processed successfully!
    pause
)
goto end

REM ============================================================================
REM MODE 4: Run Installation Test
REM ============================================================================
:run_test
echo %INFO% Running Installation Tests...
echo.
cd "%ROOT_DIR%"
if not exist "test_installation.py" (
    echo %ERROR% test_installation.py not found!
    pause
    goto end
)

python test_installation.py
set TEST_ERROR=!errorlevel!

if !TEST_ERROR! neq 0 (
    echo.
    echo %ERROR% Installation test failed!
    echo Please check the error messages above and fix any issues.
) else (
    echo.
    echo %SUCCESS% All tests passed!
)
echo.
pause
goto menu

REM ============================================================================
REM END
REM ============================================================================
:end
echo.
echo ============================================================================
echo Thank you for using Indian Sign Language Recognition!
echo ============================================================================
echo.
endlocal
exit /b 0
