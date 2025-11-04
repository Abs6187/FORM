# Batch File Runner Documentation

This document explains how to use the batch files to run the Indian Sign Language Recognition project on Windows.

## Available Batch Files

### 1. `run_project.bat` - Full-Featured Runner
**Comprehensive project runner with error handling and multiple modes**

#### Features:
- ✅ Checks Python installation
- ✅ Verifies project directory structure
- ✅ Validates model file existence
- ✅ Checks and installs dependencies
- ✅ Runs installation tests
- ✅ Interactive menu with multiple running modes
- ✅ Detailed error messages and troubleshooting hints

#### Usage:
```batch
run_project.bat
```

#### Available Modes:
1. **Web API Server** - Start FastAPI server for production use
2. **Webcam Detection** - Real-time sign language detection from webcam
3. **Command-line Processing** - Process a single video file
4. **Installation Test** - Verify all dependencies and setup
5. **Exit** - Close the application

#### When to Use:
- First-time setup
- When you need to test different modes
- When troubleshooting issues
- When you want comprehensive error checking

---

### 2. `quick_start.bat` - Fast API Launcher
**Quick launcher for the FastAPI web server**

#### Features:
- 🚀 Fast startup with minimal checks
- 🎯 Directly starts the web API server
- ⚡ No interactive menu
- ✅ Basic validation (Python, model file)

#### Usage:
```batch
quick_start.bat
```

#### When to Use:
- When you've already run the full setup
- For daily development work
- When you need to quickly start the API server
- When all dependencies are already installed

---

## Prerequisites

### System Requirements:
- **OS**: Windows 7 or higher
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for dependencies

### Required Software:
1. **Python 3.8+** - [Download from python.org](https://www.python.org/)
   - Make sure to check "Add Python to PATH" during installation
2. **Git** (optional) - For cloning the repository
3. **Webcam** (optional) - Only needed for real-time detection mode

---

## First-Time Setup

### Step 1: Verify Python Installation
Open Command Prompt and run:
```batch
python --version
```
You should see something like `Python 3.10.x`

### Step 2: Run Full Setup
Double-click `run_project.bat` or run from Command Prompt:
```batch
cd C:\path\to\FORM
run_project.bat
```

### Step 3: Install Dependencies
When prompted, choose **Yes** to install dependencies. This will install:
- FastAPI (web framework)
- TensorFlow (machine learning)
- MediaPipe (pose detection)
- OpenCV (video processing)
- And other required packages

### Step 4: Run Tests
Select option **4** from the menu to run installation tests and verify everything is working.

---

## Usage Examples

### Example 1: Start Web API Server

**Option A: Using Full Runner**
```batch
run_project.bat
```
Then select option **1** from the menu.

**Option B: Using Quick Start**
```batch
quick_start.bat
```

**Result:**
- Server starts at `http://localhost:8000`
- API documentation at `http://localhost:8000/docs`
- Upload videos via POST to `/upload-video/`

---

### Example 2: Test with Webcam

```batch
run_project.bat
```
Select option **2** from the menu.

**What happens:**
- Opens your webcam
- Captures 10 videos (5 seconds each)
- Processes each video and shows predictions
- Predictions: "Hello", "How are you", or "thank you"

---

### Example 3: Process Existing Video

```batch
run_project.bat
```
Select option **3** from the menu, then enter the path to your video:
```
Video path: C:\Users\YourName\Videos\sign_language_video.mp4
```

**Supported formats:**
- `.mp4`
- `.avi`
- `.mov`

---

## Error Handling

The batch files include comprehensive error handling for common issues:

### Error: "Python is not installed or not in PATH"
**Solution:**
1. Install Python from https://www.python.org/
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt

### Error: "Model file not found"
**Solution:**
1. Verify `Indian-Sign-Language-Recognition/lstm-model/170-0.83.hdf5` exists
2. If missing, download the model file from the repository

### Error: "Failed to install dependencies"
**Solution:**
1. Run as Administrator
2. Try manual installation:
   ```batch
   cd Indian-Sign-Language-Recognition
   pip install -r requirements.txt
   ```
3. Check internet connection

### Error: "Webcam not connected"
**Solution:**
1. Ensure webcam is plugged in
2. Close other applications using the webcam (Zoom, Skype, etc.)
3. Check webcam permissions in Windows Settings

### Error: "FastAPI server failed to start"
**Solution:**
1. Check if port 8000 is already in use
2. Close other applications using port 8000
3. Try running:
   ```batch
   netstat -ano | findstr :8000
   ```
   And kill the process using that port

---

## Advanced Configuration

### Changing the Server Port
Edit `app.py` and change the port in the `uvicorn.run()` call:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change 8000 to 8080
```

### Custom Video Input Directory
Edit `deploy-code.py` or `run-through-cmd-line.py` to change input paths.

### Modifying Error Logging
Batch files output to console. To save logs:
```batch
run_project.bat > output.log 2>&1
```

---

## API Usage After Starting Server

### Using curl (Windows)
```batch
curl -X POST "http://localhost:8000/upload-video/" -F "file=@video.mp4"
```

### Using Python requests
```python
import requests

url = "http://localhost:8000/upload-video/"
files = {"file": open("video.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using Browser
Navigate to `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

---

## Troubleshooting Checklist

- [ ] Python 3.8+ installed and in PATH
- [ ] All files in `Indian-Sign-Language-Recognition/` directory
- [ ] Model file exists: `lstm-model/170-0.83.hdf5`
- [ ] Requirements.txt exists
- [ ] Dependencies installed (`pip list` shows tensorflow, fastapi, etc.)
- [ ] No other application using port 8000
- [ ] Webcam connected (for webcam mode)
- [ ] Video file exists and in supported format (for video mode)

---

## Performance Tips

1. **First run is slow** - TensorFlow initialization takes time
2. **Webcam lag** - Normal for real-time processing, depends on CPU
3. **GPU acceleration** - Install tensorflow-gpu for faster processing
4. **Memory usage** - Close other applications if RAM is limited

---

## Getting Help

If you encounter issues not covered here:

1. Run the installation test:
   ```batch
   run_project.bat
   ```
   Select option **4**

2. Check the main documentation:
   - `ISL_DEPLOYMENT_GUIDE.md` - Complete deployment guide
   - `PROJECT_SUMMARY.md` - Project overview
   - `QUICK_FIXES.md` - Common bug fixes

3. Check error logs in the console output

---

## Technical Details

### What the Batch Files Do:

**Validation Steps:**
1. Check Python installation
2. Verify project directory structure
3. Check requirements.txt exists
4. Validate model file presence
5. Test import of critical packages

**Installation Steps:**
1. Navigate to project directory
2. Run `pip install -r requirements.txt`
3. Verify installations with test imports

**Execution Steps:**
1. Change to appropriate directory
2. Run selected Python script
3. Capture exit codes
4. Display error messages if failed

### Error Codes:
- `0` - Success
- `1` - General error
- `2` - File not found
- Other - Python script-specific errors

---

## License & Credits

This batch file system is part of the Indian Sign Language Recognition project.

**Project Components:**
- Deep Learning: TensorFlow/Keras LSTM
- Pose Detection: MediaPipe Holistic
- Web Framework: FastAPI
- Video Processing: OpenCV

**Recognized Words:**
- "Hello"
- "How are you"
- "thank you"

---

## Quick Reference

| Task | Command |
|------|---------|
| Full setup with menu | `run_project.bat` |
| Quick start API server | `quick_start.bat` |
| Install dependencies | `run_project.bat` → Option 1 → Yes |
| Test installation | `run_project.bat` → Option 4 |
| Webcam detection | `run_project.bat` → Option 2 |
| Process video file | `run_project.bat` → Option 3 |

---

**Last Updated:** 2025-11-04
**Compatible with:** Windows 7/8/10/11
**Python Version:** 3.8 - 3.10 (recommended)
