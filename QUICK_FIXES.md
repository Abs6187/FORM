# Quick Fixes for Indian Sign Language Recognition

## Critical Fixes Needed

### Fix 1: Path Separator Issues (Cross-platform Compatibility)

The code currently uses Windows-style path separators `\` which won't work on Linux/Mac.

#### Files to Fix:

**1. app.py (Line 35)**
```python
# Current (❌ Windows only)
model.load_weights(r"lstm-model\170-0.83.hdf5")

# Fixed (✅ Cross-platform)
model.load_weights("lstm-model/170-0.83.hdf5")
```

**2. deploy-code.py**

Line 28:
```python
# Current (❌)
model.load_weights(r"lstm-model\170-0.83.hdf5")

# Fixed (✅)
model.load_weights("lstm-model/170-0.83.hdf5")
```

Line 50:
```python
# Current (❌)
out= cv2.VideoWriter('input-video\input.mp4', cv2.VideoWriter_fourcc(*'DIVX'),10, (width,height))

# Fixed (✅)
out= cv2.VideoWriter('input-video/input.mp4', cv2.VideoWriter_fourcc(*'DIVX'),10, (width,height))
```

Line 67:
```python
# Current (❌)
out_np_array=convert_video_to_pose_embedded_np_array("input-video\input.mp4",remove_input=False)

# Fixed (✅)
out_np_array=convert_video_to_pose_embedded_np_array("input-video/input.mp4",remove_input=False)
```

**3. run-through-cmd-line.py (Line 47)**
```python
# Current (❌)
model.load_weights(r"lstm-model\170-0.83.hdf5")

# Fixed (✅)
model.load_weights("lstm-model/170-0.83.hdf5")
```

---

### Fix 2: Typo in helper_functions.py (Line 88)

```python
# Current (❌ - has parentheses)
key_points_shape = keypoints.shape()

# Fixed (✅ - no parentheses)
key_points_shape = keypoints.shape
```

---

## Automated Fix Script

Save this as `fix_paths.py` and run it to automatically fix all path issues:

```python
import os

def fix_file(filepath, replacements):
    """Fix path separators in a file"""
    with open(filepath, 'r') as f:
        content = f.read()

    for old, new in replacements:
        content = content.replace(old, new)

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✅ Fixed: {filepath}")

# Fix app.py
fix_file('Indian-Sign-Language-Recognition/app.py', [
    (r'r"lstm-model\170-0.83.hdf5"', '"lstm-model/170-0.83.hdf5"')
])

# Fix deploy-code.py
fix_file('Indian-Sign-Language-Recognition/deploy-code.py', [
    (r'r"lstm-model\170-0.83.hdf5"', '"lstm-model/170-0.83.hdf5"'),
    (r"'input-video\input.mp4'", "'input-video/input.mp4'")
])

# Fix run-through-cmd-line.py
fix_file('Indian-Sign-Language-Recognition/run-through-cmd-line.py', [
    (r'r"lstm-model\170-0.83.hdf5"', '"lstm-model/170-0.83.hdf5"')
])

# Fix helper_functions.py
fix_file('Indian-Sign-Language-Recognition/helper_functions.py', [
    ('key_points_shape = keypoints.shape()', 'key_points_shape = keypoints.shape')
])

print("\n✅ All fixes applied successfully!")
```

Run it:
```bash
python fix_paths.py
```

---

## Manual Testing After Fixes

### Test 1: Import and Model Loading
```python
import sys
sys.path.append('Indian-Sign-Language-Recognition')
from helper_functions import convert_video_to_pose_embedded_np_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Test model initialization
actions = np.array(["Hello","How are you","thank you"])
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Load weights - this should work now
model.load_weights("Indian-Sign-Language-Recognition/lstm-model/170-0.83.hdf5")
print("✅ Model loaded successfully!")
```

### Test 2: API Server
```bash
cd Indian-Sign-Language-Recognition
python app.py
```

Then test:
```bash
curl http://localhost:8000/test/
```

Should return: `"working"`

---

## Additional Improvements (Optional)

### 1. Use os.path.join() for Better Path Handling

More robust approach:
```python
import os

# In app.py, deploy-code.py, run-through-cmd-line.py
model_path = os.path.join("lstm-model", "170-0.83.hdf5")
model.load_weights(model_path)

# In deploy-code.py
video_dir = "input-video"
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, "input.mp4")
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))
```

### 2. Add Error Handling

```python
import os

def initialize_model():
    """Initializes lstm model and loads the trained model weight"""
    model = Sequential()
    model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(45,258)))
    model.add(LSTM(128,return_sequences=True, activation = 'relu'))
    model.add(LSTM(256,return_sequences=True,activation="relu"))
    model.add(LSTM(64, return_sequences = False,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Better error handling
    model_path = "lstm-model/170-0.83.hdf5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_weights(model_path)
    print("✅ Model loaded successfully!")

    return model
```

### 3. Configuration File

Create `config.py`:
```python
import os

# Model configuration
MODEL_DIR = "lstm-model"
MODEL_FILE = "170-0.83.hdf5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Actions
ACTIONS = ["Hello", "How are you", "Thank you"]

# Video configuration
NUM_FRAMES = 45
KEYPOINTS_PER_FRAME = 258
VIDEO_DURATION = 5  # seconds

# Paths
INPUT_VIDEO_DIR = "input-video"
TRAINING_DATA_DIR = "training-data"

# Server configuration
HOST = "0.0.0.0"
PORT = 8000

# Supported video formats
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov']
```

Then use it:
```python
from config import MODEL_PATH, ACTIONS

model.load_weights(MODEL_PATH)
```

---

## Testing Checklist

After applying fixes, test:

- [ ] Model loads without path errors
- [ ] Web API starts successfully (`python app.py`)
- [ ] Web API `/test/` endpoint works
- [ ] Webcam script runs (`python deploy-code.py`)
- [ ] Command-line script works (`python run-through-cmd-line.py -i video.mp4`)
- [ ] Video processing completes without errors
- [ ] Predictions are generated correctly
- [ ] Helper functions work on all platforms

---

## Summary

### Required Fixes:
1. ✅ Change all `\` to `/` in path strings (4 locations)
2. ✅ Fix `keypoints.shape()` to `keypoints.shape` (1 location)

### Optional Improvements:
1. Use `os.path.join()` for paths
2. Add error handling
3. Create configuration file
4. Add logging
5. Add input validation

**Time to fix**: ~5 minutes

**Impact**: Makes the code cross-platform compatible and more robust

**Priority**: HIGH - Required for Linux/Mac users
