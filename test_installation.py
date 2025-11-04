#!/usr/bin/env python3
"""
Test Script for Indian Sign Language Recognition System
This script verifies that all dependencies are installed and the model loads correctly.
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)

    packages = {
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'skvideo': 'sk-video',
        'pandas': 'pandas'
    }

    failed = []
    for package, pip_name in packages.items():
        try:
            __import__(package)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name} - NOT INSTALLED")
            failed.append(pip_name)

    if failed:
        print("\n" + "=" * 60)
        print("❌ Missing packages detected!")
        print("=" * 60)
        print("Install them with:")
        print(f"pip install {' '.join(failed)}")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True

def test_model_file():
    """Test if the model file exists"""
    print("\n" + "=" * 60)
    print("Checking Model File...")
    print("=" * 60)

    model_path = "Indian-Sign-Language-Recognition/lstm-model/170-0.83.hdf5"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Model file NOT found: {model_path}")
        print("   Please ensure you have cloned the repository correctly.")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n" + "=" * 60)
    print("Testing Model Loading...")
    print("=" * 60)

    try:
        sys.path.insert(0, 'Indian-Sign-Language-Recognition')

        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        actions = np.array(["Hello", "How are you", "thank you"])

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        model.load_weights("Indian-Sign-Language-Recognition/lstm-model/170-0.83.hdf5")

        print("✅ Model loaded successfully!")
        print(f"   Actions: {actions}")
        print(f"   Input shape: (45, 258)")
        print(f"   Output classes: {len(actions)}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def test_helper_functions():
    """Test if helper functions can be imported"""
    print("\n" + "=" * 60)
    print("Testing Helper Functions...")
    print("=" * 60)

    try:
        sys.path.insert(0, 'Indian-Sign-Language-Recognition')
        from helper_functions import mediapipe_detection, extract_keypoints, convert_video_to_pose_embedded_np_array

        print("✅ Helper functions imported successfully!")
        print("   - mediapipe_detection")
        print("   - extract_keypoints")
        print("   - convert_video_to_pose_embedded_np_array")
        return True

    except Exception as e:
        print(f"❌ Helper function import failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 Indian Sign Language Recognition - Installation Test")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Package imports
    results.append(("Package Imports", test_imports()))

    # Test 2: Model file
    results.append(("Model File", test_model_file()))

    # Test 3: Helper functions
    results.append(("Helper Functions", test_helper_functions()))

    # Test 4: Model loading
    results.append(("Model Loading", test_model_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Start the web API:")
        print("     cd Indian-Sign-Language-Recognition && python app.py")
        print("\n  2. Or test with webcam:")
        print("     cd Indian-Sign-Language-Recognition && python deploy-code.py")
        print("\n  3. Or process a video:")
        print("     cd Indian-Sign-Language-Recognition && python run-through-cmd-line.py -i video.mp4")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  1. Install missing packages:")
        print("     pip install -r Indian-Sign-Language-Recognition/requirements.txt")
        print("\n  2. Ensure you're in the correct directory")
        print("\n  3. Check if the repository was cloned completely")
        return 1

if __name__ == "__main__":
    sys.exit(main())
