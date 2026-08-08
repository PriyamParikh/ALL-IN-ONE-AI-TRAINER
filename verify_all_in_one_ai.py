
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = [
    "customtkinter",
    "cv2",
    "numpy",
    "scipy",
    "pandas",
    "PIL",
    "matplotlib",
    "sklearn",
    "joblib",
    "openpyxl",
    "xlrd",
    "yaml",
    "sounddevice",
    "torch",
    "ultralytics",
]

SCRIPTS = [
    "all_in_one_ai_5in1_launcher.py",
    "personalized_image_ai_trainer_gui_v3_manual_hog.py",
    "personalized_multi_object_ai_trainer_gui_v3.py",
    "personalized_yolo_pose_classifier_gui.py",
    "personalized_audio_classifier_gui.py",
    "all_in_one_ai_data_trainer_gui.py",
]

print("ALL IN ONE AI — INSTALLATION VERIFICATION")
print("Python:", sys.version)
print("Executable:", sys.executable)
print()

all_ok = True
for module_name in REQUIRED:
    try:
        module = importlib.import_module(module_name)
        print(f"[OK] {module_name}: {getattr(module, '__version__', 'Installed')}")
    except Exception as error:
        all_ok = False
        print(f"[ERROR] {module_name}: {error}")

print("\nSuite files:")
base = Path(__file__).resolve().parent
for script_name in SCRIPTS:
    exists = (base / script_name).exists()
    all_ok = all_ok and exists
    print(f"[{'OK' if exists else 'MISSING'}] {script_name}")

try:
    import sounddevice
    print("\nMicrophone input devices:")
    found = False
    for index, device in enumerate(sounddevice.query_devices()):
        if int(device["max_input_channels"]) > 0:
            found = True
            print(f"  {index}: {device['name']} ({device['max_input_channels']} input channels)")
    if not found:
        print("  No input microphone was reported by PortAudio.")
except Exception as error:
    print("\nMicrophone listing failed:", error)

print("\nRESULT:", "READY" if all_ok else "ACTION REQUIRED")
