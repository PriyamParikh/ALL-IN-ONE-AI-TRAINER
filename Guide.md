# ALL-IN-ONE-AI User Guide

This guide explains how to install and use all five modules of ALL-IN-ONE-AI.

## 1. Installation

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install core packages:

```bash
pip install numpy pandas scipy matplotlib pillow joblib scikit-learn opencv-python openpyxl
pip install ultralytics
pip install sounddevice soundfile librosa
```

Verify the main imports:

```bash
python -c "import cv2, numpy, pandas, sklearn, matplotlib, ultralytics, librosa; print('Core libraries OK')"
```

## 2. Launch

From the project folder:

```bash
python ALL_IN_ONE_AI.py
```

The launcher provides five studios:

1. Image AI
2. Object AI
3. Pose AI
4. Audio AI
5. Data AI

---

# 3. Image AI

## Purpose

Image AI assigns one class name to the complete image or webcam frame.

Example:

- Pin_Box
- Ink_Pad

## Step 1: Prepare Data

Folder datasets should follow:

```text
dataset/
├── Pin_Box/
│   ├── image001.jpg
│   └── ...
└── Ink_Pad/
    ├── image001.jpg
    └── ...
```

Alternatively, use webcam capture from inside the GUI.

Collect variation in:

- orientation
- distance
- lighting
- position
- background

## Step 2: Extract Features

The trainer uses HOG-based feature extraction to convert each image into a numerical vector.

If too few valid images are extracted, check that:

- the correct parent dataset folder is selected
- files are valid JPG/PNG/BMP images
- class folders contain actual images
- no corrupted files are present

## Step 3: Select and Train Models

Choose four or five models from:

- Random Forest
- Decision Tree
- SVM
- KNN
- Logistic Regression
- Neural Network

Set the test fraction, random state, cross-validation folds, and confidence threshold where available.

Select **Train and Evaluate**.

## Step 4: Evaluate

Review:

- accuracy
- precision
- recall
- specificity
- F1
- balanced accuracy
- MCC
- ROC-AUC
- PR-AUC
- confusion matrices
- ROC/PR curves
- PCA scatter
- learning curve
- feature importance

## Step 5: Save and Test

Save the selected model as `.joblib`, then open the real-time webcam test.

Image AI predicts one label for the full webcam frame.

---

# 4. Object AI: YOLO Multi-Object Detection

## Purpose

Object AI identifies both the class and spatial location of multiple objects in one image.

## Step 1: Create Classes

Example:

- Pin_Box
- Ink_Pad

## Step 2: Capture or Import Images

Include:

- single-object scenes
- multiple-object scenes
- different positions
- different scales
- different backgrounds

## Step 3: Annotate

Draw a tight bounding box around every target object and assign the correct class.

Every intended training object should be annotated.

## Step 4: Prepare the YOLO Dataset

A typical structure is:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

## Step 5: Train

Choose:

- pretrained YOLO weight
- epochs
- image size
- batch size
- device
- run name

Start training.

A successful training run normally creates:

```text
weights/
├── best.pt
└── last.pt
```

Use `best.pt` for inference.

## Step 6: Review Results

Typical YOLO outputs include:

- precision
- recall
- F1-confidence
- precision-confidence
- recall-confidence
- Precision-Recall curve
- confusion matrix
- mAP@0.5
- mAP@0.5:0.95
- training and validation losses

## Step 7: Real-Time Test

Load `best.pt`, start webcam detection, and inspect:

- bounding boxes
- labels
- confidence scores

Multiple classes can be detected simultaneously.

---

# 5. Pose AI

## Purpose

Pose AI learns user-defined body poses without manual keypoint annotation.

Example:

- Hands Up
- Hands Down

## Step 1: Create Pose Names

Enter two to five pose names.

## Step 2: Record Samples

Select one pose and record webcam samples.

Keep the body visible and include small natural variations.

## Step 3: Keypoint Extraction

YOLO Pose automatically extracts the human skeleton and keypoints.

Features can include:

- normalised keypoint coordinates
- confidence values
- joint angles
- wrist positions
- inter-joint distances

## Step 4: Train Models

Choose four or five classifiers from RF, DT, SVM, KNN, LR and NN.

Train and evaluate.

## Step 5: Review Results

Possible outputs:

- confusion matrix
- ROC/PR curves
- PCA pose scatter
- learning curves
- feature importance
- accuracy
- balanced accuracy
- F1
- AUC

## Step 6: Real-Time Recognition

Save/load the personalised model, then start webcam testing.

The program extracts keypoints and predicts the named pose with confidence.

---

# 6. Audio AI

## Purpose

Audio AI trains personalised sound classes from microphone recordings.

Example:

- Drum
- Piano

## Step 1: Create Classes

Create two to five sound classes.

## Step 2: Record Samples

Record labelled clips using the built-in or USB microphone.

For stronger datasets, vary recording sessions and avoid collecting only duplicate clips.

## Step 3: Feature Extraction

The software can use:

- MFCC
- delta MFCC
- RMS energy
- zero-crossing rate
- spectral centroid
- bandwidth
- roll-off
- flatness
- spectral flux
- dominant frequency

## Step 4: Train Models

Choose four or five classifiers from RF, DT, SVM, KNN, LR and NN.

Train and evaluate.

## Step 5: Review Results

Possible outputs include:

- confusion matrix
- ROC/PR curves
- PCA audio scatter
- learning curve
- feature importance
- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

## Step 6: Live Microphone Test

Save/load the best model and start real-time testing.

Incoming sound is classified into a trained class or an uncertain state when confidence is below the selected threshold.

---

# 7. Data AI

## Purpose

Data AI trains classification or regression models from CSV, TSV, or Excel data.

## Step 1: Import Data

Example MPU6050 dataset:

```text
AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Motion
0.12,0.02,0.98,1.2,0.3,0.7,Motion_A
...
```

Each row is one observation.

## Step 2: Inspect the Dataset

Data AI can automatically detect:

- columns
- numerical/categorical types
- missing values
- target characteristics

## Step 3: Choose the Target

For classification, choose a class column such as `Motion`.

For regression, choose a continuous output variable.

## Step 4: Automatic Preprocessing

The pipeline can perform:

- missing-value imputation
- categorical encoding
- scaling
- predictor/target separation

## Step 5: Train Models

Classification options can include RF, DT, SVM, KNN, LR and NN.

Regression can use corresponding regressors such as RF Regressor, DT Regressor, SVR, KNN Regressor, Ridge/linear models and NN regressors.

## Step 6: Evaluate

Classification:

- accuracy
- precision
- recall
- F1
- balanced accuracy
- ROC-AUC
- PR-AUC
- confusion matrix
- ROC/PR curves
- PCA
- learning curve
- feature importance

Regression:

- MAE
- RMSE
- R²
- actual vs predicted
- residual plots

## Step 7: Save Pipeline

Save preprocessing and the trained model together as `.joblib` whenever possible.

## Step 8: Batch Prediction

Load new CSV/Excel data, run predictions, and export the results.

---

# 8. Data Collection Recommendations

For all modules:

- use balanced classes where practical
- include realistic variation
- avoid duplicate samples
- retain genuinely unseen test data
- document acquisition conditions
- record class counts
- avoid data leakage

Perfect accuracy on a small controlled dataset should be interpreted as successful workflow validation, not universal generalisation.

---

# 9. Troubleshooting

## Webcam does not open

Try camera index `0`, `1`, or `2` and close other programs using the camera.

## Microphone does not record

Check OS microphone permissions and the selected input device.

To list audio devices:

```python
import sounddevice as sd
print(sd.query_devices())
```

## Too few valid images are extracted

Check:

- correct dataset folder
- valid image extensions
- readable files
- no corrupted images
- proper class-folder structure

## YOLO `best.pt` is missing

`best.pt` appears only after successful training.

Check:

- bounding-box annotations
- `data.yaml`
- pretrained model path
- training log
- run output directory

Typical location:

```text
runs/.../<run_name>/weights/best.pt
```

## `.pt` file does not load

The checkpoint may be empty, truncated, corrupted, or incompatible. Use a valid pretrained Ultralytics weight or retrain the custom model.

## Windows/PyTorch long-path error

Use a standard Python installation, short project paths, updated pip, and Windows long-path support if required.

## Pose prediction is unstable

Improve lighting, keep the body visible, record more varied samples, and avoid pose classes that are almost identical.

## Audio predicts one class continuously

Collect more varied samples, check class balance, microphone level, background noise, and confidence threshold.

## Unrealistically high Data AI accuracy

Check for data leakage, duplicated rows, target information inside predictor columns, or train/test samples from the same continuous recording.

---

# 10. Reproducibility Checklist

For research experiments, record:

```text
Python version:
Operating system:
CPU/GPU:
Module:
Dataset size:
Class counts:
Train/test split:
Random state:
Selected models:
Best model:
Hyperparameters:
Confidence threshold:
YOLO weight:
Epochs:
Image size:
Batch size:
Evaluation metrics:
```

---

# 11. GitHub Release Checklist

Before publishing:

- remove absolute local paths such as `C:\Users\...`
- remove personal/private datasets
- use relative project paths
- include `requirements.txt`
- include `README.md`
- include `Guide.md`
- add `LICENSE`
- add `.gitignore`
- add screenshots under `assets/`
- test the launcher from a fresh folder
- confirm output folders are created automatically
- remove API keys, tokens, passwords, or credentials

Suggested `.gitignore`:

```gitignore
__pycache__/
*.pyc
*.pyo
runs/
outputs/
.DS_Store
Thumbs.db
.env
```

Add `*.pt` and `*.joblib` only if you do not want trained models committed to GitHub.

---

# 12. Reporting an Issue

Include:

- operating system
- Python version
- module name
- full error message
- steps performed before the error
- screenshot
- dataset structure
- model filename

This information makes debugging significantly easier.
