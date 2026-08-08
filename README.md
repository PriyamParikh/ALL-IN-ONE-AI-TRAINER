# ALL-IN-ONE-AI

A beginner-centred 5-in-1 AI/ML desktop trainer for STEM learners, AI/ML beginners, and educators.

ALL-IN-ONE-AI integrates five learning environments in one GUI:

1. Image AI — whole-image classification
2. Object AI — YOLO multi-object detection
3. Pose AI — personalised pose classification
4. Audio AI — microphone-based sound classification
5. Data AI — CSV/Excel classification and regression

The common learning workflow is:

**Collect/Import Data → Train → Test → Evaluate → Explain → Deploy**

## Key Features

- One launcher for five AI/ML modules
- Colour-coded, guided interface
- Webcam image acquisition
- YOLO bounding-box annotation and training
- Personalised pose-name creation
- Built-in/USB microphone recording
- CSV, TSV, and Excel import
- Multiple ML algorithm comparison
- Train/test validation
- Real-time inference
- Batch prediction
- Model saving/loading
- Confusion matrices
- ROC and Precision-Recall curves
- PCA scatter plots
- Learning curves
- Feature importance
- YOLO precision, recall, F1 and mAP
- Classification and regression metrics

## Modules

| Module | Input | Core Method |
|---|---|---|
| Image AI | Webcam / image folders | HOG + RF, DT, SVM, KNN, LR, NN |
| Object AI | Images / webcam | YOLO object detection |
| Pose AI | Webcam | YOLO Pose keypoints + ML classifiers |
| Audio AI | Built-in / USB microphone | MFCC/spectral features + ML |
| Data AI | CSV / TSV / Excel | Automatic preprocessing + classification/regression |

## Supported Algorithms

Depending on the module:

- Random Forest
- Decision Tree
- Support Vector Machine
- K-Nearest Neighbours
- Logistic Regression
- Neural Network

Object AI uses YOLO. Pose AI uses YOLO Pose for keypoint extraction followed by conventional ML classification.

## System Requirements

Recommended:

- Python 3.10 or 3.11, 64-bit
- Webcam for Image AI, Object AI, and Pose AI
- Microphone for Audio AI
- CSV/Excel data for Data AI
- CPU mode is supported; GPU acceleration is optional for YOLO

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ALL-IN-ONE-AI.git
cd ALL-IN-ONE-AI
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install numpy pandas scipy matplotlib pillow joblib scikit-learn opencv-python openpyxl
pip install ultralytics
pip install sounddevice soundfile librosa
```

Tkinter is normally included with standard Windows Python. On Debian/Ubuntu-based Linux:

```bash
sudo apt install python3-tk
```

## Running the Suite

Run the main launcher:

```bash
python ALL_IN_ONE_AI.py
```

If your launcher has a different filename, run that file instead.

## Typical Workflow

### Image AI
Create/load image classes → extract HOG features → select models → train/test → inspect plots → save model → real-time webcam classification.

### Object AI
Create object classes → capture/import images → annotate bounding boxes → prepare YOLO dataset → train → inspect metrics → load `best.pt` → real-time multi-object detection.

### Pose AI
Create pose names → record webcam samples → extract YOLO Pose keypoints → train classifiers → evaluate → save model → real-time named-pose recognition.

### Audio AI
Create sound classes → record samples → extract MFCC/spectral features → train/evaluate → save model → live microphone classification.

### Data AI
Load CSV/Excel → select target → automatic preprocessing → train classification/regression models → evaluate → save pipeline → batch prediction.

## Evaluation Outputs

Module-dependent outputs include:

- Accuracy, precision, recall, specificity and F1-score
- Balanced accuracy and MCC
- ROC-AUC and PR-AUC
- Confusion matrices
- ROC/PR curves
- PCA scatter plots
- Learning curves
- Feature importance
- YOLO mAP@0.5 and mAP@0.5:0.95
- MAE, RMSE and R² for regression

## Saved Models

| Module | Typical Output |
|---|---|
| Image AI | `.joblib` |
| Object AI | `best.pt` |
| Pose AI | `.joblib` |
| Audio AI | `.joblib` |
| Data AI | `.joblib` pipeline |

## Example STEM Experiments

- Image AI: Pin box vs. ink pad
- Object AI: simultaneous pin box and ink pad detection
- Pose AI: Hands Up vs. Hands Down
- Audio AI: Drum vs. Piano
- Data AI: three MPU6050 motion classes

## Research Use

ALL-IN-ONE-AI can support studies in AI literacy, STEM education, multimodal AI learning, beginner-centred AI interfaces, model comparison, explainability, and usability evaluation.

For reproducibility, report dataset size, class distribution, train/test split, random state, preprocessing, model parameters, selected metrics, and hardware/software environment.

## Limitations

The suite is designed primarily for beginner-oriented experimentation and educational use. Results obtained from small or highly controlled datasets should not be interpreted as proof of universal generalisation.

## Future Scope

Future versions can include larger multimodal datasets, additional deep-learning architectures, automated hyperparameter tuning, expanded explainability, embedded deployment, classroom dashboards, and long-term learning analytics.

## Citation

If you use ALL-IN-ONE-AI in academic work, cite the associated paper once its final conference citation is available.

Temporary project description:

```text
ALL-IN-ONE-AI: A multimodal machine-learning trainer integrating image
classification, YOLO object detection, personalised pose classification,
audio classification, and tabular-data machine learning for STEM education.
```

## Licence

Add a `LICENSE` file before public release. A permissive licence such as MIT may be used if open reuse and modification are intended.

## Full User Guide

See [Guide.md](Guide.md) for complete module-by-module instructions and troubleshooting.
