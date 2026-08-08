
"""
PERSONALIZED IMAGE AI TRAINER
--------------------------------
A GUI for STEM learners to create and train an image-classification project
with a maximum of five classes.

Main capabilities
-----------------
1. Capture image samples directly from a webcam.
2. Load an image dataset arranged in class folders.
3. Train exactly four or five machine-learning models:
   - Random Forest
   - Decision Tree
   - Support Vector Machine
   - K-Nearest Neighbours
   - Logistic Regression
   - Neural Network (MLP)
4. Generate:
   - Training confusion matrix
   - Testing confusion matrix
   - ROC curves
   - Precision–Recall curves
   - PCA scatter plot
   - Misclassification scatter plot
   - Feature-importance graph
   - Learning curve
   - Model-comparison graph
5. Save the best trained model.
6. Perform real-time webcam classification.

Dataset format
--------------
dataset_folder/
    Class_1/
        image001.jpg
        image002.jpg
    Class_2/
        image001.jpg
        image002.jpg

Recommended Python version: 3.11, 64-bit
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import customtkinter as ctk
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


APP_NAME = "Personalized Image AI Trainer"
APP_VERSION = "1.2 Manual HOG"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageDataset:
    X: np.ndarray
    y_text: np.ndarray
    feature_names: list[str]
    image_paths: list[str]


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    train_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    classification_report_text: str
    training_seconds: float
    inference_ms_per_sample: float


@dataclass
class AppState:
    dataset_path: str = ""
    dataset: Optional[ImageDataset] = None
    label_encoder: Optional[LabelEncoder] = None
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    results: dict[str, ModelResult] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    current_figure: Optional[Figure] = None


# ---------------------------------------------------------------------------
# Image feature extraction
# ---------------------------------------------------------------------------

def read_image_robust(image_path: Path) -> np.ndarray:
    """Read an image reliably on Windows, including Unicode and long paths."""
    if not image_path.exists():
        raise FileNotFoundError(f"File does not exist: {image_path}")

    file_size = image_path.stat().st_size
    if file_size <= 0:
        raise ValueError("The image file is empty (0 bytes).")

    try:
        encoded_data = np.fromfile(str(image_path), dtype=np.uint8)
    except Exception as error:
        raise ValueError(f"Could not read image bytes: {error}") from error

    if encoded_data.size == 0:
        raise ValueError("No image bytes were read.")

    image = cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "OpenCV could not decode the file. It may be corrupt, "
            "incomplete, or not a genuine image despite its extension."
        )

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unexpected image shape: {image.shape}")

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def compute_manual_hog(
    gray_image: np.ndarray,
    cell_size: int = 8,
    block_cells: int = 2,
    orientation_bins: int = 9,
) -> np.ndarray:
    """Compute a HOG descriptor using NumPy only.

    For a 64x64 image with 8x8 cells, 2x2-cell blocks and 9 bins, the output
    length is 7 * 7 * 2 * 2 * 9 = 1764 features.
    """
    gray = np.asarray(gray_image, dtype=np.float32)

    if gray.ndim != 2:
        raise ValueError(
            f"Manual HOG expects a grayscale image, received shape {gray.shape}."
        )

    height, width = gray.shape
    if height % cell_size != 0 or width % cell_size != 0:
        raise ValueError(
            "Image dimensions must be divisible by the HOG cell size."
        )

    # Central-difference gradients using NumPy. This avoids requiring OpenCV's
    # optional HOG binding while preserving directional edge information.
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.sqrt(
        gradient_x * gradient_x + gradient_y * gradient_y
    )
    orientation = (
        np.degrees(np.arctan2(gradient_y, gradient_x)) % 180.0
    )

    cells_y = height // cell_size
    cells_x = width // cell_size
    cell_histograms = np.zeros(
        (cells_y, cells_x, orientation_bins),
        dtype=np.float32,
    )

    bin_width = 180.0 / orientation_bins

    for cell_y in range(cells_y):
        y_start = cell_y * cell_size
        y_end = y_start + cell_size

        for cell_x in range(cells_x):
            x_start = cell_x * cell_size
            x_end = x_start + cell_size

            cell_magnitude = magnitude[
                y_start:y_end,
                x_start:x_end,
            ].reshape(-1)
            cell_orientation = orientation[
                y_start:y_end,
                x_start:x_end,
            ].reshape(-1)

            # Bilinear interpolation between adjacent orientation bins.
            fractional_bin = cell_orientation / bin_width
            lower_bin = np.floor(fractional_bin).astype(int)
            upper_bin = (lower_bin + 1) % orientation_bins
            upper_weight = fractional_bin - lower_bin
            lower_weight = 1.0 - upper_weight
            lower_bin %= orientation_bins

            histogram = np.zeros(
                orientation_bins,
                dtype=np.float32,
            )

            np.add.at(
                histogram,
                lower_bin,
                cell_magnitude * lower_weight,
            )
            np.add.at(
                histogram,
                upper_bin,
                cell_magnitude * upper_weight,
            )

            cell_histograms[
                cell_y,
                cell_x,
                :,
            ] = histogram

    block_features: list[np.ndarray] = []
    epsilon = 1e-6

    for block_y in range(cells_y - block_cells + 1):
        for block_x in range(cells_x - block_cells + 1):
            block = cell_histograms[
                block_y:block_y + block_cells,
                block_x:block_x + block_cells,
                :,
            ].reshape(-1)

            # L2-Hys-like normalisation.
            block = block / np.sqrt(
                np.sum(block * block) + epsilon * epsilon
            )
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(
                np.sum(block * block) + epsilon * epsilon
            )

            block_features.append(
                block.astype(np.float32)
            )

    if not block_features:
        raise ValueError("Manual HOG generated no blocks.")

    descriptor = np.concatenate(block_features).astype(np.float32)

    if descriptor.size != 1764:
        raise ValueError(
            f"Unexpected manual HOG length: {descriptor.size}; expected 1764."
        )

    if not np.all(np.isfinite(descriptor)):
        raise ValueError(
            "Manual HOG produced NaN or infinite values."
        )

    return descriptor


def extract_image_features(image_or_path: Any) -> tuple[np.ndarray, list[str]]:
    """
    Extract manual HOG, colour-histogram and grayscale-statistical features.

    The positional HOG constructor is used for compatibility across
    OpenCV wheel versions. File images are decoded through NumPy to support
    Windows paths that cv2.imread may fail to open.
    """
    if isinstance(image_or_path, (str, Path)):
        image = read_image_robust(Path(image_or_path))
    else:
        image = np.asarray(image_or_path)

    if image is None or image.size == 0:
        raise ValueError("The image is empty.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    try:
        image = cv2.resize(
            image,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error as error:
        raise ValueError(f"OpenCV preprocessing failed: {error}") from error

    # Manual HOG implementation.
    #
    # This deliberately does not use cv2.HOGDescriptor because some OpenCV
    # 5.x Python builds do not expose that class. It produces the standard
    # 1,764-element descriptor for a 64x64 image using:
    #   - 8x8 cells
    #   - 2x2-cell blocks
    #   - 8-pixel block stride
    #   - 9 unsigned orientation bins
    hog_features = compute_manual_hog(
        gray,
        cell_size=8,
        block_cells=2,
        orientation_bins=9,
    )

    histogram_features: list[float] = []
    histogram_names: list[str] = []

    for channel_index, channel_name in enumerate(("B", "G", "R")):
        try:
            histogram = cv2.calcHist(
                [image],
                [channel_index],
                None,
                [16],
                [0, 256],
            )
            histogram = cv2.normalize(
                histogram,
                None,
            ).reshape(-1)
        except cv2.error as error:
            raise ValueError(
                f"Colour histogram extraction failed: {error}"
            ) from error

        histogram_features.extend(
            histogram.astype(np.float32).tolist()
        )
        histogram_names.extend(
            [f"{channel_name}_hist_{index}" for index in range(16)]
        )

    statistics = np.asarray(
        [
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.min(gray)),
            float(np.max(gray)),
        ],
        dtype=np.float32,
    )

    feature_vector = np.concatenate(
        [
            hog_features,
            np.asarray(histogram_features, dtype=np.float32),
            statistics,
        ]
    )

    if not np.all(np.isfinite(feature_vector)):
        raise ValueError("The extracted feature vector contains NaN or infinity.")

    feature_names = (
        [f"HOG_{index}" for index in range(len(hog_features))]
        + histogram_names
        + ["Gray_mean", "Gray_std", "Gray_min", "Gray_max"]
    )

    return feature_vector, feature_names


def discover_class_folders(dataset_root: Path) -> list[Path]:
    class_folders = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir()]
    )

    if len(class_folders) < 2:
        raise ValueError(
            "The dataset must contain at least two class folders."
        )

    if len(class_folders) > 5:
        raise ValueError(
            "This trainer supports a maximum of five classes."
        )

    return class_folders


def load_image_dataset(
    dataset_root: Path,
    progress_callback: Callable[[float, str], None],
) -> ImageDataset:
    class_folders = discover_class_folders(dataset_root)

    files: list[tuple[Path, str]] = []
    discovered_counts: dict[str, int] = {}

    for class_folder in class_folders:
        class_files = [
            image_path
            for image_path in sorted(class_folder.rglob("*"))
            if image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        discovered_counts[class_folder.name] = len(class_files)
        files.extend(
            (image_path, class_folder.name)
            for image_path in class_files
        )

    if not files:
        folder_summary = "\n".join(
            f"• {name}: {count} supported images"
            for name, count in discovered_counts.items()
        )
        raise ValueError(
            "No supported image files were found.\n\n"
            f"Dataset: {dataset_root}\n"
            f"{folder_summary}\n\n"
            "Supported extensions: .jpg, .jpeg, .png, .bmp, .tif, .tiff"
        )

    features: list[np.ndarray] = []
    labels: list[str] = []
    image_paths: list[str] = []
    feature_names: list[str] = []
    rejected_details: list[str] = []
    valid_counts: dict[str, int] = {
        folder.name: 0 for folder in class_folders
    }

    for index, (image_path, class_name) in enumerate(files, start=1):
        try:
            vector, names = extract_image_features(image_path)
            features.append(vector)
            labels.append(class_name)
            image_paths.append(str(image_path))
            feature_names = names
            valid_counts[class_name] += 1
        except Exception as error:
            rejected_details.append(
                f"{image_path} -> {type(error).__name__}: {error}"
            )

        progress_callback(
            index / len(files),
            (
                f"Extracting image features: {index}/{len(files)} | "
                f"Valid: {len(features)} | Rejected: {len(rejected_details)}"
            ),
        )

    valid_classes = [
        class_name
        for class_name, count in valid_counts.items()
        if count > 0
    ]

    if len(features) < 4 or len(valid_classes) < 2:
        class_summary = "\n".join(
            (
                f"• {class_name}: "
                f"{valid_counts[class_name]} valid / "
                f"{discovered_counts.get(class_name, 0)} discovered"
            )
            for class_name in discovered_counts
        )

        first_errors = "\n".join(
            f"• {detail}"
            for detail in rejected_details[:10]
        )
        if len(rejected_details) > 10:
            first_errors += (
                f"\n• ...and {len(rejected_details) - 10} more rejected files"
            )

        raise ValueError(
            "Image feature extraction did not produce enough usable data.\n\n"
            f"Dataset folder:\n{dataset_root}\n\n"
            f"OpenCV version: {cv2.__version__}\n"
            f"NumPy version: {np.__version__}\n\n"
            f"Class summary:\n{class_summary}\n\n"
            f"First rejection reasons:\n"
            f"{first_errors if first_errors else 'No rejection details available.'}\n\n"
            "The dataset needs at least two classes and at least four valid "
            "images in total. For reliable training, use at least 20–50 "
            "readable images in every class."
        )

    classes_with_too_few = {
        class_name: count
        for class_name, count in valid_counts.items()
        if 0 < count < 2
    }
    if classes_with_too_few:
        details = ", ".join(
            f"{name}={count}"
            for name, count in classes_with_too_few.items()
        )
        raise ValueError(
            "At least one class has fewer than two valid images: "
            f"{details}. Add more images to each class."
        )

    if rejected_details:
        diagnostic_path = dataset_root / "rejected_images_report.txt"
        try:
            diagnostic_path.write_text(
                "\n".join(rejected_details),
                encoding="utf-8",
            )
            report_note = f" Report: {diagnostic_path}"
        except Exception:
            report_note = ""

        progress_callback(
            1.0,
            (
                f"Feature extraction completed. "
                f"Valid: {len(features)} | "
                f"Rejected: {len(rejected_details)}."
                f"{report_note}"
            ),
        )

    return ImageDataset(
        X=np.vstack(features),
        y_text=np.asarray(labels),
        feature_names=feature_names,
        image_paths=image_paths,
    )


# ---------------------------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------------------------

def build_model_pipelines(random_state: int) -> dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Neural Network": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=0.0005,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def calculate_macro_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> float:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(class_count),
    )

    total = matrix.sum()
    values: list[float] = []

    for class_index in range(class_count):
        true_positive = matrix[class_index, class_index]
        false_negative = (
            matrix[class_index, :].sum() - true_positive
        )
        false_positive = (
            matrix[:, class_index].sum() - true_positive
        )
        true_negative = (
            total
            - true_positive
            - false_negative
            - false_positive
        )

        denominator = true_negative + false_positive
        values.append(
            true_negative / denominator if denominator else 0.0
        )

    return float(np.mean(values))


def calculate_probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> tuple[float, float]:
    try:
        if class_count == 2:
            positive_probabilities = probabilities[:, 1]
            roc_auc = roc_auc_score(
                y_true,
                positive_probabilities,
            )
            pr_auc = average_precision_score(
                y_true,
                positive_probabilities,
            )
        else:
            binary_labels = label_binarize(
                y_true,
                classes=np.arange(class_count),
            )
            roc_auc = roc_auc_score(
                binary_labels,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
            pr_auc = average_precision_score(
                binary_labels,
                probabilities,
                average="macro",
            )

        return float(roc_auc), float(pr_auc)

    except Exception:
        return float("nan"), float("nan")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    roc_auc, pr_auc = calculate_probability_metrics(
        y_true,
        probabilities,
        class_count,
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Specificity": calculate_macro_specificity(
            y_true,
            y_pred,
            class_count,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "MCC": matthews_corrcoef(
            y_true,
            y_pred,
        ),
        "Cohen Kappa": cohen_kappa_score(
            y_true,
            y_pred,
        ),
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
    }


def choose_best_model(results: dict[str, ModelResult]) -> str:
    best_name = ""
    best_score = -np.inf

    for model_name, result in results.items():
        metrics = result.test_metrics

        roc_auc = (
            0.0
            if np.isnan(metrics["ROC AUC"])
            else metrics["ROC AUC"]
        )
        pr_auc = (
            0.0
            if np.isnan(metrics["PR AUC"])
            else metrics["PR AUC"]
        )

        score = (
            0.35 * metrics["F1"]
            + 0.25 * metrics["Balanced Accuracy"]
            + 0.20 * roc_auc
            + 0.20 * pr_auc
        )

        if score > best_score:
            best_score = score
            best_name = model_name

    return best_name


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ImageTrainerGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} — Version {APP_VERSION}")
        self.geometry("1450x900")
        self.minsize(1180, 760)

        self.state_data = AppState()
        self.message_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_webcam_event = threading.Event()
        self.plot_canvas: Optional[FigureCanvasTkAgg] = None

        self.model_vars: dict[str, ctk.BooleanVar] = {}

        self._build_header()
        self._build_tabs()
        self._build_status_bar()

        self.after(100, self._process_message_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=12, pady=(12, 6))

        ctk.CTkLabel(
            frame,
            text="PERSONALIZED IMAGE AI TRAINER",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(side="left", padx=18, pady=12)

        ctk.CTkLabel(
            frame,
            text=(
                "Webcam Dataset • 4/5 ML Models • Evaluation • "
                "Explainability • Real-Time Prediction"
            ),
            font=ctk.CTkFont(size=14),
        ).pack(side="left", padx=10)

        self.best_model_label = ctk.CTkLabel(
            frame,
            text="Best model: Not trained",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.best_model_label.pack(side="right", padx=18)

    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=6,
        )

        for name in (
            "1. Dataset",
            "2. Webcam Capture",
            "3. Models",
            "4. Results",
            "5. Graphs",
            "6. Real-Time Test",
        ):
            self.tabs.add(name)

        self._build_dataset_tab()
        self._build_capture_tab()
        self._build_models_tab()
        self._build_results_tab()
        self._build_graphs_tab()
        self._build_realtime_tab()

    def _build_status_bar(self) -> None:
        footer = ctk.CTkFrame(self)
        footer.pack(
            fill="x",
            padx=12,
            pady=(6, 12),
        )

        self.status_label = ctk.CTkLabel(
            footer,
            text="Ready",
            anchor="w",
        )
        self.status_label.pack(
            side="left",
            padx=12,
            pady=8,
            fill="x",
            expand=True,
        )

        self.progress_bar = ctk.CTkProgressBar(
            footer,
            width=320,
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(
            side="right",
            padx=12,
            pady=8,
        )

    def _build_dataset_tab(self) -> None:
        tab = self.tabs.tab("1. Dataset")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            tab,
            text="Load an Image Dataset",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            padx=20,
            pady=(20, 15),
            sticky="w",
        )

        ctk.CTkLabel(
            tab,
            text="Dataset folder",
        ).grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="w",
        )

        self.dataset_path_entry = ctk.CTkEntry(
            tab,
            placeholder_text=(
                "Select a parent folder containing 2–5 class folders"
            ),
        )
        self.dataset_path_entry.grid(
            row=1,
            column=1,
            padx=20,
            pady=10,
            sticky="ew",
        )

        ctk.CTkButton(
            tab,
            text="Browse",
            width=120,
            command=self._browse_dataset,
        ).grid(
            row=1,
            column=2,
            padx=20,
            pady=10,
        )

        controls = ctk.CTkFrame(tab)
        controls.grid(
            row=2,
            column=0,
            columnspan=3,
            padx=20,
            pady=10,
            sticky="ew",
        )

        ctk.CTkButton(
            controls,
            text="Load and Extract Features",
            command=self._start_dataset_loading,
        ).pack(
            side="left",
            padx=10,
            pady=10,
        )

        ctk.CTkButton(
            controls,
            text="Clear Project",
            command=self._clear_project,
        ).pack(
            side="left",
            padx=10,
            pady=10,
        )

        ctk.CTkLabel(
            tab,
            text=(
                "Expected folder structure: Dataset/Class_Name/image.jpg"
            ),
        ).grid(
            row=3,
            column=0,
            columnspan=3,
            padx=20,
            pady=(5, 10),
            sticky="w",
        )

        self.dataset_summary_text = ctk.CTkTextbox(tab)
        self.dataset_summary_text.grid(
            row=4,
            column=0,
            columnspan=3,
            padx=20,
            pady=(10, 20),
            sticky="nsew",
        )
        self.dataset_summary_text.insert(
            "1.0",
            "No dataset loaded.\n\n"
            "The selected parent folder must contain between two and five "
            "class folders. Each class folder should contain image samples.",
        )

    def _build_capture_tab(self) -> None:
        tab = self.tabs.tab("2. Webcam Capture")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(
            tab,
            text="Create an Image Dataset Using the Webcam",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            padx=20,
            pady=(20, 15),
            sticky="w",
        )

        ctk.CTkLabel(
            tab,
            text="Dataset output folder",
        ).grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="w",
        )

        self.capture_folder_entry = ctk.CTkEntry(tab)
        self.capture_folder_entry.grid(
            row=1,
            column=1,
            padx=20,
            pady=10,
            sticky="ew",
        )

        ctk.CTkButton(
            tab,
            text="Browse",
            width=120,
            command=self._browse_capture_folder,
        ).grid(
            row=1,
            column=2,
            padx=20,
            pady=10,
        )

        ctk.CTkLabel(
            tab,
            text="Class name",
        ).grid(
            row=2,
            column=0,
            padx=20,
            pady=10,
            sticky="w",
        )

        self.capture_class_entry = ctk.CTkEntry(
            tab,
            placeholder_text="Example: Open Hand",
        )
        self.capture_class_entry.grid(
            row=2,
            column=1,
            padx=20,
            pady=10,
            sticky="ew",
        )

        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=20,
            pady=10,
            sticky="ew",
        )

        ctk.CTkLabel(
            settings_frame,
            text="Number of images",
        ).pack(
            side="left",
            padx=(12, 5),
        )
        self.capture_count_entry = ctk.CTkEntry(
            settings_frame,
            width=100,
        )
        self.capture_count_entry.insert(0, "200")
        self.capture_count_entry.pack(
            side="left",
            padx=8,
            pady=12,
        )

        ctk.CTkLabel(
            settings_frame,
            text="Capture interval (seconds)",
        ).pack(
            side="left",
            padx=(20, 5),
        )
        self.capture_interval_entry = ctk.CTkEntry(
            settings_frame,
            width=100,
        )
        self.capture_interval_entry.insert(0, "0.15")
        self.capture_interval_entry.pack(
            side="left",
            padx=8,
            pady=12,
        )

        ctk.CTkButton(
            settings_frame,
            text="Start Webcam Capture",
            command=lambda: self._run_in_thread(
                self._capture_images_worker
            ),
        ).pack(
            side="left",
            padx=20,
            pady=12,
        )

        ctk.CTkButton(
            settings_frame,
            text="Stop Capture",
            command=self._stop_webcam,
        ).pack(
            side="left",
            padx=8,
            pady=12,
        )

        self.capture_log = ctk.CTkTextbox(tab)
        self.capture_log.grid(
            row=5,
            column=0,
            columnspan=3,
            padx=20,
            pady=(10, 20),
            sticky="nsew",
        )
        self.capture_log.insert(
            "1.0",
            "Instructions:\n"
            "1. Select a dataset output folder.\n"
            "2. Enter one class name.\n"
            "3. Set the number of images.\n"
            "4. Start webcam capture.\n"
            "5. Repeat for each class.\n\n"
            "Press Q in the webcam window to stop early.",
        )

    def _build_models_tab(self) -> None:
        tab = self.tabs.tab("3. Models")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            tab,
            text="Select Exactly Four or Five Models",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            padx=20,
            pady=(20, 15),
            sticky="w",
        )

        model_frame = ctk.CTkFrame(tab)
        model_frame.grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="nsew",
        )

        defaults = {
            "Random Forest": True,
            "Decision Tree": True,
            "SVM": True,
            "KNN": True,
            "Logistic Regression": True,
            "Neural Network": False,
        }

        for row_index, (model_name, default_value) in enumerate(
            defaults.items()
        ):
            variable = ctk.BooleanVar(value=default_value)
            self.model_vars[model_name] = variable

            ctk.CTkCheckBox(
                model_frame,
                text=model_name,
                variable=variable,
                command=self._update_selected_model_count,
            ).grid(
                row=row_index,
                column=0,
                padx=20,
                pady=12,
                sticky="w",
            )

        self.selected_model_count_label = ctk.CTkLabel(
            model_frame,
            text="Selected models: 5",
            font=ctk.CTkFont(weight="bold"),
        )
        self.selected_model_count_label.grid(
            row=len(defaults),
            column=0,
            padx=20,
            pady=15,
            sticky="w",
        )

        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(
            row=1,
            column=1,
            padx=20,
            pady=10,
            sticky="nsew",
        )
        settings_frame.grid_columnconfigure(1, weight=1)

        settings = [
            ("Testing fraction", "0.20"),
            ("Cross-validation folds", "5"),
            ("Random state", "42"),
            ("Minimum confidence", "0.65"),
        ]

        self.setting_entries: dict[str, ctk.CTkEntry] = {}

        for row_index, (setting_name, default_value) in enumerate(
            settings
        ):
            ctk.CTkLabel(
                settings_frame,
                text=setting_name,
            ).grid(
                row=row_index,
                column=0,
                padx=16,
                pady=12,
                sticky="w",
            )

            entry = ctk.CTkEntry(settings_frame)
            entry.insert(0, default_value)
            entry.grid(
                row=row_index,
                column=1,
                padx=16,
                pady=12,
                sticky="ew",
            )

            self.setting_entries[setting_name] = entry

        self.train_button = ctk.CTkButton(
            tab,
            text="TRAIN AND EVALUATE SELECTED MODELS",
            height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._start_training,
        )
        self.train_button.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=20,
            pady=20,
            sticky="ew",
        )

        self.training_log = ctk.CTkTextbox(tab)
        self.training_log.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=20,
            pady=(0, 20),
            sticky="nsew",
        )

    def _build_results_tab(self) -> None:
        tab = self.tabs.tab("4. Results")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            tab,
            text="Testing Performance Comparison",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).grid(
            row=0,
            column=0,
            padx=20,
            pady=(20, 10),
            sticky="w",
        )

        columns = (
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "Balanced Accuracy",
            "MCC",
            "ROC AUC",
            "PR AUC",
            "Train Time",
            "Inference ms",
        )

        self.results_tree = ttk.Treeview(
            tab,
            columns=columns,
            show="headings",
            height=14,
        )

        for column_name in columns:
            self.results_tree.heading(
                column_name,
                text=column_name,
            )
            self.results_tree.column(
                column_name,
                width=155 if column_name == "Model" else 105,
                anchor="center",
            )

        x_scrollbar = ttk.Scrollbar(
            tab,
            orient="horizontal",
            command=self.results_tree.xview,
        )
        y_scrollbar = ttk.Scrollbar(
            tab,
            orient="vertical",
            command=self.results_tree.yview,
        )

        self.results_tree.configure(
            xscrollcommand=x_scrollbar.set,
            yscrollcommand=y_scrollbar.set,
        )

        self.results_tree.grid(
            row=1,
            column=0,
            padx=(20, 0),
            pady=10,
            sticky="nsew",
        )
        y_scrollbar.grid(
            row=1,
            column=1,
            padx=(0, 20),
            pady=10,
            sticky="ns",
        )
        x_scrollbar.grid(
            row=2,
            column=0,
            padx=20,
            pady=(0, 10),
            sticky="ew",
        )

        buttons_frame = ctk.CTkFrame(tab)
        buttons_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=20,
            pady=10,
            sticky="ew",
        )

        ctk.CTkButton(
            buttons_frame,
            text="Save Best Model",
            command=self._save_best_model,
        ).pack(
            side="left",
            padx=10,
            pady=10,
        )

        ctk.CTkButton(
            buttons_frame,
            text="Export Results",
            command=self._export_results,
        ).pack(
            side="left",
            padx=10,
            pady=10,
        )

        ctk.CTkButton(
            buttons_frame,
            text="View Classification Report",
            command=self._show_classification_report,
        ).pack(
            side="left",
            padx=10,
            pady=10,
        )

    def _build_graphs_tab(self) -> None:
        tab = self.tabs.tab("5. Graphs")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(tab)
        controls.grid(
            row=0,
            column=0,
            padx=20,
            pady=(20, 10),
            sticky="ew",
        )

        ctk.CTkLabel(
            controls,
            text="Model",
        ).pack(
            side="left",
            padx=(12, 5),
        )

        self.plot_model_var = ctk.StringVar(value="")
        self.plot_model_menu = ctk.CTkOptionMenu(
            controls,
            values=["No trained model"],
            variable=self.plot_model_var,
        )
        self.plot_model_menu.pack(
            side="left",
            padx=8,
            pady=12,
        )

        ctk.CTkLabel(
            controls,
            text="Graph",
        ).pack(
            side="left",
            padx=(16, 5),
        )

        self.plot_type_var = ctk.StringVar(
            value="Testing Confusion Matrix"
        )
        self.plot_type_menu = ctk.CTkOptionMenu(
            controls,
            width=260,
            values=[
                "Training Confusion Matrix",
                "Testing Confusion Matrix",
                "Training ROC",
                "Testing ROC",
                "Training Precision-Recall",
                "Testing Precision-Recall",
                "PCA Actual Classes",
                "PCA Prediction Errors",
                "Feature Importance",
                "Learning Curve",
                "Model Comparison",
            ],
            variable=self.plot_type_var,
        )
        self.plot_type_menu.pack(
            side="left",
            padx=8,
            pady=12,
        )

        ctk.CTkButton(
            controls,
            text="Generate Graph",
            command=self._generate_selected_plot,
        ).pack(
            side="left",
            padx=10,
            pady=12,
        )

        ctk.CTkButton(
            controls,
            text="Save Current Graph",
            command=self._save_current_graph,
        ).pack(
            side="left",
            padx=10,
            pady=12,
        )

        self.plot_container = ctk.CTkFrame(tab)
        self.plot_container.grid(
            row=1,
            column=0,
            padx=20,
            pady=(0, 20),
            sticky="nsew",
        )

    def _build_realtime_tab(self) -> None:
        tab = self.tabs.tab("6. Real-Time Test")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            tab,
            text="Real-Time Webcam Classification",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).grid(
            row=0,
            column=0,
            padx=20,
            pady=(20, 10),
            sticky="w",
        )

        controls = ctk.CTkFrame(tab)
        controls.grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="ew",
        )

        ctk.CTkButton(
            controls,
            text="Start Real-Time Webcam",
            command=lambda: self._run_in_thread(
                self._realtime_webcam_worker
            ),
        ).pack(
            side="left",
            padx=10,
            pady=12,
        )

        ctk.CTkButton(
            controls,
            text="Stop Webcam",
            command=self._stop_webcam,
        ).pack(
            side="left",
            padx=10,
            pady=12,
        )

        self.realtime_result_label = ctk.CTkLabel(
            tab,
            text="Prediction: —",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        self.realtime_result_label.grid(
            row=2,
            column=0,
            padx=20,
            pady=20,
            sticky="w",
        )

        self.realtime_log = ctk.CTkTextbox(tab)
        self.realtime_log.grid(
            row=3,
            column=0,
            padx=20,
            pady=(0, 20),
            sticky="nsew",
        )

    # ------------------------------------------------------------------
    # Thread helpers
    # ------------------------------------------------------------------

    def _run_in_thread(
        self,
        target: Callable[[], None],
    ) -> None:
        threading.Thread(
            target=self._thread_wrapper,
            args=(target,),
            daemon=True,
        ).start()

    def _thread_wrapper(
        self,
        target: Callable[[], None],
    ) -> None:
        try:
            target()
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            # Expected user/data errors are shown clearly without a full
            # programming traceback.
            self.message_queue.put(
                (
                    "error",
                    str(error),
                )
            )
        except Exception as error:
            self.message_queue.put(
                (
                    "error",
                    f"{error}\n\n{traceback.format_exc()}",
                )
            )

    def _process_message_queue(self) -> None:
        try:
            while True:
                message_type, payload = (
                    self.message_queue.get_nowait()
                )

                if message_type == "status":
                    progress, text = payload
                    self.progress_bar.set(
                        max(0.0, min(1.0, progress))
                    )
                    self.status_label.configure(text=text)

                elif message_type == "dataset_loaded":
                    self._finish_dataset_loading(payload)

                elif message_type == "training_log":
                    self.training_log.insert(
                        "end",
                        payload + "\n",
                    )
                    self.training_log.see("end")

                elif message_type == "training_finished":
                    self._finish_training(payload)

                elif message_type == "capture_log":
                    self.capture_log.insert(
                        "end",
                        payload + "\n",
                    )
                    self.capture_log.see("end")

                elif message_type == "realtime":
                    class_name, confidence = payload
                    self.realtime_result_label.configure(
                        text=(
                            f"Prediction: {class_name} "
                            f"({confidence:.1%})"
                        )
                    )
                    self.realtime_log.insert(
                        "end",
                        (
                            f"{time.strftime('%H:%M:%S')} — "
                            f"{class_name} — {confidence:.3f}\n"
                        ),
                    )
                    self.realtime_log.see("end")

                elif message_type == "info":
                    messagebox.showinfo(
                        APP_NAME,
                        str(payload),
                    )

                elif message_type == "error":
                    messagebox.showerror(
                        APP_NAME,
                        str(payload),
                    )
                    self.progress_bar.set(0)
                    self.status_label.configure(
                        text="Operation failed"
                    )
                    self.train_button.configure(
                        state="normal"
                    )

        except queue.Empty:
            pass

        self.after(
            100,
            self._process_message_queue,
        )

    def _set_status(
        self,
        progress: float,
        text: str,
    ) -> None:
        self.message_queue.put(
            (
                "status",
                (progress, text),
            )
        )

    # ------------------------------------------------------------------
    # Dataset methods
    # ------------------------------------------------------------------

    def _browse_dataset(self) -> None:
        selected_folder = filedialog.askdirectory(
            title="Select image dataset folder"
        )

        if selected_folder:
            self.dataset_path_entry.delete(
                0,
                "end",
            )
            self.dataset_path_entry.insert(
                0,
                selected_folder,
            )

    def _browse_capture_folder(self) -> None:
        selected_folder = filedialog.askdirectory(
            title="Select dataset output folder"
        )

        if selected_folder:
            self.capture_folder_entry.delete(
                0,
                "end",
            )
            self.capture_folder_entry.insert(
                0,
                selected_folder,
            )

    def _start_dataset_loading(self) -> None:
        path_text = self.dataset_path_entry.get().strip()

        if not path_text:
            messagebox.showwarning(
                APP_NAME,
                "Select a dataset folder.",
            )
            return

        dataset_path = Path(path_text)

        if not dataset_path.exists():
            messagebox.showerror(
                APP_NAME,
                "The selected dataset folder does not exist.",
            )
            return

        self.state_data.dataset_path = str(dataset_path)

        self._run_in_thread(
            lambda: self._load_dataset_worker(dataset_path)
        )

    def _load_dataset_worker(
        self,
        dataset_path: Path,
    ) -> None:
        self._set_status(
            0.01,
            "Loading image dataset...",
        )

        dataset = load_image_dataset(
            dataset_path,
            self._set_status,
        )

        self.message_queue.put(
            (
                "dataset_loaded",
                dataset,
            )
        )

    def _finish_dataset_loading(
        self,
        dataset: ImageDataset,
    ) -> None:
        self.state_data.dataset = dataset
        self.state_data.results.clear()
        self.state_data.best_model_name = None

        class_names, class_counts = np.unique(
            dataset.y_text,
            return_counts=True,
        )

        summary_lines = [
            f"Total valid images: {len(dataset.X)}",
            f"Features per image: {dataset.X.shape[1]}",
            f"Number of classes: {len(class_names)}",
            "",
            "Class distribution:",
        ]

        summary_lines.extend(
            [
                f"  • {class_name}: {class_count}"
                for class_name, class_count
                in zip(class_names, class_counts)
            ]
        )

        if class_counts.min() < 10:
            summary_lines.extend(
                [
                    "",
                    "Warning: At least one class contains fewer "
                    "than ten images. More images are recommended.",
                ]
            )

        self.dataset_summary_text.delete(
            "1.0",
            "end",
        )
        self.dataset_summary_text.insert(
            "1.0",
            "\n".join(summary_lines),
        )

        self.best_model_label.configure(
            text="Best model: Not trained"
        )

        self._set_status(
            1.0,
            "Image dataset loaded successfully",
        )

    def _clear_project(self) -> None:
        confirmed = messagebox.askyesno(
            APP_NAME,
            "Clear the dataset and all trained results?",
        )

        if not confirmed:
            return

        self.state_data = AppState()

        self.dataset_summary_text.delete(
            "1.0",
            "end",
        )
        self.dataset_summary_text.insert(
            "1.0",
            "No dataset loaded.",
        )

        self.training_log.delete(
            "1.0",
            "end",
        )

        self._clear_results_tree()

        self.best_model_label.configure(
            text="Best model: Not trained"
        )

        self.plot_model_menu.configure(
            values=["No trained model"]
        )
        self.plot_model_var.set("")

        self._set_status(
            0,
            "Project cleared",
        )

    # ------------------------------------------------------------------
    # Webcam dataset capture
    # ------------------------------------------------------------------

    def _capture_images_worker(self) -> None:
        root_text = self.capture_folder_entry.get().strip()
        class_name = self.capture_class_entry.get().strip()

        if not root_text:
            raise ValueError(
                "Select a dataset output folder."
            )

        if not class_name:
            raise ValueError(
                "Enter a class name."
            )

        image_count = int(
            self.capture_count_entry.get()
        )
        capture_interval = float(
            self.capture_interval_entry.get()
        )

        if image_count < 1:
            raise ValueError(
                "Number of images must be positive."
            )

        if capture_interval <= 0:
            raise ValueError(
                "Capture interval must be positive."
            )

        safe_class_name = "".join(
            character
            if character.isalnum()
            or character in (" ", "_", "-")
            else "_"
            for character in class_name
        ).strip()

        output_folder = (
            Path(root_text)
            / safe_class_name
        )
        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "The webcam could not be opened."
            )

        captured = 0
        last_capture_time = 0.0
        self.stop_webcam_event.clear()

        try:
            while (
                captured < image_count
                and not self.stop_webcam_event.is_set()
            ):
                success, frame = camera.read()

                if not success:
                    continue

                current_time = time.time()

                if (
                    current_time - last_capture_time
                    >= capture_interval
                ):
                    image_path = (
                        output_folder
                        / f"{safe_class_name}_{captured:05d}.jpg"
                    )

                    cv2.imwrite(
                        str(image_path),
                        frame,
                    )

                    captured += 1
                    last_capture_time = current_time

                    self._set_status(
                        captured / image_count,
                        (
                            f"Captured images: "
                            f"{captured}/{image_count}"
                        ),
                    )

                preview = frame.copy()

                cv2.putText(
                    preview,
                    (
                        f"Class: {safe_class_name} | "
                        f"{captured}/{image_count} | Q: stop"
                    ),
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "Personalized Image Dataset Capture",
                    preview,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()
            self.stop_webcam_event.set()

        self.message_queue.put(
            (
                "capture_log",
                (
                    f"\nCapture completed for class "
                    f"'{safe_class_name}'.\n"
                    f"Images saved: {captured}\n"
                    f"Folder: {output_folder}\n"
                ),
            )
        )

        self._set_status(
            1.0,
            "Webcam image capture completed",
        )

    def _stop_webcam(self) -> None:
        self.stop_webcam_event.set()
        self._set_status(
            0,
            "Webcam stopped",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _update_selected_model_count(self) -> None:
        selected_count = sum(
            variable.get()
            for variable
            in self.model_vars.values()
        )

        self.selected_model_count_label.configure(
            text=f"Selected models: {selected_count}"
        )

    def _read_setting(
        self,
        setting_name: str,
        data_type: Callable[[str], Any],
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Any:
        value = data_type(
            self.setting_entries[setting_name].get()
        )

        if (
            minimum is not None
            and value < minimum
        ):
            raise ValueError(
                f"{setting_name} must be at least {minimum}."
            )

        if (
            maximum is not None
            and value > maximum
        ):
            raise ValueError(
                f"{setting_name} must not exceed {maximum}."
            )

        return value

    def _start_training(self) -> None:
        if self.state_data.dataset is None:
            messagebox.showwarning(
                APP_NAME,
                "Load an image dataset before training.",
            )
            return

        selected_models = [
            model_name
            for model_name, variable
            in self.model_vars.items()
            if variable.get()
        ]

        if len(selected_models) not in (4, 5):
            messagebox.showwarning(
                APP_NAME,
                "Select exactly four or five models.",
            )
            return

        self.train_button.configure(
            state="disabled"
        )
        self.training_log.delete(
            "1.0",
            "end",
        )
        self._clear_results_tree()

        self._run_in_thread(
            lambda: self._training_worker(
                selected_models
            )
        )

    def _training_worker(
        self,
        selected_models: list[str],
    ) -> None:
        dataset = self.state_data.dataset

        if dataset is None:
            raise RuntimeError(
                "The dataset is unavailable."
            )

        testing_fraction = self._read_setting(
            "Testing fraction",
            float,
            0.10,
            0.40,
        )
        requested_folds = self._read_setting(
            "Cross-validation folds",
            int,
            2,
            10,
        )
        random_state = self._read_setting(
            "Random state",
            int,
            0,
            None,
        )

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(
            dataset.y_text
        )
        class_count = len(
            label_encoder.classes_
        )

        class_counts = np.bincount(
            y_encoded
        )

        if class_counts.min() < 2:
            raise ValueError(
                "Every class needs at least two images."
            )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            dataset.X,
            y_encoded,
            test_size=testing_fraction,
            stratify=y_encoded,
            random_state=random_state,
        )

        training_class_counts = np.bincount(
            y_train,
            minlength=class_count,
        )

        folds = min(
            requested_folds,
            int(training_class_counts.min()),
        )

        if folds < 2:
            raise ValueError(
                "Insufficient training images per class "
                "for cross-validation."
            )

        cross_validation = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        all_pipelines = build_model_pipelines(
            random_state
        )
        results: dict[str, ModelResult] = {}

        self.message_queue.put(
            (
                "training_log",
                (
                    f"Dataset: {len(dataset.X)} images × "
                    f"{dataset.X.shape[1]} features"
                ),
            )
        )
        self.message_queue.put(
            (
                "training_log",
                (
                    f"Classes: "
                    f"{list(label_encoder.classes_)}"
                ),
            )
        )
        self.message_queue.put(
            (
                "training_log",
                (
                    f"Training images: {len(X_train)} | "
                    f"Testing images: {len(X_test)} | "
                    f"Cross-validation folds: {folds}"
                ),
            )
        )

        total_models = len(selected_models)

        for model_index, model_name in enumerate(
            selected_models,
            start=1,
        ):
            pipeline = clone(
                all_pipelines[model_name]
            )

            self._set_status(
                (model_index - 1) / total_models,
                f"Training {model_name}...",
            )

            self.message_queue.put(
                (
                    "training_log",
                    (
                        f"\n[{model_index}/{total_models}] "
                        f"{model_name}"
                    ),
                )
            )

            training_start = time.perf_counter()

            train_predictions = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=cross_validation,
                method="predict",
            )

            train_probabilities = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=cross_validation,
                method="predict_proba",
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            training_seconds = (
                time.perf_counter()
                - training_start
            )

            inference_start = time.perf_counter()

            test_predictions = pipeline.predict(
                X_test
            )

            test_probabilities = pipeline.predict_proba(
                X_test
            )

            inference_seconds = (
                time.perf_counter()
                - inference_start
            )

            inference_ms_per_sample = (
                1000.0
                * inference_seconds
                / max(len(X_test), 1)
            )

            train_metrics = calculate_metrics(
                y_train,
                train_predictions,
                train_probabilities,
                class_count,
            )

            test_metrics = calculate_metrics(
                y_test,
                test_predictions,
                test_probabilities,
                class_count,
            )

            report_text = classification_report(
                y_test,
                test_predictions,
                target_names=label_encoder.classes_,
                zero_division=0,
            )

            results[model_name] = ModelResult(
                name=model_name,
                pipeline=pipeline,
                train_predictions=train_predictions,
                test_predictions=test_predictions,
                train_probabilities=train_probabilities,
                test_probabilities=test_probabilities,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                classification_report_text=report_text,
                training_seconds=training_seconds,
                inference_ms_per_sample=inference_ms_per_sample,
            )

            self.message_queue.put(
                (
                    "training_log",
                    (
                        f"Testing accuracy: "
                        f"{test_metrics['Accuracy']:.4f} | "
                        f"F1: {test_metrics['F1']:.4f} | "
                        f"ROC-AUC: "
                        f"{test_metrics['ROC AUC']:.4f} | "
                        f"PR-AUC: "
                        f"{test_metrics['PR AUC']:.4f}"
                    ),
                )
            )

        best_model_name = choose_best_model(
            results
        )

        self.message_queue.put(
            (
                "training_finished",
                {
                    "label_encoder": label_encoder,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "results": results,
                    "best_model_name": best_model_name,
                },
            )
        )

    def _finish_training(
        self,
        payload: dict[str, Any],
    ) -> None:
        for key, value in payload.items():
            setattr(
                self.state_data,
                key,
                value,
            )

        self.best_model_label.configure(
            text=(
                f"Best model: "
                f"{self.state_data.best_model_name}"
            )
        )

        self._populate_results_tree()

        model_names = list(
            self.state_data.results.keys()
        )

        self.plot_model_menu.configure(
            values=model_names
        )

        if model_names:
            self.plot_model_var.set(
                self.state_data.best_model_name
                or model_names[0]
            )

        self._set_status(
            1.0,
            "Training and evaluation completed",
        )

        self.train_button.configure(
            state="normal"
        )

        self.message_queue.put(
            (
                "training_log",
                (
                    f"\nBest model: "
                    f"{self.state_data.best_model_name}"
                ),
            )
        )

        self.tabs.set(
            "4. Results"
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _clear_results_tree(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def _populate_results_tree(self) -> None:
        self._clear_results_tree()

        for model_name, result in (
            self.state_data.results.items()
        ):
            metrics = result.test_metrics

            self.results_tree.insert(
                "",
                "end",
                values=(
                    model_name,
                    f"{metrics['Accuracy']:.4f}",
                    f"{metrics['Precision']:.4f}",
                    f"{metrics['Recall']:.4f}",
                    f"{metrics['Specificity']:.4f}",
                    f"{metrics['F1']:.4f}",
                    (
                        f"{metrics['Balanced Accuracy']:.4f}"
                    ),
                    f"{metrics['MCC']:.4f}",
                    f"{metrics['ROC AUC']:.4f}",
                    f"{metrics['PR AUC']:.4f}",
                    f"{result.training_seconds:.3f}",
                    (
                        f"{result.inference_ms_per_sample:.3f}"
                    ),
                ),
            )

    def _results_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for model_name, result in (
            self.state_data.results.items()
        ):
            row: dict[str, Any] = {
                "Model": model_name
            }

            row.update(
                {
                    f"Train {metric_name}": value
                    for metric_name, value
                    in result.train_metrics.items()
                }
            )

            row.update(
                {
                    f"Test {metric_name}": value
                    for metric_name, value
                    in result.test_metrics.items()
                }
            )

            row["Training seconds"] = (
                result.training_seconds
            )
            row["Inference ms/sample"] = (
                result.inference_ms_per_sample
            )
            row["Selected as best"] = (
                model_name
                == self.state_data.best_model_name
            )

            rows.append(row)

        return pd.DataFrame(rows)

    def _save_best_model(self) -> None:
        if not self.state_data.best_model_name:
            messagebox.showwarning(
                APP_NAME,
                "Train the models first.",
            )
            return

        output_path = filedialog.asksaveasfilename(
            title="Save best image model",
            defaultextension=".joblib",
            filetypes=[
                ("Joblib model", "*.joblib")
            ],
            initialfile=(
                "personalized_image_ai_model.joblib"
            ),
        )

        if not output_path:
            return

        dataset = self.state_data.dataset
        label_encoder = self.state_data.label_encoder

        if (
            dataset is None
            or label_encoder is None
        ):
            raise RuntimeError(
                "Model metadata is unavailable."
            )

        best_result = self.state_data.results[
            self.state_data.best_model_name
        ]

        model_package = {
            "application": APP_NAME,
            "version": APP_VERSION,
            "created_at": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "model_name": (
                self.state_data.best_model_name
            ),
            "pipeline": best_result.pipeline,
            "label_encoder": label_encoder,
            "feature_names": dataset.feature_names,
            "minimum_confidence": float(
                self.setting_entries[
                    "Minimum confidence"
                ].get()
            ),
            "feature_extractor": (
                "NumPy manual HOG + colour histogram "
                "+ grayscale statistics"
            ),
        }

        joblib.dump(
            model_package,
            output_path,
        )

        messagebox.showinfo(
            APP_NAME,
            f"Best model saved:\n{output_path}",
        )

    def _export_results(self) -> None:
        if not self.state_data.results:
            messagebox.showwarning(
                APP_NAME,
                "No trained results are available.",
            )
            return

        output_folder = filedialog.askdirectory(
            title="Select result export folder"
        )

        if not output_folder:
            return

        folder = Path(output_folder)
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._results_dataframe().to_csv(
            folder / "image_model_comparison.csv",
            index=False,
        )

        for model_name, result in (
            self.state_data.results.items()
        ):
            safe_name = (
                model_name.lower()
                .replace(" ", "_")
            )

            (
                folder
                / f"{safe_name}_classification_report.txt"
            ).write_text(
                result.classification_report_text,
                encoding="utf-8",
            )

        summary = {
            "dataset_path": (
                self.state_data.dataset_path
            ),
            "best_model": (
                self.state_data.best_model_name
            ),
            "classes": (
                self.state_data.label_encoder.classes_.tolist()
                if self.state_data.label_encoder
                is not None
                else []
            ),
        }

        (
            folder
            / "image_project_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        messagebox.showinfo(
            APP_NAME,
            f"Results exported to:\n{folder}",
        )

    def _show_classification_report(self) -> None:
        model_name = self.plot_model_var.get()

        if model_name not in self.state_data.results:
            messagebox.showwarning(
                APP_NAME,
                "Select a trained model.",
            )
            return

        window = ctk.CTkToplevel(self)
        window.title(
            f"Classification Report — {model_name}"
        )
        window.geometry(
            "780x600"
        )

        textbox = ctk.CTkTextbox(
            window,
            font=ctk.CTkFont(
                family="Courier",
                size=14,
            ),
        )
        textbox.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15,
        )

        textbox.insert(
            "1.0",
            self.state_data.results[
                model_name
            ].classification_report_text,
        )

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    def _generate_selected_plot(self) -> None:
        if not self.state_data.results:
            messagebox.showwarning(
                APP_NAME,
                "Train the models before generating graphs.",
            )
            return

        plot_type = self.plot_type_var.get()
        model_name = self.plot_model_var.get()

        try:
            if plot_type == "Model Comparison":
                figure = (
                    self._plot_model_comparison()
                )

            else:
                if model_name not in (
                    self.state_data.results
                ):
                    raise ValueError(
                        "Select a trained model."
                    )

                result = self.state_data.results[
                    model_name
                ]

                if plot_type == (
                    "Training Confusion Matrix"
                ):
                    figure = self._plot_confusion_matrix(
                        result,
                        training=True,
                    )

                elif plot_type == (
                    "Testing Confusion Matrix"
                ):
                    figure = self._plot_confusion_matrix(
                        result,
                        training=False,
                    )

                elif plot_type == "Training ROC":
                    figure = self._plot_roc(
                        result,
                        training=True,
                    )

                elif plot_type == "Testing ROC":
                    figure = self._plot_roc(
                        result,
                        training=False,
                    )

                elif plot_type == (
                    "Training Precision-Recall"
                ):
                    figure = (
                        self._plot_precision_recall(
                            result,
                            training=True,
                        )
                    )

                elif plot_type == (
                    "Testing Precision-Recall"
                ):
                    figure = (
                        self._plot_precision_recall(
                            result,
                            training=False,
                        )
                    )

                elif plot_type == (
                    "PCA Actual Classes"
                ):
                    figure = self._plot_pca(
                        result,
                        errors_only=False,
                    )

                elif plot_type == (
                    "PCA Prediction Errors"
                ):
                    figure = self._plot_pca(
                        result,
                        errors_only=True,
                    )

                elif plot_type == (
                    "Feature Importance"
                ):
                    figure = (
                        self._plot_feature_importance(
                            result
                        )
                    )

                elif plot_type == "Learning Curve":
                    figure = (
                        self._plot_learning_curve(
                            result
                        )
                    )

                else:
                    raise ValueError(
                        f"Unsupported graph: {plot_type}"
                    )

            self._display_figure(figure)

        except Exception as error:
            messagebox.showerror(
                APP_NAME,
                f"Graph generation failed:\n{error}",
            )

    def _display_figure(
        self,
        figure: Figure,
    ) -> None:
        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()

        if self.state_data.current_figure is not None:
            plt.close(
                self.state_data.current_figure
            )

        self.state_data.current_figure = figure

        self.plot_canvas = FigureCanvasTkAgg(
            figure,
            master=self.plot_container,
        )
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8,
        )

    def _plot_confusion_matrix(
        self,
        result: ModelResult,
        training: bool,
    ) -> Figure:
        label_encoder = self.state_data.label_encoder

        if label_encoder is None:
            raise RuntimeError(
                "Class labels are unavailable."
            )

        y_true = (
            self.state_data.y_train
            if training
            else self.state_data.y_test
        )

        y_pred = (
            result.train_predictions
            if training
            else result.test_predictions
        )

        if y_true is None:
            raise RuntimeError(
                "Evaluation labels are unavailable."
            )

        figure, axis = plt.subplots(
            figsize=(8.5, 6.5)
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(
                len(label_encoder.classes_)
            ),
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=label_encoder.classes_,
        )

        display.plot(
            ax=axis,
            values_format="d",
            colorbar=False,
        )

        split_name = (
            "Cross-validated training"
            if training
            else "Testing"
        )

        axis.set_title(
            (
                f"{split_name} confusion matrix "
                f"— {result.name}"
            )
        )

        figure.tight_layout()
        return figure

    def _plot_roc(
        self,
        result: ModelResult,
        training: bool,
    ) -> Figure:
        label_encoder = self.state_data.label_encoder

        if label_encoder is None:
            raise RuntimeError(
                "Class labels are unavailable."
            )

        y_true = (
            self.state_data.y_train
            if training
            else self.state_data.y_test
        )

        probabilities = (
            result.train_probabilities
            if training
            else result.test_probabilities
        )

        if y_true is None:
            raise RuntimeError(
                "Evaluation labels are unavailable."
            )

        class_count = len(
            label_encoder.classes_
        )

        figure, axis = plt.subplots(
            figsize=(9, 6.5)
        )

        if class_count == 2:
            false_positive_rate, true_positive_rate, _ = (
                roc_curve(
                    y_true,
                    probabilities[:, 1],
                )
            )

            auc_value = roc_auc_score(
                y_true,
                probabilities[:, 1],
            )

            axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=(
                    f"{label_encoder.classes_[1]} "
                    f"(AUC={auc_value:.3f})"
                ),
            )

        else:
            binary_labels = label_binarize(
                y_true,
                classes=np.arange(class_count),
            )

            for class_index, class_name in enumerate(
                label_encoder.classes_
            ):
                (
                    false_positive_rate,
                    true_positive_rate,
                    _,
                ) = roc_curve(
                    binary_labels[:, class_index],
                    probabilities[:, class_index],
                )

                auc_value = roc_auc_score(
                    binary_labels[:, class_index],
                    probabilities[:, class_index],
                )

                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    label=(
                        f"{class_name} "
                        f"(AUC={auc_value:.3f})"
                    ),
                )

            micro_false_positive_rate, micro_true_positive_rate, _ = (
                roc_curve(
                    binary_labels.ravel(),
                    probabilities.ravel(),
                )
            )

            micro_auc = roc_auc_score(
                binary_labels,
                probabilities,
                average="micro",
                multi_class="ovr",
            )

            axis.plot(
                micro_false_positive_rate,
                micro_true_positive_rate,
                linestyle="--",
                label=(
                    f"Micro average "
                    f"(AUC={micro_auc:.3f})"
                ),
            )

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
        )
        axis.set_xlabel(
            "False positive rate"
        )
        axis.set_ylabel(
            "True positive rate"
        )
        axis.set_title(
            (
                f"{'Training' if training else 'Testing'} "
                f"ROC curves — {result.name}"
            )
        )
        axis.legend(
            loc="lower right"
        )
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()
        return figure

    def _plot_precision_recall(
        self,
        result: ModelResult,
        training: bool,
    ) -> Figure:
        label_encoder = self.state_data.label_encoder

        if label_encoder is None:
            raise RuntimeError(
                "Class labels are unavailable."
            )

        y_true = (
            self.state_data.y_train
            if training
            else self.state_data.y_test
        )

        probabilities = (
            result.train_probabilities
            if training
            else result.test_probabilities
        )

        if y_true is None:
            raise RuntimeError(
                "Evaluation labels are unavailable."
            )

        class_count = len(
            label_encoder.classes_
        )

        figure, axis = plt.subplots(
            figsize=(9, 6.5)
        )

        if class_count == 2:
            precision_values, recall_values, _ = (
                precision_recall_curve(
                    y_true,
                    probabilities[:, 1],
                )
            )

            average_precision = (
                average_precision_score(
                    y_true,
                    probabilities[:, 1],
                )
            )

            axis.plot(
                recall_values,
                precision_values,
                label=(
                    f"{label_encoder.classes_[1]} "
                    f"(AP={average_precision:.3f})"
                ),
            )

        else:
            binary_labels = label_binarize(
                y_true,
                classes=np.arange(class_count),
            )

            for class_index, class_name in enumerate(
                label_encoder.classes_
            ):
                (
                    precision_values,
                    recall_values,
                    _,
                ) = precision_recall_curve(
                    binary_labels[:, class_index],
                    probabilities[:, class_index],
                )

                average_precision = (
                    average_precision_score(
                        binary_labels[:, class_index],
                        probabilities[:, class_index],
                    )
                )

                axis.plot(
                    recall_values,
                    precision_values,
                    label=(
                        f"{class_name} "
                        f"(AP={average_precision:.3f})"
                    ),
                )

            micro_precision, micro_recall, _ = (
                precision_recall_curve(
                    binary_labels.ravel(),
                    probabilities.ravel(),
                )
            )

            micro_average_precision = (
                average_precision_score(
                    binary_labels,
                    probabilities,
                    average="micro",
                )
            )

            axis.plot(
                micro_recall,
                micro_precision,
                linestyle="--",
                label=(
                    f"Micro average "
                    f"(AP={micro_average_precision:.3f})"
                ),
            )

        axis.set_xlabel(
            "Recall"
        )
        axis.set_ylabel(
            "Precision"
        )
        axis.set_title(
            (
                f"{'Training' if training else 'Testing'} "
                f"Precision–Recall curves — {result.name}"
            )
        )
        axis.legend(
            loc="lower left"
        )
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()
        return figure

    def _plot_pca(
        self,
        result: ModelResult,
        errors_only: bool,
    ) -> Figure:
        X_test = self.state_data.X_test
        y_test = self.state_data.y_test
        label_encoder = self.state_data.label_encoder

        if (
            X_test is None
            or y_test is None
            or label_encoder is None
        ):
            raise RuntimeError(
                "Testing data is unavailable."
            )

        scaled_features = (
            StandardScaler()
            .fit_transform(X_test)
        )

        pca = PCA(
            n_components=2,
            random_state=42,
        )

        points = pca.fit_transform(
            scaled_features
        )

        figure, axis = plt.subplots(
            figsize=(9, 6.5)
        )

        if errors_only:
            correct_mask = (
                y_test
                == result.test_predictions
            )

            axis.scatter(
                points[correct_mask, 0],
                points[correct_mask, 1],
                alpha=0.65,
                label="Correct prediction",
            )

            axis.scatter(
                points[~correct_mask, 0],
                points[~correct_mask, 1],
                marker="x",
                s=80,
                label="Misclassified",
            )

            axis.set_title(
                (
                    f"PCA prediction-error scatter "
                    f"— {result.name}"
                )
            )

        else:
            for class_index, class_name in enumerate(
                label_encoder.classes_
            ):
                class_mask = (
                    y_test == class_index
                )

                axis.scatter(
                    points[class_mask, 0],
                    points[class_mask, 1],
                    alpha=0.7,
                    label=class_name,
                )

            axis.set_title(
                (
                    f"PCA feature scatter — {result.name}\n"
                    f"Explained variance: "
                    f"{pca.explained_variance_ratio_.sum():.2%}"
                )
            )

        axis.set_xlabel(
            "Principal component 1"
        )
        axis.set_ylabel(
            "Principal component 2"
        )
        axis.legend()
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()
        return figure

    def _plot_feature_importance(
        self,
        result: ModelResult,
    ) -> Figure:
        dataset = self.state_data.dataset
        X_test = self.state_data.X_test
        y_test = self.state_data.y_test

        if (
            dataset is None
            or X_test is None
            or y_test is None
        ):
            raise RuntimeError(
                "Evaluation data is unavailable."
            )

        estimator = result.pipeline.named_steps[
            "model"
        ]

        if hasattr(
            estimator,
            "feature_importances_",
        ):
            importance = np.asarray(
                estimator.feature_importances_
            )
            source_name = (
                "Built-in feature importance"
            )

        elif hasattr(
            estimator,
            "coef_",
        ):
            coefficients = np.asarray(
                estimator.coef_
            )
            importance = np.mean(
                np.abs(coefficients),
                axis=0,
            )
            source_name = (
                "Absolute coefficient importance"
            )

        else:
            permutation = permutation_importance(
                result.pipeline,
                X_test,
                y_test,
                scoring="f1_macro",
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )

            importance = (
                permutation.importances_mean
            )
            source_name = (
                "Permutation importance"
            )

        top_feature_count = min(
            20,
            len(importance),
        )

        top_indices = np.argsort(
            importance
        )[-top_feature_count:]

        top_names = [
            dataset.feature_names[index]
            for index in top_indices
        ]

        top_values = importance[
            top_indices
        ]

        figure, axis = plt.subplots(
            figsize=(10, 7)
        )

        axis.barh(
            top_names,
            top_values,
        )
        axis.set_xlabel(
            "Importance"
        )
        axis.set_title(
            (
                f"Top image-feature importance "
                f"— {result.name}\n"
                f"{source_name}"
            )
        )
        axis.grid(
            True,
            axis="x",
            alpha=0.3,
        )

        figure.tight_layout()
        return figure

    def _plot_learning_curve(
        self,
        result: ModelResult,
    ) -> Figure:
        X_train = self.state_data.X_train
        y_train = self.state_data.y_train

        if (
            X_train is None
            or y_train is None
        ):
            raise RuntimeError(
                "Training data is unavailable."
            )

        minimum_class_count = np.bincount(
            y_train
        ).min()

        folds = min(
            5,
            int(minimum_class_count),
        )

        if folds < 2:
            raise ValueError(
                "Insufficient images for a learning curve."
            )

        cross_validation = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42,
        )

        (
            training_sizes,
            training_scores,
            validation_scores,
        ) = learning_curve(
            clone(result.pipeline),
            X_train,
            y_train,
            cv=cross_validation,
            scoring="f1_macro",
            train_sizes=np.linspace(
                0.25,
                1.0,
                5,
            ),
        )

        training_mean = (
            training_scores.mean(axis=1)
        )
        validation_mean = (
            validation_scores.mean(axis=1)
        )

        training_std = (
            training_scores.std(axis=1)
        )
        validation_std = (
            validation_scores.std(axis=1)
        )

        figure, axis = plt.subplots(
            figsize=(9, 6.5)
        )

        axis.plot(
            training_sizes,
            training_mean,
            marker="o",
            label="Training macro-F1",
        )

        axis.plot(
            training_sizes,
            validation_mean,
            marker="o",
            label="Cross-validation macro-F1",
        )

        axis.fill_between(
            training_sizes,
            training_mean - training_std,
            training_mean + training_std,
            alpha=0.15,
        )

        axis.fill_between(
            training_sizes,
            validation_mean - validation_std,
            validation_mean + validation_std,
            alpha=0.15,
        )

        axis.set_xlabel(
            "Training images"
        )
        axis.set_ylabel(
            "Macro-F1"
        )
        axis.set_title(
            f"Learning curve — {result.name}"
        )
        axis.legend()
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()
        return figure

    def _plot_model_comparison(
        self,
    ) -> Figure:
        model_names = list(
            self.state_data.results.keys()
        )

        metric_names = [
            "Accuracy",
            "F1",
            "Balanced Accuracy",
            "ROC AUC",
            "PR AUC",
        ]

        x_positions = np.arange(
            len(model_names)
        )
        bar_width = 0.15

        figure, axis = plt.subplots(
            figsize=(11, 6.5)
        )

        for metric_index, metric_name in enumerate(
            metric_names
        ):
            values = [
                self.state_data.results[
                    model_name
                ].test_metrics[metric_name]
                for model_name in model_names
            ]

            axis.bar(
                (
                    x_positions
                    + (metric_index - 2)
                    * bar_width
                ),
                values,
                bar_width,
                label=metric_name,
            )

        axis.set_xticks(
            x_positions
        )
        axis.set_xticklabels(
            model_names,
            rotation=15,
            ha="right",
        )
        axis.set_ylim(
            0,
            1.05,
        )
        axis.set_ylabel(
            "Score"
        )
        axis.set_title(
            "Testing-performance comparison"
        )
        axis.legend(
            ncol=3
        )
        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        figure.tight_layout()
        return figure

    def _save_current_graph(self) -> None:
        figure = self.state_data.current_figure

        if figure is None:
            messagebox.showwarning(
                APP_NAME,
                "Generate a graph first.",
            )
            return

        output_path = filedialog.asksaveasfilename(
            title="Save graph",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG image", "*.svg"),
            ],
        )

        if output_path:
            figure.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )

            messagebox.showinfo(
                APP_NAME,
                f"Graph saved:\n{output_path}",
            )

    # ------------------------------------------------------------------
    # Real-time webcam
    # ------------------------------------------------------------------

    def _realtime_webcam_worker(self) -> None:
        if not self.state_data.best_model_name:
            raise RuntimeError(
                "Train the models before starting real-time testing."
            )

        label_encoder = self.state_data.label_encoder

        if label_encoder is None:
            raise RuntimeError(
                "Class labels are unavailable."
            )

        best_result = self.state_data.results[
            self.state_data.best_model_name
        ]

        confidence_threshold = float(
            self.setting_entries[
                "Minimum confidence"
            ].get()
        )

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "The webcam could not be opened."
            )

        self.stop_webcam_event.clear()

        try:
            while not self.stop_webcam_event.is_set():
                success, frame = camera.read()

                if not success:
                    continue

                feature_vector, _ = (
                    extract_image_features(frame)
                )

                probabilities = (
                    best_result.pipeline
                    .predict_proba(
                        [feature_vector]
                    )[0]
                )

                predicted_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[predicted_index]
                )

                if confidence >= confidence_threshold:
                    predicted_class = str(
                        label_encoder.inverse_transform(
                            [predicted_index]
                        )[0]
                    )
                else:
                    predicted_class = "Uncertain"

                self.message_queue.put(
                    (
                        "realtime",
                        (
                            predicted_class,
                            confidence,
                        ),
                    )
                )

                preview = frame.copy()

                cv2.putText(
                    preview,
                    (
                        f"{predicted_class}: "
                        f"{confidence:.1%}"
                    ),
                    (12, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    preview,
                    "Press Q to stop",
                    (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "Real-Time Image Classification",
                    preview,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()
            self.stop_webcam_event.set()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self.stop_webcam_event.set()
        self.destroy()


def main() -> None:
    application = ImageTrainerGUI()
    application.mainloop()


if __name__ == "__main__":
    main()
