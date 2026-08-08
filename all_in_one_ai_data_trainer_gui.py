
"""
ALL IN ONE AI — TABULAR DATA TRAINER
====================================

Production-oriented GUI for supervised machine learning on CSV and Excel data.

Main capabilities
-----------------
1. Load CSV, TSV, XLSX or XLS files.
2. Inspect row count, column names, data types, missing values and unique values.
3. Handle large CSV files by chunked scanning and bounded random sampling.
4. Automatically suggest classification or regression from the selected target.
5. Automatically preprocess numeric, categorical, Boolean and date/time columns.
6. Create independent training, validation and testing sets.
7. Compare exactly four or five machine-learning models.
8. Generate task-appropriate metrics and research graphs.
9. Save/load complete preprocessing + model pipelines as .joblib.
10. Batch-predict a new CSV/Excel file and export the results.

Important note
--------------
No desktop program can guarantee unlimited file size independent of computer
memory. This GUI addresses large CSV files with chunked reading and a user-set
row cap. Set the cap to 0 only when enough RAM is available for all rows.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable, Optional

import customtkinter as ctk
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


APP_NAME = "All in One AI — Data Trainer"
APP_VERSION = "1.0 Production"
ACCENT = "#0891B2"
ACCENT_HOVER = "#0E7490"

CLASSIFICATION_MODELS = [
    "Random Forest",
    "Decision Tree",
    "SVM",
    "KNN",
    "Logistic Regression",
    "Neural Network",
]

REGRESSION_MODELS = [
    "Random Forest",
    "Decision Tree",
    "SVR",
    "KNN Regressor",
    "Ridge Regression",
    "Neural Network",
]

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


def safe_output_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return cleaned.strip("_") or "model"


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a bounded encoder across old/new scikit-learn releases."""
    common = {
        "handle_unknown": "ignore",
        "min_frequency": 2,
        "max_categories": 50,
    }
    try:
        return OneHotEncoder(sparse_output=True, **common)
    except TypeError:
        return OneHotEncoder(sparse=True, **common)


def detect_delimiter(file_path: Path, encoding: str) -> str:
    if file_path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with file_path.open("r", encoding=encoding, errors="replace", newline="") as stream:
            sample = stream.read(65536)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def detect_encoding(file_path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            with file_path.open("r", encoding=encoding) as stream:
                stream.read(65536)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin1"


def read_large_csv(
    file_path: Path,
    max_rows: int,
    random_state: int,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[pd.DataFrame, int, str, str]:
    """Read a CSV in chunks; randomly retain at most max_rows rows."""
    encoding = detect_encoding(file_path)
    delimiter = detect_delimiter(file_path, encoding)
    rng = np.random.default_rng(random_state)

    chunks = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        chunksize=50000,
        low_memory=False,
        on_bad_lines="skip",
    )

    total_rows = 0
    retained: Optional[pd.DataFrame] = None
    all_chunks: list[pd.DataFrame] = []

    for chunk in chunks:
        total_rows += len(chunk)
        if progress_callback is not None:
            progress_callback(total_rows)

        if max_rows == 0:
            all_chunks.append(chunk)
            continue

        chunk = chunk.copy()
        chunk["__sample_priority__"] = rng.random(len(chunk))

        if retained is None:
            retained = chunk
        else:
            retained = pd.concat([retained, chunk], ignore_index=True)

        if len(retained) > max_rows:
            retained = retained.nsmallest(max_rows, "__sample_priority__")

    if max_rows == 0:
        dataframe = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    else:
        dataframe = retained if retained is not None else pd.DataFrame()
        if "__sample_priority__" in dataframe.columns:
            dataframe = dataframe.drop(columns="__sample_priority__")
        dataframe = dataframe.reset_index(drop=True)

    return dataframe, total_rows, encoding, delimiter


def excel_sheet_names(file_path: Path) -> list[str]:
    with pd.ExcelFile(file_path) as workbook:
        return list(workbook.sheet_names)


def read_excel_data(
    file_path: Path,
    sheet_name: str,
    max_rows: int,
) -> tuple[pd.DataFrame, int]:
    nrows = None if max_rows == 0 else max_rows
    dataframe = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        nrows=nrows,
    )
    # pandas does not expose a cheap universal exact Excel row count. The GUI
    # reports the loaded count and labels it as capped when a limit was used.
    return dataframe, len(dataframe)


def infer_semantic_type(series: pd.Series) -> str:
    non_missing = series.dropna()
    if non_missing.empty:
        return "All missing"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"
    if pd.api.types.is_numeric_dtype(series):
        unique_count = non_missing.nunique(dropna=True)
        if unique_count <= 2:
            return "Binary numeric"
        return "Numeric"

    sample = non_missing.astype(str).head(500)
    name_hint = str(series.name).lower()
    date_hint = any(token in name_hint for token in ("date", "time", "timestamp", "year"))
    if date_hint:
        parsed = pd.to_datetime(sample, errors="coerce")
        if float(parsed.notna().mean()) >= 0.75:
            return "Date/Time"

    unique_count = non_missing.nunique(dropna=True)
    ratio = unique_count / max(len(non_missing), 1)
    if ratio > 0.80 and unique_count > 50:
        return "High-cardinality text / ID"
    return "Categorical / text"


def suggest_target_column(dataframe: pd.DataFrame) -> str:
    if dataframe.empty or not len(dataframe.columns):
        return ""
    preferred_tokens = (
        "target", "label", "class", "outcome", "result", "response", "y"
    )
    for column in dataframe.columns:
        lower = str(column).strip().lower()
        if lower in preferred_tokens or any(lower.endswith("_" + token) for token in preferred_tokens):
            return str(column)
    return str(dataframe.columns[-1])


def suggest_excluded_columns(dataframe: pd.DataFrame, target_column: str) -> list[str]:
    suggestions: list[str] = []
    for column in dataframe.columns:
        column_name = str(column)
        if column_name == target_column:
            continue
        series = dataframe[column]
        non_missing = series.dropna()
        if non_missing.empty or non_missing.nunique(dropna=True) <= 1:
            suggestions.append(column_name)
            continue
        lower = column_name.lower()
        unique_ratio = non_missing.nunique(dropna=True) / max(len(non_missing), 1)
        id_hint = lower in {"id", "index", "serial", "uuid", "guid"} or lower.endswith("_id")
        if id_hint and unique_ratio > 0.80:
            suggestions.append(column_name)
    return suggestions


def infer_task_type(target: pd.Series) -> str:
    non_missing = target.dropna()
    if non_missing.empty:
        return "Classification"
    unique_count = int(non_missing.nunique(dropna=True))
    row_count = len(non_missing)

    if not pd.api.types.is_numeric_dtype(non_missing):
        return "Classification"

    classification_limit = max(20, int(math.sqrt(max(row_count, 1))))
    if unique_count <= classification_limit:
        return "Classification"
    return "Regression"


def identify_datetime_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if infer_semantic_type(dataframe[column]) == "Date/Time"
    ]


def add_datetime_features(
    dataframe: pd.DataFrame,
    datetime_columns: list[str],
) -> pd.DataFrame:
    output = dataframe.copy()
    for column in datetime_columns:
        if column not in output.columns:
            continue
        parsed = pd.to_datetime(output[column], errors="coerce")
        output[f"{column}__year"] = parsed.dt.year
        output[f"{column}__month"] = parsed.dt.month
        output[f"{column}__day"] = parsed.dt.day
        output[f"{column}__dayofweek"] = parsed.dt.dayofweek
        output[f"{column}__hour"] = parsed.dt.hour
        output = output.drop(columns=[column])
    return output


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = [
        str(column)
        for column in feature_frame.columns
        if pd.api.types.is_numeric_dtype(feature_frame[column])
        and not pd.api.types.is_bool_dtype(feature_frame[column])
    ]
    categorical_columns = [
        str(column)
        for column in feature_frame.columns
        if str(column) not in numeric_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns remain after configuration.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, numeric_columns, categorical_columns


def classification_estimator(name: str, random_state: int) -> Any:
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        )
    if name == "SVM":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        )
    if name == "Neural Network":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1200,
            early_stopping=True,
            random_state=random_state,
        )
    raise ValueError(f"Unknown classification model: {name}")


def regression_estimator(name: str, random_state: int) -> Any:
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=random_state)
    if name == "SVR":
        return SVR(kernel="rbf")
    if name == "KNN Regressor":
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    if name == "Ridge Regression":
        return Ridge(alpha=1.0)
    if name == "Neural Network":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=1500,
            early_stopping=True,
            random_state=random_state,
        )
    raise ValueError(f"Unknown regression model: {name}")


def probability_scores(
    estimator: Pipeline,
    X: pd.DataFrame,
    class_count: int,
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        return np.asarray(probabilities, dtype=float)
    decision = np.asarray(estimator.decision_function(X), dtype=float)
    if decision.ndim == 1:
        probability_one = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack([1.0 - probability_one, probability_one])
    decision = decision - decision.max(axis=1, keepdims=True)
    exponential = np.exp(decision)
    return exponential / exponential.sum(axis=1, keepdims=True)


def classification_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> tuple[float, float]:
    try:
        binary = label_binarize(y_true, classes=np.arange(class_count))
        if class_count == 2:
            binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
        return (
            float(
                roc_auc_score(
                    binary,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                )
            ),
            float(average_precision_score(binary, probabilities, average="macro")),
        )
    except Exception:
        return float("nan"), float("nan")


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    validation_probabilities: Optional[np.ndarray]
    test_probabilities: Optional[np.ndarray]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    training_seconds: float


@dataclass
class TrainerState:
    file_path: Optional[Path] = None
    sheet_name: str = ""
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_rows_seen: int = 0
    load_note: str = ""
    semantic_types: dict[str, str] = field(default_factory=dict)

    task: str = ""
    target_column: str = ""
    excluded_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    input_feature_columns: list[str] = field(default_factory=list)
    prepared_feature_columns: list[str] = field(default_factory=list)

    X_train: Optional[pd.DataFrame] = None
    X_validation: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    y_train: Optional[np.ndarray] = None
    y_validation: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    label_encoder: Optional[LabelEncoder] = None

    results: dict[str, ModelResult] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    loaded_model_package: Optional[dict[str, Any]] = None

    current_figure: Optional[Figure] = None
    current_canvas: Optional[FigureCanvasTkAgg] = None


class DataTrainerGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} — Version {APP_VERSION}")
        self.geometry("1580x940")
        self.minsize(1280, 780)

        self.state_data = TrainerState()
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()

        self._build_header()
        self._build_tabs()
        self._build_status_bar()
        self.after(100, self._process_messages)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # --------------------------- GUI construction ---------------------------

    def _accent_button(self, *args, **kwargs) -> ctk.CTkButton:
        kwargs.setdefault("fg_color", ACCENT)
        kwargs.setdefault("hover_color", ACCENT_HOVER)
        return ctk.CTkButton(*args, **kwargs)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=12, pady=(12, 6))

        stripe = ctk.CTkFrame(header, width=10, fg_color=ACCENT)
        stripe.pack(side="left", fill="y", padx=(0, 10), pady=0)

        ctk.CTkLabel(
            header,
            text="ALL IN ONE AI — DATA TRAINER",
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(side="left", padx=12, pady=12)

        ctk.CTkLabel(
            header,
            text="Upload Data • Auto-Inspect • Train • Validate • Test • Explain",
            font=ctk.CTkFont(size=14),
        ).pack(side="left", padx=12)

        self.file_status_label = ctk.CTkLabel(
            header,
            text="Data: Not loaded",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.file_status_label.pack(side="right", padx=18)

    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=6)

        for tab_name in (
            "1. Load & Inspect",
            "2. Configure Experiment",
            "3. Train & Validate",
            "4. Results & Graphs",
            "5. Save Model & Predict",
        ):
            self.tabs.add(tab_name)

        self._build_load_tab()
        self._build_config_tab()
        self._build_train_tab()
        self._build_results_tab()
        self._build_predict_tab()

    def _build_status_bar(self) -> None:
        footer = ctk.CTkFrame(self)
        footer.pack(fill="x", padx=12, pady=(6, 12))

        self.status_label = ctk.CTkLabel(footer, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=12, pady=8, fill="x", expand=True)

        self.progress_bar = ctk.CTkProgressBar(footer, width=350, progress_color=ACCENT)
        self.progress_bar.set(0)
        self.progress_bar.pack(side="right", padx=12, pady=8)

    def _build_load_tab(self) -> None:
        tab = self.tabs.tab("1. Load & Inspect")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(
            tab,
            text="Step 1 — Upload a CSV or Excel Dataset",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkLabel(tab, text="Data file").grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.file_entry = ctk.CTkEntry(tab)
        self.file_entry.grid(row=1, column=1, columnspan=2, padx=20, pady=8, sticky="ew")
        self._accent_button(tab, text="Browse", width=120, command=self._browse_file).grid(
            row=1, column=3, padx=20, pady=8
        )

        ctk.CTkLabel(tab, text="Excel sheet").grid(row=2, column=0, padx=20, pady=8, sticky="w")
        self.sheet_var = ctk.StringVar(value="Not applicable")
        self.sheet_menu = ctk.CTkOptionMenu(
            tab,
            values=["Not applicable"],
            variable=self.sheet_var,
            fg_color=ACCENT,
            button_color=ACCENT_HOVER,
            button_hover_color=ACCENT,
        )
        self.sheet_menu.grid(row=2, column=1, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(tab, text="Maximum rows retained").grid(
            row=2, column=2, padx=20, pady=8, sticky="e"
        )
        self.max_rows_entry = ctk.CTkEntry(tab, width=140)
        self.max_rows_entry.insert(0, "50000")
        self.max_rows_entry.grid(row=2, column=3, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(
            tab,
            text=(
                "For very large CSV files, the whole file is scanned in chunks and a random sample is retained. "
                "Enter 0 only when the computer has enough RAM for every row."
            ),
            wraplength=1200,
            justify="left",
        ).grid(row=3, column=0, columnspan=4, padx=20, pady=6, sticky="w")

        controls = ctk.CTkFrame(tab)
        controls.grid(row=4, column=0, columnspan=4, padx=20, pady=10, sticky="ew")
        self._accent_button(
            controls,
            text="LOAD AND INSPECT DATA",
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_load,
        ).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Clear Data", command=self._clear_data).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Export Column Profile", command=self._export_profile).pack(
            side="left", padx=10, pady=10
        )

        self.dataset_summary_label = ctk.CTkLabel(
            tab,
            text="Rows: — | Columns: — | Missing cells: — | Duplicates: —",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        self.dataset_summary_label.grid(row=5, column=0, columnspan=4, padx=20, pady=8, sticky="ew")

        inspection = ctk.CTkFrame(tab)
        inspection.grid(row=7, column=0, columnspan=4, padx=20, pady=(8, 20), sticky="nsew")
        inspection.grid_columnconfigure(0, weight=2)
        inspection.grid_columnconfigure(1, weight=3)
        inspection.grid_rowconfigure(0, weight=1)

        profile_frame = ctk.CTkFrame(inspection)
        profile_frame.grid(row=0, column=0, padx=(8, 4), pady=8, sticky="nsew")
        profile_frame.grid_rowconfigure(1, weight=1)
        profile_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            profile_frame,
            text="Automatic Column Profile",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=8, pady=8, sticky="w")

        profile_columns = ("Column", "Detected type", "Pandas dtype", "Missing", "Unique", "Example")
        self.profile_tree = ttk.Treeview(profile_frame, columns=profile_columns, show="headings")
        for column in profile_columns:
            self.profile_tree.heading(column, text=column)
            width = 170 if column in ("Column", "Detected type") else 100
            self.profile_tree.column(column, width=width, anchor="w")
        self.profile_tree.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        profile_scroll = ttk.Scrollbar(profile_frame, orient="vertical", command=self.profile_tree.yview)
        profile_scroll.grid(row=1, column=1, sticky="ns")
        self.profile_tree.configure(yscrollcommand=profile_scroll.set)

        preview_frame = ctk.CTkFrame(inspection)
        preview_frame.grid(row=0, column=1, padx=(4, 8), pady=8, sticky="nsew")
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            preview_frame,
            text="Data Preview — first 100 loaded rows",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.preview_tree = ttk.Treeview(preview_frame, show="headings")
        self.preview_tree.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        preview_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        preview_y.grid(row=1, column=1, sticky="ns")
        preview_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        preview_x.grid(row=2, column=0, sticky="ew")
        self.preview_tree.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)

    def _build_config_tab(self) -> None:
        tab = self.tabs.tab("2. Configure Experiment")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            tab,
            text="Step 2 — Choose What the AI Should Predict",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkLabel(tab, text="Target column").grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.target_var = ctk.StringVar(value="Load data first")
        self.target_menu = ctk.CTkOptionMenu(
            tab,
            values=["Load data first"],
            variable=self.target_var,
            command=lambda _value: self._refresh_configuration_suggestions(),
            fg_color=ACCENT,
            button_color=ACCENT_HOVER,
            button_hover_color=ACCENT,
        )
        self.target_menu.grid(row=1, column=1, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(tab, text="Learning task").grid(row=1, column=2, padx=20, pady=8, sticky="e")
        self.task_var = ctk.StringVar(value="Auto Detect")
        self.task_menu = ctk.CTkOptionMenu(
            tab,
            values=["Auto Detect", "Classification", "Regression"],
            variable=self.task_var,
            command=lambda _value: self._update_model_choices(),
            fg_color=ACCENT,
            button_color=ACCENT_HOVER,
            button_hover_color=ACCENT,
        )
        self.task_menu.grid(row=1, column=3, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(tab, text="Columns to exclude").grid(row=2, column=0, padx=20, pady=8, sticky="w")
        self.exclude_entry = ctk.CTkEntry(
            tab,
            placeholder_text="Optional: ID, serial_number, free_text",
        )
        self.exclude_entry.grid(row=2, column=1, columnspan=3, padx=20, pady=8, sticky="ew")
        ctk.CTkLabel(
            tab,
            text="Comma-separated. Constant and all-missing columns are removed automatically.",
        ).grid(row=3, column=1, columnspan=3, padx=20, pady=(0, 8), sticky="w")

        split_frame = ctk.CTkFrame(tab)
        split_frame.grid(row=4, column=0, columnspan=4, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(split_frame, text="Train %").pack(side="left", padx=(12, 4))
        self.train_percent_entry = ctk.CTkEntry(split_frame, width=75)
        self.train_percent_entry.insert(0, "70")
        self.train_percent_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(split_frame, text="Validation %").pack(side="left", padx=(16, 4))
        self.validation_percent_entry = ctk.CTkEntry(split_frame, width=75)
        self.validation_percent_entry.insert(0, "15")
        self.validation_percent_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(split_frame, text="Test %").pack(side="left", padx=(16, 4))
        self.test_percent_entry = ctk.CTkEntry(split_frame, width=75)
        self.test_percent_entry.insert(0, "15")
        self.test_percent_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(split_frame, text="Random state").pack(side="left", padx=(16, 4))
        self.random_state_entry = ctk.CTkEntry(split_frame, width=80)
        self.random_state_entry.insert(0, "42")
        self.random_state_entry.pack(side="left", padx=6, pady=12)

        self._accent_button(
            tab,
            text="VALIDATE EXPERIMENT CONFIGURATION",
            height=44,
            command=self._validate_configuration,
        ).grid(row=5, column=0, columnspan=4, padx=20, pady=12, sticky="ew")

        self.configuration_label = ctk.CTkLabel(
            tab,
            text="Configuration: Load a dataset first.",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
            wraplength=1200,
            justify="left",
        )
        self.configuration_label.grid(row=6, column=0, columnspan=4, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(
            tab,
            text="Model choices will update automatically for classification or regression.",
        ).grid(row=7, column=0, columnspan=4, padx=20, pady=8, sticky="w")

        self.config_text = ctk.CTkTextbox(tab)
        self.config_text.grid(row=10, column=0, columnspan=4, padx=20, pady=(8, 20), sticky="nsew")
        self.config_text.insert(
            "1.0",
            "Select the target column—the answer the model should learn to predict.\n\n"
            "Auto Detect treats text/labels and low-cardinality numeric targets as classification, "
            "and continuous numeric targets as regression.",
        )

    def _build_train_tab(self) -> None:
        tab = self.tabs.tab("3. Train & Validate")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(
            tab,
            text="Step 3 — Compare Four or Five Machine-Learning Models",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")

        self.models_frame = ctk.CTkFrame(tab)
        self.models_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.model_vars: list[ctk.BooleanVar] = []
        self.model_checkboxes: list[ctk.CTkCheckBox] = []
        for index in range(6):
            variable = ctk.BooleanVar(value=index < 5)
            checkbox = ctk.CTkCheckBox(
                self.models_frame,
                text=CLASSIFICATION_MODELS[index],
                variable=variable,
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
            )
            checkbox.grid(row=index // 3, column=index % 3, padx=18, pady=12, sticky="w")
            self.model_vars.append(variable)
            self.model_checkboxes.append(checkbox)

        self._accent_button(
            tab,
            text="TRAIN, VALIDATE AND TEST SELECTED MODELS",
            height=50,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_training,
        ).grid(row=2, column=0, padx=20, pady=15, sticky="ew")

        self.training_summary_label = ctk.CTkLabel(
            tab,
            text="Training status: Configuration not validated.",
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        self.training_summary_label.grid(row=3, column=0, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(
            tab,
            text=(
                "The best model is selected from validation performance. The untouched test set is then used for final reporting."
            ),
            wraplength=1200,
            justify="left",
        ).grid(row=4, column=0, padx=20, pady=6, sticky="w")

        self.training_log = ctk.CTkTextbox(tab)
        self.training_log.grid(row=5, column=0, padx=20, pady=(8, 20), sticky="nsew")

    def _build_results_tab(self) -> None:
        tab = self.tabs.tab("4. Results & Graphs")
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=3)
        tab.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")

        ctk.CTkLabel(controls, text="Model").pack(side="left", padx=(12, 4))
        self.result_model_var = ctk.StringVar(value="No trained models")
        self.result_model_menu = ctk.CTkOptionMenu(
            controls,
            values=["No trained models"],
            variable=self.result_model_var,
            width=190,
            fg_color=ACCENT,
            button_color=ACCENT_HOVER,
            button_hover_color=ACCENT,
        )
        self.result_model_menu.pack(side="left", padx=8, pady=10)

        ctk.CTkLabel(controls, text="Graph").pack(side="left", padx=(16, 4))
        self.graph_type_var = ctk.StringVar(value="Model Comparison")
        self.graph_type_menu = ctk.CTkOptionMenu(
            controls,
            values=["Model Comparison"],
            variable=self.graph_type_var,
            width=250,
            fg_color=ACCENT,
            button_color=ACCENT_HOVER,
            button_hover_color=ACCENT,
        )
        self.graph_type_menu.pack(side="left", padx=8, pady=10)

        self._accent_button(controls, text="Generate Graph", command=self._generate_graph).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Save Graph", command=self._save_graph).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Export Metrics", command=self._export_metrics).pack(
            side="left", padx=10, pady=10
        )

        table_frame = ctk.CTkFrame(tab)
        table_frame.grid(row=1, column=0, padx=(20, 10), pady=(0, 20), sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(table_frame, show="headings")
        self.results_tree.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        result_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        result_y.grid(row=0, column=1, sticky="ns")
        result_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.results_tree.xview)
        result_x.grid(row=1, column=0, sticky="ew")
        self.results_tree.configure(yscrollcommand=result_y.set, xscrollcommand=result_x.set)

        self.best_model_label = ctk.CTkLabel(
            table_frame,
            text="Best model: —",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        )
        self.best_model_label.grid(row=2, column=0, padx=10, pady=8, sticky="ew")

        self.graph_frame = ctk.CTkFrame(tab)
        self.graph_frame.grid(row=1, column=1, padx=(10, 20), pady=(0, 20), sticky="nsew")
        self.graph_frame.grid_columnconfigure(0, weight=1)
        self.graph_frame.grid_rowconfigure(0, weight=1)

    def _build_predict_tab(self) -> None:
        tab = self.tabs.tab("5. Save Model & Predict")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(
            tab,
            text="Step 5 — Save the AI and Predict a New File",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkLabel(tab, text="Active model").grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.active_model_label = ctk.CTkLabel(
            tab,
            text="Train a model or load a saved .joblib model.",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.active_model_label.grid(row=1, column=1, padx=20, pady=8, sticky="ew")
        ctk.CTkButton(tab, text="Load .joblib", width=120, command=self._load_model).grid(
            row=1, column=2, padx=20, pady=8
        )

        model_controls = ctk.CTkFrame(tab)
        model_controls.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        self._accent_button(
            model_controls,
            text="SAVE BEST MODEL",
            height=44,
            command=self._save_best_model,
        ).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(
            model_controls,
            text="Export Experiment Report",
            command=self._export_experiment_report,
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkLabel(tab, text="New CSV / Excel file").grid(
            row=3, column=0, padx=20, pady=8, sticky="w"
        )
        self.batch_file_entry = ctk.CTkEntry(tab)
        self.batch_file_entry.grid(row=3, column=1, padx=20, pady=8, sticky="ew")
        ctk.CTkButton(tab, text="Browse", width=120, command=self._browse_batch_file).grid(
            row=3, column=2, padx=20, pady=8
        )

        self._accent_button(
            tab,
            text="PREDICT NEW FILE AND EXPORT RESULTS",
            height=46,
            command=self._start_batch_prediction,
        ).grid(row=4, column=0, columnspan=3, padx=20, pady=12, sticky="ew")

        self.batch_status_label = ctk.CTkLabel(
            tab,
            text="Batch prediction: Waiting for a model and file.",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.batch_status_label.grid(row=5, column=0, columnspan=3, padx=20, pady=8, sticky="ew")

        self.predict_log = ctk.CTkTextbox(tab)
        self.predict_log.grid(row=6, column=0, columnspan=3, padx=20, pady=(8, 20), sticky="nsew")
        self.predict_log.insert(
            "1.0",
            "The saved model contains preprocessing, missing-value handling, categorical encoding, date conversion and the trained estimator.\n\n"
            "For batch prediction, the new file must contain the same input feature column names used during training.",
        )

    # ------------------------------ Loading ------------------------------

    def _browse_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select CSV or Excel dataset",
            filetypes=[
                ("Supported data", "*.csv *.tsv *.txt *.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        path = Path(selected)
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, str(path))
        self.state_data.file_path = path

        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                sheets = excel_sheet_names(path)
                if not sheets:
                    sheets = ["Sheet1"]
                self.sheet_menu.configure(values=sheets)
                self.sheet_var.set(sheets[0])
            except Exception as error:
                messagebox.showerror(APP_NAME, f"Could not read Excel sheet names:\n{error}")
        else:
            self.sheet_menu.configure(values=["Not applicable"])
            self.sheet_var.set("Not applicable")

    def _start_load(self) -> None:
        try:
            path = Path(self.file_entry.get().strip())
            if not path.exists():
                raise FileNotFoundError("Select a valid CSV or Excel file.")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError("Supported formats: CSV, TSV, TXT, XLSX and XLS.")
            max_rows = int(self.max_rows_entry.get())
            if max_rows < 0:
                raise ValueError("Maximum rows must be 0 or a positive integer.")

            settings = {
                "path": path,
                "sheet": self.sheet_var.get(),
                "max_rows": max_rows,
                "random_state": 42,
            }
            threading.Thread(
                target=self._thread_wrapper,
                args=(lambda: self._load_worker(settings),),
                daemon=True,
            ).start()
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _load_worker(self, settings: dict[str, Any]) -> None:
        path: Path = settings["path"]
        self._set_status(0.05, "Loading and inspecting data...")

        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            dataframe, total_rows, encoding, delimiter = read_large_csv(
                path,
                settings["max_rows"],
                settings["random_state"],
                progress_callback=lambda rows: self.messages.put(("load_scan", rows)),
            )
            load_note = (
                f"CSV encoding={encoding}, delimiter={repr(delimiter)}. "
                f"Scanned {total_rows:,} rows; retained {len(dataframe):,}."
            )
            sheet_name = ""
        else:
            sheet_name = settings["sheet"]
            dataframe, loaded_rows = read_excel_data(path, sheet_name, settings["max_rows"])
            total_rows = loaded_rows
            cap_text = " (row cap applied)" if settings["max_rows"] and loaded_rows >= settings["max_rows"] else ""
            load_note = (
                f"Excel sheet={sheet_name}. Loaded {loaded_rows:,} rows{cap_text}."
            )

        if dataframe.empty:
            raise RuntimeError("The selected file contains no readable data rows.")
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        if len(set(dataframe.columns)) != len(dataframe.columns):
            raise RuntimeError("Duplicate column names were detected. Rename them before training.")

        semantic_types = {
            str(column): infer_semantic_type(dataframe[column])
            for column in dataframe.columns
        }

        self.state_data.file_path = path
        self.state_data.sheet_name = sheet_name
        self.state_data.dataframe = dataframe
        self.state_data.total_rows_seen = total_rows
        self.state_data.load_note = load_note
        self.state_data.semantic_types = semantic_types
        self.state_data.results.clear()
        self.state_data.best_model_name = None
        self.state_data.loaded_model_package = None

        self.messages.put(("data_loaded", None))
        self._set_status(1.0, "Dataset loaded and inspected")

    def _clear_data(self) -> None:
        self.state_data = TrainerState()
        self.file_entry.delete(0, "end")
        self.file_status_label.configure(text="Data: Not loaded")
        self.dataset_summary_label.configure(text="Rows: — | Columns: — | Missing cells: — | Duplicates: —")
        for tree in (self.profile_tree, self.preview_tree, self.results_tree):
            for item in tree.get_children():
                tree.delete(item)
        self.target_menu.configure(values=["Load data first"])
        self.target_var.set("Load data first")
        self.configuration_label.configure(text="Configuration: Load a dataset first.")
        self._clear_figure()

    def _display_loaded_data(self) -> None:
        dataframe = self.state_data.dataframe
        missing_cells = int(dataframe.isna().sum().sum())
        duplicate_rows = int(dataframe.duplicated().sum())
        self.file_status_label.configure(text=f"Data: {self.state_data.file_path.name}")
        self.dataset_summary_label.configure(
            text=(
                f"Rows retained: {len(dataframe):,} / scanned: {self.state_data.total_rows_seen:,} | "
                f"Columns: {len(dataframe.columns):,} | Missing cells: {missing_cells:,} | Duplicates: {duplicate_rows:,}"
            )
        )

        for item in self.profile_tree.get_children():
            self.profile_tree.delete(item)
        for column in dataframe.columns:
            series = dataframe[column]
            sample_values = series.dropna().astype(str)
            example = sample_values.iloc[0][:60] if not sample_values.empty else ""
            self.profile_tree.insert(
                "",
                "end",
                values=(
                    str(column),
                    self.state_data.semantic_types[str(column)],
                    str(series.dtype),
                    int(series.isna().sum()),
                    int(series.nunique(dropna=True)),
                    example,
                ),
            )

        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        preview_columns = list(dataframe.columns[:40])
        self.preview_tree.configure(columns=preview_columns)
        for column in preview_columns:
            self.preview_tree.heading(column, text=column)
            self.preview_tree.column(column, width=130, anchor="w")
        for _, row in dataframe[preview_columns].head(100).iterrows():
            values = ["" if pd.isna(value) else str(value)[:100] for value in row]
            self.preview_tree.insert("", "end", values=values)

        column_names = [str(column) for column in dataframe.columns]
        self.target_menu.configure(values=column_names)
        suggested_target = suggest_target_column(dataframe)
        self.target_var.set(suggested_target)
        self._refresh_configuration_suggestions()
        self.config_text.delete("1.0", "end")
        self.config_text.insert(
            "1.0",
            self.state_data.load_note
            + "\n\nThe target and possible ID columns have been suggested automatically. Review them before training.",
        )

    def _export_profile(self) -> None:
        if self.state_data.dataframe.empty:
            messagebox.showwarning(APP_NAME, "Load a dataset first.")
            return
        output = filedialog.asksaveasfilename(
            title="Export column profile",
            defaultextension=".csv",
            initialfile="column_profile.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not output:
            return
        rows = []
        dataframe = self.state_data.dataframe
        for column in dataframe.columns:
            series = dataframe[column]
            rows.append(
                {
                    "column": str(column),
                    "detected_type": self.state_data.semantic_types[str(column)],
                    "pandas_dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "missing_percent": float(series.isna().mean() * 100.0),
                    "unique_count": int(series.nunique(dropna=True)),
                }
            )
        pd.DataFrame(rows).to_csv(output, index=False)
        messagebox.showinfo(APP_NAME, f"Column profile exported:\n{output}")

    # ----------------------------- Configuration -----------------------------

    def _refresh_configuration_suggestions(self) -> None:
        if self.state_data.dataframe.empty:
            return
        target = self.target_var.get()
        if target not in self.state_data.dataframe.columns:
            return
        suggested_task = infer_task_type(self.state_data.dataframe[target])
        if self.task_var.get() == "Auto Detect":
            display_task = suggested_task
        else:
            display_task = self.task_var.get()
        suggestions = suggest_excluded_columns(self.state_data.dataframe, target)
        self.exclude_entry.delete(0, "end")
        self.exclude_entry.insert(0, ", ".join(suggestions))
        self.configuration_label.configure(
            text=(
                f"Suggested task: {suggested_task} | Current effective task: {display_task} | "
                f"Target unique values: {self.state_data.dataframe[target].nunique(dropna=True):,}"
            )
        )
        self._update_model_choices()

    def _effective_task(self) -> str:
        requested = self.task_var.get()
        if requested != "Auto Detect":
            return requested
        target = self.target_var.get()
        if target not in self.state_data.dataframe.columns:
            return "Classification"
        return infer_task_type(self.state_data.dataframe[target])

    def _update_model_choices(self) -> None:
        task = self._effective_task() if not self.state_data.dataframe.empty else "Classification"
        names = CLASSIFICATION_MODELS if task == "Classification" else REGRESSION_MODELS
        for checkbox, name in zip(self.model_checkboxes, names):
            checkbox.configure(text=name)

    def _parse_exclusions(self) -> list[str]:
        return [
            value.strip()
            for value in self.exclude_entry.get().split(",")
            if value.strip()
        ]

    def _validate_configuration(self) -> None:
        try:
            configuration = self._build_configuration()
            self._apply_configuration(configuration)
            self.tabs.set("3. Train & Validate")
            messagebox.showinfo(
                APP_NAME,
                "Configuration validated successfully. You can now train the models.",
            )
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _build_configuration(self) -> dict[str, Any]:
        dataframe = self.state_data.dataframe
        if dataframe.empty:
            raise RuntimeError("Load a dataset first.")
        target = self.target_var.get()
        if target not in dataframe.columns:
            raise ValueError("Select a valid target column.")
        task = self._effective_task()
        exclusions = self._parse_exclusions()
        unknown = [column for column in exclusions if column not in dataframe.columns]
        if unknown:
            raise ValueError(f"Excluded columns not found: {', '.join(unknown)}")
        if target in exclusions:
            raise ValueError("The target column cannot also be excluded.")

        train_percent = float(self.train_percent_entry.get())
        validation_percent = float(self.validation_percent_entry.get())
        test_percent = float(self.test_percent_entry.get())
        if any(value <= 0 for value in (train_percent, validation_percent, test_percent)):
            raise ValueError("Train, validation and test percentages must all be positive.")
        if abs(train_percent + validation_percent + test_percent - 100.0) > 1e-6:
            raise ValueError("Train, validation and test percentages must sum to 100.")
        if train_percent < 50:
            raise ValueError("Training percentage should be at least 50%.")

        random_state = int(self.random_state_entry.get())
        return {
            "target": target,
            "task": task,
            "exclusions": exclusions,
            "train_fraction": train_percent / 100.0,
            "validation_fraction": validation_percent / 100.0,
            "test_fraction": test_percent / 100.0,
            "random_state": random_state,
        }

    def _apply_configuration(self, configuration: dict[str, Any]) -> None:
        dataframe = self.state_data.dataframe.copy()
        target = configuration["target"]
        task = configuration["task"]

        dataframe = dataframe.dropna(subset=[target]).reset_index(drop=True)
        if len(dataframe) < 20:
            raise RuntimeError("At least 20 rows with a non-missing target are required.")

        feature_columns = [
            str(column)
            for column in dataframe.columns
            if str(column) != target and str(column) not in configuration["exclusions"]
        ]
        if not feature_columns:
            raise RuntimeError("No feature columns remain.")

        # Remove columns with no information.
        automatically_removed: list[str] = []
        for column in list(feature_columns):
            series = dataframe[column]
            if series.dropna().empty or series.nunique(dropna=True) <= 1:
                feature_columns.remove(column)
                automatically_removed.append(column)
        if not feature_columns:
            raise RuntimeError("Every available feature is constant or empty.")

        X_original = dataframe[feature_columns].copy()
        datetime_columns = identify_datetime_columns(X_original)
        X = add_datetime_features(X_original, datetime_columns)

        if task == "Classification":
            y_text = dataframe[target].astype(str).to_numpy()
            encoder = LabelEncoder()
            y = encoder.fit_transform(y_text)
            class_counts = pd.Series(y).value_counts()
            if len(encoder.classes_) < 2:
                raise RuntimeError("Classification requires at least two target classes.")
            if len(encoder.classes_) > 100:
                raise RuntimeError(
                    "More than 100 target classes were detected. Verify that the selected target is a class label."
                )
            if int(class_counts.min()) < 3:
                raise RuntimeError("Every target class needs at least three rows for train/validation/test splitting.")
            stratify = y
            self.state_data.label_encoder = encoder
        else:
            numeric_target = pd.to_numeric(dataframe[target], errors="coerce")
            valid = numeric_target.notna().to_numpy()
            X = X.loc[valid].reset_index(drop=True)
            X_original = X_original.loc[valid].reset_index(drop=True)
            y = numeric_target.loc[valid].to_numpy(dtype=float)
            stratify = None
            self.state_data.label_encoder = None
            if len(y) < 20:
                raise RuntimeError("Regression requires at least 20 numeric target rows.")
            if float(np.nanstd(y)) <= 1e-12:
                raise RuntimeError("The regression target is constant.")

        temporary_fraction = configuration["validation_fraction"] + configuration["test_fraction"]
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=temporary_fraction,
                random_state=configuration["random_state"],
                stratify=stratify,
            )
            validation_share_of_temp = configuration["validation_fraction"] / temporary_fraction
            stratify_temp = y_temp if task == "Classification" else None
            X_validation, X_test, y_validation, y_test = train_test_split(
                X_temp,
                y_temp,
                train_size=validation_share_of_temp,
                random_state=configuration["random_state"],
                stratify=stratify_temp,
            )
        except ValueError as error:
            raise RuntimeError(
                "The requested split could not preserve every class. Add more rows or adjust percentages.\n\n"
                + str(error)
            ) from error

        self.state_data.task = task
        self.state_data.target_column = target
        self.state_data.excluded_columns = configuration["exclusions"] + automatically_removed
        self.state_data.datetime_columns = datetime_columns
        self.state_data.input_feature_columns = feature_columns
        self.state_data.prepared_feature_columns = [str(column) for column in X.columns]
        self.state_data.X_train = X_train
        self.state_data.X_validation = X_validation
        self.state_data.X_test = X_test
        self.state_data.y_train = np.asarray(y_train)
        self.state_data.y_validation = np.asarray(y_validation)
        self.state_data.y_test = np.asarray(y_test)
        self.state_data.results.clear()
        self.state_data.best_model_name = None
        self.state_data.loaded_model_package = None

        task_detail = ""
        if task == "Classification" and self.state_data.label_encoder is not None:
            task_detail = f" | Classes: {', '.join(map(str, self.state_data.label_encoder.classes_))}"
        self.configuration_label.configure(
            text=(
                f"Validated: {task} | Target: {target} | Features: {len(feature_columns)} original / "
                f"{X.shape[1]} after date expansion | Train: {len(X_train):,} | Validation: {len(X_validation):,} | "
                f"Test: {len(X_test):,}{task_detail}"
            )
        )
        self.training_summary_label.configure(
            text=(
                f"Ready: {task} | Train {len(X_train):,} | Validation {len(X_validation):,} | Test {len(X_test):,}"
            )
        )
        self._update_model_choices()
        self.config_text.delete("1.0", "end")
        self.config_text.insert(
            "1.0",
            (
                f"Task: {task}\nTarget: {target}\nOriginal feature columns: {len(feature_columns)}\n"
                f"Date/time columns expanded: {datetime_columns or 'None'}\n"
                f"Excluded/automatic removals: {self.state_data.excluded_columns or 'None'}\n"
                f"Training rows: {len(X_train):,}\nValidation rows: {len(X_validation):,}\nTest rows: {len(X_test):,}"
            ),
        )

    # ------------------------------- Training -------------------------------

    def _selected_model_names(self) -> list[str]:
        names = CLASSIFICATION_MODELS if self.state_data.task == "Classification" else REGRESSION_MODELS
        selected = [name for name, variable in zip(names, self.model_vars) if variable.get()]
        if len(selected) not in (4, 5):
            raise ValueError("Select exactly four or five models.")
        return selected

    def _start_training(self) -> None:
        try:
            configuration = self._build_configuration()
            # Re-apply in case the user changed settings after validation.
            self._apply_configuration(configuration)
            selected_models = self._selected_model_names()
            settings = {
                "models": selected_models,
                "random_state": configuration["random_state"],
            }
            threading.Thread(
                target=self._thread_wrapper,
                args=(lambda: self._training_worker(settings),),
                daemon=True,
            ).start()
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _training_worker(self, settings: dict[str, Any]) -> None:
        X_train = self.state_data.X_train
        X_validation = self.state_data.X_validation
        X_test = self.state_data.X_test
        y_train = self.state_data.y_train
        y_validation = self.state_data.y_validation
        y_test = self.state_data.y_test
        if any(value is None for value in (X_train, X_validation, X_test, y_train, y_validation, y_test)):
            raise RuntimeError("Validate the experiment configuration first.")

        preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
        self.messages.put(
            (
                "training_log",
                (
                    f"Task: {self.state_data.task}\n"
                    f"Numeric prepared columns: {len(numeric_columns)}\n"
                    f"Categorical prepared columns: {len(categorical_columns)}\n"
                    f"Training rows: {len(X_train):,}\nValidation rows: {len(X_validation):,}\nTest rows: {len(X_test):,}\n"
                ),
            )
        )

        results: dict[str, ModelResult] = {}

        for index, model_name in enumerate(settings["models"], start=1):
            self._set_status(
                (index - 1) / len(settings["models"]),
                f"Training {model_name}...",
            )
            self.messages.put(("training_log", f"Training {model_name}..."))

            estimator = (
                classification_estimator(model_name, settings["random_state"])
                if self.state_data.task == "Classification"
                else regression_estimator(model_name, settings["random_state"])
            )
            pipeline = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", estimator),
                ]
            )

            # Protect STEM laptops from accidentally fitting kernel models on
            # extremely large sampled datasets while preserving unbiased holdouts.
            fit_X = X_train
            fit_y = y_train
            expensive = model_name in {"SVM", "SVR", "KNN", "KNN Regressor", "Neural Network"}
            if expensive and len(X_train) > 25000:
                rng = np.random.default_rng(settings["random_state"])
                chosen = rng.choice(len(X_train), size=25000, replace=False)
                fit_X = X_train.iloc[chosen]
                fit_y = y_train[chosen]
                self.messages.put(
                    (
                        "training_log",
                        f"{model_name}: fitting on a random 25,000-row subset for computational safety; validation/test remain unchanged.",
                    )
                )

            start_time = time.perf_counter()
            pipeline.fit(fit_X, fit_y)
            training_seconds = time.perf_counter() - start_time

            validation_predictions = pipeline.predict(X_validation)
            test_predictions = pipeline.predict(X_test)

            if self.state_data.task == "Classification":
                class_count = len(self.state_data.label_encoder.classes_)
                validation_probabilities = probability_scores(pipeline, X_validation, class_count)
                test_probabilities = probability_scores(pipeline, X_test, class_count)
                validation_metrics = self._classification_metrics(
                    y_validation, validation_predictions, validation_probabilities
                )
                test_metrics = self._classification_metrics(y_test, test_predictions, test_probabilities)
                summary_metric = (
                    f"Validation F1={validation_metrics['f1']:.4f}, Test F1={test_metrics['f1']:.4f}"
                )
            else:
                validation_probabilities = None
                test_probabilities = None
                validation_metrics = self._regression_metrics(y_validation, validation_predictions)
                test_metrics = self._regression_metrics(y_test, test_predictions)
                summary_metric = (
                    f"Validation R2={validation_metrics['r2']:.4f}, Test R2={test_metrics['r2']:.4f}"
                )

            results[model_name] = ModelResult(
                name=model_name,
                pipeline=pipeline,
                validation_predictions=np.asarray(validation_predictions),
                test_predictions=np.asarray(test_predictions),
                validation_probabilities=validation_probabilities,
                test_probabilities=test_probabilities,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                training_seconds=training_seconds,
            )
            self.messages.put(("training_log", f"{model_name} completed — {summary_metric}\n"))

        if self.state_data.task == "Classification":
            best_name = max(
                results,
                key=lambda name: (
                    results[name].validation_metrics["f1"],
                    results[name].validation_metrics["balanced_accuracy"],
                ),
            )
        else:
            best_name = max(
                results,
                key=lambda name: (
                    results[name].validation_metrics["r2"],
                    -results[name].validation_metrics["rmse"],
                ),
            )

        self.state_data.results = results
        self.state_data.best_model_name = best_name
        self.state_data.loaded_model_package = None
        self.messages.put(("training_complete", {"models": list(results), "best": best_name}))
        self._set_status(1.0, f"Training completed — best validation model: {best_name}")

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict[str, float]:
        roc_auc, pr_auc = classification_auc(
            y_true,
            probabilities,
            len(self.state_data.label_encoder.classes_),
        )
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "mcc": float(matthews_corrcoef(y_true, predictions)),
            "kappa": float(cohen_kappa_score(y_true, predictions)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        rmse = float(math.sqrt(mean_squared_error(y_true, predictions)))
        try:
            mape = float(mean_absolute_percentage_error(y_true, predictions))
        except Exception:
            mape = float("nan")
        return {
            "r2": float(r2_score(y_true, predictions)),
            "mae": float(mean_absolute_error(y_true, predictions)),
            "rmse": rmse,
            "mape": mape,
            "median_ae": float(median_absolute_error(y_true, predictions)),
            "explained_variance": float(explained_variance_score(y_true, predictions)),
        }

    # --------------------------- Results and graphs ---------------------------

    def _refresh_results_table(self) -> None:
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if self.state_data.task == "Classification":
            columns = (
                "Model",
                "Val F1",
                "Test Accuracy",
                "Test Precision",
                "Test Recall",
                "Test F1",
                "Balanced",
                "ROC-AUC",
                "PR-AUC",
                "Train sec",
            )
        else:
            columns = (
                "Model",
                "Val R2",
                "Test R2",
                "Test MAE",
                "Test RMSE",
                "Test MAPE",
                "Median AE",
                "Explained Var",
                "Train sec",
            )
        self.results_tree.configure(columns=columns)
        for column in columns:
            self.results_tree.heading(column, text=column)
            self.results_tree.column(column, width=110 if column != "Model" else 150, anchor="center")

        for name, result in self.state_data.results.items():
            if self.state_data.task == "Classification":
                metrics = result.test_metrics
                values = (
                    name,
                    f"{result.validation_metrics['f1']:.4f}",
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['balanced_accuracy']:.4f}",
                    f"{metrics['roc_auc']:.4f}" if np.isfinite(metrics["roc_auc"]) else "N/A",
                    f"{metrics['pr_auc']:.4f}" if np.isfinite(metrics["pr_auc"]) else "N/A",
                    f"{result.training_seconds:.2f}",
                )
            else:
                metrics = result.test_metrics
                values = (
                    name,
                    f"{result.validation_metrics['r2']:.4f}",
                    f"{metrics['r2']:.4f}",
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['mape']:.4f}" if np.isfinite(metrics["mape"]) else "N/A",
                    f"{metrics['median_ae']:.4f}",
                    f"{metrics['explained_variance']:.4f}",
                    f"{result.training_seconds:.2f}",
                )
            self.results_tree.insert("", "end", values=values)

    def _graph_options(self) -> list[str]:
        common = [
            "Model Comparison",
            "Target Distribution",
            "Missing Values",
            "Numeric Correlation Heatmap",
            "PCA / SVD Feature Scatter",
            "Permutation Feature Importance",
            "Learning Curve",
        ]
        if self.state_data.task == "Classification":
            return [
                "Testing Confusion Matrix",
                "Validation Confusion Matrix",
                "Testing ROC Curve",
                "Testing Precision-Recall Curve",
            ] + common
        return [
            "Testing Actual vs Predicted",
            "Validation Actual vs Predicted",
            "Testing Residual Plot",
            "Testing Residual Distribution",
        ] + common

    def _selected_result(self) -> ModelResult:
        name = self.result_model_var.get()
        if name not in self.state_data.results:
            raise RuntimeError("Select a trained model.")
        return self.state_data.results[name]

    def _clear_figure(self) -> None:
        if self.state_data.current_canvas is not None:
            self.state_data.current_canvas.get_tk_widget().destroy()
            self.state_data.current_canvas = None
        if self.state_data.current_figure is not None:
            plt.close(self.state_data.current_figure)
            self.state_data.current_figure = None

    def _display_figure(self, figure: Figure) -> None:
        self._clear_figure()
        canvas = FigureCanvasTkAgg(figure, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.state_data.current_figure = figure
        self.state_data.current_canvas = canvas

    def _generate_graph(self) -> None:
        try:
            graph_type = self.graph_type_var.get()
            result = self._selected_result()
            figure = Figure(figsize=(8.6, 6.4), dpi=100)
            axis = figure.add_subplot(111)

            if graph_type in {"Testing Confusion Matrix", "Validation Confusion Matrix"}:
                y_true = self.state_data.y_test if graph_type.startswith("Testing") else self.state_data.y_validation
                predictions = result.test_predictions if graph_type.startswith("Testing") else result.validation_predictions
                class_names = self.state_data.label_encoder.classes_
                matrix = confusion_matrix(y_true, predictions, labels=np.arange(len(class_names)))
                image = axis.imshow(matrix)
                figure.colorbar(image, ax=axis)
                axis.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=35, ha="right")
                axis.set_yticks(np.arange(len(class_names)), labels=class_names)
                axis.set_xlabel("Predicted class")
                axis.set_ylabel("Actual class")
                axis.set_title(f"{graph_type} — {result.name}")
                threshold = matrix.max() / 2.0 if matrix.size else 0
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        axis.text(
                            column_index,
                            row_index,
                            str(matrix[row_index, column_index]),
                            ha="center",
                            va="center",
                            color="white" if matrix[row_index, column_index] > threshold else "black",
                        )

            elif graph_type == "Testing ROC Curve":
                class_names = self.state_data.label_encoder.classes_
                class_count = len(class_names)
                binary = label_binarize(self.state_data.y_test, classes=np.arange(class_count))
                if class_count == 2:
                    binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
                for index, class_name in enumerate(class_names):
                    fpr, tpr, _ = roc_curve(binary[:, index], result.test_probabilities[:, index])
                    axis.plot(fpr, tpr, label=str(class_name))
                axis.plot([0, 1], [0, 1], linestyle="--")
                axis.set_xlabel("False Positive Rate")
                axis.set_ylabel("True Positive Rate")
                axis.set_title(f"Testing ROC Curve — {result.name}")
                axis.legend(fontsize=8)

            elif graph_type == "Testing Precision-Recall Curve":
                class_names = self.state_data.label_encoder.classes_
                class_count = len(class_names)
                binary = label_binarize(self.state_data.y_test, classes=np.arange(class_count))
                if class_count == 2:
                    binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
                for index, class_name in enumerate(class_names):
                    precision, recall, _ = precision_recall_curve(
                        binary[:, index], result.test_probabilities[:, index]
                    )
                    axis.plot(recall, precision, label=str(class_name))
                axis.set_xlabel("Recall")
                axis.set_ylabel("Precision")
                axis.set_title(f"Testing Precision–Recall Curve — {result.name}")
                axis.legend(fontsize=8)

            elif graph_type in {"Testing Actual vs Predicted", "Validation Actual vs Predicted"}:
                y_true = self.state_data.y_test if graph_type.startswith("Testing") else self.state_data.y_validation
                predictions = result.test_predictions if graph_type.startswith("Testing") else result.validation_predictions
                axis.scatter(y_true, predictions, alpha=0.65)
                minimum = min(float(np.min(y_true)), float(np.min(predictions)))
                maximum = max(float(np.max(y_true)), float(np.max(predictions)))
                axis.plot([minimum, maximum], [minimum, maximum], linestyle="--")
                axis.set_xlabel("Actual target")
                axis.set_ylabel("Predicted target")
                axis.set_title(f"{graph_type} — {result.name}")

            elif graph_type == "Testing Residual Plot":
                residuals = self.state_data.y_test - result.test_predictions
                axis.scatter(result.test_predictions, residuals, alpha=0.65)
                axis.axhline(0.0, linestyle="--")
                axis.set_xlabel("Predicted target")
                axis.set_ylabel("Residual: actual − predicted")
                axis.set_title(f"Testing Residual Plot — {result.name}")

            elif graph_type == "Testing Residual Distribution":
                residuals = self.state_data.y_test - result.test_predictions
                axis.hist(residuals, bins=30)
                axis.set_xlabel("Residual")
                axis.set_ylabel("Frequency")
                axis.set_title(f"Testing Residual Distribution — {result.name}")

            elif graph_type == "Model Comparison":
                names = list(self.state_data.results)
                positions = np.arange(len(names))
                width = 0.36
                if self.state_data.task == "Classification":
                    validation = [self.state_data.results[name].validation_metrics["f1"] for name in names]
                    testing = [self.state_data.results[name].test_metrics["f1"] for name in names]
                    axis.set_ylabel("Weighted F1")
                else:
                    validation = [self.state_data.results[name].validation_metrics["r2"] for name in names]
                    testing = [self.state_data.results[name].test_metrics["r2"] for name in names]
                    axis.set_ylabel("R²")
                axis.bar(positions - width / 2, validation, width, label="Validation")
                axis.bar(positions + width / 2, testing, width, label="Testing")
                axis.set_xticks(positions, labels=names, rotation=30, ha="right")
                axis.set_title("Validation and Testing Model Comparison")
                axis.legend()

            elif graph_type == "Target Distribution":
                target = self.state_data.dataframe[self.state_data.target_column].dropna()
                if self.state_data.task == "Classification":
                    counts = target.astype(str).value_counts().head(30)
                    axis.bar(np.arange(len(counts)), counts.values)
                    axis.set_xticks(np.arange(len(counts)), labels=counts.index, rotation=40, ha="right")
                    axis.set_ylabel("Rows")
                else:
                    numeric = pd.to_numeric(target, errors="coerce").dropna()
                    axis.hist(numeric, bins=30)
                    axis.set_xlabel(self.state_data.target_column)
                    axis.set_ylabel("Frequency")
                axis.set_title("Target Distribution")

            elif graph_type == "Missing Values":
                missing = self.state_data.dataframe.isna().sum().sort_values(ascending=False)
                missing = missing[missing > 0].head(30)
                if missing.empty:
                    axis.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
                    axis.set_axis_off()
                else:
                    axis.bar(np.arange(len(missing)), missing.values)
                    axis.set_xticks(np.arange(len(missing)), labels=missing.index, rotation=40, ha="right")
                    axis.set_ylabel("Missing cells")
                    axis.set_title("Columns with Missing Values")

            elif graph_type == "Numeric Correlation Heatmap":
                numeric = self.state_data.dataframe.select_dtypes(include=[np.number]).copy()
                if numeric.shape[1] < 2:
                    raise RuntimeError("At least two numeric columns are required for a correlation heatmap.")
                numeric = numeric.loc[:, numeric.var(numeric_only=True).sort_values(ascending=False).head(20).index]
                correlation = numeric.corr()
                image = axis.imshow(correlation, vmin=-1, vmax=1)
                figure.colorbar(image, ax=axis)
                axis.set_xticks(np.arange(len(correlation.columns)), labels=correlation.columns, rotation=45, ha="right")
                axis.set_yticks(np.arange(len(correlation.columns)), labels=correlation.columns)
                axis.set_title("Numeric Correlation Heatmap — top 20 variable columns")

            elif graph_type == "PCA / SVD Feature Scatter":
                X = pd.concat(
                    [self.state_data.X_train, self.state_data.X_validation, self.state_data.X_test],
                    ignore_index=True,
                )
                y = np.concatenate(
                    [self.state_data.y_train, self.state_data.y_validation, self.state_data.y_test]
                )
                if len(X) > 5000:
                    rng = np.random.default_rng(42)
                    chosen = rng.choice(len(X), 5000, replace=False)
                    X = X.iloc[chosen]
                    y = y[chosen]
                transformed = result.pipeline.named_steps["preprocessor"].transform(X)
                if sparse.issparse(transformed):
                    projection = TruncatedSVD(n_components=2, random_state=42).fit_transform(transformed)
                else:
                    projection = PCA(n_components=2, random_state=42).fit_transform(np.asarray(transformed))
                if self.state_data.task == "Classification":
                    for class_index, class_name in enumerate(self.state_data.label_encoder.classes_):
                        mask = y == class_index
                        axis.scatter(projection[mask, 0], projection[mask, 1], label=str(class_name), alpha=0.65)
                    axis.legend(fontsize=8)
                else:
                    scatter_plot = axis.scatter(projection[:, 0], projection[:, 1], c=y, alpha=0.65)
                    figure.colorbar(scatter_plot, ax=axis, label=self.state_data.target_column)
                axis.set_xlabel("Component 1")
                axis.set_ylabel("Component 2")
                axis.set_title("Two-Dimensional Projection of Prepared Features")

            elif graph_type == "Permutation Feature Importance":
                X_holdout = self.state_data.X_test
                y_holdout = self.state_data.y_test
                if len(X_holdout) > 2000:
                    rng = np.random.default_rng(42)
                    chosen = rng.choice(len(X_holdout), 2000, replace=False)
                    X_holdout = X_holdout.iloc[chosen]
                    y_holdout = y_holdout[chosen]
                scoring = "f1_weighted" if self.state_data.task == "Classification" else "r2"
                importance = permutation_importance(
                    result.pipeline,
                    X_holdout,
                    y_holdout,
                    scoring=scoring,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=1,
                )
                top_count = min(20, len(X_holdout.columns))
                top_indices = np.argsort(importance.importances_mean)[-top_count:]
                axis.barh(
                    np.arange(top_count),
                    importance.importances_mean[top_indices],
                    xerr=importance.importances_std[top_indices],
                )
                axis.set_yticks(np.arange(top_count), labels=np.asarray(X_holdout.columns)[top_indices])
                axis.set_xlabel("Decrease in holdout score after shuffling")
                axis.set_title(f"Permutation Feature Importance — {result.name}")

            elif graph_type == "Learning Curve":
                X = pd.concat([self.state_data.X_train, self.state_data.X_validation], ignore_index=True)
                y = np.concatenate([self.state_data.y_train, self.state_data.y_validation])
                if len(X) > 10000:
                    rng = np.random.default_rng(42)
                    chosen = rng.choice(len(X), 10000, replace=False)
                    X = X.iloc[chosen]
                    y = y[chosen]
                scoring = "f1_weighted" if self.state_data.task == "Classification" else "r2"
                cv_folds = 3
                train_sizes, train_scores, validation_scores = learning_curve(
                    clone(result.pipeline),
                    X,
                    y,
                    cv=cv_folds,
                    train_sizes=np.linspace(0.25, 1.0, 5),
                    scoring=scoring,
                    n_jobs=1,
                    shuffle=True,
                    random_state=42,
                )
                axis.plot(train_sizes, np.mean(train_scores, axis=1), marker="o", label="Training")
                axis.plot(train_sizes, np.mean(validation_scores, axis=1), marker="o", label="Cross-validation")
                axis.set_xlabel("Training rows")
                axis.set_ylabel("Weighted F1" if self.state_data.task == "Classification" else "R²")
                axis.set_title(f"Learning Curve — {result.name}")
                axis.legend()

            figure.tight_layout()
            self._display_figure(figure)
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _save_graph(self) -> None:
        figure = self.state_data.current_figure
        if figure is None:
            messagebox.showwarning(APP_NAME, "Generate a graph first.")
            return
        output = filedialog.asksaveasfilename(
            title="Save graph",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("JPEG", "*.jpg")],
        )
        if output:
            figure.savefig(output, dpi=300, bbox_inches="tight")
            messagebox.showinfo(APP_NAME, f"Graph saved:\n{output}")

    def _metrics_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, result in self.state_data.results.items():
            row = {"model": name, "training_seconds": result.training_seconds}
            row.update({f"validation_{key}": value for key, value in result.validation_metrics.items()})
            row.update({f"test_{key}": value for key, value in result.test_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def _export_metrics(self) -> None:
        if not self.state_data.results:
            messagebox.showwarning(APP_NAME, "Train models first.")
            return
        output = filedialog.asksaveasfilename(
            title="Export metrics",
            defaultextension=".csv",
            initialfile="model_metrics.csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")],
        )
        if not output:
            return
        metrics = self._metrics_dataframe()
        if Path(output).suffix.lower() == ".xlsx":
            metrics.to_excel(output, index=False)
        else:
            metrics.to_csv(output, index=False)
        messagebox.showinfo(APP_NAME, f"Metrics exported:\n{output}")

    # ---------------------------- Save and predict ----------------------------

    def _active_package(self) -> dict[str, Any]:
        if self.state_data.loaded_model_package is not None:
            return self.state_data.loaded_model_package
        if self.state_data.best_model_name is None:
            raise RuntimeError("Train models or load a saved model first.")
        result = self.state_data.results[self.state_data.best_model_name]
        return {
            "application": APP_NAME,
            "version": APP_VERSION,
            "task": self.state_data.task,
            "target_column": self.state_data.target_column,
            "pipeline": result.pipeline,
            "model_name": result.name,
            "input_feature_columns": self.state_data.input_feature_columns,
            "datetime_columns": self.state_data.datetime_columns,
            "prepared_feature_columns": self.state_data.prepared_feature_columns,
            "excluded_columns": self.state_data.excluded_columns,
            "label_encoder": self.state_data.label_encoder,
            "classes": (
                list(map(str, self.state_data.label_encoder.classes_))
                if self.state_data.label_encoder is not None
                else None
            ),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _save_best_model(self) -> None:
        try:
            package = self._active_package()
            output = filedialog.asksaveasfilename(
                title="Save complete ML model",
                defaultextension=".joblib",
                initialfile=f"{safe_output_name(package['target_column'])}_{safe_output_name(package['task'])}_model.joblib",
                filetypes=[("Joblib model", "*.joblib")],
            )
            if not output:
                return
            joblib.dump(package, output)
            messagebox.showinfo(APP_NAME, f"Complete model pipeline saved:\n{output}")
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _load_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load saved All in One AI data model",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            package = joblib.load(selected)
            required = {
                "task",
                "target_column",
                "pipeline",
                "model_name",
                "input_feature_columns",
                "datetime_columns",
            }
            missing = required.difference(package.keys())
            if missing:
                raise ValueError(f"Invalid model package; missing: {sorted(missing)}")
            self.state_data.loaded_model_package = package
            class_text = ""
            if package["task"] == "Classification" and package.get("classes"):
                class_text = " | Classes: " + ", ".join(package["classes"])
            self.active_model_label.configure(
                text=(
                    f"{package['model_name']} | {package['task']} | Target: {package['target_column']}{class_text}"
                )
            )
            messagebox.showinfo(APP_NAME, "Saved data model loaded successfully.")
        except Exception as error:
            messagebox.showerror(APP_NAME, f"Model loading failed:\n{error}")

    def _browse_batch_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a new CSV or Excel file",
            filetypes=[
                ("Supported data", "*.csv *.tsv *.txt *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.batch_file_entry.delete(0, "end")
            self.batch_file_entry.insert(0, selected)

    def _start_batch_prediction(self) -> None:
        try:
            package = self._active_package()
            path = Path(self.batch_file_entry.get().strip())
            if not path.exists():
                raise FileNotFoundError("Select a valid new CSV or Excel file.")

            output = filedialog.asksaveasfilename(
                title="Save batch predictions",
                defaultextension=".csv",
                initialfile=f"predictions_{path.stem}.csv",
                filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")],
            )
            if not output:
                return

            threading.Thread(
                target=self._thread_wrapper,
                args=(lambda: self._batch_prediction_worker(package, path, Path(output)),),
                daemon=True,
            ).start()
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _read_batch_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            encoding = detect_encoding(path)
            delimiter = detect_delimiter(path, encoding)
            return pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False, on_bad_lines="skip")
        sheets = excel_sheet_names(path)
        sheet = sheets[0]
        if len(sheets) > 1:
            self.messages.put(("predict_log", f"Multiple Excel sheets found; using first sheet: {sheet}"))
        return pd.read_excel(path, sheet_name=sheet)

    def _batch_prediction_worker(
        self,
        package: dict[str, Any],
        path: Path,
        output: Path,
    ) -> None:
        self._set_status(0.1, "Loading new data for prediction...")
        dataframe = self._read_batch_file(path)
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        required_columns = list(package["input_feature_columns"])
        missing = [column for column in required_columns if column not in dataframe.columns]
        if missing:
            raise RuntimeError(
                "The new file is missing required feature columns:\n" + "\n".join(missing)
            )
        X_original = dataframe[required_columns].copy()
        X = add_datetime_features(X_original, list(package.get("datetime_columns", [])))
        expected_prepared = list(package.get("prepared_feature_columns", X.columns))
        for column in expected_prepared:
            if column not in X.columns:
                X[column] = np.nan
        X = X[expected_prepared]

        predictions = package["pipeline"].predict(X)
        output_data = dataframe.copy()

        if package["task"] == "Classification":
            encoder = package.get("label_encoder")
            if encoder is not None:
                labels = encoder.inverse_transform(np.asarray(predictions, dtype=int))
            else:
                labels = predictions
            output_data[f"predicted_{package['target_column']}"] = labels
            probabilities = probability_scores(
                package["pipeline"],
                X,
                len(package.get("classes") or np.unique(predictions)),
            )
            output_data["prediction_confidence"] = probabilities.max(axis=1)
            classes = package.get("classes") or [str(index) for index in range(probabilities.shape[1])]
            for index, class_name in enumerate(classes):
                output_data[f"probability_{safe_output_name(str(class_name))}"] = probabilities[:, index]
        else:
            output_data[f"predicted_{package['target_column']}"] = predictions

        if output.suffix.lower() == ".xlsx":
            output_data.to_excel(output, index=False)
        else:
            output_data.to_csv(output, index=False)
        self.messages.put(("batch_complete", (len(output_data), str(output))))
        self._set_status(1.0, "Batch prediction completed")

    def _export_experiment_report(self) -> None:
        if not self.state_data.results:
            messagebox.showwarning(APP_NAME, "Train models first.")
            return
        output = filedialog.asksaveasfilename(
            title="Export experiment report",
            defaultextension=".xlsx",
            initialfile="all_in_one_ai_experiment_report.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if not output:
            return
        profile_rows = []
        for column in self.state_data.dataframe.columns:
            series = self.state_data.dataframe[column]
            profile_rows.append(
                {
                    "column": str(column),
                    "detected_type": self.state_data.semantic_types.get(str(column), ""),
                    "dtype": str(series.dtype),
                    "missing": int(series.isna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                }
            )
        configuration = pd.DataFrame(
            [
                {"parameter": "file", "value": str(self.state_data.file_path)},
                {"parameter": "sheet", "value": self.state_data.sheet_name},
                {"parameter": "rows_retained", "value": len(self.state_data.dataframe)},
                {"parameter": "rows_scanned", "value": self.state_data.total_rows_seen},
                {"parameter": "task", "value": self.state_data.task},
                {"parameter": "target", "value": self.state_data.target_column},
                {"parameter": "best_model", "value": self.state_data.best_model_name},
                {"parameter": "input_features", "value": ", ".join(self.state_data.input_feature_columns)},
                {"parameter": "excluded_columns", "value": ", ".join(self.state_data.excluded_columns)},
            ]
        )
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            configuration.to_excel(writer, sheet_name="Experiment", index=False)
            pd.DataFrame(profile_rows).to_excel(writer, sheet_name="Column Profile", index=False)
            self._metrics_dataframe().to_excel(writer, sheet_name="Model Metrics", index=False)
            self.state_data.dataframe.head(1000).to_excel(writer, sheet_name="Data Preview", index=False)
        messagebox.showinfo(APP_NAME, f"Experiment report exported:\n{output}")

    # ----------------------------- Message loop -----------------------------

    def _thread_wrapper(self, target: Callable[[], None]) -> None:
        try:
            target()
        except (ValueError, RuntimeError, FileNotFoundError, MemoryError) as error:
            self.messages.put(("error", str(error)))
        except Exception as error:
            self.messages.put(("error", f"{error}\n\n{traceback.format_exc()}"))

    def _set_status(self, progress: float, text: str) -> None:
        self.messages.put(("status", (progress, text)))

    def _process_messages(self) -> None:
        try:
            while True:
                message_type, payload = self.messages.get_nowait()
                if message_type == "status":
                    progress, text = payload
                    self.progress_bar.set(max(0.0, min(1.0, float(progress))))
                    self.status_label.configure(text=str(text))
                elif message_type == "load_scan":
                    self.status_label.configure(text=f"Scanning CSV rows: {int(payload):,}")
                elif message_type == "data_loaded":
                    self._display_loaded_data()
                    self.tabs.set("2. Configure Experiment")
                elif message_type == "training_log":
                    self.training_log.insert("end", str(payload) + "\n")
                    self.training_log.see("end")
                elif message_type == "training_complete":
                    model_names = payload["models"]
                    best_name = payload["best"]
                    self.result_model_menu.configure(values=model_names)
                    self.result_model_var.set(best_name)
                    options = self._graph_options()
                    self.graph_type_menu.configure(values=options)
                    self.graph_type_var.set("Model Comparison")
                    self.best_model_label.configure(text=f"Best validation model: {best_name}")
                    self.active_model_label.configure(
                        text=(
                            f"Newly trained: {best_name} | {self.state_data.task} | Target: {self.state_data.target_column}"
                        )
                    )
                    self._refresh_results_table()
                    self.tabs.set("4. Results & Graphs")
                    messagebox.showinfo(
                        APP_NAME,
                        f"Training, validation and testing completed.\n\nBest validation model: {best_name}",
                    )
                elif message_type == "predict_log":
                    self.predict_log.insert("end", str(payload) + "\n")
                    self.predict_log.see("end")
                elif message_type == "batch_complete":
                    rows, output = payload
                    self.batch_status_label.configure(
                        text=f"Batch prediction completed: {rows:,} rows → {output}"
                    )
                    self.predict_log.insert("end", f"\nPredicted {rows:,} rows.\nSaved: {output}\n")
                    self.predict_log.see("end")
                    messagebox.showinfo(APP_NAME, f"Predictions exported:\n{output}")
                elif message_type == "error":
                    self.progress_bar.set(0)
                    self.status_label.configure(text="Operation failed")
                    messagebox.showerror(APP_NAME, str(payload))
        except queue.Empty:
            pass
        self.after(100, self._process_messages)

    def _close(self) -> None:
        self._clear_figure()
        self.destroy()


if __name__ == "__main__":
    DataTrainerGUI().mainloop()
