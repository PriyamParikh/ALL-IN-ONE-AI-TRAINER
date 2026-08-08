from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Optional

import customtkinter as ctk
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import dct
from scipy.io import wavfile
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

APP_NAME = "Personalized Audio Classification GUI"
APP_VERSION = "1.0"
PROJECT_FILE = "audio_project.json"
SAMPLES_FILE = "audio_features.csv"
MODEL_NAMES = [
    "Random Forest", "Decision Tree", "SVM", "KNN",
    "Logistic Regression", "Neural Network",
]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_") or "Sound"


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def _frame_audio(signal: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    count = 1 + int(np.ceil((signal.size - frame_length) / hop))
    required = (count - 1) * hop + frame_length
    if signal.size < required:
        signal = np.pad(signal, (0, required - signal.size))
    indices = np.arange(frame_length)[None, :] + hop * np.arange(count)[:, None]
    return signal[indices]


def _mel_filterbank(sample_rate: int, n_fft: int, count: int = 26) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(20.0), _hz_to_mel(sample_rate / 2), count + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((count, n_fft // 2 + 1), dtype=np.float32)
    for index in range(1, count + 1):
        left, centre, right = map(int, bins[index - 1:index + 2])
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        centre = min(centre, n_fft // 2)
        right = min(right, n_fft // 2)
        for frequency_bin in range(left, centre):
            bank[index - 1, frequency_bin] = (frequency_bin - left) / max(centre - left, 1)
        for frequency_bin in range(centre, right):
            bank[index - 1, frequency_bin] = (right - frequency_bin) / max(right - centre, 1)
    return bank


def feature_names() -> list[str]:
    names: list[str] = []
    for index in range(13):
        names += [
            f"mfcc_{index + 1}_mean", f"mfcc_{index + 1}_std",
            f"mfcc_{index + 1}_min", f"mfcc_{index + 1}_max",
        ]
    for index in range(13):
        names += [f"delta_mfcc_{index + 1}_mean", f"delta_mfcc_{index + 1}_std"]
    names += [
        "rms_mean", "rms_std", "rms_max", "zcr_mean", "zcr_std",
        "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
        "rolloff_mean", "rolloff_std", "flatness_mean", "flatness_std",
        "flux_mean", "flux_std", "dominant_frequency_mean",
        "dominant_frequency_std", "duration_seconds",
    ]
    return names


FEATURE_NAMES = feature_names()


def extract_audio_features(
    audio: np.ndarray,
    sample_rate: int,
    minimum_rms: float = 0.002,
) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("The recorded audio is empty.")
    signal = np.nan_to_num(signal)
    signal -= np.mean(signal)
    rms_global = float(np.sqrt(np.mean(signal * signal) + 1e-12))
    if rms_global < minimum_rms:
        raise ValueError("Audio is too quiet. Produce the sound closer to the microphone.")

    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - 0.97 * signal[:-1]
    frame_length = max(64, int(round(0.025 * sample_rate)))
    hop = max(32, int(round(0.010 * sample_rate)))
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2

    frames = _frame_audio(emphasized, frame_length, hop)
    windowed = frames * np.hamming(frame_length).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1)).astype(np.float32)
    power = magnitude * magnitude / n_fft
    mel_energy = power @ _mel_filterbank(sample_rate, n_fft).T
    mfcc = dct(np.log(np.maximum(mel_energy, 1e-12)), type=2, axis=1, norm="ortho")[:, :13]
    delta = np.gradient(mfcc, axis=0)

    values: list[float] = []
    for coefficient in mfcc.T:
        values += [float(np.mean(coefficient)), float(np.std(coefficient)),
                   float(np.min(coefficient)), float(np.max(coefficient))]
    for coefficient in delta.T:
        values += [float(np.mean(coefficient)), float(np.std(coefficient))]

    raw_frames = _frame_audio(signal, frame_length, hop)
    rms = np.sqrt(np.mean(raw_frames * raw_frames, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(raw_frames), axis=1)), axis=1)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate).astype(np.float32)
    magnitude_sum = np.sum(magnitude, axis=1) + 1e-12
    centroid = (magnitude @ frequencies) / magnitude_sum
    bandwidth = np.sqrt(
        np.sum(magnitude * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1)
        / magnitude_sum
    )
    cumulative = np.cumsum(power, axis=1)
    rolloff = np.zeros(power.shape[0], dtype=np.float32)
    for frame_index in range(power.shape[0]):
        bin_index = int(np.searchsorted(cumulative[frame_index], 0.85 * cumulative[frame_index, -1]))
        rolloff[frame_index] = frequencies[min(bin_index, frequencies.size - 1)]
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=1)) / (np.mean(power, axis=1) + 1e-12)
    normalized_magnitude = magnitude / magnitude_sum[:, None]
    flux = (
        np.sqrt(np.sum(np.diff(normalized_magnitude, axis=0) ** 2, axis=1))
        if normalized_magnitude.shape[0] > 1 else np.zeros(1, dtype=np.float32)
    )
    dominant = frequencies[np.argmax(magnitude, axis=1)]

    values += [
        float(np.mean(rms)), float(np.std(rms)), float(np.max(rms)),
        float(np.mean(zcr)), float(np.std(zcr)),
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(bandwidth)), float(np.std(bandwidth)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(flatness)), float(np.std(flatness)),
        float(np.mean(flux)), float(np.std(flux)),
        float(np.mean(dominant)), float(np.std(dominant)),
        float(signal.size / sample_rate),
    ]
    vector = np.asarray(values, dtype=np.float32)
    if vector.size != len(FEATURE_NAMES):
        raise ValueError(f"Unexpected feature length: {vector.size}; expected {len(FEATURE_NAMES)}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Feature vector contains NaN or infinity.")
    return vector


@dataclass
class Evaluation:
    name: str
    estimator: Any
    y_train: np.ndarray
    y_test: np.ndarray
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    train_probabilities: np.ndarray
    test_probabilities: np.ndarray
    metrics: dict[str, float]


@dataclass
class State:
    project_dir: Optional[Path] = None
    class_names: list[str] = field(default_factory=list)
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_encoder: Optional[LabelEncoder] = None
    evaluations: dict[str, Evaluation] = field(default_factory=dict)
    best_model_name: Optional[str] = None
    loaded_model: Optional[dict[str, Any]] = None
    current_figure: Optional[Figure] = None
    current_canvas: Optional[FigureCanvasTkAgg] = None
    last_audio: Optional[np.ndarray] = None
    last_sample_rate: int = 16000


class AudioClassifierGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.title(f"{APP_NAME} — Version {APP_VERSION}")
        self.geometry("1500x900")
        self.minsize(1180, 740)

        self.state_data = State()
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.capture_stop = threading.Event()
        self.realtime_stop = threading.Event()
        self.device_map: dict[str, Optional[int]] = {"System default input": None}

        self._build_header()
        self._build_tabs()
        self._build_footer()
        self._refresh_devices()
        self.after(100, self._process_messages)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(
            header,
            text="PERSONALIZED AUDIO CLASSIFICATION",
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(side="left", padx=18, pady=12)
        ctk.CTkLabel(
            header,
            text="Built-in Microphone • MFCC Features • Model Comparison • Live Test",
            font=ctk.CTkFont(size=14),
        ).pack(side="left", padx=12)
        self.project_label = ctk.CTkLabel(
            header,
            text="Project: Not created",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.project_label.pack(side="right", padx=18)

    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=6)
        for tab_name in (
            "1. Setup & Classes", "2. Record Samples", "3. Train Models",
            "4. Results & Graphs", "5. Real-Time Test",
        ):
            self.tabs.add(tab_name)
        self._build_setup_tab()
        self._build_capture_tab()
        self._build_training_tab()
        self._build_results_tab()
        self._build_realtime_tab()

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self)
        footer.pack(fill="x", padx=12, pady=(6, 12))
        self.status_label = ctk.CTkLabel(footer, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=12, pady=8, fill="x", expand=True)
        self.progress = ctk.CTkProgressBar(footer, width=330)
        self.progress.set(0)
        self.progress.pack(side="right", padx=12, pady=8)

    def _build_setup_tab(self) -> None:
        tab = self.tabs.tab("1. Setup & Classes")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(7, weight=1)
        ctk.CTkLabel(
            tab,
            text="Step 1 — Create a Project and Name the Sounds",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkLabel(tab, text="Project folder").grid(row=1, column=0, padx=20, pady=9, sticky="w")
        self.project_entry = ctk.CTkEntry(tab)
        self.project_entry.grid(row=1, column=1, padx=20, pady=9, sticky="ew")
        ctk.CTkButton(tab, text="Browse", width=120, command=self._browse_project).grid(
            row=1, column=2, padx=20, pady=9
        )

        ctk.CTkLabel(tab, text="Sound class names").grid(row=2, column=0, padx=20, pady=9, sticky="w")
        self.classes_entry = ctk.CTkEntry(tab, placeholder_text="Clap, Whistle, Background")
        self.classes_entry.grid(row=2, column=1, padx=20, pady=9, sticky="ew")
        ctk.CTkLabel(tab, text="Enter 2–5 class names separated by commas.").grid(
            row=3, column=1, padx=20, pady=(0, 8), sticky="w"
        )

        ctk.CTkLabel(tab, text="Input microphone").grid(row=4, column=0, padx=20, pady=9, sticky="w")
        self.setup_device_var = ctk.StringVar(value="System default input")
        self.setup_device_menu = ctk.CTkOptionMenu(
            tab, values=["System default input"], variable=self.setup_device_var
        )
        self.setup_device_menu.grid(row=4, column=1, padx=20, pady=9, sticky="ew")
        device_buttons = ctk.CTkFrame(tab)
        device_buttons.grid(row=4, column=2, padx=20, pady=9)
        ctk.CTkButton(device_buttons, text="Refresh", width=86, command=self._refresh_devices).pack(
            side="left", padx=4, pady=4
        )
        ctk.CTkButton(device_buttons, text="Test Mic", width=86, command=self._start_mic_test).pack(
            side="left", padx=4, pady=4
        )

        controls = ctk.CTkFrame(tab)
        controls.grid(row=5, column=0, columnspan=3, padx=20, pady=12, sticky="ew")
        ctk.CTkButton(controls, text="CREATE NEW PROJECT", height=42, command=self._create_project).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="LOAD EXISTING PROJECT", height=42, command=self._load_project).pack(
            side="left", padx=10, pady=10
        )

        self.mic_test_label = ctk.CTkLabel(
            tab, text="Microphone test: Not performed", anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.mic_test_label.grid(row=6, column=0, columnspan=3, padx=20, pady=8, sticky="ew")
        self.setup_text = ctk.CTkTextbox(tab)
        self.setup_text.grid(row=7, column=0, columnspan=3, padx=20, pady=(8, 20), sticky="nsew")
        self.setup_text.insert(
            "1.0",
            "Suggested first classes: Clap, Whistle, Background.\n\n"
            "A Background class is recommended because it helps the real-time model "
            "avoid calling every quiet clip a clap or whistle.\n\n"
            "Select the built-in microphone, test it, then create the project.",
        )

    def _build_capture_tab(self) -> None:
        tab = self.tabs.tab("2. Record Samples")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(8, weight=1)
        ctk.CTkLabel(
            tab,
            text="Step 2 — Select a Class and Record Audio Clips",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkLabel(tab, text="Sound class").grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.capture_class_var = ctk.StringVar(value="Create a project first")
        self.capture_class_menu = ctk.CTkOptionMenu(
            tab, values=["Create a project first"], variable=self.capture_class_var
        )
        self.capture_class_menu.grid(row=1, column=1, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(tab, text="Target accepted clips").grid(row=2, column=0, padx=20, pady=8, sticky="w")
        self.target_entry = ctk.CTkEntry(tab)
        self.target_entry.insert(0, "50")
        self.target_entry.grid(row=2, column=1, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(tab, text="Clip duration (seconds)").grid(row=3, column=0, padx=20, pady=8, sticky="w")
        self.duration_entry = ctk.CTkEntry(tab)
        self.duration_entry.insert(0, "1.0")
        self.duration_entry.grid(row=3, column=1, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(tab, text="Pause between clips (seconds)").grid(row=4, column=0, padx=20, pady=8, sticky="w")
        self.pause_entry = ctk.CTkEntry(tab)
        self.pause_entry.insert(0, "0.35")
        self.pause_entry.grid(row=4, column=1, padx=20, pady=8, sticky="ew")

        settings = ctk.CTkFrame(tab)
        settings.grid(row=5, column=0, columnspan=4, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(settings, text="Microphone").pack(side="left", padx=(12, 4))
        self.capture_device_var = ctk.StringVar(value="System default input")
        self.capture_device_menu = ctk.CTkOptionMenu(
            settings, values=["System default input"], variable=self.capture_device_var, width=310
        )
        self.capture_device_menu.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(settings, text="Sample rate").pack(side="left", padx=(16, 4))
        self.sample_rate_entry = ctk.CTkEntry(settings, width=80)
        self.sample_rate_entry.insert(0, "16000")
        self.sample_rate_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(settings, text="Minimum RMS").pack(side="left", padx=(16, 4))
        self.min_rms_entry = ctk.CTkEntry(settings, width=80)
        self.min_rms_entry.insert(0, "0.002")
        self.min_rms_entry.pack(side="left", padx=6, pady=12)
        self.save_wav_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(settings, text="Save WAV files", variable=self.save_wav_var).pack(
            side="left", padx=16, pady=12
        )

        controls = ctk.CTkFrame(tab)
        controls.grid(row=6, column=0, columnspan=4, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="START RECORDING", height=44, command=self._start_capture).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="STOP", height=44, command=self._stop_capture).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="CLEAR SELECTED CLASS", command=self._clear_class).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="EXPORT FEATURE CSV", command=self._export_features).pack(
            side="left", padx=10, pady=10
        )
        self.capture_count_label = ctk.CTkLabel(
            controls, text="Accepted this session: 0/0", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.capture_count_label.pack(side="right", padx=15)

        self.dataset_label = ctk.CTkLabel(
            tab, text="Current dataset: No project", anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.dataset_label.grid(row=7, column=0, columnspan=4, padx=20, pady=8, sticky="ew")
        self.capture_log = ctk.CTkTextbox(tab)
        self.capture_log.grid(row=8, column=0, columnspan=4, padx=20, pady=(8, 20), sticky="nsew")
        self.capture_log.insert(
            "1.0",
            "During RECORDING NOW, produce the selected sound once.\n"
            "During PAUSE, prepare for the next clip.\n\n"
            "For Background, remain quiet and allow normal room noise.\n"
            "Record each class in two or three separate sessions for stronger testing.",
        )

    def _build_training_tab(self) -> None:
        tab = self.tabs.tab("3. Train Models")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(
            tab,
            text="Step 3 — Train and Compare Audio Classifiers",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")

        models_frame = ctk.CTkFrame(tab)
        models_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.model_vars: dict[str, ctk.BooleanVar] = {}
        for index, name in enumerate(MODEL_NAMES):
            variable = ctk.BooleanVar(value=index < 5)
            self.model_vars[name] = variable
            ctk.CTkCheckBox(models_frame, text=name, variable=variable).grid(
                row=index // 3, column=index % 3, padx=18, pady=12, sticky="w"
            )

        settings = ctk.CTkFrame(tab)
        settings.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(settings, text="Test fraction").pack(side="left", padx=(12, 4))
        self.test_fraction_entry = ctk.CTkEntry(settings, width=80)
        self.test_fraction_entry.insert(0, "0.20")
        self.test_fraction_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(settings, text="Random state").pack(side="left", padx=(16, 4))
        self.random_state_entry = ctk.CTkEntry(settings, width=80)
        self.random_state_entry.insert(0, "42")
        self.random_state_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(settings, text="Real-time threshold").pack(side="left", padx=(16, 4))
        self.saved_threshold_entry = ctk.CTkEntry(settings, width=80)
        self.saved_threshold_entry.insert(0, "0.65")
        self.saved_threshold_entry.pack(side="left", padx=6, pady=12)

        ctk.CTkButton(
            tab,
            text="TRAIN AND EVALUATE SELECTED MODELS",
            height=48,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_training,
        ).grid(row=3, column=0, padx=20, pady=15, sticky="ew")

        self.training_dataset_label = ctk.CTkLabel(
            tab, text="Dataset: No project", anchor="w", font=ctk.CTkFont(size=15, weight="bold")
        )
        self.training_dataset_label.grid(row=4, column=0, padx=20, pady=8, sticky="ew")
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
            controls, values=["No trained models"], variable=self.result_model_var, width=190
        )
        self.result_model_menu.pack(side="left", padx=8, pady=10)
        ctk.CTkLabel(controls, text="Graph").pack(side="left", padx=(16, 4))
        self.graph_var = ctk.StringVar(value="Testing Confusion Matrix")
        self.graph_menu = ctk.CTkOptionMenu(
            controls,
            values=[
                "Training Confusion Matrix", "Testing Confusion Matrix",
                "Testing ROC Curve", "Testing Precision-Recall Curve",
                "PCA Audio Scatter", "Model Comparison",
                "Feature Importance", "Learning Curve",
            ],
            variable=self.graph_var,
            width=235,
        )
        self.graph_menu.pack(side="left", padx=8, pady=10)
        ctk.CTkButton(controls, text="Generate Graph", command=self._generate_graph).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Save Graph", command=self._save_graph).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Save Best Model", command=self._save_best_model).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Load Model", command=self._load_model).pack(
            side="left", padx=10, pady=10
        )

        table_frame = ctk.CTkFrame(tab)
        table_frame.grid(row=1, column=0, padx=(20, 10), pady=(0, 20), sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
        columns = ("Model", "Accuracy", "Precision", "Recall", "F1", "Balanced", "ROC-AUC", "PR-AUC")
        self.result_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        for column in columns:
            self.result_tree.heading(column, text=column)
            self.result_tree.column(column, width=105 if column != "Model" else 150, anchor="center")
        self.result_tree.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.best_model_label = ctk.CTkLabel(
            table_frame, text="Best model: —", anchor="w", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.best_model_label.grid(row=1, column=0, padx=10, pady=8, sticky="ew")

        self.graph_frame = ctk.CTkFrame(tab)
        self.graph_frame.grid(row=1, column=1, padx=(10, 20), pady=(0, 20), sticky="nsew")
        self.graph_frame.grid_columnconfigure(0, weight=1)
        self.graph_frame.grid_rowconfigure(0, weight=1)

    def _build_realtime_tab(self) -> None:
        tab = self.tabs.tab("5. Real-Time Test")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(
            tab,
            text="Step 5 — Classify Live Microphone Audio",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 12), sticky="w")

        ctk.CTkLabel(tab, text="Active classifier").grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.active_model_label = ctk.CTkLabel(
            tab, text="Train a model in Step 3 or load a .joblib model.", anchor="w"
        )
        self.active_model_label.grid(row=1, column=1, padx=20, pady=8, sticky="ew")
        ctk.CTkButton(tab, text="Load .joblib", width=120, command=self._load_model).grid(
            row=1, column=2, padx=20, pady=8
        )

        settings = ctk.CTkFrame(tab)
        settings.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(settings, text="Microphone").pack(side="left", padx=(12, 4))
        self.realtime_device_var = ctk.StringVar(value="System default input")
        self.realtime_device_menu = ctk.CTkOptionMenu(
            settings, values=["System default input"], variable=self.realtime_device_var, width=320
        )
        self.realtime_device_menu.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(settings, text="Confidence threshold").pack(side="left", padx=(16, 4))
        self.realtime_threshold_entry = ctk.CTkEntry(settings, width=80)
        self.realtime_threshold_entry.insert(0, "0.65")
        self.realtime_threshold_entry.pack(side="left", padx=6, pady=12)
        ctk.CTkLabel(settings, text="Delay between predictions").pack(side="left", padx=(16, 4))
        self.realtime_delay_entry = ctk.CTkEntry(settings, width=80)
        self.realtime_delay_entry.insert(0, "0.10")
        self.realtime_delay_entry.pack(side="left", padx=6, pady=12)

        controls = ctk.CTkFrame(tab)
        controls.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(
            controls, text="START REAL-TIME AUDIO CLASSIFICATION", height=44, command=self._start_realtime
        ).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="STOP", height=44, command=self._stop_realtime).pack(
            side="left", padx=10, pady=10
        )
        ctk.CTkButton(controls, text="Save Last WAV", command=self._save_last_wav).pack(
            side="left", padx=10, pady=10
        )
        self.realtime_result_label = ctk.CTkLabel(
            controls, text="Prediction: —", font=ctk.CTkFont(size=18, weight="bold")
        )
        self.realtime_result_label.pack(side="right", padx=15)

        self.level_label = ctk.CTkLabel(
            tab, text="Microphone RMS level: 0.00000", anchor="w", font=ctk.CTkFont(size=15, weight="bold")
        )
        self.level_label.grid(row=4, column=0, columnspan=3, padx=20, pady=8, sticky="ew")
        self.realtime_log = ctk.CTkTextbox(tab)
        self.realtime_log.grid(row=5, column=0, columnspan=3, padx=20, pady=(8, 20), sticky="nsew")

    # ------------------------------------------------------------------
    # Devices and recording
    # ------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        try:
            labels = ["System default input"]
            mapping: dict[str, Optional[int]] = {"System default input": None}
            for index, device in enumerate(sd.query_devices()):
                if int(device["max_input_channels"]) > 0:
                    label = f"{index}: {device['name']} ({device['max_input_channels']} input ch)"
                    labels.append(label)
                    mapping[label] = index
            self.device_map = mapping
            for menu_name, variable_name in (
                ("setup_device_menu", "setup_device_var"),
                ("capture_device_menu", "capture_device_var"),
                ("realtime_device_menu", "realtime_device_var"),
            ):
                menu = getattr(self, menu_name, None)
                variable = getattr(self, variable_name, None)
                if menu is not None and variable is not None:
                    menu.configure(values=labels)
                    if variable.get() not in labels:
                        variable.set(labels[0])
        except Exception as error:
            messagebox.showerror(APP_NAME, f"Microphone enumeration failed:\n{error}")

    def _device_index(self, label: str) -> Optional[int]:
        return self.device_map.get(label)

    @staticmethod
    def _record_audio(duration: float, sample_rate: int, device: Optional[int]) -> np.ndarray:
        frame_count = int(round(duration * sample_rate))
        recording = sd.rec(
            frame_count,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            blocking=True,
        )
        return np.asarray(recording, dtype=np.float32).reshape(-1)

    def _start_mic_test(self) -> None:
        label = self.setup_device_var.get()
        self._run_thread(lambda: self._mic_test_worker(label, self._device_index(label)))

    def _mic_test_worker(self, label: str, device: Optional[int]) -> None:
        self._status(0.1, "Recording a two-second microphone test...")
        audio = self._record_audio(2.0, 16000, device)
        rms_value = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        peak_value = float(np.max(np.abs(audio)))
        self.state_data.last_audio = audio
        self.state_data.last_sample_rate = 16000
        self.messages.put(("mic_test", (label, rms_value, peak_value)))
        self._status(1.0, "Microphone test completed")

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def _browse_project(self) -> None:
        selected = filedialog.askdirectory(title="Select audio project folder")
        if selected:
            self.project_entry.delete(0, "end")
            self.project_entry.insert(0, selected)

    def _parse_classes(self) -> list[str]:
        values = [value.strip() for value in self.classes_entry.get().split(",") if value.strip()]
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        if not 2 <= len(unique) <= 5:
            raise ValueError("Enter between two and five unique sound class names.")
        return unique

    @staticmethod
    def _empty_samples() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["session_id", "class_name", "sample_id", "wav_path"] + FEATURE_NAMES
        )

    def _create_project(self) -> None:
        try:
            folder_text = self.project_entry.get().strip()
            if not folder_text:
                raise ValueError("Select a project folder.")
            class_names = self._parse_classes()
            project_dir = Path(folder_text)
            project_dir.mkdir(parents=True, exist_ok=True)
            audio_root = project_dir / "recorded_audio"
            audio_root.mkdir(parents=True, exist_ok=True)
            for class_name in class_names:
                (audio_root / safe_name(class_name)).mkdir(parents=True, exist_ok=True)
            config = {
                "application": APP_NAME,
                "version": APP_VERSION,
                "class_names": class_names,
                "input_device_label": self.setup_device_var.get(),
                "feature_names": FEATURE_NAMES,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (project_dir / PROJECT_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
            samples = self._empty_samples()
            samples.to_csv(project_dir / SAMPLES_FILE, index=False)
            self.state_data = State(project_dir=project_dir, class_names=class_names, samples=samples)
            selected_device = self.setup_device_var.get()
            if selected_device in self.device_map:
                self.capture_device_var.set(selected_device)
                self.realtime_device_var.set(selected_device)
            self._apply_project()
            messagebox.showinfo(APP_NAME, f"Audio project created:\n{project_dir}")
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _load_project(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select audio_project.json",
            filetypes=[("Audio project", PROJECT_FILE), ("JSON file", "*.json")],
        )
        if not selected:
            return
        try:
            config_path = Path(selected)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            class_names = list(config["class_names"])
            if not 2 <= len(class_names) <= 5:
                raise ValueError("The project has an invalid class list.")
            project_dir = config_path.parent
            samples_path = project_dir / SAMPLES_FILE
            samples = pd.read_csv(samples_path) if samples_path.exists() else self._empty_samples()
            self.state_data = State(project_dir=project_dir, class_names=class_names, samples=samples)
            self.project_entry.delete(0, "end")
            self.project_entry.insert(0, str(project_dir))
            self.classes_entry.delete(0, "end")
            self.classes_entry.insert(0, ", ".join(class_names))
            saved_device = str(config.get("input_device_label", "System default input"))
            if saved_device in self.device_map:
                self.setup_device_var.set(saved_device)
                self.capture_device_var.set(saved_device)
                self.realtime_device_var.set(saved_device)
            self._apply_project()
        except Exception as error:
            messagebox.showerror(APP_NAME, f"Project loading failed:\n{error}")

    def _apply_project(self) -> None:
        if self.state_data.project_dir is None:
            return
        self.capture_class_menu.configure(values=self.state_data.class_names)
        self.capture_class_var.set(self.state_data.class_names[0])
        self.project_label.configure(text=f"Project: {self.state_data.project_dir.name}")
        self.setup_text.delete("1.0", "end")
        self.setup_text.insert(
            "1.0",
            f"Project: {self.state_data.project_dir}\n"
            f"Sound classes: {', '.join(self.state_data.class_names)}\n"
            f"Microphone: {self.setup_device_var.get()}\n"
            f"Features per clip: {len(FEATURE_NAMES)}\n\n"
            "Next: open Step 2 and record every class.",
        )
        self._update_counts()

    def _require_project(self) -> Path:
        if self.state_data.project_dir is None:
            raise RuntimeError("Create or load a project first.")
        return self.state_data.project_dir

    def _save_samples(self) -> None:
        self.state_data.samples.to_csv(self._require_project() / SAMPLES_FILE, index=False)

    def _update_counts(self) -> None:
        if not self.state_data.class_names:
            return
        parts: list[str] = []
        total = 0
        for class_name in self.state_data.class_names:
            count = (
                int((self.state_data.samples["class_name"] == class_name).sum())
                if not self.state_data.samples.empty else 0
            )
            parts.append(f"{class_name}: {count}")
            total += count
        text = f"{' | '.join(parts)} | Total: {total}"
        self.dataset_label.configure(text=f"Current dataset: {text}")
        self.training_dataset_label.configure(text=f"Dataset: {text}")

    # ------------------------------------------------------------------
    # Sample recording
    # ------------------------------------------------------------------

    def _start_capture(self) -> None:
        try:
            project_dir = self._require_project()
            class_name = self.capture_class_var.get()
            if class_name not in self.state_data.class_names:
                raise ValueError("Select a valid sound class.")
            settings = {
                "project_dir": project_dir,
                "class_name": class_name,
                "target": int(self.target_entry.get()),
                "duration": float(self.duration_entry.get()),
                "pause": float(self.pause_entry.get()),
                "sample_rate": int(self.sample_rate_entry.get()),
                "minimum_rms": float(self.min_rms_entry.get()),
                "save_wav": bool(self.save_wav_var.get()),
                "device_label": self.capture_device_var.get(),
                "device": self._device_index(self.capture_device_var.get()),
            }
            if settings["target"] < 5:
                raise ValueError("Record at least five clips per session.")
            if not 0.25 <= settings["duration"] <= 10.0:
                raise ValueError("Clip duration must be between 0.25 and 10 seconds.")
            if settings["pause"] < 0:
                raise ValueError("Pause cannot be negative.")
            if not 8000 <= settings["sample_rate"] <= 48000:
                raise ValueError("Sample rate must be between 8000 and 48000 Hz.")
            self.capture_stop.clear()
            self._run_thread(lambda: self._capture_worker(settings))
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _capture_worker(self, settings: dict[str, Any]) -> None:
        session_id = time.strftime("%Y%m%d_%H%M%S")
        output_folder = settings["project_dir"] / "recorded_audio" / safe_name(settings["class_name"])
        output_folder.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        accepted = 0
        rejected = 0
        self.messages.put((
            "capture_log",
            f"Recording class: {settings['class_name']}\n"
            f"Microphone: {settings['device_label']}\n"
            f"Duration: {settings['duration']} s | Target: {settings['target']}",
        ))
        while not self.capture_stop.is_set() and accepted < settings["target"]:
            self.messages.put(("capture_state", ("RECORDING NOW — produce the selected sound", accepted, settings["target"])))
            audio = self._record_audio(settings["duration"], settings["sample_rate"], settings["device"])
            self.state_data.last_audio = audio.copy()
            self.state_data.last_sample_rate = settings["sample_rate"]
            rms_value = float(np.sqrt(np.mean(audio * audio) + 1e-12))
            try:
                vector = extract_audio_features(audio, settings["sample_rate"], settings["minimum_rms"])
                sample_id = f"{session_id}_{accepted + 1:05d}"
                wav_path_text = ""
                if settings["save_wav"]:
                    wav_path = output_folder / f"{safe_name(settings['class_name'])}_{sample_id}.wav"
                    wavfile.write(
                        wav_path,
                        settings["sample_rate"],
                        (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16),
                    )
                    wav_path_text = str(wav_path)
                row: dict[str, Any] = {
                    "session_id": session_id,
                    "class_name": settings["class_name"],
                    "sample_id": sample_id,
                    "wav_path": wav_path_text,
                }
                row.update({name: float(value) for name, value in zip(FEATURE_NAMES, vector)})
                rows.append(row)
                accepted += 1
                self.messages.put(("capture_progress", (accepted, settings["target"], rejected, rms_value)))
            except ValueError as error:
                rejected += 1
                self.messages.put(("capture_log", f"Rejected: {error} | RMS={rms_value:.5f}"))

            if not self.capture_stop.is_set() and accepted < settings["target"] and settings["pause"] > 0:
                self.messages.put(("capture_state", ("PAUSE — prepare for the next clip", accepted, settings["target"])))
                time.sleep(settings["pause"])

        self.capture_stop.set()
        if rows:
            self.state_data.samples = pd.concat(
                [self.state_data.samples, pd.DataFrame(rows)], ignore_index=True
            )
            self._save_samples()
        self.messages.put(("capture_complete", (settings["class_name"], accepted, rejected)))

    def _stop_capture(self) -> None:
        self.capture_stop.set()
        try:
            sd.stop()
        except Exception:
            pass

    def _clear_class(self) -> None:
        try:
            self._require_project()
            class_name = self.capture_class_var.get()
            if class_name not in self.state_data.class_names:
                raise ValueError("Select a valid sound class.")
            if not messagebox.askyesno(APP_NAME, f"Delete all feature samples for:\n{class_name}?"):
                return
            self.state_data.samples = self.state_data.samples[
                self.state_data.samples["class_name"] != class_name
            ].reset_index(drop=True)
            self._save_samples()
            self._update_counts()
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _export_features(self) -> None:
        if self.state_data.samples.empty:
            messagebox.showwarning(APP_NAME, "No audio samples are available.")
            return
        output = filedialog.asksaveasfilename(
            title="Export audio features", defaultextension=".csv",
            initialfile="personalized_audio_features.csv", filetypes=[("CSV file", "*.csv")]
        )
        if output:
            self.state_data.samples.to_csv(output, index=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _selected_models(self) -> list[str]:
        selected = [name for name, variable in self.model_vars.items() if variable.get()]
        if len(selected) not in (4, 5):
            raise ValueError("Select exactly four or five models.")
        return selected

    @staticmethod
    def _make_model(name: str, random_state: int) -> Any:
        if name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )
        if name == "Decision Tree":
            return DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
        if name == "SVM":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf", probability=True,
                    class_weight="balanced", random_state=random_state,
                )),
            ])
        if name == "KNN":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ])
        if name == "Logistic Regression":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=random_state,
                )),
            ])
        if name == "Neural Network":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=1500,
                    early_stopping=True, random_state=random_state,
                )),
            ])
        raise ValueError(f"Unknown model: {name}")

    @staticmethod
    def _auc_scores(
        y_true: np.ndarray,
        probabilities: np.ndarray,
        class_count: int,
    ) -> tuple[float, float]:
        try:
            binary = label_binarize(y_true, classes=np.arange(class_count))
            if class_count == 2:
                binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
            return (
                float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")),
                float(average_precision_score(binary, probabilities, average="macro")),
            )
        except Exception:
            return float("nan"), float("nan")

    def _start_training(self) -> None:
        try:
            self._require_project()
            settings = {
                "models": self._selected_models(),
                "test_fraction": float(self.test_fraction_entry.get()),
                "random_state": int(self.random_state_entry.get()),
            }
            if not 0.10 <= settings["test_fraction"] <= 0.40:
                raise ValueError("Test fraction must be between 0.10 and 0.40.")
            self._run_thread(lambda: self._training_worker(settings))
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _training_worker(self, settings: dict[str, Any]) -> None:
        samples = self.state_data.samples.copy()
        if samples.empty:
            raise RuntimeError("No audio samples are available. Complete Step 2 first.")
        counts = samples["class_name"].value_counts()
        insufficient = [
            class_name for class_name in self.state_data.class_names
            if int(counts.get(class_name, 0)) < 10
        ]
        if insufficient:
            raise RuntimeError(
                "Record at least 10 accepted clips for every class.\n\n"
                f"Insufficient: {', '.join(insufficient)}"
            )

        X = samples[FEATURE_NAMES].to_numpy(dtype=np.float32)
        encoder = LabelEncoder()
        y = encoder.fit_transform(samples["class_name"].astype(str))
        session_counts = samples.groupby("class_name")["session_id"].nunique()
        use_group_split = (
            len(session_counts) == len(self.state_data.class_names)
            and int(session_counts.min()) >= 2
        )

        if use_group_split:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=settings["test_fraction"],
                random_state=settings["random_state"],
            )
            train_index, test_index = next(
                splitter.split(X, y, groups=samples["session_id"].astype(str))
            )
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            split_text = "Session-wise split"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=settings["test_fraction"],
                random_state=settings["random_state"], stratify=y,
            )
            split_text = "Stratified random split (record multiple sessions for stronger validation)"

        if len(np.unique(y_train)) != len(encoder.classes_) or len(np.unique(y_test)) != len(encoder.classes_):
            raise RuntimeError(
                "The train/test split does not contain every class. "
                "Record more independent sessions or reduce the test fraction."
            )

        self.messages.put((
            "training_log",
            f"Classes: {', '.join(encoder.classes_)}\n"
            f"Training clips: {len(X_train)}\nTesting clips: {len(X_test)}\n"
            f"Features per clip: {X.shape[1]}\nSplit: {split_text}\n",
        ))

        evaluations: dict[str, Evaluation] = {}
        for model_index, name in enumerate(settings["models"], start=1):
            self._status((model_index - 1) / len(settings["models"]), f"Training {name}...")
            self.messages.put(("training_log", f"Training {name}..."))
            estimator = self._make_model(name, settings["random_state"])
            estimator.fit(X_train, y_train)
            train_predictions = estimator.predict(X_train)
            test_predictions = estimator.predict(X_test)
            train_probabilities = estimator.predict_proba(X_train)
            test_probabilities = estimator.predict_proba(X_test)
            roc_auc, pr_auc = self._auc_scores(y_test, test_probabilities, len(encoder.classes_))
            metrics = {
                "accuracy": float(accuracy_score(y_test, test_predictions)),
                "precision": float(precision_score(
                    y_test, test_predictions, average="weighted", zero_division=0
                )),
                "recall": float(recall_score(
                    y_test, test_predictions, average="weighted", zero_division=0
                )),
                "f1": float(f1_score(
                    y_test, test_predictions, average="weighted", zero_division=0
                )),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, test_predictions)),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
            evaluations[name] = Evaluation(
                name=name, estimator=estimator, y_train=y_train, y_test=y_test,
                train_predictions=train_predictions, test_predictions=test_predictions,
                train_probabilities=train_probabilities, test_probabilities=test_probabilities,
                metrics=metrics,
            )
            self.messages.put((
                "training_log",
                f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}\n",
            ))

        best_name = max(
            evaluations,
            key=lambda item: (
                evaluations[item].metrics["f1"],
                evaluations[item].metrics["balanced_accuracy"],
            ),
        )
        self.state_data.label_encoder = encoder
        self.state_data.evaluations = evaluations
        self.state_data.best_model_name = best_name
        self.state_data.loaded_model = None
        self.messages.put(("training_complete", {"models": list(evaluations), "best": best_name}))
        self._status(1.0, f"Training completed — best model: {best_name}")

    # ------------------------------------------------------------------
    # Results, graphs and model files
    # ------------------------------------------------------------------

    def _refresh_result_table(self) -> None:
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for name, evaluation in self.state_data.evaluations.items():
            metrics = evaluation.metrics
            self.result_tree.insert(
                "", "end",
                values=(
                    name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['balanced_accuracy']:.4f}",
                    f"{metrics['roc_auc']:.4f}" if np.isfinite(metrics["roc_auc"]) else "N/A",
                    f"{metrics['pr_auc']:.4f}" if np.isfinite(metrics["pr_auc"]) else "N/A",
                ),
            )

    def _clear_figure(self) -> None:
        if self.state_data.current_canvas is not None:
            self.state_data.current_canvas.get_tk_widget().destroy()
            self.state_data.current_canvas = None
        if self.state_data.current_figure is not None:
            plt.close(self.state_data.current_figure)
            self.state_data.current_figure = None

    def _show_figure(self, figure: Figure) -> None:
        self._clear_figure()
        canvas = FigureCanvasTkAgg(figure, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.state_data.current_figure = figure
        self.state_data.current_canvas = canvas

    def _selected_evaluation(self) -> Evaluation:
        name = self.result_model_var.get()
        if name not in self.state_data.evaluations:
            raise RuntimeError("Select a trained model.")
        return self.state_data.evaluations[name]

    def _generate_graph(self) -> None:
        try:
            graph_type = self.graph_var.get()
            evaluation = self._selected_evaluation()
            encoder = self.state_data.label_encoder
            if encoder is None:
                raise RuntimeError("Class information is unavailable.")
            figure = Figure(figsize=(8.2, 6.2), dpi=100)
            axis = figure.add_subplot(111)

            if graph_type in ("Training Confusion Matrix", "Testing Confusion Matrix"):
                if graph_type.startswith("Training"):
                    y_true, predictions = evaluation.y_train, evaluation.train_predictions
                else:
                    y_true, predictions = evaluation.y_test, evaluation.test_predictions
                matrix = confusion_matrix(
                    y_true, predictions, labels=np.arange(len(encoder.classes_))
                )
                image = axis.imshow(matrix)
                figure.colorbar(image, ax=axis)
                axis.set_xticks(
                    np.arange(len(encoder.classes_)), labels=encoder.classes_, rotation=35, ha="right"
                )
                axis.set_yticks(np.arange(len(encoder.classes_)), labels=encoder.classes_)
                axis.set_xlabel("Predicted class")
                axis.set_ylabel("Actual class")
                axis.set_title(f"{graph_type} — {evaluation.name}")
                threshold = matrix.max() / 2 if matrix.size else 0
                for row in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        axis.text(
                            column, row, str(matrix[row, column]), ha="center", va="center",
                            color="white" if matrix[row, column] > threshold else "black",
                        )

            elif graph_type == "Testing ROC Curve":
                count = len(encoder.classes_)
                binary = label_binarize(evaluation.y_test, classes=np.arange(count))
                if count == 2:
                    binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
                for class_index, class_name in enumerate(encoder.classes_):
                    fpr, tpr, _ = roc_curve(
                        binary[:, class_index], evaluation.test_probabilities[:, class_index]
                    )
                    axis.plot(fpr, tpr, label=str(class_name))
                axis.plot([0, 1], [0, 1], linestyle="--")
                axis.set_xlabel("False Positive Rate")
                axis.set_ylabel("True Positive Rate")
                axis.set_title(f"Testing ROC Curve — {evaluation.name}")
                axis.legend()

            elif graph_type == "Testing Precision-Recall Curve":
                count = len(encoder.classes_)
                binary = label_binarize(evaluation.y_test, classes=np.arange(count))
                if count == 2:
                    binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
                for class_index, class_name in enumerate(encoder.classes_):
                    precision_values, recall_values, _ = precision_recall_curve(
                        binary[:, class_index], evaluation.test_probabilities[:, class_index]
                    )
                    axis.plot(recall_values, precision_values, label=str(class_name))
                axis.set_xlabel("Recall")
                axis.set_ylabel("Precision")
                axis.set_title(f"Testing Precision–Recall Curve — {evaluation.name}")
                axis.legend()

            elif graph_type == "PCA Audio Scatter":
                samples = self.state_data.samples
                X = samples[FEATURE_NAMES].to_numpy(dtype=np.float32)
                labels = samples["class_name"].astype(str).to_numpy()
                transformed = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
                for class_name in self.state_data.class_names:
                    mask = labels == class_name
                    axis.scatter(transformed[mask, 0], transformed[mask, 1], label=class_name, alpha=0.7)
                axis.set_xlabel("Principal Component 1")
                axis.set_ylabel("Principal Component 2")
                axis.set_title("PCA Scatter of Audio Samples")
                axis.legend()

            elif graph_type == "Model Comparison":
                names = list(self.state_data.evaluations)
                accuracy = [self.state_data.evaluations[name].metrics["accuracy"] for name in names]
                f1_values = [self.state_data.evaluations[name].metrics["f1"] for name in names]
                positions = np.arange(len(names))
                width = 0.36
                axis.bar(positions - width / 2, accuracy, width, label="Accuracy")
                axis.bar(positions + width / 2, f1_values, width, label="Weighted F1")
                axis.set_xticks(positions, labels=names, rotation=30, ha="right")
                axis.set_ylim(0, 1.05)
                axis.set_ylabel("Score")
                axis.set_title("Audio Classifier Comparison")
                axis.legend()

            elif graph_type == "Feature Importance":
                estimator = evaluation.estimator
                if isinstance(estimator, Pipeline):
                    estimator = estimator.steps[-1][1]
                importance: Optional[np.ndarray] = None
                if hasattr(estimator, "feature_importances_"):
                    importance = np.asarray(estimator.feature_importances_, dtype=float)
                elif hasattr(estimator, "coef_"):
                    importance = np.mean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
                elif isinstance(estimator, MLPClassifier):
                    importance = np.mean(np.abs(estimator.coefs_[0]), axis=1)
                if importance is None:
                    raise RuntimeError(f"Feature importance is unavailable for {evaluation.name}.")
                top_count = min(20, importance.size)
                indices = np.argsort(importance)[-top_count:]
                axis.barh(np.arange(top_count), importance[indices])
                axis.set_yticks(np.arange(top_count), labels=[FEATURE_NAMES[index] for index in indices])
                axis.set_xlabel("Importance")
                axis.set_title(f"Top Audio Features — {evaluation.name}")

            elif graph_type == "Learning Curve":
                samples = self.state_data.samples
                X = samples[FEATURE_NAMES].to_numpy(dtype=np.float32)
                y = LabelEncoder().fit_transform(samples["class_name"].astype(str))
                folds = max(2, min(5, int(samples["class_name"].value_counts().min())))
                train_sizes, train_scores, validation_scores = learning_curve(
                    clone(evaluation.estimator), X, y, cv=folds,
                    train_sizes=np.linspace(0.25, 1.0, 5), scoring="accuracy", n_jobs=1,
                )
                axis.plot(train_sizes, np.mean(train_scores, axis=1), marker="o", label="Training accuracy")
                axis.plot(
                    train_sizes, np.mean(validation_scores, axis=1), marker="o",
                    label="Cross-validation accuracy",
                )
                axis.set_xlabel("Training clips")
                axis.set_ylabel("Accuracy")
                axis.set_ylim(0, 1.05)
                axis.set_title(f"Learning Curve — {evaluation.name}")
                axis.legend()

            figure.tight_layout()
            self._show_figure(figure)
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _save_graph(self) -> None:
        figure = self.state_data.current_figure
        if figure is None:
            messagebox.showwarning(APP_NAME, "Generate a graph first.")
            return
        output = filedialog.asksaveasfilename(
            title="Save audio graph", defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("JPEG image", "*.jpg")],
        )
        if output:
            figure.savefig(output, dpi=300, bbox_inches="tight")

    def _save_best_model(self) -> None:
        if self.state_data.best_model_name is None or self.state_data.label_encoder is None:
            messagebox.showwarning(APP_NAME, "Train the models first.")
            return
        output = filedialog.asksaveasfilename(
            title="Save personalized audio classifier", defaultextension=".joblib",
            initialfile="personalized_audio_classifier.joblib",
            filetypes=[("Joblib model", "*.joblib")],
        )
        if not output:
            return
        evaluation = self.state_data.evaluations[self.state_data.best_model_name]
        package = {
            "application": APP_NAME,
            "version": APP_VERSION,
            "estimator": evaluation.estimator,
            "label_encoder": self.state_data.label_encoder,
            "model_name": evaluation.name,
            "feature_names": FEATURE_NAMES,
            "sample_rate": int(self.sample_rate_entry.get()),
            "clip_duration": float(self.duration_entry.get()),
            "minimum_rms": float(self.min_rms_entry.get()),
            "confidence_threshold": float(self.saved_threshold_entry.get()),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        joblib.dump(package, output)
        messagebox.showinfo(APP_NAME, f"Audio classifier saved:\n{output}")

    def _load_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load personalized audio classifier",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            package = joblib.load(selected)
            required = {
                "estimator", "label_encoder", "model_name", "feature_names",
                "sample_rate", "clip_duration",
            }
            missing = required.difference(package)
            if missing:
                raise ValueError(f"Invalid model. Missing fields: {sorted(missing)}")
            if list(package["feature_names"]) != FEATURE_NAMES:
                raise ValueError("This model uses a different audio feature definition.")
            self.state_data.loaded_model = package
            classes = ", ".join(map(str, package["label_encoder"].classes_))
            self.active_model_label.configure(text=f"{package['model_name']} | Classes: {classes}")
            self.realtime_threshold_entry.delete(0, "end")
            self.realtime_threshold_entry.insert(0, str(package.get("confidence_threshold", 0.65)))
            messagebox.showinfo(APP_NAME, f"Audio classifier loaded.\n\nClasses: {classes}")
        except Exception as error:
            messagebox.showerror(APP_NAME, f"Model loading failed:\n{error}")

    # ------------------------------------------------------------------
    # Real-time microphone classification
    # ------------------------------------------------------------------

    def _active_package(self) -> dict[str, Any]:
        if self.state_data.loaded_model is not None:
            return self.state_data.loaded_model
        if self.state_data.best_model_name is not None and self.state_data.label_encoder is not None:
            evaluation = self.state_data.evaluations[self.state_data.best_model_name]
            return {
                "estimator": evaluation.estimator,
                "label_encoder": self.state_data.label_encoder,
                "model_name": evaluation.name,
                "feature_names": FEATURE_NAMES,
                "sample_rate": int(self.sample_rate_entry.get()),
                "clip_duration": float(self.duration_entry.get()),
                "minimum_rms": float(self.min_rms_entry.get()),
                "confidence_threshold": float(self.saved_threshold_entry.get()),
            }
        raise RuntimeError("Train the models or load a saved .joblib classifier first.")

    def _start_realtime(self) -> None:
        try:
            package = self._active_package()
            settings = {
                "package": package,
                "device_label": self.realtime_device_var.get(),
                "device": self._device_index(self.realtime_device_var.get()),
                "threshold": float(self.realtime_threshold_entry.get()),
                "delay": float(self.realtime_delay_entry.get()),
            }
            if not 0.0 < settings["threshold"] < 1.0:
                raise ValueError("Confidence threshold must be between 0 and 1.")
            if settings["delay"] < 0:
                raise ValueError("Prediction delay cannot be negative.")
            self.realtime_stop.clear()
            self._run_thread(lambda: self._realtime_worker(settings))
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))

    def _realtime_worker(self, settings: dict[str, Any]) -> None:
        package = settings["package"]
        estimator = package["estimator"]
        encoder = package["label_encoder"]
        sample_rate = int(package["sample_rate"])
        duration = float(package["clip_duration"])
        minimum_rms = float(package.get("minimum_rms", 0.0))
        previous_summary = ""
        self.messages.put((
            "realtime_log",
            f"Started real-time classification.\n"
            f"Microphone: {settings['device_label']}\n"
            f"Clip duration: {duration} s | Sample rate: {sample_rate} Hz",
        ))

        while not self.realtime_stop.is_set():
            self.messages.put(("realtime_state", "Listening..."))
            audio = self._record_audio(duration, sample_rate, settings["device"])
            self.state_data.last_audio = audio.copy()
            self.state_data.last_sample_rate = sample_rate
            rms_value = float(np.sqrt(np.mean(audio * audio) + 1e-12))
            try:
                vector = extract_audio_features(audio, sample_rate, minimum_rms)
                probabilities = estimator.predict_proba([vector])[0]
                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index])
                class_name = str(encoder.inverse_transform([predicted_index])[0])
                display_name = class_name if confidence >= settings["threshold"] else "Uncertain"
                summary = f"{display_name} ({confidence * 100:.1f}%)"
            except ValueError as error:
                summary = f"Too quiet / invalid: {error}"

            self.messages.put(("realtime_result", (summary, rms_value)))
            if summary != previous_summary:
                self.messages.put((
                    "realtime_log",
                    f"{time.strftime('%H:%M:%S')} — {summary} | RMS={rms_value:.5f}",
                ))
                previous_summary = summary
            if not self.realtime_stop.is_set() and settings["delay"] > 0:
                time.sleep(settings["delay"])
        self._status(0, "Real-time audio classification stopped")

    def _stop_realtime(self) -> None:
        self.realtime_stop.set()
        try:
            sd.stop()
        except Exception:
            pass

    def _save_last_wav(self) -> None:
        if self.state_data.last_audio is None:
            messagebox.showwarning(APP_NAME, "No microphone clip is available.")
            return
        output = filedialog.asksaveasfilename(
            title="Save last microphone clip", defaultextension=".wav",
            initialfile=f"microphone_clip_{time.strftime('%Y%m%d_%H%M%S')}.wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        if output:
            wavfile.write(
                output,
                self.state_data.last_sample_rate,
                (np.clip(self.state_data.last_audio, -1.0, 1.0) * 32767.0).astype(np.int16),
            )

    # ------------------------------------------------------------------
    # Threads and messages
    # ------------------------------------------------------------------

    def _run_thread(self, target: Callable[[], None]) -> None:
        threading.Thread(
            target=self._thread_wrapper,
            args=(target,),
            daemon=True,
        ).start()

    def _thread_wrapper(self, target: Callable[[], None]) -> None:
        try:
            target()
        except (ValueError, RuntimeError, FileNotFoundError, sd.PortAudioError) as error:
            self.messages.put(("error", str(error)))
        except Exception as error:
            self.messages.put(("error", f"{error}\n\n{traceback.format_exc()}"))

    def _status(self, progress: float, text: str) -> None:
        self.messages.put(("status", (progress, text)))

    def _process_messages(self) -> None:
        try:
            while True:
                message_type, payload = self.messages.get_nowait()
                if message_type == "status":
                    progress, text = payload
                    self.progress.set(max(0.0, min(1.0, float(progress))))
                    self.status_label.configure(text=str(text))
                elif message_type == "mic_test":
                    label, rms_value, peak_value = payload
                    self.mic_test_label.configure(
                        text=f"Microphone test: {label} | RMS={rms_value:.5f} | Peak={peak_value:.5f}"
                    )
                    self.setup_text.insert(
                        "end",
                        f"\nMicrophone test completed. RMS={rms_value:.5f}, Peak={peak_value:.5f}\n",
                    )
                    self.setup_text.see("end")
                elif message_type == "capture_log":
                    self.capture_log.insert("end", "\n" + str(payload) + "\n")
                    self.capture_log.see("end")
                elif message_type == "capture_state":
                    text, accepted, target = payload
                    self.status_label.configure(text=text)
                    self.capture_count_label.configure(text=f"Accepted this session: {accepted}/{target}")
                elif message_type == "capture_progress":
                    accepted, target, rejected, rms_value = payload
                    self.capture_count_label.configure(
                        text=(
                            f"Accepted this session: {accepted}/{target} | "
                            f"Rejected: {rejected} | RMS: {rms_value:.5f}"
                        )
                    )
                    self.progress.set(accepted / max(target, 1))
                elif message_type == "capture_complete":
                    class_name, accepted, rejected = payload
                    self.capture_log.insert(
                        "end",
                        f"\nRecording completed for {class_name}. "
                        f"Accepted={accepted}, Rejected={rejected}.\n",
                    )
                    self.capture_log.see("end")
                    self._update_counts()
                    self._status(1.0, f"Recording completed for {class_name}")
                elif message_type == "training_log":
                    self.training_log.insert("end", str(payload) + "\n")
                    self.training_log.see("end")
                elif message_type == "training_complete":
                    names = payload["models"]
                    best = payload["best"]
                    self.result_model_menu.configure(values=names)
                    self.result_model_var.set(best)
                    self.best_model_label.configure(text=f"Best model: {best}")
                    classes = ", ".join(self.state_data.label_encoder.classes_)
                    self.active_model_label.configure(text=f"Newly trained: {best} | Classes: {classes}")
                    self._refresh_result_table()
                    self.tabs.set("4. Results & Graphs")
                    messagebox.showinfo(APP_NAME, f"Audio training completed.\n\nBest model: {best}")
                elif message_type == "realtime_state":
                    self.status_label.configure(text=str(payload))
                elif message_type == "realtime_result":
                    summary, rms_value = payload
                    self.realtime_result_label.configure(text=f"Prediction: {summary}")
                    self.level_label.configure(text=f"Microphone RMS level: {rms_value:.5f}")
                elif message_type == "realtime_log":
                    self.realtime_log.insert("end", str(payload) + "\n")
                    self.realtime_log.see("end")
                elif message_type == "error":
                    self.status_label.configure(text="Operation failed")
                    self.progress.set(0)
                    messagebox.showerror(APP_NAME, str(payload))
        except queue.Empty:
            pass
        self.after(100, self._process_messages)

    def _close(self) -> None:
        self.capture_stop.set()
        self.realtime_stop.set()
        try:
            sd.stop()
        except Exception:
            pass
        self._clear_figure()
        self.destroy()


if __name__ == "__main__":
    AudioClassifierGUI().mainloop()
