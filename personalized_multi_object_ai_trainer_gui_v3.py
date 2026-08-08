"""Personalized Multi-Object AI Trainer.

Custom YOLO object detection GUI for 2-5 classes. It supports image import,
bounding-box annotation, train/validation preparation, Ultralytics training,
result viewing, model loading, and simultaneous multi-class webcam detection.
"""
from __future__ import annotations

import json
import os
import queue
import random
import shutil
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import customtkinter as ctk
import cv2
import pandas as pd
import yaml
from PIL import Image, ImageTk
from tkinter import Canvas, filedialog, messagebox, ttk

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

APP_NAME = "Personalized Multi-Object AI Trainer"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OFFICIAL_BASE_MODELS = {"yolo11n.pt", "yolo26n.pt", "yolov8n.pt"}
CORRUPT_CHECKPOINT_MARKERS = (
    "pytorchstreamreader",
    "failed finding central directory",
    "failed reading zip archive",
    "not a loadable checkpoint",
    "checkpoint file is corrupted",
    "invalid header",
    "unexpected eof",
)


@dataclass
class BoundingBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class State:
    project_dir: Optional[Path] = None
    class_names: list[str] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    image_index: int = 0
    boxes: list[BoundingBox] = field(default_factory=list)
    original_image: Optional[Image.Image] = None
    photo: Optional[ImageTk.PhotoImage] = None
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    drag_start: Optional[tuple[float, float]] = None
    temp_rect: Optional[int] = None
    run_dir: Optional[Path] = None
    weights_path: Optional[Path] = None
    detector: Any = None
    result_photo: Optional[ImageTk.PhotoImage] = None


class MultiObjectTrainer(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.title(APP_NAME)
        self.geometry("1500x920")
        self.minsize(1200, 760)
        self.state_data = State()
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self._build_header()
        self._build_tabs()
        self._build_footer()
        self.after(100, self._process_messages)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_header(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(
            frame,
            text="PERSONALIZED MULTI-OBJECT AI TRAINER",
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(side="left", padx=18, pady=12)
        ctk.CTkLabel(
            frame,
            text="Bounding-box annotation • Custom YOLO • Multiple classes per frame",
        ).pack(side="left", padx=12)
        self.model_label = ctk.CTkLabel(
            frame, text="Model: Not loaded", font=ctk.CTkFont(weight="bold")
        )
        self.model_label.pack(side="right", padx=18)

    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=6)
        names = (
            "1. Project",
            "2. Images",
            "3. Annotate",
            "4. Prepare",
            "5. Train",
            "6. Results",
            "7. Multi-Object Test",
        )
        for name in names:
            self.tabs.add(name)
        self._project_tab()
        self._images_tab()
        self._annotation_tab()
        self._prepare_tab()
        self._train_tab()
        self._results_tab()
        self._test_tab()

    def _build_footer(self) -> None:
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=12, pady=(6, 12))
        self.status = ctk.CTkLabel(frame, text="Ready", anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=12, pady=8)
        self.progress = ctk.CTkProgressBar(frame, width=320)
        self.progress.set(0)
        self.progress.pack(side="right", padx=12, pady=8)

    def _project_tab(self) -> None:
        tab = self.tabs.tab("1. Project")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(5, weight=1)
        ctk.CTkLabel(
            tab,
            text="Create or Load an Object-Detection Project",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 15), sticky="w")
        ctk.CTkLabel(tab, text="Project folder").grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.project_entry = ctk.CTkEntry(tab)
        self.project_entry.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(tab, text="Browse", command=self._browse_project).grid(row=1, column=2, padx=20, pady=10)
        ctk.CTkLabel(tab, text="Class names").grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.classes_entry = ctk.CTkEntry(tab, placeholder_text="Bottle, Mobile, Book")
        self.classes_entry.grid(row=2, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(tab, text="Enter 2-5 unique names separated by commas.").grid(row=3, column=1, padx=20, sticky="w")
        controls = ctk.CTkFrame(tab)
        controls.grid(row=4, column=0, columnspan=3, padx=20, pady=15, sticky="ew")
        ctk.CTkButton(controls, text="Create Project", command=self._create_project).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Load project.json", command=self._load_project).pack(side="left", padx=10, pady=10)
        self.project_info = ctk.CTkTextbox(tab)
        self.project_info.grid(row=5, column=0, columnspan=3, padx=20, pady=(0, 20), sticky="nsew")
        self.project_info.insert("1.0", "A true multi-object detector requires a bounding box around every visible training object.")

    def _images_tab(self) -> None:
        tab = self.tabs.tab("2. Images")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(tab, text="Import Training Images", font=ctk.CTkFont(size=21, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        controls = ctk.CTkFrame(tab)
        controls.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(controls, text="Import Files", command=self._import_files).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Import Folder", command=self._import_folder).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Capture Webcam Images", command=lambda: self._thread(self._capture_images)).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Refresh", command=self._refresh_images).pack(side="left", padx=10, pady=10)
        self.image_log = ctk.CTkTextbox(tab)
        self.image_log.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.image_log.insert("1.0", "Import images containing one or more target objects. Include different combinations, distances, backgrounds and lighting conditions.")

    def _annotation_tab(self) -> None:
        tab = self.tabs.tab("3. Annotate")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        ctk.CTkLabel(controls, text="Active class").pack(side="left", padx=(10, 5), pady=10)
        self.class_var = ctk.StringVar(value="No classes")
        self.class_menu = ctk.CTkOptionMenu(controls, values=["No classes"], variable=self.class_var, width=180)
        self.class_menu.pack(side="left", padx=8, pady=10)
        ctk.CTkButton(controls, text="Previous", command=self._previous).pack(side="left", padx=5)
        ctk.CTkButton(controls, text="Next", command=self._next).pack(side="left", padx=5)
        ctk.CTkButton(controls, text="Save Boxes", command=self._save_boxes).pack(side="left", padx=8)
        ctk.CTkButton(controls, text="Undo", command=self._undo).pack(side="left", padx=8)
        ctk.CTkButton(controls, text="Clear", command=self._clear_boxes).pack(side="left", padx=8)
        self.image_position = ctk.CTkLabel(controls, text="Image 0/0", font=ctk.CTkFont(weight="bold"))
        self.image_position.pack(side="right", padx=15)
        holder = ctk.CTkFrame(tab)
        holder.grid(row=1, column=0, padx=12, pady=(6, 12), sticky="nsew")
        holder.grid_columnconfigure(0, weight=1)
        holder.grid_rowconfigure(0, weight=1)
        self.canvas = Canvas(holder, background="#202020", cursor="crosshair", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<Configure>", lambda _event: self._display_image())

    def _prepare_tab(self) -> None:
        tab = self.tabs.tab("4. Prepare")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(4, weight=1)
        ctk.CTkLabel(tab, text="Prepare YOLO Dataset", font=ctk.CTkFont(size=21, weight="bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 15), sticky="w")
        ctk.CTkLabel(tab, text="Validation fraction").grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.val_entry = ctk.CTkEntry(tab)
        self.val_entry.insert(0, "0.20")
        self.val_entry.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(tab, text="Random seed").grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.seed_entry = ctk.CTkEntry(tab)
        self.seed_entry.insert(0, "42")
        self.seed_entry.grid(row=2, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(tab, text="PREPARE TRAIN/VALIDATION DATASET", height=45, command=lambda: self._thread(self._prepare_dataset)).grid(row=3, column=0, columnspan=2, padx=20, pady=15, sticky="ew")
        self.prepare_log = ctk.CTkTextbox(tab)
        self.prepare_log.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")

    def _train_tab(self) -> None:
        tab = self.tabs.tab("5. Train")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(9, weight=1)
        ctk.CTkLabel(tab, text="Train Custom YOLO Detector", font=ctk.CTkFont(size=21, weight="bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 15), sticky="w")
        defaults = [
            ("Pretrained weights", "yolo11n.pt"),
            ("Epochs", "10"),
            ("Image size", "640"),
            ("Batch size", "4"),
            ("Device", "cpu"),
            ("Run name", "custom_detector"),
        ]
        self.train_entries: dict[str, ctk.CTkEntry] = {}
        for row, (name, value) in enumerate(defaults, start=1):
            ctk.CTkLabel(tab, text=name).grid(row=row, column=0, padx=20, pady=8, sticky="w")
            entry = ctk.CTkEntry(tab)
            entry.insert(0, value)
            entry.grid(row=row, column=1, padx=20, pady=8, sticky="ew")
            self.train_entries[name] = entry
        ctk.CTkLabel(
            tab,
            text=(
                "Training base: use yolo11n.pt. After training, webcam testing "
                "must use the generated runs/.../weights/best.pt file."
            ),
            wraplength=900,
            justify="left",
        ).grid(
            row=7,
            column=0,
            columnspan=2,
            padx=20,
            pady=(8, 2),
            sticky="w",
        )
        ctk.CTkButton(
            tab,
            text="TRAIN MULTI-OBJECT DETECTOR",
            height=45,
            font=ctk.CTkFont(weight="bold"),
            command=lambda: self._thread(self._train_model),
        ).grid(
            row=8,
            column=0,
            columnspan=2,
            padx=20,
            pady=15,
            sticky="ew",
        )
        self.train_log = ctk.CTkTextbox(tab)
        self.train_log.grid(row=9, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")

    def _results_tab(self) -> None:
        tab = self.tabs.tab("6. Results")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=3)
        tab.grid_rowconfigure(1, weight=1)
        controls = ctk.CTkFrame(tab)
        controls.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        ctk.CTkButton(controls, text="Refresh", command=self._refresh_results).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Load Results Folder", command=self._load_results_folder).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(controls, text="Load best.pt", command=self._load_weights).pack(side="left", padx=10, pady=10)
        self.result_tree = ttk.Treeview(tab, columns=("File",), show="headings")
        self.result_tree.heading("File", text="Generated result")
        self.result_tree.column("File", width=250)
        self.result_tree.grid(row=1, column=0, padx=(20, 10), pady=(0, 20), sticky="nsew")
        self.result_tree.bind("<<TreeviewSelect>>", self._select_result)
        right = ctk.CTkFrame(tab)
        right.grid(row=1, column=1, padx=(10, 20), pady=(0, 20), sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self.result_canvas = Canvas(right, background="#202020", highlightthickness=0)
        self.result_canvas.grid(row=0, column=0, sticky="nsew")
        self.metric_text = ctk.CTkTextbox(right, height=190)
        self.metric_text.grid(row=1, column=0, sticky="ew", padx=8, pady=8)

    def _test_tab(self) -> None:
        tab = self.tabs.tab("7. Multi-Object Test")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(4, weight=1)
        ctk.CTkLabel(tab, text="Simultaneous Multi-Class Webcam Detection", font=ctk.CTkFont(size=21, weight="bold")).grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 15), sticky="w")
        ctk.CTkLabel(tab, text="Model weights").grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.weights_entry = ctk.CTkEntry(tab)
        self.weights_entry.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(tab, text="Browse", command=self._load_weights).grid(row=1, column=2, padx=20, pady=10)
        controls = ctk.CTkFrame(tab)
        controls.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(controls, text="Confidence").pack(side="left", padx=(12, 5))
        self.conf_entry = ctk.CTkEntry(controls, width=90)
        self.conf_entry.insert(0, "0.40")
        self.conf_entry.pack(side="left", padx=8, pady=12)
        ctk.CTkLabel(controls, text="Camera index").pack(side="left", padx=(18, 5))
        self.camera_entry = ctk.CTkEntry(controls, width=90)
        self.camera_entry.insert(0, "0")
        self.camera_entry.pack(side="left", padx=8, pady=12)
        ctk.CTkButton(controls, text="Start Multi-Object Webcam", command=lambda: self._thread(self._webcam_detection)).pack(side="left", padx=18, pady=12)
        ctk.CTkButton(controls, text="Stop", command=self._stop_webcam).pack(side="left", padx=8, pady=12)
        self.detection_label = ctk.CTkLabel(tab, text="Detections: —", font=ctk.CTkFont(size=24, weight="bold"))
        self.detection_label.grid(row=3, column=0, columnspan=3, padx=20, pady=15, sticky="w")
        self.test_log = ctk.CTkTextbox(tab)
        self.test_log.grid(row=4, column=0, columnspan=3, padx=20, pady=(0, 20), sticky="nsew")

    # ---------- generic helpers ----------
    def _thread(self, target: Callable[[], None]) -> None:
        threading.Thread(target=self._thread_wrapper, args=(target,), daemon=True).start()

    def _thread_wrapper(self, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception as exc:
            self.messages.put(("error", f"{exc}\n\n{traceback.format_exc()}"))

    def _set_status(self, value: float, text: str) -> None:
        self.messages.put(("status", (value, text)))

    def _process_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "status":
                    value, text = payload
                    self.progress.set(max(0.0, min(1.0, float(value))))
                    self.status.configure(text=str(text))
                elif kind == "image_log":
                    self.image_log.insert("end", str(payload) + "\n")
                    self.image_log.see("end")
                elif kind == "prepare_log":
                    self.prepare_log.insert("end", str(payload) + "\n")
                    self.prepare_log.see("end")
                elif kind == "train_log":
                    self.train_log.insert("end", str(payload) + "\n")
                    self.train_log.see("end")
                elif kind == "trained":
                    self._training_complete(Path(payload))
                elif kind == "detections":
                    summary = str(payload)
                    self.detection_label.configure(text=f"Detections: {summary}")
                    self.test_log.insert("end", f"{time.strftime('%H:%M:%S')} — {summary}\n")
                    self.test_log.see("end")
                elif kind == "refresh_images":
                    self._refresh_images()
                elif kind == "error":
                    messagebox.showerror(APP_NAME, str(payload))
                    self.progress.set(0)
        except queue.Empty:
            pass
        self.after(100, self._process_messages)

    # ---------- project ----------
    def _browse_project(self) -> None:
        folder = filedialog.askdirectory(title="Select project folder")
        if folder:
            self.project_entry.delete(0, "end")
            self.project_entry.insert(0, folder)

    def _parse_classes(self) -> list[str]:
        names = [name.strip() for name in self.classes_entry.get().split(",") if name.strip()]
        unique: list[str] = []
        for name in names:
            if name not in unique:
                unique.append(name)
        if not 2 <= len(unique) <= 5:
            raise ValueError("Enter 2-5 unique class names.")
        return unique

    def _create_project(self) -> None:
        try:
            folder = self.project_entry.get().strip()
            if not folder:
                raise ValueError("Select a project folder.")
            project = Path(folder)
            classes = self._parse_classes()
            for relative in (
                "images_raw",
                "labels_raw",
                "dataset/images/train",
                "dataset/images/val",
                "dataset/labels/train",
                "dataset/labels/val",
                "runs",
            ):
                (project / relative).mkdir(parents=True, exist_ok=True)
            (project / "project.json").write_text(
                json.dumps({"class_names": classes, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                encoding="utf-8",
            )
            self.state_data.project_dir = project
            self.state_data.class_names = classes
            self._apply_project()
            messagebox.showinfo(APP_NAME, f"Project created:\n{project}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def _load_project(self) -> None:
        selected = filedialog.askopenfilename(title="Select project.json", filetypes=[("Project", "project.json"), ("JSON", "*.json")])
        if not selected:
            return
        try:
            path = Path(selected)
            data = json.loads(path.read_text(encoding="utf-8"))
            classes = list(data["class_names"])
            if not 2 <= len(classes) <= 5:
                raise ValueError("Invalid class list.")
            self.state_data.project_dir = path.parent
            self.state_data.class_names = classes
            self.project_entry.delete(0, "end")
            self.project_entry.insert(0, str(path.parent))
            self.classes_entry.delete(0, "end")
            self.classes_entry.insert(0, ", ".join(classes))
            self._apply_project()
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Project loading failed:\n{exc}")

    def _apply_project(self) -> None:
        project = self._require_project()
        self.class_menu.configure(values=self.state_data.class_names)
        self.class_var.set(self.state_data.class_names[0])
        self.project_info.delete("1.0", "end")
        self.project_info.insert("1.0", f"Project: {project}\nClasses: {', '.join(self.state_data.class_names)}\n\nImport images and annotate every object.")
        self._refresh_images()

    def _require_project(self) -> Path:
        if self.state_data.project_dir is None:
            raise RuntimeError("Create or load a project first.")
        return self.state_data.project_dir

    # ---------- image import ----------
    def _copy_unique(self, source: Path, destination: Path) -> Path:
        target = destination / source.name
        counter = 1
        while target.exists():
            target = destination / f"{source.stem}_{counter}{source.suffix.lower()}"
            counter += 1
        shutil.copy2(source, target)
        return target

    def _import_files(self) -> None:
        try:
            project = self._require_project()
            files = filedialog.askopenfilenames(title="Select images", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")])
            if not files:
                return
            count = 0
            for name in files:
                path = Path(name)
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self._copy_unique(path, project / "images_raw")
                    count += 1
            self.image_log.insert("end", f"\nImported {count} images.\n")
            self._refresh_images()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def _import_folder(self) -> None:
        try:
            project = self._require_project()
            folder = filedialog.askdirectory(title="Select image folder")
            if not folder:
                return
            files = [path for path in Path(folder).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
            for path in files:
                self._copy_unique(path, project / "images_raw")
            self.image_log.insert("end", f"\nImported {len(files)} images.\n")
            self._refresh_images()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def _capture_images(self) -> None:
        project = self._require_project()
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Webcam could not be opened.")
        self.stop_event.clear()
        captured = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    continue
                preview = frame.copy()
                cv2.putText(preview, "SPACE: capture | Q: stop", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Capture Detection Images", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    output = project / "images_raw" / f"webcam_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(str(output), frame)
                    captured += 1
                    self._set_status(0.5, f"Captured images: {captured}")
                elif key == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
            self.stop_event.set()
        self.messages.put(("image_log", f"Captured {captured} webcam images."))
        self.messages.put(("refresh_images", None))

    def _refresh_images(self) -> None:
        if self.state_data.project_dir is None:
            return
        folder = self.state_data.project_dir / "images_raw"
        self.state_data.images = sorted([path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])
        if self.state_data.images:
            self.state_data.image_index = min(self.state_data.image_index, len(self.state_data.images) - 1)
            self._load_current_image()
        else:
            self.canvas.delete("all")
            self.image_position.configure(text="Image 0/0")
        annotated = sum(1 for path in self.state_data.images if self._label_path(path).exists())
        self.image_log.insert("end", f"\nProject images: {len(self.state_data.images)} | Annotated: {annotated}\n")
        self.image_log.see("end")

    # ---------- annotation ----------
    def _label_path(self, image_path: Path) -> Path:
        return self._require_project() / "labels_raw" / f"{image_path.stem}.txt"

    def _load_current_image(self) -> None:
        if not self.state_data.images:
            return
        path = self.state_data.images[self.state_data.image_index]
        self.state_data.original_image = Image.open(path).convert("RGB")
        self.state_data.boxes = self._read_boxes(path)
        self.image_position.configure(text=f"Image {self.state_data.image_index + 1}/{len(self.state_data.images)} — {path.name}")
        self._display_image()

    def _read_boxes(self, image_path: Path) -> list[BoundingBox]:
        label = self._label_path(image_path)
        if not label.exists():
            return []
        with Image.open(image_path) as image:
            width, height = image.size
        boxes: list[BoundingBox] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
            boxes.append(BoundingBox(class_id, (xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height))
        return boxes

    def _save_boxes(self) -> None:
        if not self.state_data.images or self.state_data.original_image is None:
            return
        image_path = self.state_data.images[self.state_data.image_index]
        width, height = self.state_data.original_image.size
        lines: list[str] = []
        for box in self.state_data.boxes:
            x1 = max(0.0, min(float(width), box.x1))
            y1 = max(0.0, min(float(height), box.y1))
            x2 = max(0.0, min(float(width), box.x2))
            y2 = max(0.0, min(float(height), box.y2))
            bw = x2 - x1
            bh = y2 - y1
            if bw < 2 or bh < 2:
                continue
            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            lines.append(f"{box.class_id} {xc:.6f} {yc:.6f} {bw / width:.6f} {bh / height:.6f}")
        self._label_path(image_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._set_status(1.0, f"Saved {len(lines)} boxes for {image_path.name}")

    def _display_image(self) -> None:
        image = self.state_data.original_image
        if image is None:
            return
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        scale = min(cw / image.width, ch / image.height)
        dw, dh = max(1, int(image.width * scale)), max(1, int(image.height * scale))
        resized = image.resize((dw, dh), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        ox, oy = (cw - dw) / 2, (ch - dh) / 2
        self.state_data.photo = photo
        self.state_data.scale = scale
        self.state_data.offset_x = ox
        self.state_data.offset_y = oy
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=photo)
        for index, box in enumerate(self.state_data.boxes):
            x1, y1 = ox + box.x1 * scale, oy + box.y1 * scale
            x2, y2 = ox + box.x2 * scale, oy + box.y2 * scale
            name = self.state_data.class_names[box.class_id] if 0 <= box.class_id < len(self.state_data.class_names) else str(box.class_id)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=3)
            self.canvas.create_text(x1 + 4, y1 + 4, anchor="nw", text=f"{index + 1}: {name}", fill="yellow", font=("Arial", 13, "bold"))

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        image = self.state_data.original_image
        if image is None or self.state_data.scale <= 0:
            return 0.0, 0.0
        ix = (x - self.state_data.offset_x) / self.state_data.scale
        iy = (y - self.state_data.offset_y) / self.state_data.scale
        return max(0.0, min(float(image.width), ix)), max(0.0, min(float(image.height), iy))

    def _mouse_down(self, event: Any) -> None:
        if self.state_data.original_image is None:
            return
        self.state_data.drag_start = (event.x, event.y)
        self.state_data.temp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="cyan", width=2)

    def _mouse_drag(self, event: Any) -> None:
        if self.state_data.drag_start is None or self.state_data.temp_rect is None:
            return
        sx, sy = self.state_data.drag_start
        self.canvas.coords(self.state_data.temp_rect, sx, sy, event.x, event.y)

    def _mouse_up(self, event: Any) -> None:
        if self.state_data.drag_start is None:
            return
        sx, sy = self.state_data.drag_start
        x1, y1 = self._canvas_to_image(min(sx, event.x), min(sy, event.y))
        x2, y2 = self._canvas_to_image(max(sx, event.x), max(sy, event.y))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            name = self.class_var.get()
            if name in self.state_data.class_names:
                self.state_data.boxes.append(BoundingBox(self.state_data.class_names.index(name), x1, y1, x2, y2))
        self.state_data.drag_start = None
        self.state_data.temp_rect = None
        self._display_image()

    def _undo(self) -> None:
        if self.state_data.boxes:
            self.state_data.boxes.pop()
            self._display_image()

    def _clear_boxes(self) -> None:
        if messagebox.askyesno(APP_NAME, "Clear all boxes in this image?"):
            self.state_data.boxes.clear()
            self._display_image()

    def _previous(self) -> None:
        if not self.state_data.images:
            return
        self._save_boxes()
        self.state_data.image_index = max(0, self.state_data.image_index - 1)
        self._load_current_image()

    def _next(self) -> None:
        if not self.state_data.images:
            return
        self._save_boxes()
        self.state_data.image_index = min(len(self.state_data.images) - 1, self.state_data.image_index + 1)
        self._load_current_image()

    # ---------- prepare dataset ----------
    def _prepare_dataset(self) -> None:
        project = self._require_project()
        val_fraction = float(self.val_entry.get())
        seed = int(self.seed_entry.get())
        if not 0 < val_fraction < 0.5:
            raise ValueError("Validation fraction must be between 0 and 0.5.")
        pairs: list[tuple[Path, Path]] = []
        skipped = 0
        for image in sorted((project / "images_raw").iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = self._label_path(image)
            if label.exists() and label.read_text(encoding="utf-8").strip():
                pairs.append((image, label))
            else:
                skipped += 1
        if len(pairs) < 4:
            raise ValueError("At least four annotated images are required; substantially more are recommended.")
        random.Random(seed).shuffle(pairs)
        val_count = max(1, round(len(pairs) * val_fraction))
        val_pairs, train_pairs = pairs[:val_count], pairs[val_count:]
        destinations = {
            "train_images": project / "dataset/images/train",
            "val_images": project / "dataset/images/val",
            "train_labels": project / "dataset/labels/train",
            "val_labels": project / "dataset/labels/val",
        }
        for folder in destinations.values():
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
        for image, label in train_pairs:
            shutil.copy2(image, destinations["train_images"] / image.name)
            shutil.copy2(label, destinations["train_labels"] / label.name)
        for image, label in val_pairs:
            shutil.copy2(image, destinations["val_images"] / image.name)
            shutil.copy2(label, destinations["val_labels"] / label.name)
        dataset = project / "dataset"
        data_yaml = {
            "path": str(dataset.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(self.state_data.class_names)},
        }
        (dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self.messages.put(("prepare_log", f"Dataset prepared.\nTraining images: {len(train_pairs)}\nValidation images: {len(val_pairs)}\nSkipped unannotated images: {skipped}\nConfiguration: {dataset / 'data.yaml'}"))
        self._set_status(1.0, "Dataset prepared")

    # ---------- train ----------
    def _require_yolo(self) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Run: python -m pip install ultralytics")

    def _load_yolo_checkpoint(
        self,
        weights_text: str,
        *,
        operation_name: str,
    ) -> Any:
        """Load a YOLO checkpoint and repair corrupt official base weights.

        Official base checkpoints such as yolo11n.pt may be downloaded again.
        Custom trained files such as best.pt are never deleted automatically.
        """
        self._require_yolo()

        text = weights_text.strip()
        if not text:
            raise RuntimeError(
                f"No weights were selected for {operation_name}."
            )

        candidate = Path(text).expanduser()
        official_name = candidate.name.lower() in OFFICIAL_BASE_MODELS

        # A simple official filename is intentionally allowed even when it
        # does not exist. Ultralytics will download it from its official asset.
        is_simple_name = (
            candidate.name == text
            and candidate.parent == Path(".")
        )

        if not candidate.exists() and not (official_name and is_simple_name):
            raise RuntimeError(
                f"Weights file not found:\n{candidate}\n\n"
                "For initial training, enter only an official base model name "
                "such as yolo11n.pt. After custom training, load the generated "
                "runs/.../weights/best.pt file for webcam detection."
            )

        try:
            if candidate.exists():
                if candidate.stat().st_size == 0:
                    raise RuntimeError(
                        f"The checkpoint is empty: {candidate}"
                    )
                return YOLO(str(candidate))

            # No local file: let Ultralytics fetch the official checkpoint.
            return YOLO(candidate.name)

        except Exception as error:
            error_text = str(error).lower()
            looks_corrupt = any(
                marker_text in error_text
                for marker_text in CORRUPT_CHECKPOINT_MARKERS
            )

            if candidate.exists() and official_name and looks_corrupt:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                corrupt_backup = candidate.with_name(
                    f"{candidate.name}.corrupt_{timestamp}"
                )

                try:
                    candidate.replace(corrupt_backup)
                except Exception as move_error:
                    raise RuntimeError(
                        f"The official checkpoint is corrupted:\n{candidate}\n\n"
                        "Close any program using this file, delete it manually, "
                        "and try again.\n\n"
                        f"Original loading error: {error}\n"
                        f"Could not move damaged file: {move_error}"
                    ) from error

                self.messages.put(
                    (
                        "train_log",
                        (
                            f"Damaged checkpoint moved to:\n{corrupt_backup}\n"
                            f"Downloading a clean official {candidate.name}..."
                        ),
                    )
                )
                self._set_status(
                    0.05,
                    f"Downloading clean {candidate.name}...",
                )

                previous_directory = Path.cwd()
                try:
                    candidate.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    os.chdir(candidate.parent)
                    repaired_model = YOLO(candidate.name)
                except Exception as download_error:
                    raise RuntimeError(
                        f"The damaged checkpoint was moved to:\n"
                        f"{corrupt_backup}\n\n"
                        f"A clean {candidate.name} could not be downloaded. "
                        "Check the internet connection and run the operation "
                        "again.\n\n"
                        f"Download error: {download_error}"
                    ) from download_error
                finally:
                    os.chdir(previous_directory)

                if not candidate.exists():
                    raise RuntimeError(
                        f"Ultralytics did not create the expected file:\n"
                        f"{candidate}"
                    )

                self.messages.put(
                    (
                        "train_log",
                        f"Clean checkpoint downloaded successfully:\n{candidate}",
                    )
                )
                return repaired_model

            if looks_corrupt:
                raise RuntimeError(
                    f"The selected checkpoint is corrupted or incomplete:\n"
                    f"{candidate}\n\n"
                    "If this is an official base model, delete it and enter "
                    "only its model name, for example yolo11n.pt, so that "
                    "Ultralytics downloads a clean copy.\n\n"
                    "If this is a custom best.pt file, retrain the detector or "
                    "restore a valid copy; a damaged custom checkpoint cannot "
                    "be reconstructed.\n\n"
                    f"Technical detail: {error}"
                ) from error

            raise RuntimeError(
                f"Could not load the checkpoint for {operation_name}:\n"
                f"{candidate}\n\n{error}"
            ) from error

    def _train_model(self) -> None:
        self._require_yolo()
        project = self._require_project()
        data_yaml = project / "dataset/data.yaml"

        if not data_yaml.exists():
            raise RuntimeError(
                "Prepare the dataset before training."
            )

        weights = self.train_entries[
            "Pretrained weights"
        ].get().strip()
        epochs = int(
            self.train_entries["Epochs"].get()
        )
        imgsz = int(
            self.train_entries["Image size"].get()
        )
        batch = int(
            self.train_entries["Batch size"].get()
        )
        device = self.train_entries[
            "Device"
        ].get().strip()
        run_name = (
            self.train_entries["Run name"].get().strip()
            or "custom_detector"
        )

        if epochs < 1:
            raise ValueError("Epochs must be at least 1.")
        if imgsz < 160:
            raise ValueError("Image size must be at least 160.")
        if batch < 1:
            raise ValueError("Batch size must be at least 1.")

        run_dir_expected = (
            project
            / "runs"
            / run_name
        )

        self.messages.put(
            (
                "train_log",
                (
                    f"Loading base checkpoint: {weights}\n"
                    f"Training data: {data_yaml}\n"
                    f"Epochs: {epochs}\n"
                    f"Image size: {imgsz}\n"
                    f"Batch: {batch}\n"
                    f"Device: {device or 'automatic'}\n"
                    f"Expected output folder: {run_dir_expected}\n"
                    "Windows data workers: 0\n"
                    "Checkpoint saving: enabled\n"
                ),
            )
        )
        self._set_status(
            0.03,
            "Loading the pretrained checkpoint...",
        )

        model = self._load_yolo_checkpoint(
            weights,
            operation_name="training",
        )

        self.messages.put(
            (
                "train_log",
                (
                    "Base checkpoint loaded successfully.\n"
                    "Initialising the YOLO trainer..."
                ),
            )
        )
        self._set_status(
            0.07,
            "Initialising training...",
        )

        # GUI-safe training callbacks. Ultralytics invokes these inside the
        # training thread, so all Tk updates are sent through the message queue.
        def on_train_start(trainer: Any) -> None:
            save_dir = Path(trainer.save_dir)
            self.messages.put(
                (
                    "train_log",
                    (
                        "Training has started.\n"
                        f"Output directory: {save_dir}\n"
                        "best.pt is normally written after the first "
                        "successful validation epoch."
                    ),
                )
            )
            self._set_status(
                0.10,
                "Training epoch 1...",
            )

        def on_fit_epoch_end(trainer: Any) -> None:
            completed_epoch = int(trainer.epoch) + 1
            total_epochs = int(
                getattr(trainer, "epochs", epochs)
            )
            metrics = getattr(
                trainer,
                "metrics",
                {},
            ) or {}

            metric_parts: list[str] = []
            preferred_metric_names = (
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
            )

            for metric_name in preferred_metric_names:
                if metric_name in metrics:
                    try:
                        metric_parts.append(
                            f"{metric_name}="
                            f"{float(metrics[metric_name]):.4f}"
                        )
                    except Exception:
                        pass

            metric_text = (
                " | ".join(metric_parts)
                if metric_parts
                else "Validation completed"
            )

            best_path = Path(
                getattr(
                    trainer,
                    "best",
                    Path(trainer.save_dir)
                    / "weights"
                    / "best.pt",
                )
            )
            last_path = Path(
                getattr(
                    trainer,
                    "last",
                    Path(trainer.save_dir)
                    / "weights"
                    / "last.pt",
                )
            )

            self.messages.put(
                (
                    "train_log",
                    (
                        f"Epoch {completed_epoch}/{total_epochs} completed\n"
                        f"{metric_text}\n"
                        f"best.pt exists: {best_path.exists()}\n"
                        f"last.pt exists: {last_path.exists()}\n"
                    ),
                )
            )

            progress = (
                0.10
                + 0.85
                * completed_epoch
                / max(total_epochs, 1)
            )
            self._set_status(
                min(progress, 0.95),
                (
                    f"Completed epoch "
                    f"{completed_epoch}/{total_epochs}"
                ),
            )

        def on_model_save(trainer: Any) -> None:
            best_path = Path(
                getattr(
                    trainer,
                    "best",
                    Path(trainer.save_dir)
                    / "weights"
                    / "best.pt",
                )
            )
            last_path = Path(
                getattr(
                    trainer,
                    "last",
                    Path(trainer.save_dir)
                    / "weights"
                    / "last.pt",
                )
            )
            self.messages.put(
                (
                    "train_log",
                    (
                        "Checkpoint saved.\n"
                        f"Best: {best_path}\n"
                        f"Last: {last_path}"
                    ),
                )
            )

        def on_train_end(trainer: Any) -> None:
            self.messages.put(
                (
                    "train_log",
                    (
                        "Ultralytics training loop finished.\n"
                        f"Run directory: {trainer.save_dir}"
                    ),
                )
            )

        model.add_callback(
            "on_train_start",
            on_train_start,
        )
        model.add_callback(
            "on_fit_epoch_end",
            on_fit_epoch_end,
        )
        model.add_callback(
            "on_model_save",
            on_model_save,
        )
        model.add_callback(
            "on_train_end",
            on_train_end,
        )

        training_arguments: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": str(project / "runs"),
            "name": run_name,
            "exist_ok": True,
            "plots": True,
            "verbose": True,
            "workers": 0,
            "save": True,
            "save_period": 1,
            "val": True,
            "cache": False,
        }

        if device:
            training_arguments["device"] = device

        try:
            model.train(
                **training_arguments
            )
        except Exception as error:
            partial_run_dir = (
                Path(model.trainer.save_dir)
                if getattr(
                    model,
                    "trainer",
                    None,
                ) is not None
                else run_dir_expected
            )
            last_checkpoint = (
                partial_run_dir
                / "weights"
                / "last.pt"
            )
            best_checkpoint = (
                partial_run_dir
                / "weights"
                / "best.pt"
            )

            raise RuntimeError(
                "YOLO training stopped before successful completion.\n\n"
                f"Run directory: {partial_run_dir}\n"
                f"best.pt exists: {best_checkpoint.exists()}\n"
                f"last.pt exists: {last_checkpoint.exists()}\n\n"
                f"Original error: {error}"
            ) from error

        run_dir = Path(
            model.trainer.save_dir
        )
        best_checkpoint = (
            run_dir
            / "weights"
            / "best.pt"
        )
        last_checkpoint = (
            run_dir
            / "weights"
            / "last.pt"
        )

        if best_checkpoint.exists():
            selected_checkpoint = (
                best_checkpoint
            )
            completion_message = (
                "Training completed successfully.\n"
                f"Best model: {best_checkpoint}"
            )
        elif last_checkpoint.exists():
            selected_checkpoint = (
                last_checkpoint
            )
            completion_message = (
                "Training completed, but best.pt was not produced.\n"
                f"Using last.pt instead: {last_checkpoint}"
            )
        else:
            raise RuntimeError(
                "Training returned, but no model checkpoint was found.\n\n"
                f"Expected folder: {run_dir / 'weights'}"
            )

        self.messages.put(
            (
                "train_log",
                completion_message,
            )
        )
        self.messages.put(
            (
                "trained",
                str(run_dir),
            )
        )
        self._set_status(
            1.0,
            (
                "Training completed — "
                f"{selected_checkpoint.name} generated"
            ),
        )

    def _training_complete(self, run_dir: Path) -> None:
        self.state_data.run_dir = run_dir

        best_checkpoint = (
            run_dir
            / "weights"
            / "best.pt"
        )
        last_checkpoint = (
            run_dir
            / "weights"
            / "last.pt"
        )

        if best_checkpoint.exists():
            selected_checkpoint = best_checkpoint
            checkpoint_label = "Best model"
        elif last_checkpoint.exists():
            selected_checkpoint = last_checkpoint
            checkpoint_label = "Last model"
        else:
            messagebox.showerror(
                APP_NAME,
                (
                    "Training finished, but no checkpoint was found in:\n"
                    f"{run_dir / 'weights'}"
                ),
            )
            return

        self.state_data.weights_path = (
            selected_checkpoint
        )
        self.state_data.detector = None

        self.weights_entry.delete(
            0,
            "end",
        )
        self.weights_entry.insert(
            0,
            str(selected_checkpoint),
        )

        self.model_label.configure(
            text=(
                f"Model: "
                f"{selected_checkpoint.name}"
            )
        )

        self._refresh_results()
        self.tabs.set("6. Results")

        messagebox.showinfo(
            APP_NAME,
            (
                "Training completed.\n\n"
                f"{checkpoint_label}:\n"
                f"{selected_checkpoint}"
            ),
        )

    # ---------- results ----------
    def _load_results_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select YOLO run folder")
        if folder:
            self.state_data.run_dir = Path(folder)
            self._refresh_results()

    def _refresh_results(self) -> None:
        run_dir = self.state_data.run_dir
        if run_dir is None or not run_dir.exists():
            return
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        preferred = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png", "labels.jpg", "labels_correlogram.jpg"]
        for name in preferred:
            if (run_dir / name).exists():
                self.result_tree.insert("", "end", values=(name,))
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                frame = pd.read_csv(csv_path)
                if not frame.empty:
                    row = frame.iloc[-1]
                    text = "Final recorded metrics:\n\n" + "\n".join(f"{column.strip()}: {value}" for column, value in row.items() if pd.notna(value))
                    self.metric_text.delete("1.0", "end")
                    self.metric_text.insert("1.0", text)
            except Exception as exc:
                self.metric_text.delete("1.0", "end")
                self.metric_text.insert("1.0", f"Could not read results.csv:\n{exc}")

    def _select_result(self, _event: Any) -> None:
        selection = self.result_tree.selection()
        if not selection or self.state_data.run_dir is None:
            return
        name = self.result_tree.item(selection[0], "values")[0]
        image = Image.open(self.state_data.run_dir / name).convert("RGB")
        cw = max(self.result_canvas.winfo_width(), 300)
        ch = max(self.result_canvas.winfo_height(), 300)
        scale = min(cw / image.width, ch / image.height)
        resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self.state_data.result_photo = photo
        self.result_canvas.delete("all")
        self.result_canvas.create_image(cw / 2, ch / 2, anchor="center", image=photo)

    # ---------- model load and webcam ----------
    def _load_weights(self) -> None:
        selected = filedialog.askopenfilename(title="Select YOLO weights", filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if not selected:
            return
        self.state_data.weights_path = Path(selected)
        self.state_data.detector = None
        self.weights_entry.delete(0, "end")
        self.weights_entry.insert(0, selected)
        self.model_label.configure(text=f"Model: {Path(selected).name}")

    def _get_detector(self) -> Any:
        text = self.weights_entry.get().strip()
        if not text and self.state_data.weights_path is not None:
            text = str(self.state_data.weights_path)

        if not text:
            raise RuntimeError(
                "Train a detector or load a valid .pt model."
            )

        if self.state_data.detector is None:
            self.state_data.detector = self._load_yolo_checkpoint(
                text,
                operation_name="webcam detection",
            )

            resolved_path = Path(text).expanduser()
            if resolved_path.exists():
                self.state_data.weights_path = resolved_path

        return self.state_data.detector

    def _webcam_detection(self) -> None:
        model = self._get_detector()
        confidence = float(self.conf_entry.get())
        camera_index = int(self.camera_entry.get())
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1.")
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError("Webcam could not be opened.")
        self.stop_event.clear()
        previous = ""
        try:
            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    continue
                result = model.predict(source=frame, conf=confidence, verbose=False)[0]
                annotated = result.plot()
                counts: Counter[str] = Counter()
                if result.boxes is not None and len(result.boxes) > 0:
                    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                    for class_id in class_ids:
                        counts[str(result.names[class_id])] += 1
                summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items())) if counts else "No trained object detected"
                if summary != previous:
                    self.messages.put(("detections", summary))
                    previous = summary
                cv2.putText(annotated, "Multiple objects/classes detected simultaneously", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(annotated, "Press Q to stop", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("Multi-Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
            self.stop_event.set()

    def _stop_webcam(self) -> None:
        self.stop_event.set()
        self._set_status(0, "Webcam stopped")

    def _close(self) -> None:
        self.stop_event.set()
        self.destroy()


def main() -> None:
    MultiObjectTrainer().mainloop()


if __name__ == "__main__":
    main()
