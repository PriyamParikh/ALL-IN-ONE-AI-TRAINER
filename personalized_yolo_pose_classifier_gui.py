"""Easy personalized pose classifier using YOLO keypoints + scikit-learn."""
from __future__ import annotations

import json, math, os, queue, re, threading, time, traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import customtkinter as ctk
import cv2, joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

APP_NAME = "Personalized YOLO Pose Classifier"
APP_VERSION = "2.0 Easy Pose"
PROJECT_FILE = "pose_project.json"
SAMPLES_FILE = "pose_samples.csv"
DEFAULT_WEIGHTS = "yolo11n-pose.pt"
MODEL_NAMES = ["Random Forest", "Decision Tree", "SVM", "KNN", "Logistic Regression", "Neural Network"]
KEYPOINTS = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
OFFICIAL_MODELS = {f"yolo11{x}-pose.pt" for x in "nsm lx".replace(" ","")} | {f"yolo26{x}-pose.pt" for x in "nsm lx".replace(" ","")}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_") or "Pose"


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a-b, c-b
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if den <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, float(np.dot(ba, bc) / den)))
    return math.degrees(math.acos(cosine)) / 180.0


def feature_names() -> list[str]:
    names = [f"{k}_{v}" for k in KEYPOINTS for v in ("x","y")]
    names += [f"{k}_confidence" for k in KEYPOINTS]
    names += ["left_elbow_angle","right_elbow_angle","left_shoulder_angle","right_shoulder_angle","left_knee_angle","right_knee_angle","left_hip_angle","right_hip_angle","wrist_distance","ankle_distance","shoulder_distance","hip_distance","left_forearm_length","right_forearm_length","left_wrist_above_shoulder","right_wrist_above_shoulder","left_wrist_above_nose","right_wrist_above_nose"]
    return names

FEATURE_NAMES = feature_names()


def extract_features(xyn: np.ndarray, conf: Optional[np.ndarray]) -> np.ndarray:
    xy = np.asarray(xyn, dtype=np.float32)
    if xy.shape != (17,2):
        raise ValueError(f"Expected 17 keypoints, received {xy.shape}.")
    confidence = np.ones(17, np.float32) if conf is None else np.asarray(conf, np.float32).reshape(-1)
    if confidence.size != 17:
        raise ValueError("Expected 17 keypoint confidence values.")
    confidence = np.clip(confidence, 0, 1)
    visible = confidence >= 0.15
    if np.count_nonzero(visible) < 7:
        raise ValueError("Too few keypoints visible. Move farther from camera.")
    shoulder_mid = (xy[5]+xy[6])/2
    hip_mid = (xy[11]+xy[12])/2
    centre = hip_mid if visible[11] and visible[12] else shoulder_mid
    scales = [float(np.max(np.ptp(xy[visible], axis=0)))]
    if visible[5] and visible[6]: scales.append(float(np.linalg.norm(xy[5]-xy[6])))
    if visible[11] and visible[12]: scales.append(float(np.linalg.norm(xy[11]-xy[12])))
    scale = max([s for s in scales if np.isfinite(s) and s > 1e-4] or [1.0])
    norm = (xy-centre)/scale
    norm[~visible] = 0
    def dist(i,j): return float(np.linalg.norm(norm[i]-norm[j])) if visible[i] and visible[j] else 0.0
    def ang(i,j,k): return angle(norm[i],norm[j],norm[k]) if visible[i] and visible[j] and visible[k] else 0.0
    extras = np.array([ang(5,7,9),ang(6,8,10),ang(7,5,11),ang(8,6,12),ang(11,13,15),ang(12,14,16),ang(5,11,13),ang(6,12,14),dist(9,10),dist(15,16),dist(5,6),dist(11,12),dist(7,9),dist(8,10),float(norm[5,1]-norm[9,1]) if visible[5] and visible[9] else 0,float(norm[6,1]-norm[10,1]) if visible[6] and visible[10] else 0,float(norm[0,1]-norm[9,1]) if visible[0] and visible[9] else 0,float(norm[0,1]-norm[10,1]) if visible[0] and visible[10] else 0], np.float32)
    vector = np.concatenate([norm.reshape(-1), confidence, extras]).astype(np.float32)
    if vector.size != len(FEATURE_NAMES) or not np.all(np.isfinite(vector)):
        raise ValueError("Invalid pose feature vector.")
    return vector


@dataclass
class Evaluation:
    name: str
    model: Any
    y_train: np.ndarray
    y_test: np.ndarray
    train_pred: np.ndarray
    test_pred: np.ndarray
    test_prob: np.ndarray
    metrics: dict[str,float]


@dataclass
class State:
    project: Optional[Path] = None
    classes: list[str] = field(default_factory=list)
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    yolo: Any = None
    yolo_weights: str = ""
    encoder: Optional[LabelEncoder] = None
    evaluations: dict[str,Evaluation] = field(default_factory=dict)
    best: Optional[str] = None
    loaded_package: Optional[dict[str,Any]] = None
    figure: Optional[Figure] = None
    canvas: Optional[FigureCanvasTkAgg] = None
    last_frame: Optional[np.ndarray] = None

class PoseGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue")
        self.title(f"{APP_NAME} — Version {APP_VERSION}"); self.geometry("1500x900"); self.minsize(1180,740)
        self.state_data = State(); self.messages = queue.Queue(); self.capture_stop = threading.Event(); self.live_stop = threading.Event()
        self._header(); self._tabs(); self._status(); self.after(100,self._poll); self.protocol("WM_DELETE_WINDOW",self._close)

    def _header(self):
        f=ctk.CTkFrame(self); f.pack(fill="x",padx=12,pady=(12,6))
        ctk.CTkLabel(f,text="PERSONALIZED YOLO POSE CLASSIFIER",font=ctk.CTkFont(size=25,weight="bold")).pack(side="left",padx=18,pady=12)
        ctk.CTkLabel(f,text="Name Poses • Record Samples • Train • Recognise Live").pack(side="left",padx=12)
        self.project_label=ctk.CTkLabel(f,text="Project: Not created",font=ctk.CTkFont(weight="bold")); self.project_label.pack(side="right",padx=18)

    def _tabs(self):
        self.tabs=ctk.CTkTabview(self); self.tabs.pack(fill="both",expand=True,padx=12,pady=6)
        for n in ("1. Setup & Pose Names","2. Capture Samples","3. Train Models","4. Results & Graphs","5. Real-Time Test"): self.tabs.add(n)
        self._setup_tab(); self._capture_tab(); self._train_tab(); self._results_tab(); self._live_tab()

    def _status(self):
        f=ctk.CTkFrame(self); f.pack(fill="x",padx=12,pady=(6,12))
        self.status=ctk.CTkLabel(f,text="Ready",anchor="w"); self.status.pack(side="left",padx=12,pady=8,fill="x",expand=True)
        self.progress=ctk.CTkProgressBar(f,width=330); self.progress.set(0); self.progress.pack(side="right",padx=12,pady=8)

    def _setup_tab(self):
        t=self.tabs.tab("1. Setup & Pose Names"); t.grid_columnconfigure(1,weight=1); t.grid_rowconfigure(7,weight=1)
        ctk.CTkLabel(t,text="Step 1 — Create Project and Enter Pose Names",font=ctk.CTkFont(size=22,weight="bold")).grid(row=0,column=0,columnspan=3,padx=20,pady=(20,12),sticky="w")
        ctk.CTkLabel(t,text="Project folder").grid(row=1,column=0,padx=20,pady=9,sticky="w")
        self.project_entry=ctk.CTkEntry(t); self.project_entry.grid(row=1,column=1,padx=20,pady=9,sticky="ew")
        ctk.CTkButton(t,text="Browse",width=120,command=self._browse_project).grid(row=1,column=2,padx=20,pady=9)
        ctk.CTkLabel(t,text="Pose class names").grid(row=2,column=0,padx=20,pady=9,sticky="w")
        self.class_entry=ctk.CTkEntry(t,placeholder_text="Hands Up, Hands on Waist"); self.class_entry.grid(row=2,column=1,padx=20,pady=9,sticky="ew")
        ctk.CTkLabel(t,text="Enter 2–5 pose names separated by commas.").grid(row=3,column=1,padx=20,pady=(0,8),sticky="w")
        ctk.CTkLabel(t,text="YOLO pose model").grid(row=4,column=0,padx=20,pady=9,sticky="w")
        self.weights_entry=ctk.CTkEntry(t); self.weights_entry.insert(0,DEFAULT_WEIGHTS); self.weights_entry.grid(row=4,column=1,padx=20,pady=9,sticky="ew")
        ctk.CTkButton(t,text="Browse .pt",width=120,command=self._browse_weights).grid(row=4,column=2,padx=20,pady=9)
        f=ctk.CTkFrame(t); f.grid(row=5,column=0,columnspan=3,padx=20,pady=12,sticky="ew")
        ctk.CTkButton(f,text="CREATE NEW PROJECT",height=42,command=self._create_project).pack(side="left",padx=10,pady=10)
        ctk.CTkButton(f,text="LOAD EXISTING PROJECT",height=42,command=self._load_project).pack(side="left",padx=10,pady=10)
        ctk.CTkButton(f,text="LOAD / TEST YOLO MODEL",height=42,command=self._load_yolo_async).pack(side="left",padx=10,pady=10)
        ctk.CTkLabel(t,text="Example: Hands Up, Hands on Waist. Record at least 100 accepted samples per pose.",wraplength=1100,justify="left").grid(row=6,column=0,columnspan=3,padx=20,pady=8,sticky="w")
        self.setup_log=ctk.CTkTextbox(t); self.setup_log.grid(row=7,column=0,columnspan=3,padx=20,pady=(8,20),sticky="nsew")
        self.setup_log.insert("1.0","1. Select a folder.\n2. Enter 2–5 pose names.\n3. Create the project.\n4. Go to Step 2 and record every pose.\n\nNo manual keypoint annotation or data.yaml is required.")

    def _capture_tab(self):
        t=self.tabs.tab("2. Capture Samples"); t.grid_columnconfigure(1,weight=1); t.grid_rowconfigure(7,weight=1)
        ctk.CTkLabel(t,text="Step 2 — Select a Pose and Record Samples",font=ctk.CTkFont(size=22,weight="bold")).grid(row=0,column=0,columnspan=4,padx=20,pady=(20,12),sticky="w")
        ctk.CTkLabel(t,text="Pose class").grid(row=1,column=0,padx=20,pady=8,sticky="w")
        self.capture_class=ctk.StringVar(value="Create a project first"); self.class_menu=ctk.CTkOptionMenu(t,values=["Create a project first"],variable=self.capture_class); self.class_menu.grid(row=1,column=1,padx=20,pady=8,sticky="ew")
        ctk.CTkLabel(t,text="Accepted samples required").grid(row=2,column=0,padx=20,pady=8,sticky="w")
        self.target_entry=ctk.CTkEntry(t); self.target_entry.insert(0,"100"); self.target_entry.grid(row=2,column=1,padx=20,pady=8,sticky="ew")
        ctk.CTkLabel(t,text="Sampling interval (seconds)").grid(row=3,column=0,padx=20,pady=8,sticky="w")
        self.interval_entry=ctk.CTkEntry(t); self.interval_entry.insert(0,"0.15"); self.interval_entry.grid(row=3,column=1,padx=20,pady=8,sticky="ew")
        f=ctk.CTkFrame(t); f.grid(row=4,column=0,columnspan=4,padx=20,pady=10,sticky="ew")
        self.camera_entry=self._small_field(f,"Camera","0"); self.conf_entry=self._small_field(f,"YOLO confidence","0.35"); self.imgsz_entry=self._small_field(f,"Image size","640"); self.device_entry=self._small_field(f,"Device","cpu")
        self.mirror_var=ctk.BooleanVar(value=True); ctk.CTkCheckBox(f,text="Mirror camera",variable=self.mirror_var).pack(side="left",padx=14,pady=12)
        self.save_images_var=ctk.BooleanVar(value=True); ctk.CTkCheckBox(f,text="Save accepted images",variable=self.save_images_var).pack(side="left",padx=8,pady=12)
        c=ctk.CTkFrame(t); c.grid(row=5,column=0,columnspan=4,padx=20,pady=10,sticky="ew")
        ctk.CTkButton(c,text="START CAPTURE",height=44,command=self._start_capture).pack(side="left",padx=10,pady=10)
        ctk.CTkButton(c,text="STOP CAPTURE",height=44,command=self.capture_stop.set).pack(side="left",padx=10,pady=10)
        ctk.CTkButton(c,text="CLEAR SELECTED CLASS",command=self._clear_class).pack(side="left",padx=10,pady=10)
        ctk.CTkButton(c,text="EXPORT CSV",command=self._export_csv).pack(side="left",padx=10,pady=10)
        self.capture_count=ctk.CTkLabel(c,text="Accepted this session: 0/0",font=ctk.CTkFont(weight="bold")); self.capture_count.pack(side="right",padx=15)
        self.count_label=ctk.CTkLabel(t,text="Current dataset: No project",font=ctk.CTkFont(weight="bold"),anchor="w"); self.count_label.grid(row=6,column=0,columnspan=4,padx=20,pady=8,sticky="ew")
        self.capture_log=ctk.CTkTextbox(t); self.capture_log.grid(row=7,column=0,columnspan=4,padx=20,pady=(8,20),sticky="nsew")
        self.capture_log.insert("1.0","Select a class and click START CAPTURE. Maintain the pose while making small natural variations. Samples are accepted only when enough body keypoints are visible. Press Q to stop.")

    def _small_field(self,parent,label,default):
        ctk.CTkLabel(parent,text=label).pack(side="left",padx=(12,4)); e=ctk.CTkEntry(parent,width=78); e.insert(0,default); e.pack(side="left",padx=6,pady=12); return e

    def _train_tab(self):
        t=self.tabs.tab("3. Train Models"); t.grid_columnconfigure(0,weight=1); t.grid_rowconfigure(5,weight=1)
        ctk.CTkLabel(t,text="Step 3 — Train Four or Five Pose Classifiers",font=ctk.CTkFont(size=22,weight="bold")).grid(row=0,column=0,padx=20,pady=(20,12),sticky="w")
        f=ctk.CTkFrame(t); f.grid(row=1,column=0,padx=20,pady=10,sticky="ew")
        self.model_vars={}
        for i,name in enumerate(MODEL_NAMES):
            var=ctk.BooleanVar(value=i<5); self.model_vars[name]=var
            ctk.CTkCheckBox(f,text=name,variable=var).grid(row=i//3,column=i%3,padx=18,pady=12,sticky="w")
        s=ctk.CTkFrame(t); s.grid(row=2,column=0,padx=20,pady=10,sticky="ew")
        self.test_entry=self._small_field(s,"Test fraction","0.20"); self.random_entry=self._small_field(s,"Random state","42"); self.default_threshold_entry=self._small_field(s,"Real-time threshold","0.65")
        ctk.CTkButton(t,text="TRAIN AND EVALUATE SELECTED MODELS",height=48,font=ctk.CTkFont(size=15,weight="bold"),command=self._start_training).grid(row=3,column=0,padx=20,pady=15,sticky="ew")
        self.train_count_label=ctk.CTkLabel(t,text="Dataset: No project",font=ctk.CTkFont(weight="bold"),anchor="w"); self.train_count_label.grid(row=4,column=0,padx=20,pady=8,sticky="ew")
        self.train_log=ctk.CTkTextbox(t); self.train_log.grid(row=5,column=0,padx=20,pady=(8,20),sticky="nsew")

    def _results_tab(self):
        t=self.tabs.tab("4. Results & Graphs"); t.grid_columnconfigure(0,weight=2); t.grid_columnconfigure(1,weight=3); t.grid_rowconfigure(1,weight=1)
        c=ctk.CTkFrame(t); c.grid(row=0,column=0,columnspan=2,padx=20,pady=(20,10),sticky="ew")
        ctk.CTkLabel(c,text="Model").pack(side="left",padx=(12,4)); self.result_model=ctk.StringVar(value="No trained models"); self.result_menu=ctk.CTkOptionMenu(c,values=["No trained models"],variable=self.result_model,width=190); self.result_menu.pack(side="left",padx=8,pady=10)
        ctk.CTkLabel(c,text="Graph").pack(side="left",padx=(16,4)); self.graph_type=ctk.StringVar(value="Testing Confusion Matrix")
        self.graph_menu=ctk.CTkOptionMenu(c,values=["Training Confusion Matrix","Testing Confusion Matrix","Testing ROC Curve","Testing Precision-Recall Curve","PCA Pose Scatter","Model Comparison","Feature Importance"],variable=self.graph_type,width=235); self.graph_menu.pack(side="left",padx=8,pady=10)
        ctk.CTkButton(c,text="Generate Graph",command=self._generate_graph).pack(side="left",padx=8,pady=10); ctk.CTkButton(c,text="Save Graph",command=self._save_graph).pack(side="left",padx=8,pady=10); ctk.CTkButton(c,text="Save Best Model",command=self._save_model).pack(side="left",padx=8,pady=10); ctk.CTkButton(c,text="Load Model",command=self._load_model).pack(side="left",padx=8,pady=10)
        lf=ctk.CTkFrame(t); lf.grid(row=1,column=0,padx=(20,10),pady=(0,20),sticky="nsew"); lf.grid_rowconfigure(0,weight=1); lf.grid_columnconfigure(0,weight=1)
        cols=("Model","Accuracy","Precision","Recall","F1","Balanced","ROC-AUC","PR-AUC"); self.tree=ttk.Treeview(lf,columns=cols,show="headings")
        for col in cols: self.tree.heading(col,text=col); self.tree.column(col,width=105 if col!="Model" else 150,anchor="center")
        self.tree.grid(row=0,column=0,padx=8,pady=8,sticky="nsew"); self.best_label=ctk.CTkLabel(lf,text="Best model: —",font=ctk.CTkFont(weight="bold"),anchor="w"); self.best_label.grid(row=1,column=0,padx=10,pady=8,sticky="ew")
        self.graph_frame=ctk.CTkFrame(t); self.graph_frame.grid(row=1,column=1,padx=(10,20),pady=(0,20),sticky="nsew"); self.graph_frame.grid_rowconfigure(0,weight=1); self.graph_frame.grid_columnconfigure(0,weight=1)

    def _live_tab(self):
        t=self.tabs.tab("5. Real-Time Test"); t.grid_columnconfigure(1,weight=1); t.grid_rowconfigure(4,weight=1)
        ctk.CTkLabel(t,text="Step 5 — Recognise Your Named Poses",font=ctk.CTkFont(size=22,weight="bold")).grid(row=0,column=0,columnspan=3,padx=20,pady=(20,12),sticky="w")
        ctk.CTkLabel(t,text="Active classifier").grid(row=1,column=0,padx=20,pady=8,sticky="w"); self.active_model=ctk.CTkLabel(t,text="Train a model or load a .joblib file.",anchor="w"); self.active_model.grid(row=1,column=1,padx=20,pady=8,sticky="ew"); ctk.CTkButton(t,text="Load .joblib",width=120,command=self._load_model).grid(row=1,column=2,padx=20,pady=8)
        f=ctk.CTkFrame(t); f.grid(row=2,column=0,columnspan=3,padx=20,pady=10,sticky="ew")
        self.live_camera=self._small_field(f,"Camera","0"); self.live_yolo_conf=self._small_field(f,"YOLO confidence","0.35"); self.live_threshold=self._small_field(f,"Pose threshold","0.65"); self.live_imgsz=self._small_field(f,"Image size","640"); self.live_device=self._small_field(f,"Device","cpu")
        self.live_mirror=ctk.BooleanVar(value=True); ctk.CTkCheckBox(f,text="Mirror camera",variable=self.live_mirror).pack(side="left",padx=14,pady=12)
        c=ctk.CTkFrame(t); c.grid(row=3,column=0,columnspan=3,padx=20,pady=10,sticky="ew")
        ctk.CTkButton(c,text="START REAL-TIME RECOGNITION",height=44,command=self._start_live).pack(side="left",padx=10,pady=10); ctk.CTkButton(c,text="STOP",height=44,command=self.live_stop.set).pack(side="left",padx=10,pady=10); ctk.CTkButton(c,text="Save Snapshot",command=self._save_snapshot).pack(side="left",padx=10,pady=10)
        self.live_summary=ctk.CTkLabel(c,text="Predictions: —",font=ctk.CTkFont(weight="bold")); self.live_summary.pack(side="right",padx=15)
        self.live_log=ctk.CTkTextbox(t); self.live_log.grid(row=4,column=0,columnspan=3,padx=20,pady=(8,20),sticky="nsew")

    # Project management
    def _browse_project(self):
        path=filedialog.askdirectory(title="Select pose project folder")
        if path: self.project_entry.delete(0,"end"); self.project_entry.insert(0,path)

    def _browse_weights(self):
        path=filedialog.askopenfilename(title="Select YOLO pose weights",filetypes=[("PyTorch weights","*.pt"),("All files","*.*")])
        if path: self.weights_entry.delete(0,"end"); self.weights_entry.insert(0,path); self.state_data.yolo=None; self.state_data.yolo_weights=""

    def _parse_classes(self):
        values=[]
        for text in self.class_entry.get().split(","):
            text=text.strip()
            if text and text not in values: values.append(text)
        if not 2 <= len(values) <= 5: raise ValueError("Enter 2–5 unique pose class names.")
        return values

    def _empty_samples(self): return pd.DataFrame(columns=["session_id","class_name","sample_id"]+FEATURE_NAMES)

    def _create_project(self):
        try:
            if not self.project_entry.get().strip(): raise ValueError("Select a project folder.")
            project=Path(self.project_entry.get().strip()); classes=self._parse_classes(); project.mkdir(parents=True,exist_ok=True); (project/"captured_images").mkdir(exist_ok=True)
            for name in classes: (project/"captured_images"/safe_name(name)).mkdir(parents=True,exist_ok=True)
            config={"application":APP_NAME,"version":APP_VERSION,"class_names":classes,"pose_weights":self.weights_entry.get().strip() or DEFAULT_WEIGHTS,"feature_names":FEATURE_NAMES,"created_at":time.strftime("%Y-%m-%d %H:%M:%S")}
            (project/PROJECT_FILE).write_text(json.dumps(config,indent=2),encoding="utf-8"); samples=self._empty_samples(); samples.to_csv(project/SAMPLES_FILE,index=False)
            self.state_data.project=project; self.state_data.classes=classes; self.state_data.samples=samples; self.state_data.evaluations={}; self.state_data.best=None; self.state_data.loaded_package=None; self._apply_project(); messagebox.showinfo(APP_NAME,f"Pose project created:\n{project}")
        except Exception as e: messagebox.showerror(APP_NAME,str(e))

    def _load_project(self):
        path=filedialog.askopenfilename(title="Select pose_project.json",filetypes=[("Pose project",PROJECT_FILE),("JSON","*.json")])
        if not path: return
        try:
            config_path=Path(path); config=json.loads(config_path.read_text(encoding="utf-8")); classes=list(config["class_names"]); project=config_path.parent; samples=pd.read_csv(project/SAMPLES_FILE) if (project/SAMPLES_FILE).exists() else self._empty_samples()
            self.state_data.project=project; self.state_data.classes=classes; self.state_data.samples=samples; self.state_data.evaluations={}; self.state_data.best=None; self.state_data.loaded_package=None
            self.project_entry.delete(0,"end"); self.project_entry.insert(0,str(project)); self.class_entry.delete(0,"end"); self.class_entry.insert(0,", ".join(classes)); self.weights_entry.delete(0,"end"); self.weights_entry.insert(0,config.get("pose_weights",DEFAULT_WEIGHTS)); self._apply_project()
        except Exception as e: messagebox.showerror(APP_NAME,f"Project loading failed:\n{e}")

    def _apply_project(self):
        self.class_menu.configure(values=self.state_data.classes); self.capture_class.set(self.state_data.classes[0]); self.project_label.configure(text=f"Project: {self.state_data.project.name}")
        self.setup_log.delete("1.0","end"); self.setup_log.insert("1.0",f"Project: {self.state_data.project}\nPose names: {', '.join(self.state_data.classes)}\nYOLO model: {self.weights_entry.get().strip()}\nFeatures per sample: {len(FEATURE_NAMES)}\n\nNext: record every pose in Step 2."); self._update_counts()

    def _require_project(self):
        if self.state_data.project is None: raise RuntimeError("Create or load a project first.")
        return self.state_data.project

    def _save_samples(self): self.state_data.samples.to_csv(self._require_project()/SAMPLES_FILE,index=False)

    def _update_counts(self):
        parts=[]; total=0
        for name in self.state_data.classes:
            count=int((self.state_data.samples["class_name"]==name).sum()) if not self.state_data.samples.empty else 0; parts.append(f"{name}: {count}"); total+=count
        text=" | ".join(parts)+f" | Total: {total}"; self.count_label.configure(text=f"Current dataset: {text}"); self.train_count_label.configure(text=f"Dataset: {text}")

    # YOLO loading
    def _load_yolo_async(self):
        threading.Thread(target=self._thread,args=(lambda:self._load_yolo_worker(self.weights_entry.get().strip() or DEFAULT_WEIGHTS),),daemon=True).start()

    def _load_yolo_worker(self,weights):
        self._set_status(.1,"Loading YOLO pose model..."); self._get_yolo(weights); self.messages.put(("setup_log",f"YOLO pose model loaded successfully:\n{weights}")); self._set_status(1,"YOLO pose model ready")

    def _get_yolo(self,weights):
        if self.state_data.yolo is None or self.state_data.yolo_weights != weights:
            if YOLO is None: raise RuntimeError("Ultralytics is not installed. Run: python -m pip install ultralytics")
            candidate=Path(weights).expanduser(); official=candidate.name.lower() in OFFICIAL_MODELS; simple=candidate.name==weights and candidate.parent==Path(".")
            if not candidate.exists() and not (official and simple): raise RuntimeError(f"Pose model not found:\n{candidate}\nUse {DEFAULT_WEIGHTS} or a valid -pose.pt file.")
            try: model=YOLO(str(candidate)) if candidate.exists() else YOLO(candidate.name)
            except Exception as e: raise RuntimeError(f"Could not load YOLO pose model:\n{weights}\n\n{e}") from e
            task=str(getattr(model,"task","")).lower()
            if task and task!="pose": raise RuntimeError(f"Selected model task is '{task}', not pose. Use a -pose.pt model.")
            self.state_data.yolo=model; self.state_data.yolo_weights=weights
        return self.state_data.yolo

    # Capture
    def _start_capture(self):
        try:
            project=self._require_project(); class_name=self.capture_class.get()
            if class_name not in self.state_data.classes: raise ValueError("Select a valid pose class.")
            settings={"project":project,"class":class_name,"target":int(self.target_entry.get()),"interval":float(self.interval_entry.get()),"camera":int(self.camera_entry.get()),"conf":float(self.conf_entry.get()),"imgsz":int(self.imgsz_entry.get()),"device":self.device_entry.get().strip(),"mirror":bool(self.mirror_var.get()),"save_images":bool(self.save_images_var.get()),"weights":self.weights_entry.get().strip() or DEFAULT_WEIGHTS}
            if settings["target"]<10: raise ValueError("Capture at least 10 samples per session.")
            if settings["interval"]<.03: raise ValueError("Sampling interval must be at least 0.03 seconds.")
            if not 0<settings["conf"]<1: raise ValueError("YOLO confidence must be between 0 and 1.")
            self.capture_stop.clear(); self.capture_count.configure(text=f"Accepted this session: 0/{settings['target']}")
            threading.Thread(target=self._thread,args=(lambda:self._capture_worker(settings),),daemon=True).start()
        except Exception as e: messagebox.showerror(APP_NAME,str(e))

    def _capture_worker(self,s):
        model=self._get_yolo(s["weights"]); cap=cv2.VideoCapture(s["camera"],cv2.CAP_DSHOW)
        if not cap.isOpened(): cap.release(); cap=cv2.VideoCapture(s["camera"])
        if not cap.isOpened(): raise RuntimeError(f"Webcam {s['camera']} could not be opened.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
        session=time.strftime("%Y%m%d_%H%M%S"); rows=[]; accepted=rejected=0; last=0.0; image_dir=s["project"]/"captured_images"/safe_name(s["class"]); image_dir.mkdir(parents=True,exist_ok=True)
        self.messages.put(("capture_log",f"Capture started for {s['class']}. Target: {s['target']} accepted samples. Only the largest person is recorded."))
        try:
            while not self.capture_stop.is_set() and accepted<s["target"]:
                ok,frame=cap.read()
                if not ok: continue
                if s["mirror"]: frame=cv2.flip(frame,1)
                args={"source":frame,"conf":s["conf"],"imgsz":s["imgsz"],"verbose":False}
                if s["device"]: args["device"]=s["device"]
                result=model.predict(**args)[0]; annotated=result.plot(); status="Waiting for one visible person"
                if result.keypoints is not None and result.boxes is not None and len(result.keypoints)>0:
                    boxes=result.boxes.xyxy.detach().cpu().numpy(); person=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))); now=time.perf_counter()
                    if now-last>=s["interval"]:
                        xyn=result.keypoints.xyn[person].detach().cpu().numpy(); ct=getattr(result.keypoints,"conf",None); conf=ct[person].detach().cpu().numpy() if ct is not None else None
                        try:
                            vec=extract_features(xyn,conf); sample_id=f"{session}_{accepted+1:05d}"; row={"session_id":session,"class_name":s["class"],"sample_id":sample_id}; row.update({n:float(v) for n,v in zip(FEATURE_NAMES,vec)}); rows.append(row); accepted+=1; last=now; status="Sample accepted"
                            if s["save_images"]: cv2.imwrite(str(image_dir/f"{safe_name(s['class'])}_{sample_id}.jpg"),frame)
                            self.messages.put(("capture_progress",(accepted,s["target"],rejected)))
                        except ValueError as e: rejected+=1; status=str(e)
                cv2.putText(annotated,f"Pose: {s['class']} | Accepted: {accepted}/{s['target']}",(12,30),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(annotated,status[:90],(12,60),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(annotated,"Maintain pose + make small variations | Q = stop",(12,88),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow("Personalized Pose Sample Capture",annotated)
                if cv2.waitKey(1)&0xFF==ord("q"): break
        finally: cap.release(); cv2.destroyAllWindows(); self.capture_stop.set()
        if rows: self.state_data.samples=pd.concat([self.state_data.samples,pd.DataFrame(rows)],ignore_index=True); self._save_samples()
        self.messages.put(("capture_complete",(s["class"],accepted,rejected)))

    def _clear_class(self):
        try:
            self._require_project(); name=self.capture_class.get()
            if not messagebox.askyesno(APP_NAME,f"Delete all samples for:\n{name}?"): return
            self.state_data.samples=self.state_data.samples[self.state_data.samples["class_name"]!=name].reset_index(drop=True); self._save_samples(); self._update_counts()
        except Exception as e: messagebox.showerror(APP_NAME,str(e))

    def _export_csv(self):
        if self.state_data.samples.empty: messagebox.showwarning(APP_NAME,"No pose samples are available."); return
        path=filedialog.asksaveasfilename(title="Export pose samples",defaultextension=".csv",initialfile="personalized_pose_samples.csv",filetypes=[("CSV","*.csv")])
        if path: self.state_data.samples.to_csv(path,index=False); messagebox.showinfo(APP_NAME,f"Samples exported:\n{path}")

    # Training
    def _selected_models(self):
        selected=[n for n,v in self.model_vars.items() if v.get()]
        if len(selected) not in (4,5): raise ValueError("Select exactly four or five models.")
        return selected

    def _make_model(self,name,seed):
        if name=="Random Forest": return RandomForestClassifier(n_estimators=250,class_weight="balanced",random_state=seed,n_jobs=-1)
        if name=="Decision Tree": return DecisionTreeClassifier(class_weight="balanced",random_state=seed)
        if name=="SVM": return Pipeline([("scaler",StandardScaler()),("classifier",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=seed))])
        if name=="KNN": return Pipeline([("scaler",StandardScaler()),("classifier",KNeighborsClassifier(n_neighbors=5,weights="distance"))])
        if name=="Logistic Regression": return Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=seed))])
        if name=="Neural Network": return Pipeline([("scaler",StandardScaler()),("classifier",MLPClassifier(hidden_layer_sizes=(128,64),max_iter=1200,early_stopping=True,random_state=seed))])
        raise ValueError(name)

    def _auc(self,y,prob,n):
        try:
            b=label_binarize(y,classes=np.arange(n)); b=np.column_stack([1-b[:,0],b[:,0]]) if n==2 else b
            return float(roc_auc_score(b,prob,average="macro",multi_class="ovr")),float(average_precision_score(b,prob,average="macro"))
        except Exception: return float("nan"),float("nan")

    def _start_training(self):
        try:
            self._require_project(); settings={"models":self._selected_models(),"test":float(self.test_entry.get()),"seed":int(self.random_entry.get())}
            if not .1<=settings["test"]<=.4: raise ValueError("Test fraction must be 0.10–0.40.")
            threading.Thread(target=self._thread,args=(lambda:self._training_worker(settings),),daemon=True).start()
        except Exception as e: messagebox.showerror(APP_NAME,str(e))

    def _training_worker(self,s):
        data=self.state_data.samples.copy()
        if data.empty: raise RuntimeError("No samples. Complete Step 2 first.")
        counts=data["class_name"].value_counts(); bad=[n for n in self.state_data.classes if int(counts.get(n,0))<20]
        if bad: raise RuntimeError("Record at least 20 accepted samples for every class.\nInsufficient: "+", ".join(bad))
        X=data[FEATURE_NAMES].to_numpy(np.float32); encoder=LabelEncoder(); y=encoder.fit_transform(data["class_name"].astype(str)); session_counts=data.groupby("class_name")["session_id"].nunique(); use_groups=len(session_counts)==len(self.state_data.classes) and session_counts.min()>=2
        if use_groups:
            splitter=GroupShuffleSplit(n_splits=1,test_size=s["test"],random_state=s["seed"]); tr,te=next(splitter.split(X,y,groups=data["session_id"].astype(str))); Xtr,Xte,ytr,yte=X[tr],X[te],y[tr],y[te]; split="session-wise"
        else:
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=s["test"],random_state=s["seed"],stratify=y); split="stratified random"
        self.messages.put(("train_log",f"Classes: {', '.join(encoder.classes_)}\nTraining: {len(Xtr)} | Testing: {len(Xte)} | Split: {split}\nFeatures: {X.shape[1]}")); evaluations={}
        for i,name in enumerate(s["models"],1):
            self._set_status((i-1)/len(s["models"]),f"Training {name}..."); model=self._make_model(name,s["seed"]); model.fit(Xtr,ytr); train_pred=model.predict(Xtr); test_pred=model.predict(Xte); prob=model.predict_proba(Xte); roc_auc,pr_auc=self._auc(yte,prob,len(encoder.classes_)); metrics={"accuracy":float(accuracy_score(yte,test_pred)),"precision":float(precision_score(yte,test_pred,average="weighted",zero_division=0)),"recall":float(recall_score(yte,test_pred,average="weighted",zero_division=0)),"f1":float(f1_score(yte,test_pred,average="weighted",zero_division=0)),"balanced":float(balanced_accuracy_score(yte,test_pred)),"roc_auc":roc_auc,"pr_auc":pr_auc}; evaluations[name]=Evaluation(name,model,ytr,yte,train_pred,test_pred,prob,metrics); self.messages.put(("train_log",f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"))
        best=max(evaluations,key=lambda n:(evaluations[n].metrics["f1"],evaluations[n].metrics["balanced"])); self.state_data.encoder=encoder; self.state_data.evaluations=evaluations; self.state_data.best=best; self.state_data.loaded_package=None; self.messages.put(("training_complete",{"models":list(evaluations),"best":best})); self._set_status(1,f"Training complete — {best}")

    # Results
    def _refresh_tree(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for name,e in self.state_data.evaluations.items():
            m=e.metrics; self.tree.insert("","end",values=(name,f"{m['accuracy']:.4f}",f"{m['precision']:.4f}",f"{m['recall']:.4f}",f"{m['f1']:.4f}",f"{m['balanced']:.4f}",f"{m['roc_auc']:.4f}" if np.isfinite(m['roc_auc']) else "N/A",f"{m['pr_auc']:.4f}" if np.isfinite(m['pr_auc']) else "N/A"))

    def _clear_figure(self):
        if self.state_data.canvas is not None: self.state_data.canvas.get_tk_widget().destroy(); self.state_data.canvas=None
        if self.state_data.figure is not None: plt.close(self.state_data.figure); self.state_data.figure=None

    def _show_figure(self,fig):
        self._clear_figure(); canvas=FigureCanvasTkAgg(fig,master=self.graph_frame); canvas.draw(); canvas.get_tk_widget().grid(row=0,column=0,padx=8,pady=8,sticky="nsew"); self.state_data.figure=fig; self.state_data.canvas=canvas

    def _evaluation(self):
        name=self.result_model.get()
        if name not in self.state_data.evaluations: raise RuntimeError("Select a trained model.")
        return self.state_data.evaluations[name]

    def _generate_graph(self):
        try:
            graph=self.graph_type.get(); e=self._evaluation(); enc=self.state_data.encoder
            if enc is None: raise RuntimeError("No label encoder.")
            fig=Figure(figsize=(8.2,6.2),dpi=100); ax=fig.add_subplot(111)
            if graph in ("Training Confusion Matrix","Testing Confusion Matrix"):
                y=e.y_train if graph.startswith("Training") else e.y_test; p=e.train_pred if graph.startswith("Training") else e.test_pred; cm=confusion_matrix(y,p,labels=np.arange(len(enc.classes_))); im=ax.imshow(cm); fig.colorbar(im,ax=ax); ax.set_xticks(np.arange(len(enc.classes_)),labels=enc.classes_,rotation=35,ha="right"); ax.set_yticks(np.arange(len(enc.classes_)),labels=enc.classes_); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{graph} — {e.name}")
                threshold=cm.max()/2 if cm.size else 0
                for r in range(cm.shape[0]):
                    for c in range(cm.shape[1]): ax.text(c,r,str(cm[r,c]),ha="center",va="center",color="white" if cm[r,c]>threshold else "black")
            elif graph=="Testing ROC Curve":
                n=len(enc.classes_); binary=label_binarize(e.y_test,classes=np.arange(n)); binary=np.column_stack([1-binary[:,0],binary[:,0]]) if n==2 else binary
                for i,name in enumerate(enc.classes_): fpr,tpr,_=roc_curve(binary[:,i],e.test_prob[:,i]); ax.plot(fpr,tpr,label=str(name))
                ax.plot([0,1],[0,1],linestyle="--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(f"Testing ROC — {e.name}"); ax.legend()
            elif graph=="Testing Precision-Recall Curve":
                n=len(enc.classes_); binary=label_binarize(e.y_test,classes=np.arange(n)); binary=np.column_stack([1-binary[:,0],binary[:,0]]) if n==2 else binary
                for i,name in enumerate(enc.classes_): precision,recall,_=precision_recall_curve(binary[:,i],e.test_prob[:,i]); ax.plot(recall,precision,label=str(name))
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Testing Precision–Recall — {e.name}"); ax.legend()
            elif graph=="PCA Pose Scatter":
                data=self.state_data.samples; X=StandardScaler().fit_transform(data[FEATURE_NAMES].to_numpy(np.float32)); pts=PCA(n_components=2).fit_transform(X); labels=data["class_name"].astype(str).to_numpy()
                for name in self.state_data.classes: mask=labels==name; ax.scatter(pts[mask,0],pts[mask,1],label=name,alpha=.7)
                ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2"); ax.set_title("PCA Scatter of Pose Samples"); ax.legend()
            elif graph=="Model Comparison":
                names=list(self.state_data.evaluations); acc=[self.state_data.evaluations[n].metrics["accuracy"] for n in names]; f1=[self.state_data.evaluations[n].metrics["f1"] for n in names]; x=np.arange(len(names)); w=.36; ax.bar(x-w/2,acc,w,label="Accuracy"); ax.bar(x+w/2,f1,w,label="F1"); ax.set_xticks(x,labels=names,rotation=30,ha="right"); ax.set_ylim(0,1.05); ax.set_title("Pose Classifier Comparison"); ax.legend()
            elif graph=="Feature Importance":
                est=e.model.steps[-1][1] if isinstance(e.model,Pipeline) else e.model; importance=None
                if hasattr(est,"feature_importances_"): importance=np.asarray(est.feature_importances_)
                elif hasattr(est,"coef_"): importance=np.mean(np.abs(np.asarray(est.coef_)),axis=0)
                elif isinstance(est,MLPClassifier): importance=np.mean(np.abs(est.coefs_[0]),axis=1)
                if importance is None: raise RuntimeError(f"Feature importance unavailable for {e.name}.")
                idx=np.argsort(importance)[-20:]; ax.barh(np.arange(len(idx)),importance[idx]); ax.set_yticks(np.arange(len(idx)),labels=[FEATURE_NAMES[i] for i in idx]); ax.set_title(f"Top Pose Features — {e.name}")
            fig.tight_layout(); self._show_figure(fig)
        except Exception as ex: messagebox.showerror(APP_NAME,str(ex))

    def _save_graph(self):
        if self.state_data.figure is None: messagebox.showwarning(APP_NAME,"Generate a graph first."); return
        path=filedialog.asksaveasfilename(title="Save graph",defaultextension=".png",filetypes=[("PNG","*.png"),("PDF","*.pdf"),("JPEG","*.jpg")])
        if path: self.state_data.figure.savefig(path,dpi=300,bbox_inches="tight"); messagebox.showinfo(APP_NAME,f"Graph saved:\n{path}")

    def _save_model(self):
        if self.state_data.best is None or self.state_data.encoder is None: messagebox.showwarning(APP_NAME,"Train models first."); return
        path=filedialog.asksaveasfilename(title="Save personalized pose classifier",defaultextension=".joblib",initialfile="personalized_pose_classifier.joblib",filetypes=[("Joblib","*.joblib")])
        if not path: return
        e=self.state_data.evaluations[self.state_data.best]; package={"application":APP_NAME,"version":APP_VERSION,"model":e.model,"encoder":self.state_data.encoder,"model_name":e.name,"feature_names":FEATURE_NAMES,"pose_weights":self.weights_entry.get().strip() or DEFAULT_WEIGHTS,"mirror":bool(self.mirror_var.get()),"threshold":float(self.default_threshold_entry.get()),"saved_at":time.strftime("%Y-%m-%d %H:%M:%S")}; joblib.dump(package,path); messagebox.showinfo(APP_NAME,f"Model saved:\n{path}")

    def _load_model(self):
        path=filedialog.askopenfilename(title="Load personalized pose classifier",filetypes=[("Joblib","*.joblib"),("All files","*.*")])
        if not path: return
        try:
            package=joblib.load(path)
            for key in ("model","encoder","model_name","feature_names"):
                if key not in package: raise ValueError(f"Missing field: {key}")
            if list(package["feature_names"])!=FEATURE_NAMES: raise ValueError("Incompatible pose feature definition.")
            self.state_data.loaded_package=package; classes=", ".join(map(str,package["encoder"].classes_)); self.active_model.configure(text=f"{package['model_name']} | Classes: {classes}"); self.weights_entry.delete(0,"end"); self.weights_entry.insert(0,package.get("pose_weights",DEFAULT_WEIGHTS)); self.live_threshold.delete(0,"end"); self.live_threshold.insert(0,str(package.get("threshold",.65))); self.live_mirror.set(bool(package.get("mirror",True))); messagebox.showinfo(APP_NAME,f"Pose classifier loaded.\nModel: {package['model_name']}\nClasses: {classes}")
        except Exception as ex: messagebox.showerror(APP_NAME,f"Model loading failed:\n{ex}")

    # Real-time
    def _active_package(self):
        if self.state_data.loaded_package is not None: return self.state_data.loaded_package
        if self.state_data.best is not None and self.state_data.encoder is not None:
            e=self.state_data.evaluations[self.state_data.best]; return {"model":e.model,"encoder":self.state_data.encoder,"model_name":e.name,"feature_names":FEATURE_NAMES,"pose_weights":self.weights_entry.get().strip() or DEFAULT_WEIGHTS,"mirror":bool(self.mirror_var.get()),"threshold":float(self.default_threshold_entry.get())}
        raise RuntimeError("Train a model or load a .joblib classifier first.")

    def _start_live(self):
        try:
            settings={"package":self._active_package(),"camera":int(self.live_camera.get()),"yolo_conf":float(self.live_yolo_conf.get()),"threshold":float(self.live_threshold.get()),"imgsz":int(self.live_imgsz.get()),"device":self.live_device.get().strip(),"mirror":bool(self.live_mirror.get())}
            if not 0<settings["yolo_conf"]<1 or not 0<settings["threshold"]<1: raise ValueError("Confidence values must be between 0 and 1.")
            self.live_stop.clear(); threading.Thread(target=self._thread,args=(lambda:self._live_worker(settings),),daemon=True).start()
        except Exception as ex: messagebox.showerror(APP_NAME,str(ex))

    def _live_worker(self,s):
        package=s["package"]; model=package["model"]; encoder=package["encoder"]; yolo=self._get_yolo(package.get("pose_weights",DEFAULT_WEIGHTS)); cap=cv2.VideoCapture(s["camera"],cv2.CAP_DSHOW)
        if not cap.isOpened(): cap.release(); cap=cv2.VideoCapture(s["camera"])
        if not cap.isOpened(): raise RuntimeError(f"Webcam {s['camera']} could not be opened.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540); previous=""
        try:
            while not self.live_stop.is_set():
                ok,frame=cap.read()
                if not ok: continue
                if s["mirror"]: frame=cv2.flip(frame,1)
                args={"source":frame,"conf":s["yolo_conf"],"imgsz":s["imgsz"],"verbose":False}
                if s["device"]: args["device"]=s["device"]
                result=yolo.predict(**args)[0]; annotated=result.plot(); summaries=[]
                if result.keypoints is not None and result.boxes is not None and len(result.keypoints)>0:
                    boxes=result.boxes.xyxy.detach().cpu().numpy(); xyn=result.keypoints.xyn.detach().cpu().numpy(); ct=getattr(result.keypoints,"conf",None); conf=ct.detach().cpu().numpy() if ct is not None else None
                    for i in range(len(xyn)):
                        try:
                            vec=extract_features(xyn[i],conf[i] if conf is not None else None); prob=model.predict_proba([vec])[0]; idx=int(np.argmax(prob)); score=float(prob[idx]); name=str(encoder.inverse_transform([idx])[0]); display=name if score>=s["threshold"] else "Uncertain"; x1,y1,x2,y2=boxes[i].astype(int); cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,255),2); cv2.putText(annotated,f"{display}: {score*100:.1f}%",(x1,max(25,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA); summaries.append(f"Person {i+1}: {display} ({score*100:.1f}%)")
                        except ValueError: summaries.append(f"Person {i+1}: insufficient keypoints")
                summary=" | ".join(summaries) if summaries else "No valid pose detected"; cv2.putText(annotated,"Press Q to stop",(12,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA); self.state_data.last_frame=annotated.copy(); self.messages.put(("live_summary",summary))
                if summary!=previous: self.messages.put(("live_log",f"{time.strftime('%H:%M:%S')} — {summary}")); previous=summary
                cv2.imshow("Personalized Named Pose Recognition",annotated)
                if cv2.waitKey(1)&0xFF==ord("q"): break
        finally: cap.release(); cv2.destroyAllWindows(); self.live_stop.set(); self._set_status(0,"Real-time recognition stopped")

    def _save_snapshot(self):
        if self.state_data.last_frame is None: messagebox.showwarning(APP_NAME,"No recognition frame available."); return
        path=filedialog.asksaveasfilename(title="Save recognition snapshot",defaultextension=".png",initialfile=f"recognized_pose_{time.strftime('%Y%m%d_%H%M%S')}.png",filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if path: cv2.imwrite(path,self.state_data.last_frame); messagebox.showinfo(APP_NAME,f"Snapshot saved:\n{path}")

    # Thread / UI messages
    def _thread(self,target):
        try: target()
        except (ValueError,RuntimeError,FileNotFoundError) as ex: self.messages.put(("error",str(ex)))
        except Exception as ex: self.messages.put(("error",f"{ex}\n\n{traceback.format_exc()}"))

    def _set_status(self,value,text): self.messages.put(("status",(value,text)))

    def _poll(self):
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=="status": value,text=payload; self.progress.set(max(0,min(1,float(value)))); self.status.configure(text=str(text))
                elif kind=="setup_log": self.setup_log.insert("end","\n"+str(payload)+"\n"); self.setup_log.see("end")
                elif kind=="capture_log": self.capture_log.insert("end","\n"+str(payload)+"\n"); self.capture_log.see("end")
                elif kind=="capture_progress": accepted,target,rejected=payload; self.capture_count.configure(text=f"Accepted this session: {accepted}/{target} | Rejected: {rejected}"); self.progress.set(accepted/max(target,1))
                elif kind=="capture_complete": name,accepted,rejected=payload; self.capture_log.insert("end",f"\nCapture complete for {name}. Accepted: {accepted}, Rejected: {rejected}\n"); self._update_counts(); self._set_status(1,f"Capture complete for {name}")
                elif kind=="train_log": self.train_log.insert("end",str(payload)+"\n"); self.train_log.see("end")
                elif kind=="training_complete": models=payload["models"]; best=payload["best"]; self.result_menu.configure(values=models); self.result_model.set(best); self.best_label.configure(text=f"Best model: {best}"); self.active_model.configure(text=f"Newly trained: {best} | Classes: {', '.join(self.state_data.encoder.classes_)}"); self._refresh_tree(); self.tabs.set("4. Results & Graphs"); messagebox.showinfo(APP_NAME,f"Training complete.\nBest model: {best}")
                elif kind=="live_summary": self.live_summary.configure(text=f"Predictions: {payload}")
                elif kind=="live_log": self.live_log.insert("end",str(payload)+"\n"); self.live_log.see("end")
                elif kind=="error": self.progress.set(0); self.status.configure(text="Operation failed"); messagebox.showerror(APP_NAME,str(payload))
        except queue.Empty: pass
        self.after(100,self._poll)

    def _close(self):
        self.capture_stop.set(); self.live_stop.set(); self._clear_figure(); self.destroy()


if __name__ == "__main__":
    PoseGUI().mainloop()
