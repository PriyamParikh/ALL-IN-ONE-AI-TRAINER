
"""
ALL IN ONE AI — 5-IN-1 STEM TRAINER
===================================

Five independent AI-learning studios behind one color-coded launcher:
1. Image classification
2. YOLO multi-object detection
3. Personalized pose classification
4. Personalized audio classification
5. CSV/Excel tabular data machine learning
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

APP_NAME = "All in One AI"
APP_VERSION = "1.0 Production"

MODULES = [
    {
        "number": "01",
        "title": "IMAGE AI",
        "subtitle": "Single-Image Classifier",
        "description": "Teach the computer to assign one name to a complete image or webcam frame.",
        "features": "2–5 classes • Webcam capture • 4/5 ML models • CF, ROC, PR, PCA • .joblib",
        "script": "personalized_image_ai_trainer_gui_v3_manual_hog.py",
        "color": "#2563EB",
        "hover": "#1D4ED8",
        "requirements": ["customtkinter", "cv2", "numpy", "pandas", "matplotlib", "sklearn", "joblib"],
    },
    {
        "number": "02",
        "title": "OBJECT AI",
        "subtitle": "YOLO Multi-Object Detector",
        "description": "Draw boxes, train YOLO, and detect several objects in the same camera frame.",
        "features": "2–5 objects • Annotation • YOLO training • Evaluation curves • best.pt",
        "script": "personalized_multi_object_ai_trainer_gui_v3.py",
        "color": "#16A34A",
        "hover": "#15803D",
        "requirements": ["customtkinter", "cv2", "pandas", "PIL", "yaml", "torch", "ultralytics"],
    },
    {
        "number": "03",
        "title": "POSE AI",
        "subtitle": "Personalized Pose Classifier",
        "description": "Name your own body poses, record skeleton samples, and recognise poses live.",
        "features": "2–5 poses • 17 YOLO keypoints • 4/5 ML models • Explainable graphs • .joblib",
        "script": "personalized_yolo_pose_classifier_gui.py",
        "color": "#7C3AED",
        "hover": "#6D28D9",
        "requirements": ["customtkinter", "cv2", "numpy", "pandas", "matplotlib", "sklearn", "joblib", "torch", "ultralytics"],
    },
    {
        "number": "04",
        "title": "AUDIO AI",
        "subtitle": "Microphone Sound Classifier",
        "description": "Record sounds with the built-in microphone and recognise them continuously.",
        "features": "2–5 sounds • MFCC features • 4/5 ML models • Research graphs • .joblib",
        "script": "personalized_audio_classifier_gui.py",
        "color": "#EA580C",
        "hover": "#C2410C",
        "requirements": ["customtkinter", "sounddevice", "numpy", "scipy", "pandas", "matplotlib", "sklearn", "joblib"],
    },
    {
        "number": "05",
        "title": "DATA AI",
        "subtitle": "CSV & Excel ML Trainer",
        "description": "Upload a table, choose the answer column, and automatically train classification or regression models.",
        "features": "CSV/Excel • Auto types • Train/validate/test • 4/5 ML models • Batch prediction",
        "script": "all_in_one_ai_data_trainer_gui.py",
        "color": "#0891B2",
        "hover": "#0E7490",
        "requirements": ["customtkinter", "numpy", "scipy", "pandas", "openpyxl", "matplotlib", "sklearn", "joblib"],
    },
]


class AllInOneAILauncher(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} — 5-in-1 STEM Trainer")
        self.geometry("1500x940")
        self.minsize(1180, 760)
        self.base_dir = Path(__file__).resolve().parent

        self._build_header()
        self._build_workspace()
        self._build_footer()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, corner_radius=18)
        header.pack(fill="x", padx=18, pady=(18, 8))

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", fill="x", expand=True, padx=22, pady=18)

        ctk.CTkLabel(
            title_frame,
            text="ALL IN ONE AI",
            font=ctk.CTkFont(size=34, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_frame,
            text="A friendly 5-in-1 artificial-intelligence laboratory for STEM learners",
            font=ctk.CTkFont(size=16),
        ).pack(anchor="w", pady=(4, 0))

        badge = ctk.CTkLabel(
            header,
            text="LEARN • TRAIN • TEST • EXPLAIN",
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=12,
            fg_color=("#E2E8F0", "#334155"),
        )
        badge.pack(side="right", padx=22, pady=18, ipadx=16, ipady=8)

    def _build_workspace(self) -> None:
        self.workspace = ctk.CTkScrollableFrame(self, corner_radius=18)
        self.workspace.pack(fill="both", expand=True, padx=18, pady=8)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(1, weight=1)

        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        for module, (row, column) in zip(MODULES, positions):
            columnspan = 2 if module["number"] == "05" else 1
            self._create_module_card(module, row, column, columnspan)

    def _create_module_card(self, module: dict, row: int, column: int, columnspan: int) -> None:
        card = ctk.CTkFrame(self.workspace, corner_radius=18, border_width=1)
        card.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            padx=10,
            pady=10,
            sticky="nsew",
        )
        card.grid_columnconfigure(1, weight=1)

        color_strip = ctk.CTkFrame(card, width=12, fg_color=module["color"], corner_radius=8)
        color_strip.grid(row=0, column=0, rowspan=5, sticky="ns", padx=(0, 12), pady=0)

        number_badge = ctk.CTkLabel(
            card,
            text=module["number"],
            width=58,
            height=42,
            corner_radius=12,
            fg_color=module["color"],
            text_color="white",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        number_badge.grid(row=0, column=1, padx=(10, 8), pady=(18, 6), sticky="w")

        title_frame = ctk.CTkFrame(card, fg_color="transparent")
        title_frame.grid(row=0, column=2, padx=8, pady=(18, 6), sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            title_frame,
            text=module["title"],
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=module["color"],
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_frame,
            text=module["subtitle"],
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        ctk.CTkLabel(
            card,
            text=module["description"],
            wraplength=560 if columnspan == 1 else 1160,
            justify="left",
            font=ctk.CTkFont(size=14),
        ).grid(row=1, column=1, columnspan=2, padx=10, pady=(2, 8), sticky="w")

        ctk.CTkLabel(
            card,
            text=module["features"],
            wraplength=560 if columnspan == 1 else 1160,
            justify="left",
            font=ctk.CTkFont(size=13),
        ).grid(row=2, column=1, columnspan=2, padx=10, pady=8, sticky="w")

        button_frame = ctk.CTkFrame(card, fg_color="transparent")
        button_frame.grid(row=3, column=1, columnspan=2, padx=10, pady=(10, 18), sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            button_frame,
            text=f"OPEN {module['title']}",
            height=46,
            fg_color=module["color"],
            hover_color=module["hover"],
            font=ctk.CTkFont(size=14, weight="bold"),
            command=lambda selected=module: self._launch(selected),
        ).grid(row=0, column=0, padx=(0, 8), sticky="ew")

        ctk.CTkButton(
            button_frame,
            text="Check",
            width=90,
            height=46,
            fg_color=("#CBD5E1", "#475569"),
            hover_color=("#94A3B8", "#64748B"),
            command=lambda selected=module: self._check_module(selected),
        ).grid(row=0, column=1, padx=(8, 0))

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, corner_radius=18)
        footer.pack(fill="x", padx=18, pady=(8, 18))

        self.status_label = ctk.CTkLabel(
            footer,
            text="Choose a colored AI studio to begin.",
            anchor="w",
            font=ctk.CTkFont(size=14),
        )
        self.status_label.pack(side="left", padx=18, pady=12, fill="x", expand=True)

        ctk.CTkButton(
            footer,
            text="Verify Everything",
            width=150,
            command=self._check_all,
        ).pack(side="right", padx=8, pady=10)
        ctk.CTkButton(
            footer,
            text="Open Suite Folder",
            width=150,
            command=self._open_folder,
        ).pack(side="right", padx=8, pady=10)

    def _launch(self, module: dict) -> None:
        script_path = self.base_dir / module["script"]
        if not script_path.exists():
            messagebox.showerror(APP_NAME, f"Module file not found:\n{script_path}")
            return
        try:
            subprocess.Popen([sys.executable, str(script_path)], cwd=str(self.base_dir))
            self.status_label.configure(text=f"Opened {module['subtitle']}.")
        except Exception as error:
            messagebox.showerror(APP_NAME, f"Could not open {module['title']}:\n{error}")

    @staticmethod
    def _module_status(import_name: str) -> tuple[bool, str]:
        specification = importlib.util.find_spec(import_name)
        if specification is None:
            return False, "Not installed"
        try:
            module = __import__(import_name)
            return True, str(getattr(module, "__version__", "Installed"))
        except Exception as error:
            return False, f"Import failed: {error}"

    def _check_module(self, module: dict) -> None:
        lines = []
        all_available = True
        for import_name in module["requirements"]:
            available, detail = self._module_status(import_name)
            lines.append(f"[{'OK' if available else 'MISSING'}] {import_name}: {detail}")
            all_available = all_available and available
        lines.extend(
            [
                "",
                "This AI studio is ready." if all_available else "Install missing packages using requirements_all_in_one_ai.txt.",
            ]
        )
        if all_available:
            messagebox.showinfo(module["subtitle"], "\n".join(lines))
        else:
            messagebox.showwarning(module["subtitle"], "\n".join(lines))

    def _check_all(self) -> None:
        unique_imports = []
        for module in MODULES:
            for import_name in module["requirements"]:
                if import_name not in unique_imports:
                    unique_imports.append(import_name)
        lines = []
        all_available = True
        for import_name in unique_imports:
            available, detail = self._module_status(import_name)
            lines.append(f"[{'OK' if available else 'MISSING'}] {import_name}: {detail}")
            all_available = all_available and available
        lines.extend(
            [
                "",
                "All five AI studios are ready." if all_available else "Install missing packages using Install_All_in_One_AI.bat.",
            ]
        )
        if all_available:
            messagebox.showinfo("All in One AI Verification", "\n".join(lines))
        else:
            messagebox.showwarning("All in One AI Verification", "\n".join(lines))

    def _open_folder(self) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(self.base_dir))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.base_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.base_dir)])
        except Exception as error:
            messagebox.showerror(APP_NAME, f"Could not open suite folder:\n{error}")


def main() -> None:
    AllInOneAILauncher().mainloop()


if __name__ == "__main__":
    main()
