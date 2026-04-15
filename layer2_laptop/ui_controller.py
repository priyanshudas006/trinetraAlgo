"""Tkinter dashboard for TRINETRA mission control."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk


class UIController:
    def __init__(self, main_system) -> None:
        self.main = main_system
        self.root = tk.Tk()
        self.root.title("TRINETRA Control Dashboard")
        self.root.geometry("1220x760")

        self.state_var = tk.StringVar(value="State: IDLE")
        self.mode_var = tk.StringVar(value="target")
        self.info_var = tk.StringVar(value="No map loaded")

        self._build_layout()
        self.main.set_ui_state_callback(self.update_state)

    def _build_layout(self) -> None:
        sidebar = tk.Frame(self.root, width=280, bg="#1d1f22")
        sidebar.pack(side="left", fill="y")

        def add_btn(text, command):
            tk.Button(sidebar, text=text, command=command, bg="#2b2e33", fg="white", relief="flat").pack(fill="x", padx=12, pady=6)

        tk.Label(sidebar, text="TRINETRA", bg="#1d1f22", fg="white", font=("Segoe UI", 16, "bold")).pack(pady=12)

        tk.Radiobutton(sidebar, text="Target Mode", variable=self.mode_var, value="target", command=self._on_mode_change, bg="#1d1f22", fg="white", selectcolor="#1d1f22").pack(anchor="w", padx=14)
        tk.Radiobutton(sidebar, text="Surveillance Mode", variable=self.mode_var, value="surveillance", command=self._on_mode_change, bg="#1d1f22", fg="white", selectcolor="#1d1f22").pack(anchor="w", padx=14, pady=(0, 12))

        add_btn("Load Drone Snapshot", self.load_snapshot)
        add_btn("Select Target Node", self.select_target)
        add_btn("Upload Target Images", self.upload_images)
        add_btn("Start Mission", self.start_mission)
        add_btn("Stop Mission", self.stop_mission)
        add_btn("Refresh Camera", self.refresh_camera)

        tk.Label(sidebar, textvariable=self.state_var, bg="#1d1f22", fg="#9bd67d", wraplength=250, justify="left").pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(sidebar, textvariable=self.info_var, bg="#1d1f22", fg="#e5e5e5", wraplength=250, justify="left").pack(fill="x", padx=14)

        main_area = tk.Frame(self.root, bg="#121416")
        main_area.pack(side="right", fill="both", expand=True)

        self.map_panel = tk.Label(main_area, bg="#121416")
        self.map_panel.pack(fill="both", expand=True, padx=8, pady=8)

    def _on_mode_change(self) -> None:
        self.main.set_mode(self.mode_var.get())
        self.info_var.set(f"Mode set to: {self.mode_var.get()}")

    def update_state(self, state_name: str) -> None:
        self.state_var.set(f"State: {state_name}")

    def load_snapshot(self) -> None:
        ok, message = self.main.load_drone_snapshot()
        self.info_var.set(message)
        if ok:
            self.show_image(self.main.get_display_map())

    def select_target(self) -> None:
        ok, message = self.main.select_target_from_map()
        self.info_var.set(message)
        if ok:
            self.show_image(self.main.get_display_map())

    def upload_images(self) -> None:
        paths = filedialog.askopenfilenames(title="Select 2-5 target images")
        if not paths:
            self.info_var.set("Image upload canceled")
            return
        ok, message = self.main.set_target_images(list(paths))
        self.info_var.set(message)

    def start_mission(self) -> None:
        ok, message = self.main.start_mission()
        self.info_var.set(message)
        if not ok:
            messagebox.showerror("TRINETRA", message)

    def stop_mission(self) -> None:
        ok, message = self.main.stop_mission()
        self.info_var.set(message)

    def refresh_camera(self) -> None:
        frame = self.main.get_camera_frame()
        if frame is None:
            self.info_var.set("Camera unavailable")
            return
        self.show_image(frame)

    def show_image(self, frame) -> None:
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (900, 680))
        image = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(image)
        self.map_panel.configure(image=tk_img)
        self.map_panel.image = tk_img

    def start(self) -> None:
        self.root.mainloop()