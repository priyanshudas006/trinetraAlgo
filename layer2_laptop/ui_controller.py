"""Tkinter dashboard for TRINETRA mission control."""

from __future__ import annotations

import tkinter as tk
import time
import threading
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
        self._live_camera = False
        self._camera_thread = None
        self._latest_camera_frame = None
        self._camera_lock = threading.Lock()
        self._camera_error_count = 0
        self._live_source = "rover"

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
        add_btn("Preview Drone Camera", self.preview_drone_camera)
        add_btn("Auto Detect Drone Source", self.auto_detect_drone_source)
        add_btn("Drone Source -", self.drone_source_prev)
        add_btn("Drone Source +", self.drone_source_next)
        add_btn("Select Target Node", self.select_target)
        add_btn("Upload Target Images", self.upload_images)
        add_btn("Start Mission", self.start_mission)
        add_btn("Stop Mission", self.stop_mission)
        add_btn("Refresh Camera", self.refresh_camera)
        add_btn("Start Live Camera", self.start_live_camera)
        add_btn("Stop Live Camera", self.stop_live_camera)
        add_btn("Start Live Drone Feed", self.start_live_drone)
        add_btn("Stop Live Drone Feed", self.stop_live_drone)

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

    def preview_drone_camera(self) -> None:
        frame = self.main.get_drone_frame()
        if frame is None:
            self.info_var.set("Drone camera unavailable. Check DRONE_VIDEO_SOURCE and close Camera app.")
            return
        self.show_image(frame)
        self.info_var.set("Drone camera preview updated")

    def auto_detect_drone_source(self) -> None:
        ok, message = self.main.auto_select_drone_source()
        self.info_var.set(message)
        if ok:
            self.preview_drone_camera()

    def drone_source_prev(self) -> None:
        ok, message = self.main.cycle_drone_source(step=-1)
        self.info_var.set(message)
        if ok:
            self.preview_drone_camera()

    def drone_source_next(self) -> None:
        ok, message = self.main.cycle_drone_source(step=1)
        self.info_var.set(message)
        if ok:
            self.preview_drone_camera()

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
        self.info_var.set("Camera frame refreshed")

    def start_live_camera(self) -> None:
        self._start_live_feed("rover")

    def stop_live_camera(self) -> None:
        if self._live_source == "rover":
            self._stop_live_feed("rover")
        else:
            self.info_var.set("Rover live feed is not active")

    def start_live_drone(self) -> None:
        self._start_live_feed("drone")

    def stop_live_drone(self) -> None:
        if self._live_source == "drone":
            self._stop_live_feed("drone")
        else:
            self.info_var.set("Drone live feed is not active")

    def _start_live_feed(self, source: str) -> None:
        source_name = "drone" if source == "drone" else "rover"
        if self._live_camera:
            if self._live_source == source:
                self.info_var.set(f"Live {source_name} feed already running")
                return
            self._live_camera = False
            time.sleep(0.1)
        self._live_camera = True
        self._live_source = source
        self._camera_error_count = 0
        self._camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self._camera_thread.start()
        self._schedule_live_render()
        self.info_var.set(f"Live {source_name} feed started")

    def _stop_live_feed(self, source_name: str) -> None:
        self._live_camera = False
        self.info_var.set(f"Live {source_name} feed stopped")

    def _camera_worker(self) -> None:
        while self._live_camera:
            if self._live_source == "drone":
                frame = self.main.get_drone_frame()
            else:
                frame = self.main.get_camera_frame()
            if frame is None:
                self._camera_error_count += 1
                time.sleep(0.15)
                continue
            with self._camera_lock:
                self._latest_camera_frame = frame
            time.sleep(0.03)

    def _schedule_live_render(self) -> None:
        if not self._live_camera:
            return
        frame = None
        with self._camera_lock:
            if self._latest_camera_frame is not None:
                frame = self._latest_camera_frame.copy()
        if frame is not None:
            self.show_image(frame)
        elif self._camera_error_count > 20:
            self.info_var.set(f"Live {self._live_source} feed unstable/unavailable")
        self.root.after(90, self._schedule_live_render)

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
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        self._live_camera = False
        self.root.destroy()
