import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import sys
from ultralytics import YOLO
import requests
import json
from pathlib import Path
import threading
import time

# -----------------------------
# Load YOLO model ONCE
# -----------------------------
MODEL_PATH = "../weights/best.pt"
model = YOLO(MODEL_PATH)

# -----------------------------
# Persistent printers storage
# -----------------------------
PRINTERS_FILE = "printers.json"

def load_printers():
    if Path(PRINTERS_FILE).exists():
        try:
            with open(PRINTERS_FILE, "r") as f:
                data = json.load(f)
                return data
        except json.JSONDecodeError:
            return {}
    return {}

def save_printers(printers_dict):
    with open(PRINTERS_FILE, "w") as f:
        json.dump(printers_dict, f, indent=2)

# -----------------------------
# Printer configuration popup
# -----------------------------
class PrinterConfigPopup:
    def __init__(self, parent, cam_index, callback, name="", ip=""):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Printer config for Camera {cam_index}")
        self.callback = callback

        tk.Label(self.top, text="Printer Name:").pack(pady=5)
        self.name_entry = tk.Entry(self.top)
        self.name_entry.pack()
        self.name_entry.insert(0, name)

        tk.Label(self.top, text="Printer IP:").pack(pady=5)
        self.ip_entry = tk.Entry(self.top)
        self.ip_entry.pack()
        self.ip_entry.insert(0, ip)

        tk.Button(self.top, text="Save", command=self.save).pack(pady=10)

        self.top.transient(parent)
        self.top.grab_set()
        parent.update_idletasks()
        w, h = 300, 150
        x = parent.winfo_rootx() + (parent.winfo_width() - w)//2
        y = parent.winfo_rooty() + (parent.winfo_height() - h)//2
        self.top.geometry(f"{w}x{h}+{x}+{y}")

    def save(self):
        name = self.name_entry.get().strip()
        ip = self.ip_entry.get().strip()
        if name and ip:
            self.callback(name, ip)
            self.top.destroy()
        else:
            messagebox.showwarning("Missing info", "Please enter both Name and IP")

# -----------------------------
# Camera feed tab class
# -----------------------------
class CameraTab:
    def __init__(self, notebook, cam_index, printer_info=None, printers_dict=None):
        self.cam_index = cam_index
        self.printer_info = printer_info
        self.printers_dict = printers_dict

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.running = True

        self.current_frame = None       # raw frame from camera
        self.annotated_frame = None     # frame with YOLO annotations
        self.max_conf = 0

        tab_title = printer_info.get("name") if printer_info else f"Camera {cam_index}"
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=tab_title)

        # Left control panel
        self.control_frame = ttk.Frame(self.frame, width=200)
        self.control_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.terminate_button = ttk.Button(self.control_frame, text="Terminate Print",
                                           command=self.terminate_print)
        self.terminate_button.pack(pady=10)

        self.edit_button = ttk.Button(self.control_frame, text="Edit Printer",
                                      command=self.edit_printer)
        self.edit_button.pack(pady=10)

        self.conf_label = ttk.Label(self.control_frame, text="Max Confidence: 0%")
        self.conf_label.pack(pady=10)

        self.status_canvas = tk.Canvas(self.control_frame, width=20, height=20)
        self.status_canvas.pack(pady=5)
        self.status_circle = self.status_canvas.create_oval(2, 2, 18, 18, fill="green")

        # Camera feed on right
        self.camera_frame = ttk.Frame(self.frame)
        self.camera_frame.pack(side="right", fill="both", expand=True)

        self.display_width = 800
        self.display_height = 600
        self.label = tk.Label(self.camera_frame, width=self.display_width, height=self.display_height)
        self.label.pack(expand=True)

        # Start threads
        threading.Thread(target=self.frame_capture_loop, daemon=True).start()
        threading.Thread(target=self.yolo_loop, daemon=True).start()
        self.update_video_frame()

    def frame_capture_loop(self):
        """Continuously capture frames from camera"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
            else:
                time.sleep(0.01)

    def yolo_loop(self):
        """Continuously run YOLO on the latest frame"""
        while self.running:
            if self.current_frame is not None:
                frame_copy = self.current_frame.copy()
                results = model.predict(frame_copy, imgsz=640, verbose=False)
                annotated = results[0].plot() if results else frame_copy

                max_conf = 0
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        confidence = float(box.conf)
                        if confidence > max_conf:
                            max_conf = confidence

                self.annotated_frame = annotated
                self.max_conf = max_conf

                # Update GUI safely
                self.label.after(0, self.update_status)
            else:
                time.sleep(0.01)

    def update_video_frame(self):
        """Display latest annotated frame or raw frame"""
        if self.running:
            frame_to_show = self.annotated_frame if self.annotated_frame is not None else self.current_frame
            if frame_to_show is not None:
                resized = cv2.resize(frame_to_show, (self.display_width, self.display_height))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.config(image=imgtk)
            self.label.after(15, self.update_video_frame)  # ~60 FPS

    def update_status(self):
        self.conf_label.config(text=f"Max Confidence: {self.max_conf*100:.1f}%")
        self.status_canvas.itemconfig(self.status_circle, fill="red" if self.max_conf >= 0.85 else "green")

    def terminate_print(self):
        if not self.printer_info:
            messagebox.showerror("Error", "Printer info not set for this camera.")
            return

        printer_ip = self.printer_info.get("ip")
        if not printer_ip:
            messagebox.showerror("Error", "Printer IP missing.")
            return

        url = f"http://{printer_ip}/api/job"
        headers = {"Content-Type": "application/json"}
        payload = {"command": "cancel"}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            if response.status_code in [200, 204]:
                messagebox.showinfo("Success", f"Print terminated on {self.printer_info.get('name')}")
            else:
                messagebox.showerror("Failed", f"Printer responded with status {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to terminate print: {e}")

    def edit_printer(self):
        def save_new(name, ip):
            self.printer_info["name"] = name
            self.printer_info["ip"] = ip
            self.printers_dict[str(self.cam_index)] = self.printer_info
            save_printers(self.printers_dict)
            tab_index = self.frame.master.index(self.frame)
            self.frame.master.tab(tab_index, text=name)

        PrinterConfigPopup(self.frame, self.cam_index, save_new,
                           name=self.printer_info.get("name", ""),
                           ip=self.printer_info.get("ip", ""))

    def stop(self):
        self.running = False
        self.cap.release()

# -----------------------------
# Main App
# -----------------------------
class App:
    def __init__(self, root, camera_count):
        self.root = root
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.printers = load_printers()
        self.tabs = []

        # Home tab
        self.home_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.home_tab, text="Home")
        ttk.Label(self.home_tab, text="Available Cameras", font=("Arial", 14)).pack(pady=10)

        self.add_camera_buttons(camera_count)

    def add_camera_buttons(self, camera_count):
        for i in range(camera_count):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok, _ = cap.read()
            cap.release()
            if ok:
                button = ttk.Button(self.home_tab, text=f"Open Camera {i}", command=lambda cam=i: self.open_camera(cam))
                button.pack(pady=5)
            else:
                ttk.Label(self.home_tab, text=f"Camera {i} not found", foreground="red").pack()

    def open_camera(self, cam_index):
        key = str(cam_index)
        if key in self.printers:
            tab = CameraTab(self.notebook, cam_index, printer_info=self.printers[key], printers_dict=self.printers)
            self.tabs.append(tab)
        else:
            def save_printer(name, ip):
                self.printers[key] = {"name": name, "ip": ip}
                save_printers(self.printers)
                tab = CameraTab(self.notebook, cam_index, printer_info=self.printers[key], printers_dict=self.printers)
                self.tabs.append(tab)
            PrinterConfigPopup(self.root, cam_index, save_printer)

    def stop_all_tabs(self):
        for tab in self.tabs:
            tab.stop()

# -----------------------------
# Program start
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <camera_count>")
        sys.exit(1)

    camera_count = int(sys.argv[1])
    root = tk.Tk()
    root.title("YOLO Multi-Camera Viewer with Printer Control")
    root.geometry("1200x800")

    app = App(root, camera_count)

    def on_close():
        app.stop_all_tabs()
        save_printers(app.printers)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
