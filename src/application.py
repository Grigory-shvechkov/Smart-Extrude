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

        # Make popup modal
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
# Camera feed tab class WITH YOLO and printer controls
# -----------------------------
class CameraTab:
    def __init__(self, notebook, cam_index, printer_info=None, printers_dict=None):
        self.cam_index = cam_index
        self.printer_info = printer_info
        self.printers_dict = printers_dict
        self.running = True
        self.latest_frame = None
        self.max_conf = 0
        self.terminated_due_threshold = False

        # Auto terminate variables
        self.auto_terminate = tk.BooleanVar(value=False)
        self.confirmed_threshold = 0.85
        self.terminate_threshold_var = tk.StringVar(value=str(self.confirmed_threshold))

        # Setup camera
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        tab_title = printer_info.get("name") if printer_info else f"Camera {cam_index}"
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=tab_title)

        # Left control panel
        self.control_frame = ttk.Frame(self.frame, width=200)
        self.control_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.terminate_button = ttk.Button(self.control_frame, text="Terminate Print",
                                           command=self.terminate_print)
        self.terminate_button.pack(pady=5)

        self.restart_button = ttk.Button(self.control_frame, text="Restart Detection",
                                         command=self.restart_detection)
        self.restart_button.pack(pady=5)

        self.edit_button = ttk.Button(self.control_frame, text="Edit Printer",
                                      command=self.edit_printer)
        self.edit_button.pack(pady=5)

        ttk.Label(self.control_frame, text="Auto Terminate Threshold (0-1)").pack(pady=5)
        self.threshold_entry = tk.Entry(self.control_frame, textvariable=self.terminate_threshold_var)
        self.threshold_entry.pack(pady=5)

        self.confirm_button = ttk.Button(self.control_frame, text="Confirm Threshold", command=self.confirm_threshold)
        self.confirm_button.pack(pady=5)

        self.auto_check = ttk.Checkbutton(self.control_frame, text="Auto Terminate",
                                          variable=self.auto_terminate)
        self.auto_check.pack(pady=5)

        self.conf_label = ttk.Label(self.control_frame, text="Max Confidence: 0%")
        self.conf_label.pack(pady=10)

        self.status_canvas = tk.Canvas(self.control_frame, width=20, height=20)
        self.status_canvas.pack(pady=5)
        self.status_circle = self.status_canvas.create_oval(2, 2, 18, 18, fill="green")

        self.print_status_label = ttk.Label(self.control_frame, text="Print Status: Active")
        self.print_status_label.pack(pady=10)

        # Camera feed on right
        self.camera_frame = ttk.Frame(self.frame)
        self.camera_frame.pack(side="right", fill="both", expand=True)

        self.display_width = 800
        self.display_height = 600
        self.label = tk.Label(self.camera_frame, width=self.display_width, height=self.display_height)
        self.label.pack(expand=True)

        # Start loops
        self.update_video_frame()
        threading.Thread(target=self.run_detection_loop, daemon=True).start()

    def confirm_threshold(self):
        try:
            val = float(self.threshold_entry.get())
            if 0 <= val <= 1:
                self.confirmed_threshold = val
                messagebox.showinfo("Threshold Set", f"Auto-terminate threshold confirmed at {val}")
            else:
                messagebox.showwarning("Invalid Value", "Enter a number between 0 and 1")
        except ValueError:
            messagebox.showwarning("Invalid Value", "Enter a valid number between 0 and 1")

    def update_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            display_frame = self.latest_frame if self.latest_frame is not None else frame
            display_frame = cv2.resize(display_frame, (self.display_width, self.display_height))
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)
        if self.running:
            self.label.after(15, self.update_video_frame)

    def run_detection_loop(self):
        while self.running:
            if self.terminated_due_threshold:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if ret:
                results = model.predict(frame, imgsz=640, verbose=False)
                annotated = results[0].plot() if results else frame
                max_conf = 0
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        confidence = float(box.conf)
                        if confidence > max_conf:
                            max_conf = confidence
                self.latest_frame = annotated
                self.max_conf = max_conf
                self.label.after(0, self.update_status)

                if self.auto_terminate.get() and max_conf >= self.confirmed_threshold:
                    self.terminated_due_threshold = True
                    self.label.after(0, self.auto_terminate_print)
            time.sleep(0.02)

    def update_status(self):
        self.conf_label.config(text=f"Max Confidence: {self.max_conf*100:.1f}%")
        self.status_canvas.itemconfig(self.status_circle, fill="red" if self.max_conf >= 0.85 else "green")

    def auto_terminate_print(self):
        if self.printer_info:
            self.print_status_label.config(text="Print Status: Terminated")
        else:
            self.print_status_label.config(text="Print Status: Terminated (no printer info)")

    def restart_detection(self):
        self.terminated_due_threshold = False
        self.print_status_label.config(text="Print Status: Active")
        self.latest_frame = None

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
                self.terminated_due_threshold = True
                self.print_status_label.config(text="Print Status: Terminated")
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
        self.camera_count = camera_count

        # Home tab
        self.home_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.home_tab, text="Home")
        ttk.Label(self.home_tab, text="Available Cameras", font=("Arial", 14)).pack(pady=10)

        self.camera_frame = ttk.Frame(self.home_tab)
        self.camera_frame.pack()

        self.add_camera_buttons(camera_count)

        # Bottom buttons
        self.bottom_frame = ttk.Frame(self.home_tab)
        self.bottom_frame.pack(pady=20)
        self.reset_button = ttk.Button(self.bottom_frame, text="Reset Printer Data", command=self.reset_data)
        self.reset_button.pack(side="left", padx=10)
        self.refresh_button = ttk.Button(self.bottom_frame, text="Refresh Cameras", command=self.refresh_cameras)
        self.refresh_button.pack(side="left", padx=10)

    def add_camera_buttons(self, camera_count):
        for widget in self.camera_frame.winfo_children():
            widget.destroy()
        for i in range(camera_count):
            if any(tab.cam_index == i for tab in self.tabs):
                continue
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok, _ = cap.read()
            cap.release()
            if ok:
                button = ttk.Button(
                    self.camera_frame,
                    text=f"Open Camera {i}",
                    command=lambda cam=i: self.open_camera(cam)
                )
                button.pack(pady=5)
            else:
                ttk.Label(self.camera_frame, text=f"Camera {i} not found", foreground="red").pack()

    def refresh_cameras(self):
        self.add_camera_buttons(self.camera_count)

    def open_camera(self, cam_index):
        key = str(cam_index)
        for tab in self.tabs:
            if tab.cam_index == cam_index:
                self.notebook.select(tab.frame)
                return
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

    def reset_data(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all printer data?"):
            self.printers = {}
            save_printers(self.printers)
            for tab in self.tabs:
                tab.stop()
            self.tabs.clear()
            messagebox.showinfo("Reset", "All printer data cleared and tabs closed.")
            self.add_camera_buttons(self.camera_count)

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
