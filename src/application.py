import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import sys
from ultralytics import YOLO
import requests

# -----------------------------
# Load YOLO model ONCE
# -----------------------------
MODEL_PATH = "../weights/best.pt"
model = YOLO(MODEL_PATH)

# -----------------------------
# Printer configuration popup
# -----------------------------
class PrinterConfigPopup:
    def __init__(self, parent, cam_index, callback):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Printer config for Camera {cam_index}")
        self.callback = callback

        tk.Label(self.top, text="Printer Name:").pack(pady=5)
        self.name_entry = tk.Entry(self.top)
        self.name_entry.pack()

        tk.Label(self.top, text="Printer IP:").pack(pady=5)
        self.ip_entry = tk.Entry(self.top)
        self.ip_entry.pack()

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

# -------------------------------------------------------
# Camera feed tab class WITH controls
# -------------------------------------------------------
class CameraTab:
    def __init__(self, notebook, cam_index, printer_info=None):
        self.cam_index = cam_index
        self.printer_info = printer_info

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        tab_title = printer_info.get("name") if printer_info else f"Camera {cam_index}"
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=tab_title)

        # Left control panel
        self.control_frame = ttk.Frame(self.frame, width=200)
        self.control_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.terminate_button = ttk.Button(self.control_frame, text="Terminate Print",
                                           command=self.terminate_print)
        self.terminate_button.pack(pady=10)

        self.conf_label = ttk.Label(self.control_frame, text="Max Confidence: 0%")
        self.conf_label.pack(pady=10)

        self.status_canvas = tk.Canvas(self.control_frame, width=20, height=20)
        self.status_canvas.pack(pady=5)
        self.status_circle = self.status_canvas.create_oval(2, 2, 18, 18, fill="green")

        # Camera feed frame (fixed size)
        self.display_width = 800
        self.display_height = 600
        self.camera_frame = ttk.Frame(self.frame, width=self.display_width, height=self.display_height)
        self.camera_frame.pack(side="right", fill="both", expand=True)
        self.camera_frame.pack_propagate(False)

        # Canvas for video (avoids jitter)
        self.camera_canvas = tk.Canvas(self.camera_frame, width=self.display_width, height=self.display_height)
        self.camera_canvas.pack()
        self.camera_canvas_img = None  # canvas image reference

        self.update_frame()

    def update_frame(self):
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

            self.conf_label.config(text=f"Max Confidence: {max_conf*100:.1f}%")
            self.status_canvas.itemconfig(self.status_circle, fill="red" if max_conf >= 0.85 else "green")

            # Resize annotated frame to canvas size
            annotated = cv2.resize(annotated, (self.display_width, self.display_height))
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self.imgtk = ImageTk.PhotoImage(image=img)

            if self.camera_canvas_img is None:
                self.camera_canvas_img = self.camera_canvas.create_image(0, 0, anchor="nw", image=self.imgtk)
            else:
                self.camera_canvas.itemconfig(self.camera_canvas_img, image=self.imgtk)

        # Schedule next frame (~30 FPS)
        self.camera_canvas.after(33, self.update_frame)

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

# -------------------------------------------------------
# Main App
# -------------------------------------------------------
class App:
    def __init__(self, root, camera_count):
        self.root = root
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.printers = {}  # store printer info per camera

        # Home tab
        self.home_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.home_tab, text="Home")

        ttk.Label(self.home_tab, text="Available Cameras",
                  font=("Arial", 14)).pack(pady=10)

        self.add_camera_buttons(camera_count)

    def add_camera_buttons(self, camera_count):
        for i in range(camera_count):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok, _ = cap.read()
            cap.release()

            if ok:
                button = ttk.Button(
                    self.home_tab,
                    text=f"Open Camera {i}",
                    command=lambda cam=i: self.open_camera(cam)
                )
                button.pack(pady=5)
            else:
                ttk.Label(self.home_tab,
                          text=f"Camera {i} not found",
                          foreground="red").pack()

    def open_camera(self, cam_index):
        # Ask for printer info first
        def save_printer(name, ip):
            self.printers[cam_index] = {"name": name, "ip": ip}
            CameraTab(self.notebook, cam_index, printer_info=self.printers[cam_index])

        PrinterConfigPopup(self.root, cam_index, save_printer)

# -------------------------------------------------------
# Program start
# -------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <camera_count>")
        sys.exit(1)

    camera_count = int(sys.argv[1])

    root = tk.Tk()
    root.title("YOLO Multi-Camera Viewer")
    root.geometry("1200x800")

    app = App(root, camera_count)

    root.mainloop()
