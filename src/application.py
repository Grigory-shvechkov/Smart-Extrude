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
# Camera feed tab class WITH YOLO + Terminate Print
# -------------------------------------------------------
class CameraTab:
    def __init__(self, notebook, cam_index, printer_info=None):
        self.cam_index = cam_index
        self.printer_info = printer_info

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        # Use printer name as tab title
        tab_title = printer_info.get("name") if printer_info else f"Camera {cam_index}"
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=tab_title)

        # Camera feed label (centered)
        self.label = tk.Label(self.frame)
        self.label.pack(expand=True, anchor="center", padx=10, pady=10)

        # Terminate Print button
        self.terminate_button = ttk.Button(self.frame, text="Terminate Print",
                                           command=self.terminate_print)
        self.terminate_button.pack(pady=5)

        # Desired display size (can be adjusted)
        self.display_width = 800
        self.display_height = 600

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Run YOLO
            results = model.predict(frame, imgsz=640, verbose=False)
            annotated = results[0].plot() if results else frame

            # Print detections
            if results and results[0].boxes:
                for box in results[0].boxes:
                    cls_index = int(box.cls)
                    class_name = results[0].names[cls_index]
                    confidence = float(box.conf)
                    xyxy = box.xyxy[0].tolist()
                    print(f"[Camera {self.cam_index}] {class_name} {confidence:.2f} {xyxy}")

            # Resize annotated frame to fit display
            annotated = cv2.resize(annotated, (self.display_width, self.display_height))
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

        self.label.after(15, self.update_frame)

    def terminate_print(self):
        if not self.printer_info:
            messagebox.showerror("Error", "Printer info not set for this camera.")
            return

        printer_ip = self.printer_info.get("ip")
        if not printer_ip:
            messagebox.showerror("Error", "Printer IP missing.")
            return

        url = f"http://{printer_ip}/api/job"
        headers = {
            "Content-Type": "application/json",
            # Add your API key if needed:
            # "X-Api-Key": "<YOUR_API_KEY>"
        }
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
            print(f"[Camera {cam_index}] Printer saved: {self.printers[cam_index]}")
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
    root.geometry("1000x800")

    app = App(root, camera_count)

    root.mainloop()
