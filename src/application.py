import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import sys
from ultralytics import YOLO

# -----------------------------
# Load YOLO model ONCE
# -----------------------------
MODEL_PATH = "../weights/best.pt"
model = YOLO(MODEL_PATH)

# -------------------------------------------------------
# Camera feed tab class WITH YOLO
# -------------------------------------------------------
class CameraTab:
    def __init__(self, notebook, cam_index):
        self.cam_index = cam_index

        # Use CAP_DSHOW for Windows (reduces lag)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=f"Camera {cam_index}")

        self.label = tk.Label(self.frame)
        self.label.pack()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Run YOLO on this frame
            results = model.predict(frame, imgsz=640, verbose=False)

            # Draw YOLO boxes on the image
            annotated = results[0].plot()

            # Print detections (same style as your old script)
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_index = int(box.cls)
                    class_name = results[0].names[cls_index]
                    confidence = float(box.conf)
                    xyxy = box.xyxy[0].tolist()

                    print(f"[Camera {self.cam_index}] {class_name} "
                          f"{confidence:.2f} {xyxy}")

            # Convert frame to display in Tkinter
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

        # Schedule next frame
        self.label.after(15, self.update_frame)

# -------------------------------------------------------
# Main App
# -------------------------------------------------------
class App:
    def __init__(self, root, camera_count):
        self.root = root
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

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
                    command=lambda cam=i: CameraTab(self.notebook, cam)
                )
                button.pack(pady=5)
            else:
                ttk.Label(self.home_tab,
                          text=f"Camera {i} not found",
                          foreground="red").pack()

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
    root.geometry("900x700")

    app = App(root, camera_count)

    root.mainloop()
