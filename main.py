# ==========
# Face Detector

# Penugasan Implementasi yolo11n-face menggunakan Python 3.12.7 untuk pendeteksi dan menghitung jumlah wajah berasarkan gambar yang diberikan.

# Setup dan Instalasi Library Project
# 1. Instalasi Python 3.12.7 pada link berikut
# https://www.python.org/downloads/release/python-3127/
# 2. Buat projek folder baru
# 3. Instal library berikut

# pip install ultralytics opencv-python numpy

# atau menggunakan

# py -m pip install ultralytics opencv-python numpy

# Running Program
# Untuk menjalankan program pendeteksi wajah, jalankan program berikut

# python main.py

# ==========

from ultralytics import YOLO
import cv2
import datetime
import os

# === CONFIGURATION ===
IMAGE_PATH = "{file gambar untuk di check.png}"     # Gambar yang akan dimasukkan
MODEL_NAME = "yolov11n-face.pt"                     # Model Deteksi
IMG_SIZE = 2560                                     # Resolusi gambar, semakin tinggi semakin akurat

# Load model
model = YOLO(MODEL_NAME)

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    print(f"Image '{IMAGE_PATH}' not found.")
    exit()

# Read image
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Failed to load image.")
    exit()

print("Running detection...")

# Run detection
results = model(frame, classes=[0], imgsz=IMG_SIZE)

count = 0

for box in results[0].boxes:
    count += 1

print("Detected students:", count)

# Draw bounding boxes
annotated = results[0].plot()

# Overlay count text
cv2.putText(
    annotated,
    f"Students: {count}",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

screen_width = 1280
screen_height = 720

h, w = annotated.shape[:2]

scale = min(screen_width / w, screen_height / h)
new_w = int(w * scale)
new_h = int(h * scale)

display_image = cv2.resize(annotated, (new_w, new_h))

# Save result
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"result_{timestamp}.jpg"
os.makedirs("result", exist_ok=True)
cv2.imwrite(os.path.join("result", output_filename), annotated)

print(f"Saved result as: {output_filename}")

# Show image
cv2.imshow("Detection Result", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()