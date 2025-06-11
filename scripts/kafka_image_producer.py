# kafka_image_producer.py
import os
import cv2
import time
from kafka import KafkaProducer

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\nuscenes\samples\CAM_FRONT"  # Updated for nuScenes
KAFKA_TOPIC = "adas_camera_stream"
KAFKA_BROKER = "localhost:9092"

# --- Kafka Producer Setup ---
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: v  # send raw JPEG bytes
)

# --- Helper: Encode Image as JPEG Bytes ---
def encode_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    success, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes() if success else None

# --- Main Loop: Stream images ---
image_files = sorted(os.listdir(IMAGE_FOLDER))
for img_file in image_files:
    if not img_file.lower().endswith(".jpg"):
        continue
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    encoded = encode_image(img_path)
    if encoded:
        producer.send(KAFKA_TOPIC, value=encoded)
        print(f"Sent: {img_file}")
        time.sleep(0.1)  # simulate ~10 FPS
    else:
        print(f"Failed to read: {img_file}")

producer.flush()
producer.close()
