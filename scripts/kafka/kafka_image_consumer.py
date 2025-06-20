# kafka_image_consumer.py
import os
import cv2
import numpy as np
from kafka import KafkaConsumer
from db_utils import create_table, insert_metrics

# --- Configuration ---
KAFKA_TOPIC = "adas_camera_stream"
KAFKA_BROKER = "localhost:9092"

# --- Initialize SQLite table ---
create_table()

# --- Kafka Consumer Setup ---
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: x
)

def analyze_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = -np.sum((p := cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel() / gray.size) * np.log2(p + 1e-10))

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur": round(blur, 2),
        "entropy": round(entropy, 2)
    }

print("[INFO] Listening to Kafka topic...")

for message in consumer:
    result = analyze_image(message.value)
    if result:
        # image_id could be None or derived from stream, here left as placeholder
        insert_metrics(
            image_id="frame_unknown",  # TODO: optionally pass filename from producer
            brightness=result["brightness"],
            contrast=result["contrast"],
            blur=result["blur"],
            entropy=result["entropy"]
        )
        print(f"[LOGGED] -> {result}")
    else:
        print("[WARN] Skipped unreadable frame")
