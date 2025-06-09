# ADAS-Sensor-Degradation
Sensor degradation monitoring using DENSE Dataset.

📘 Project Setup Guide: ADAS Sensor Degradation Detection
This guide documents the full setup and execution process for simulating camera image streams and performing predictive degradation analysis using Kafka and the DENSE dataset.

🧰 1. Git + GitHub Setup
# Clone your GitHub repository
git clone https://github.com/YOUR_USERNAME/ADAS-Sensor-Degradation.git
cd ADAS-Sensor-Degradation

# Create project folder structure
mkdir -p data scripts models kafka pipeline notebooks
touch requirements.txt .env

# Track empty folders with .gitkeep
touch data/.gitkeep scripts/.gitkeep models/.gitkeep kafka/.gitkeep pipeline/.gitkeep notebooks/.gitkeep

# Add and commit
git add .
git commit -m "Initial folder structure and base files"
git push

🛠 2. Kafka + Zookeeper Setup (via Docker)
# Go to kafka folder
cd kafka

# Create docker-compose.yml
# (Use the YAML config provided below)

# Start Kafka and Zookeeper containers
docker compose up -d

# Confirm both containers are running
docker ps


version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: zookeeper
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

📡 3. Kafka Producer — Stream Camera Images
# Go to scripts folder
cd scripts

# Create Kafka producer script
touch kafka_image_producer.py

# Paste the Python code provided below into this file

# Install dependencies
pip install kafka-python opencv-python

# Run the producer
python kafka_image_producer.py


kafka_image_producer.py snippet:

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



---

### 📡 Streaming nuScenes Images via Kafka

#### Dataset Setup:
```bash
# Navigate to data folder
cd data/nuscenes

# Download mini dataset (v1.0-mini.tgz) manually or using:
# Invoke-WebRequest -Uri https://www.nuscenes.org/data/v1.0-mini.tgz -OutFile v1.0-mini.tgz

# Extract
tar -xvzf v1.0-mini.tgz


# Update script path to use CAM_FRONT images
vim scripts/kafka_image_producer.py

# Run the producer
python scripts/kafka_image_producer.py
