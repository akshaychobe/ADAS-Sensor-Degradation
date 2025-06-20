# ADAS Sensor Degradation Monitoring (Real-Time AI Pipeline)

This project simulates a real-time ADAS camera stream using the nuScenes dataset and performs predictive sensor degradation analysis using a production-grade AI pipeline. The goal is to detect, monitor, and forecast degradation trends like fogging, glare, blur, and loss of contrast in camera sensors—crucial for safe autonomous driving.

---

## 📦 Project Overview

| Phase   | Functionality                                     | Technologies                           |
|---------|---------------------------------------------------|----------------------------------------|
| Phase 1 | Real-time sensor image ingestion + metric logging | Kafka, OpenCV, SQLite                  |
| Phase 2 | Forecast sensor health degradation                | PyTorch, LSTM/Transformer              |
| Phase 3 | Trigger drift detection & retraining              | Evidently, Airflow, FastAPI            |
| Phase 4 | Expose health endpoint for live dashboarding      | FastAPI, SQLite                        |

---

## 🧰 1. Git + GitHub Setup

```bash
# Clone your GitHub repository
git clone https://github.com/YOUR_USERNAME/ADAS-Sensor-Degradation.git
cd ADAS-Sensor-Degradation

# Create project folder structure
mkdir -p data scripts models kafka pipeline notebooks
touch requirements.txt .env

# Track empty folders
touch data/.gitkeep scripts/.gitkeep models/.gitkeep kafka/.gitkeep pipeline/.gitkeep notebooks/.gitkeep

# Stage and commit
git add .
git commit -m "Initial folder structure and base files"
git push
```

---

## 🐳 2. Kafka + Zookeeper Setup (via Docker)

```bash
cd kafka
docker compose up -d
docker ps
```

`docker-compose.yml`:

```yaml
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
```

---

## 📡 3. Kafka Producer — Stream nuScenes Camera Images

### Dataset Setup

```bash
cd data
mkdir nuscenes
cd nuscenes

# Download manually from https://www.nuscenes.org/download
# File: v1.0-mini.tgz
tar -xvzf v1.0-mini.tgz
```

### Kafka Image Producer Setup

```bash
cd scripts
touch kafka_image_producer.py
pip install kafka-python opencv-python
```

`scripts/kafka_image_producer.py`

```python
import os, cv2, time
from kafka import KafkaProducer

IMAGE_FOLDER = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\nuscenes\samples\CAM_FRONT"
KAFKA_TOPIC = "adas_camera_stream"
KAFKA_BROKER = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: v
)

def encode_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    success, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes() if success else None

image_files = sorted(os.listdir(IMAGE_FOLDER))
while True:
    for img_file in image_files:
        if not img_file.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        encoded = encode_image(img_path)
        if encoded:
            producer.send(KAFKA_TOPIC, value=encoded)
            print(f"Sent: {img_file}")
            time.sleep(0.1)
```

---

## 🔁 4. Kafka Consumer + Real-Time Sensor Metric Logging (SQLite)

```bash
cd scripts
touch kafka_image_consumer.py
touch db_utils.py
```

`scripts/db_utils.py`

```python
import sqlite3
from datetime import datetime

DB_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\sensor_health.db"

def create_table():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_id TEXT,
            brightness REAL,
            contrast REAL,
            blur REAL,
            entropy REAL
        )
        """)
        conn.commit()

def insert_metrics(image_id, brightness, contrast, blur, entropy):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO sensor_metrics (timestamp, image_id, brightness, contrast, blur, entropy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, image_id, brightness, contrast, blur, entropy))
        conn.commit()
```

`scripts/kafka_image_consumer.py`

```python
import os, cv2, numpy as np
from kafka import KafkaConsumer
from db_utils import create_table, insert_metrics

create_table()

KAFKA_TOPIC = "adas_camera_stream"
KAFKA_BROKER = "localhost:9092"

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
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = -np.sum((p := cv2.calcHist([gray], [0], None, [256], [0,256]).ravel() / gray.size) * np.log2(p + 1e-10))
    return brightness, contrast, laplacian_var, entropy

print("[INFO] Listening to Kafka topic...")
for message in consumer:
    metrics = analyze_image(message.value)
    if metrics:
        brightness, contrast, blur, entropy = metrics
        insert_metrics("frame", brightness, contrast, blur, entropy)
        print(f"[LOG] -> Bright: {brightness:.2f}, Blur: {blur:.2f}")
    else:
        print("[WARN] Skipped unreadable frame")
```

---

## 🧩 Phase 1: Real-Time Sensor Metrics Logging (SQLite Architecture)

This system logs visual degradation metrics (blur, brightness, contrast, entropy) extracted from front camera frames into a time-series SQLite database — built for scalable, portable sensor health analytics.

### 🗃️ Database: `sensor_health.db`

| Column     | Type | Description                           |
|------------|------|---------------------------------------|
| timestamp  | TEXT | Frame timestamp                       |
| image_id   | TEXT | Frame identifier                      |
| brightness | REAL | Average intensity (dark vs. light)    |
| contrast   | REAL | Intensity spread (sharp vs. flat)     |
| blur       | REAL | Laplacian variance (sharpness)        |
| entropy    | REAL | Histogram entropy (visual complexity) |

### 🔍 Sample Query

```sql
SELECT * FROM sensor_metrics
WHERE blur < 150 AND entropy < 7.0
ORDER BY timestamp DESC
LIMIT 10;
```

✅ SQLite enables secure, reliable, and analytics-ready logging—forming the foundation for Phase 2: sensor health forecasting and drift prediction.


---

## 🔮 Phase 2: Sensor Health Forecasting & Drift Monitoring

This phase extends Phase 1 with predictive modeling (LSTM), live monitoring (FastAPI + Prometheus), and drift alerting, laying the groundwork for real-time dashboards.

---

### 📷 Architecture Visuals

- `Figure_1.png`: Block diagram showing entire architecture
- `data/forecast_visualization.png`: Forecast vs. actual drift plots

---

## 🧠 LSTM Forecasting Pipeline

### 1. 📊 Simulate Degradation Data

**Script**: `scripts/simulate_degradation_data.py`

Generates synthetic sensor data (blur, brightness, entropy) to populate the database:

```bash
python scripts/simulate_degradation_data.py
```

Effect:
- Adds realistic time-series degradation into `sensor_metrics` table
- Helps train & test forecasting logic

---

### 2. 📦 Extract Training Data

**Script**: `scripts/extract_training_data.py`

Pulls past sensor metrics and exports CSV:

```bash
python scripts/extract_training_data.py
```

Output:
- `data/image_quality_metrics.csv`

---

### 3. 🧠 Train LSTM Forecast Model

**Script**: `scripts/train_lstm_forecast.py`

Trains a sequence-to-sequence LSTM model on sensor blur values:

```bash
python scripts/train_lstm_forecast.py
```

Output:
- `models/lstm_forecaster.pth`

---

### 4. 📈 Forecast Future Drift

**Script**: `scripts/forecast_lstm_predict.py`

Predicts future sensor values:

```bash
python scripts/forecast_lstm_predict.py
```

Example Output:
```text
Forecast (next 4 blur values): [302.1, 298.6, 294.3, 289.0]
```

---

### 5. 📊 Visualize Forecast

**Script**: `scripts/forecast_visualizer.py`

Draws forecast vs. actual:

```bash
python scripts/forecast_visualizer.py
```

Output:
- `data/forecast_visualization.png`

---

### 6. ✅ Check DB for Enough Samples

**Script**: `scripts/check_db_count.py`

Verifies if `sensor_health.db` has enough rows:

```bash
python scripts/check_db_count.py
```

---

### 7. 🚀 One-Time Forecast Debug Runs

| Script                | Purpose                               |
|-----------------------|---------------------------------------|
| `one_time.py`         | Prints prediction from latest blur    |
| `one_time_2.py`       | Logs result with timestamp/thresholds |

---

## 🌐 FastAPI Forecast Service (JSON API)

**Script**: `scripts/trial3_fastapi_monitor.py`

```bash
uvicorn scripts.trial3_fastapi_monitor:app --reload --port 8000
```

Then open in browser:

```
http://localhost:8000/forecast
```

**Returns:**

```json
{
  "metric": "blur",
  "forecast": [290.4, 288.9, 286.1],
  "timestamp": "2025-06-12T12:30:00"
}
```

---

## 📊 Prometheus Integration (Sensor Drift Metrics)

**Script**: `scripts/trial_3_fastapi_sensor_health.py`

Start server:

```bash
uvicorn scripts/trial_3_fastapi_sensor_health:app --reload --port 8000
```

Open Prometheus browser:

```
http://localhost:9090
```

Sample metrics at `http://localhost:8000/metrics`:

```
sensor_forecast_blur 298.45
sensor_entropy_drift 0.01
sensor_brightness_drift 7.93
```

---

### ⚙️ Prometheus Config (scraping FastAPI)

**File**: `prometheus/prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'sensor-drift-monitor'
    static_configs:
      - targets: ['localhost:8000']
```

Run Prometheus:

```bash
cd prometheus
./prometheus --config.file=prometheus.yml
```

---

## 📈 Grafana Setup (Live Dashboards)

### 1. Download and unzip Grafana
- Use: [https://grafana.com/grafana/download](https://grafana.com/grafana/download)
- Extract and run:

```bash
cd grafana/grafana-v12.0.x/bin
grafana-server.exe
```

### 2. Open browser:
```
http://localhost:3000
```
Default Login: `admin` / `admin`

### 3. Add Prometheus as data source:
- Type: **Prometheus**
- URL: `http://localhost:9090`
- Save & Test

### 4. Create dashboard panel
- Panel → Query → `sensor_forecast_blur`
- Visualization → Time Series
- Save Dashboard

---

## 🚨 Auto Trigger Drift Detection

**Script**: `scripts/auto_trigger_drift.py`

Runs in background and:
- Pulls predictions
- Compares with thresholds
- Logs alert if drift detected

---

## 🧪 Drift Detection Trials

### ✅ Trial 1: Static Thresholds

**Script**: `scripts/trial_1_drift_threshold_alert.py`

- Compare against manually defined thresholds (e.g., blur > 290)
- Logs alert in console

---

### 📊 Trial 2: Evidently Report

**Script**: `scripts/trial_2_drift_evidently.py`

- Generates HTML drift report
- Output: `scripts/trial_2_drift_report.html`

---

### 🧠 Trial 3: Combined Monitoring API

**Script**: `scripts/trial3_fastapi_monitor.py`

- Combines prediction and Prometheus exposure
- Lightweight JSON + /metrics API

---

## 🔁 Folder Overview

```
📁 data/
   ├── sensor_health.db
   ├── forecast_visualization.png
   └── image_quality_metrics.csv

📁 models/
   └── lstm_forecaster.pth

📁 scripts/
   ├── simulate_degradation_data.py
   ├── extract_training_data.py
   ├── train_lstm_forecast.py
   ├── forecast_lstm_predict.py
   ├── forecast_visualizer.py
   ├── one_time.py
   ├── one_time_2.py
   ├── auto_trigger_drift.py
   ├── trial3_fastapi_monitor.py
   ├── trial_3_fastapi_sensor_health.py
   ├── trial_1_drift_threshold_alert.py
   ├── trial_2_drift_evidently.py
   ├── check_db_count.py
   └── db_utils.py

📁 prometheus/
   └── prometheus.yml

📁 grafana/
   └── grafana-server.exe (excluded from Git)
```

---

## ✅ Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- kafka-python
- opencv-python
- numpy
- fastapi
- uvicorn
- sqlite3
- torch
- matplotlib
- evidently



---

## 🔁 Phase 3: Sensor Retraining + Automation Pipeline

This phase introduces model retraining logic and background scheduling tools to keep the system robust over time.

---

### 📌 Why Retrain?

- Over time, real-world sensor drift may evolve
- Models trained on older data may lose accuracy
- Periodic retraining ensures:
  - ✅ Adaptability to new patterns
  - ✅ Better forecast accuracy
  - ✅ Fewer false alerts

---

## ⚙️ Retraining Strategy

### 1. Update Data with Latest Window

Use already ingested metrics (via Kafka + SQLite) to:
- Pull recent sensor blur values
- Extract clean segments (no drift)
- Update CSV or input for training

```bash
python scripts/extract_training_data.py
```

---

### 2. Train Updated Model

Re-run the training logic on new clean segments:

```bash
python scripts/train_lstm_forecast.py
```

Or use improved version with validation logic:

```bash
python scripts/forecast_lstm_train.py
```

Result:
- Updated `models/lstm_forecaster.pth` (used in live forecast)

---

## 🧪 Drift Feedback Loop

If too many drift events are triggered (via `auto_trigger_drift.py`):
- Analyze DB using SQLite
- Identify stable windows
- Use them for retraining

---

## ⏱️ Background Scheduling Options

To automate retraining or drift-checking:

### 1. Use Python + `schedule` or `APScheduler`

```python
import schedule
import time
from subprocess import call

def retrain_job():
    call(['python', 'scripts/train_lstm_forecast.py'])

schedule.every().day.at("02:00").do(retrain_job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

### 2. Use Windows Task Scheduler

Schedule any `.bat` file to run every night:

```bat
@echo off
cd C:\Users\Lenovo\ADAS-Sensor-Degradation
python scripts/train_lstm_forecast.py
```

---

### 3. Use Cron (on Linux)

Add to crontab:

```bash
0 2 * * * cd /path/to/project && python3 scripts/train_lstm_forecast.py
```

---

## 🧠 Phase 3 Summary

| Task                    | Script                          | Purpose                             |
|-------------------------|----------------------------------|-------------------------------------|
| Extract fresh training  | `extract_training_data.py`      | Clean training data from DB         |
| Train model             | `train_lstm_forecast.py`        | Train LSTM from extracted metrics   |
| Auto run forecast       | `auto_trigger_drift.py`         | Alert if blur > threshold           |
| Schedule jobs           | `schedule`, `.bat`, `cron`      | Run training or drift logic on time |

---

## 📦 Optional Enhancements

| Feature                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| SQLite backup rotation          | Auto-save `.db` file daily or weekly                                        |
| Drift log table                 | Create another table to store timestamps of drift detections                |
| Notification on drift           | Email, Slack, or webhook trigger                                           |
| LSTM + Transformer hybrid       | Try transformer-based forecasters for long-term trends                     |
| Multi-metric model              | Use multi-input LSTM (blur + brightness + entropy)                         |

---

## 📊 Dataset: nuScenes

### 🔗 Official Download

https://www.nuscenes.org/download

Choose:
- **nuScenes v1.0-mini**
- File: `v1.0-mini.tgz`

### 📁 Folder Structure

```
data/
├── nuscenes/
│   └── samples/
│       └── CAM_FRONT/
│           ├── 000001.jpg
│           ├── 000002.jpg
│           └── ...
```

This folder is read by:
- `kafka_image_producer.py` to stream image frames

---

## 🔐 `.gitignore` and Clean Git Setup

Make sure these are excluded:

```
# Ignore datasets and binaries
data/nuscenes/
models/
*.pth
*.db
*.csv
*.png
__pycache__/
.venv/
*.zip

# Grafana binaries
grafana/grafana-v*/bin/
```

You can manually add `.gitkeep` to keep folders versioned.

---

## ✅ GitHub Maintenance Strategy

| Action                              | Command / Tip                                        |
|-------------------------------------|------------------------------------------------------|
| Create new branch                   | `git checkout -b phase3-retraining`                  |
| Backup before refactor              | `git checkout -b backup-before-refactor`             |
| Push updated branch                 | `git push origin phase3-retraining`                 |
| Remove large file from history      | `git filter-branch` or `git lfs migrate`             |
| Cleanup local files                 | `git clean -fd` and `git prune`                      |
| Tag stable release                  | `git tag v2.0-phase2` → `git push origin --tags`     |

---

## 📂 Final Folder Tree (So Far)

```
ADAS-Sensor-Degradation/
├── data/
│   ├── sensor_health.db
│   ├── image_quality_metrics.csv
│   └── forecast_visualization.png
├── models/
│   └── lstm_forecaster.pth
├── scripts/
│   ├── kafka_image_producer.py
│   ├── kafka_image_consumer.py
│   ├── simulate_degradation_data.py
│   ├── extract_training_data.py
│   ├── train_lstm_forecast.py
│   ├── forecast_lstm_predict.py
│   ├── forecast_visualizer.py
│   ├── trial3_fastapi_monitor.py
│   ├── trial_3_fastapi_sensor_health.py
│   ├── auto_trigger_drift.py
│   └── ...
├── prometheus/
│   └── prometheus.yml
├── grafana/
├── kafka/
├── notebooks/
├── pipeline/
├── requirements.txt
├── README.md
└── LICENSE
```

---
---

## 📊 Phase 4: Real-Time Visualization with Grafana

This final phase allows us to visualize drift predictions, metric trends, and model behavior via Grafana dashboards in real time — built on FastAPI + Prometheus architecture.

---

### 📌 Why Grafana?

- Grafana connects directly to Prometheus
- Supports live graphs, alerts, and dashboards
- Lightweight and portable (runs locally)

---

## 🖥️ Grafana Installation (Windows)

1. Download from [https://grafana.com/grafana/download](https://grafana.com/grafana/download)
   - Choose **Windows > ZIP version**

2. Extract to:
   ```
   C:\Users\Lenovo\ADAS-Sensor-Degradation\grafana-12.0.2.windows-amd64
   ```

3. Launch Grafana:
   ```bash
   cd grafana-12.0.2.windows-amd64\bin
   grafana-server.exe
   ```

4. Access in browser:
   ```
   http://localhost:3000
   ```

---

## 🔐 Default Grafana Credentials

| Username | Password  |
|----------|-----------|
| admin    | admin     |

Change password on first login.

---

## 🔧 Configure Prometheus as Data Source

1. Go to **Gear Icon > Data Sources**
2. Click **Add data source**
3. Choose **Prometheus**
4. URL: `http://localhost:9090`
5. Click **Save & Test**

---

## 📈 Create Dashboard

1. Go to **+ > Dashboard > New Panel**
2. In "Query":
   - Select Metric: `sensor_forecast_blur`
3. Choose "Time series" visualization
4. Click **Apply**

Repeat for:
- `sensor_brightness_drift`
- `sensor_entropy_drift`
- `sensor_contrast_drift`

---

## 🎯 Customize Panels

Each panel can display:
- Predicted blur values over time
- Brightness/contrast trend
- Entropy deviation

Add titles, legends, and time ranges as needed.

---

## ⚠️ Alerting (Optional)

Set up **alerts** on panels:
- If `sensor_forecast_blur > 290`, send alert
- Email, webhook, or Slack supported (if configured)

---

## 🧪 Live Monitoring Endpoint

FastAPI provides JSON output at:

```
http://localhost:8000/forecast
```

Returns:

```json
{
  "metric": "blur",
  "forecast": [295.0, 293.7, 291.2],
  "timestamp": "2025-06-20T18:45:00"
}
```

Prometheus fetches from:

```
http://localhost:8000/metrics
```

Sample Metrics:

```
sensor_blur_drift 38.29
sensor_brightness_drift 92.5
sensor_entropy_drift 0.12
```

---

## 🔁 Prometheus Scrape Config

Edit `prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'sensor-drift-monitor'
    static_configs:
      - targets: ['localhost:8000']
```

Launch Prometheus:

```bash
cd prometheus
./prometheus --config.file=prometheus.yml
```

Access: [http://localhost:9090](http://localhost:9090)

---

## ✅ Final End-to-End Flow

```
Kafka Producer (images) ➝ Kafka Consumer
 ➝ SQLite logging ➝ LSTM model ➝ Prediction
 ➝ FastAPI JSON + Prometheus metrics ➝ Grafana dashboard
```

---

## 💡 Future Work

| Task                            | Description                                             |
|----------------------------------|---------------------------------------------------------|
| CI/CD with GitHub Actions        | Auto-test model, API on push                           |
| Dockerize whole pipeline         | Use Docker Compose for Prometheus + API + Kafka        |
| Add Prometheus alerts            | Trigger on threshold breaches                          |
| Integrate Evidently in Grafana   | Convert Evidently HTML reports to live Prometheus stats|
| Use ONNX for model deployment    | Convert PyTorch model to ONNX for faster inference     |
| Sensor fusion forecasting        | Add LiDAR or radar streams for better predictions      |

---

## 🚀 Recommended Project Repo Structure

```
ADAS-Sensor-Degradation/
├── data/                  # SQLite DB, CSV, PNG outputs
├── models/                # LSTM .pth model
├── scripts/               # Kafka, API, forecast, utils
├── prometheus/            # Prometheus config
├── grafana/               # Grafana config files (NO BINARIES)
├── kafka/                 # docker-compose.yml
├── notebooks/             # (Optional) Jupyter experiments
├── requirements.txt
├── README.md
├── .gitignore
```

---

## 🧾 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 🙌 Acknowledgements

- [nuScenes Dataset](https://www.nuscenes.org/)
- [Grafana Labs](https://grafana.com/)
- [Prometheus](https://prometheus.io/)
- [EvidentlyAI](https://www.evidentlyai.com/)
- [PyTorch](https://pytorch.org/)
- [Kafka](https://kafka.apache.org/)
