import time
import requests
import random

while True:
    try:
        # Optional: force random drift in DB (you can also patch SQLite if needed)
        response = requests.get("http://localhost:8000/drift")
        print(f"[{time.strftime('%H:%M:%S')}] Drift Triggered: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")
    time.sleep(5)
