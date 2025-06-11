# trial2_drift_evidently.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Path to CSV
CSV_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\image_quality_metrics.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# Reference = first 50 rows, Current = last 50 rows
reference = df.iloc[:50].drop(columns=["timestamp"])
current = df.iloc[-50:].drop(columns=["timestamp"])

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# Save HTML report
report.save_html("trial_2_drift_report.html")
print("[DONE] Drift report saved to data/trial_2_drift_report.html")
