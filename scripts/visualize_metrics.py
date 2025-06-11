# visualize_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Path to CSV
CSV_PATH = r"C:\Users\Lenovo\ADAS-Sensor-Degradation\data\image_quality_metrics.csv"

# Load data
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# Set timestamp as index
df.set_index("timestamp", inplace=True)

# Plot all metrics
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(df.index, df["brightness"], label="Brightness", color='gold')
plt.ylabel("Brightness")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(df.index, df["contrast"], label="Contrast", color='blue')
plt.ylabel("Contrast")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(df.index, df["blur"], label="Blur", color='green')
plt.ylabel("Blur")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(df.index, df["entropy"], label="Entropy", color='red')
plt.ylabel("Entropy")
plt.xlabel("Timestamp")
plt.grid(True)

plt.tight_layout()
plt.show()
