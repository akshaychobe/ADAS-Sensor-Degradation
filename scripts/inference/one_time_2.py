import matplotlib.pyplot as plt
import numpy as np
from one_time import load_sensor_data

data = load_sensor_data()
plt.hist(data[:, 2], bins=50)  # Blur index is column 2
plt.title("Blur Value Distribution")
plt.show()
