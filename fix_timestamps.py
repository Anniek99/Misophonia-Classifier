import pandas as pd
from datetime import datetime, timedelta

# ===============================
# USER SETTINGS — CHANGE IF NEEDED
# ===============================
filename = "CoolTerm Capture (Untitled_0) 2025-12-08 01-15-38-777.txt"

# Start timestamp of recording:
start_time_str = "2025-12-08 01:15:38.777"
sample_interval = 0.05  # seconds (20 Hz)
# ===============================

# Parse start time
start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")

# Read data (no header, comma separated)
df = pd.read_csv(filename, header=None)
df.columns = ["mic1", "mic2", "mic3", "temp", "broken_unix"]

# Calculate NEW timestamps
df["time_sec"] = df.index * sample_interval
df["new_datetime"] = df["time_sec"].apply(lambda t: start_time + timedelta(seconds=t))
df["new_unix"] = df["new_datetime"].astype("int64") // 10**9  # convert to UNIX seconds

# drop broken timestamps
df = df.drop(columns=["broken_unix"])

# Save to CSV
output_name = filename.replace(".txt", "_FIXED.csv")
df.to_csv(output_name, index=False)

print(f"\n✅ Done! Saved as: {output_name}\n")
