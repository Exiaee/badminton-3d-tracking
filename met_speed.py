import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
from pathlib import Path
from datetime import datetime
from pathlib import Path
# 4 cameras in 4 corners, id: 0, 1, 2, 3
date = datetime.now().strftime("%Y%m%d_%H%M%S")

def speed_to_met(speed):
    if speed < 0.1:
        return 4
    
    elif speed < 2.0:
        return 6

    elif speed < 3.0:
        return 8
    else:
        return 10

weight_kg = 70
#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"
INPUT_PATH  = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"
SELECTED_PAIR = 0 # 0 for pair (0, 1), 1 for pair (2, 3)
VIDEO_A = f"{INPUT_PATH}/CameraReader_0.mp4"
csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\Video_2026-04-09_19-13-28\p_0p01_m_0p05_Akman_maintain\right\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260518_000049.csv"
folder_name = str(Path(csv_path).parent.relative_to(Path(csv_path).parents[1]))


# From 2024 Adult Compendium of Physical Activities - Running
# mph -> m/s
speed_met_table = pd.DataFrame({
    "speed_mph": [2.6, 4.0, 4.3, 5.0, 5.5, 6.0, 6.7, 7.0, 7.5, 8.0, 8.6, 9.0],
    "MET":       [3.3, 6.5, 7.8, 8.5, 9.0, 9.3,10.5,11.0,11.8,12.0,12.5,13.0]
})

speed_met_table["speed_mps"] = speed_met_table["speed_mph"] * 0.44704

def speed_to_met_compendium(speed_mps):
    return np.interp(
        speed_mps,
        speed_met_table["speed_mps"],
        speed_met_table["MET"],
        left=1.5,                       # very slow / idle
        right=speed_met_table["MET"].iloc[-1]
    )

safe_folder_name = folder_name.replace("\\", "_")
capA = cv2.VideoCapture(VIDEO_A)

df = pd.read_csv(csv_path)
fps = capA.get(cv2.CAP_PROP_FPS)
#fps = 30
dt = 1 / fps
df["time_sec"] = df["frame_id"] / fps
df["dx"] = df["x"].diff()
df["dy"] = df["y"].diff()
df["dist_m"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
df["dist_m"] = savgol_filter(
    df["dist_m"],
    window_length=15,
    polyorder=2
)
df["kalman_v"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
df["speed_mps"] = df["dist_m"] / dt
df["speed_mps"] = savgol_filter(
    df["speed_mps"],
    window_length=21,
    polyorder=2
)
df["kalman_v"] = 0.2*df["kalman_v"] + 0.8*df["speed_mps"]
df["speed_kmh"] = df["speed_mps"] * 3.6
df["kalman_v"] = savgol_filter(
    df["kalman_v"],
    window_length=21,
    polyorder=2
)
#df["MET"] = df["kalman_v"].apply(speed_to_met)
df["MET"] = df["kalman_v"].apply(speed_to_met_compendium)
avg_met = df["MET"].mean()
print(f"Average MET: {avg_met:.2f}")
df["kcal_per_frame"] = (
    df["MET"]
    # * weight_kg*(1/fps/3600)
)
#df["calories_cumsum"] = df["kcal_per_frame"].cumsum()
df["calories_cumsum"] = df["kcal_per_frame"]
plt.figure(figsize=(12,4))

plt.plot(
    df["time_sec"],
    df["calories_cumsum"],
    linewidth=2
)

plt.xlabel("Time (sec)")
plt.ylabel("Calories (kcal)")
plt.title("Estimated Calorie Burn")

plt.grid(True)

plt.savefig(
    f"calories_vs_time_{safe_folder_name}_{date}.png",
    dpi=300
)

plt.show()

print(
    f"Total Calories: "
    f"{df['calories_cumsum'].iloc[-1]:.2f} kcal"
)
plt.figure(figsize=(12, 4))
#plt.plot(df["frame_id"], df["speed_mps"], linewidth=2, label="Speed")
plt.plot(df["time_sec"], df["speed_mps"], linewidth=2, label="Speed")
#plt.xlabel("Frame")
plt.xlabel("Time (sec)")
plt.ylabel("Speed (m/s)")
plt.title("Player Speed vs Frame")
plt.grid(True)
plt.legend()
plt.savefig(f"speed_vs_frame_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
#plt.plot(df["frame_id"], df["kalman_v"], linewidth=2, label="Speed")
plt.plot(df["time_sec"], df["kalman_v"], linewidth=2, label="Speed")
#plt.xlabel("Frame")
plt.xlabel("Time (sec)")
plt.ylabel("Kalman Speed (m/s)")
plt.title("Player Speed vs Frame")
plt.grid(True)
plt.legend()
plt.savefig(f"kalman_v_vs_frame.png_{safe_folder_name}_{date}", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))

plt.plot(
    #df["frame_id"],
    df["time_sec"],
    df["speed_mps"],
    linewidth=2,
    label="Diff Speed + Savitzky-Golay"
)

plt.plot(
    #df["frame_id"],
    df["time_sec"],
    df["kalman_v"],
    linewidth=2,
    label="Kalman Velocity"
)

#plt.xlabel("Frame")
plt.xlabel("Time (sec)")
plt.ylabel("Speed (km/h)")
plt.title("Player Speed Comparison")
plt.grid(True)
plt.legend()

plt.savefig(f"speed_overlay_vs_frame_{safe_folder_name}_{date}.png", dpi=300)
plt.show()
df.to_csv(f"Player1_trajectory_with_speed__{safe_folder_name}_{date}.csv", index=False)

print("Saved speed_vs_frame.png")
print("Saved Player1_trajectory_with_speed.csv")