import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
from pathlib import Path
from datetime import datetime

date = datetime.now().strftime("%Y%m%d_%H%M%S")
JUMP_ROPE_SLOW_MET = 8.3
JUMP_ROPE_MOD_MET = 11.8
JUMP_ROPE_FAST_MET = 12.3


# From 2024 Adult Compendium of Physical Activities - Running
# mph -> m/s
speed_met_table = pd.DataFrame({
    "speed_mph": [2.6, 4.0, 4.3, 5.0, 5.5, 6.0, 6.7, 7.0, 7.5, 8.0, 8.6, 9.0],
    "MET":       [3.3, 6.5, 7.8, 8.5, 9.0, 9.3,10.5,11.0,11.8,12.0,12.5,13.0]
})
def speed_to_met_compendium(speed_mps):
    return np.interp(
        speed_mps,
        speed_met_table["speed_mps"],
        speed_met_table["MET"],
        #left=1.5,                       # very slow / idle
        left= 2.0,
        right=speed_met_table["MET"].iloc[-1]
    )
speed_met_table["speed_mps"] = speed_met_table["speed_mph"] * 0.44704
weight_kg = 70
height_m = 1.75

INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"
VIDEO_A = f"{INPUT_PATH}/CameraReader_0.mp4"

csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260520_004440_with_swing.csv"

folder_name = str(Path(csv_path).parent.relative_to(Path(csv_path).parents[1]))
safe_folder_name = folder_name.replace("\\", "_")

capA = cv2.VideoCapture(VIDEO_A)
fps = capA.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

dt = 1 / fps

df = pd.read_csv(csv_path)

# =========================
# Basic time
# =========================
df["time_sec"] = df["frame_id"] / fps

# =========================
# Speed calculation
# =========================
df["dx"] = df["x"].diff().fillna(0)
df["dy"] = df["y"].diff().fillna(0)

df["dist_m"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

df["dist_m"] = savgol_filter(
    df["dist_m"],
    window_length=15,
    polyorder=2
)

df["speed_mps"] = df["dist_m"] / dt

df["speed_mps"] = savgol_filter(
    df["speed_mps"],
    window_length=21,
    polyorder=2
)

df["kalman_v"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
df["kalman_v"] = 0.2 * df["kalman_v"] + 0.8 * df["speed_mps"]

df["kalman_v"] = savgol_filter(
    df["kalman_v"],
    window_length=21,
    polyorder=2
)

df["speed_kmh"] = df["kalman_v"] * 3.6

# =========================
# Jump detection
# =========================
df["z_smooth"] = savgol_filter(
    df["z"].interpolate().bfill().ffill(),
    window_length=11,
    polyorder=2
)

df["vz_calc"] = df["z_smooth"].diff().fillna(0) * fps

z_baseline = df["z_smooth"].rolling(
    int(fps),
    center=True,
    min_periods=1
).median()

JUMP_HEIGHT_TH = 0.05
JUMP_VEL_TH = 0.35

df["is_jump_raw"] = (
    (df["z_smooth"] > z_baseline + JUMP_HEIGHT_TH)
    &
    (df["vz_calc"] > JUMP_VEL_TH)
)

MIN_JUMP_FRAMES = max(2, int(0.10 * fps))

'''df["is_jump"] = (
    df["is_jump_raw"]
    .rolling(MIN_JUMP_FRAMES, center=True, min_periods=1)
    .sum()
    >= 1
)'''

# =========================
# Swing angular velocity
# =========================
# 如果 CSV 已經有 active_ang_vel，就直接用
# 如果沒有，就用 right/left elbow angular velocity 產生

if "active_ang_vel" not in df.columns:
    if (
        "right_elbow_angle (deg)" in df.columns
        and "left_elbow_angle (deg)" in df.columns
    ):
        df["right_elbow_angle_smooth"] = savgol_filter(
            df["right_elbow_angle (deg)"].interpolate().bfill().ffill(),
            window_length=7,
            polyorder=2
        )

        df["left_elbow_angle_smooth"] = savgol_filter(
            df["left_elbow_angle (deg)"].interpolate().bfill().ffill(),
            window_length=7,
            polyorder=2
        )

        df["right_elbow_ang_vel"] = (
            pd.Series(df["right_elbow_angle_smooth"])
            .diff()
            .abs()
            .fillna(0)
            * fps
        )

        df["left_elbow_ang_vel"] = (
            pd.Series(df["left_elbow_angle_smooth"])
            .diff()
            .abs()
            .fillna(0)
            * fps
        )

        df["right_elbow_ang_vel"] = df["right_elbow_ang_vel"].clip(0, 4000)
        df["left_elbow_ang_vel"] = df["left_elbow_ang_vel"].clip(0, 4000)

        df["active_ang_vel"] = np.maximum(
            df["right_elbow_ang_vel"],
            df["left_elbow_ang_vel"]
        )

        df["active_angle"] = np.where(
            df["right_elbow_ang_vel"] >= df["left_elbow_ang_vel"],
            df["right_elbow_angle_smooth"],
            df["left_elbow_angle_smooth"]
        )

        df["is_swing"] = (
            (df["active_angle"] < 140)
            &
            (df["active_ang_vel"] > 80)
        )
    else:
        df["active_ang_vel"] = 0
        df["is_swing"] = False

else:
    df["active_ang_vel"] = df["active_ang_vel"].fillna(0).clip(0, 4000)

    if "is_swing" not in df.columns:
        df["is_swing"] = df["active_ang_vel"] > 80

# =========================
# Rotational energy from active_ang_vel
# =========================
# forearm + hand mass
m_arm = 0.0223 * weight_kg

# simplified center of mass distance from elbow
L_forearm = 0.146 * height_m
L_hand = 0.108 * height_m
r = 0.43 * L_forearm + L_hand

I_elbow = m_arm * r**2

df["omega_rad"] = np.deg2rad(
    df["active_ang_vel"].clip(0, 4000)
)

df["rot_energy_J"] = (
    0.5
    * I_elbow
    * df["omega_rad"]**2
)

df["rot_energy_J"] = np.where(
    df["is_swing"],
    df["rot_energy_J"],
    0
)

ETA = 0.10

df["rot_metabolic_J"] = df["rot_energy_J"] / ETA
df["rot_power_W"] = df["rot_metabolic_J"] / dt

# 1 MET = 1.225 W/kg
df["MET_swing_rot"] = (
    df["rot_power_W"]
    /
    (1.225 * weight_kg)
)

df["MET_swing_rot"] = df["MET_swing_rot"].clip(0, 4.0)

# =========================
# Badminton MET model
# =========================
BADMINTON_BASE_MET = 5.5
BADMINTON_MATCH_MET = 9.0


speed_norm = df["kalman_v"].clip(0, 3) / 3

df["MET_movement"] = (
    BADMINTON_BASE_MET
    +
    speed_norm * (BADMINTON_MATCH_MET - BADMINTON_BASE_MET)
)
df["MET"] = np.where(df["jump"],JUMP_ROPE_MOD_MET, df["kalman_v"].apply(speed_to_met_compendium))
df["MET"] = df["MET"] + df["MET_swing_rot"]
'''df["MET"] = (
    df["MET_movement"]
    +
    df["MET_swing_rot"]
)

# jump uses rope jumping as upper reference
df["MET"] = np.where(
    df["is_jump"],
    np.maximum(df["MET"], JUMP_ROPE_MOD_MET),
    df["MET"]
)

# jump + swing = jump smash
df["is_jump_smash"] = df["is_jump"] & df["is_swing"]

df["MET"] = np.where(
    df["is_jump_smash"],
    np.maximum(df["MET"], JUMP_ROPE_FAST_MET),
    df["MET"]
)'''

df["MET"] = df["MET"].clip(1.0, 12.3)

# =========================
# Calories
# =========================
df["kcal_per_frame"] = (
    df["MET"]
    * weight_kg
    * dt
    / 3600
)

df["calories_cumsum"] = df["kcal_per_frame"].cumsum()

avg_met = df["MET"].mean()
total_kcal = df["calories_cumsum"].iloc[-1]

print(f"FPS: {fps:.2f}")
print(f"Average MET: {avg_met:.2f}")
print(f"Total Calories: {total_kcal:.2f} kcal")
print(f"Jump frames: {df['jump'].sum()}")
print(f"Swing frames: {df['is_swing'].sum()}")
#print(f"Jump smash frames: {df['is_jump_smash'].sum()}")

# =========================
# Plots
# =========================
plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["calories_cumsum"], linewidth=2)
plt.xlabel("Time (sec)")
plt.ylabel("Calories (kcal)")
plt.title("Estimated Calorie Burn")
plt.grid(True)
plt.savefig(f"calories_vs_time_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["MET"], linewidth=2, label="Dynamic MET")
plt.xlabel("Time (sec)")
plt.ylabel("MET")
plt.title("Dynamic MET")
plt.grid(True)
plt.legend()
plt.savefig(f"met_vs_time_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["kalman_v"], linewidth=2, label="Kalman Speed")
plt.xlabel("Time (sec)")
plt.ylabel("Speed (m/s)")
plt.title("Player Speed")
plt.grid(True)
plt.legend()
plt.savefig(f"kalman_speed_vs_time_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["MET_swing_rot"], linewidth=2, label="Swing Rotational MET")
plt.xlabel("Time (sec)")
plt.ylabel("MET")
plt.title("Swing Rotational MET from Elbow Angular Velocity")
plt.grid(True)
plt.legend()
plt.savefig(f"swing_rot_met_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

# =========================
# Save CSV
# =========================
out_csv = f"Player1_trajectory_with_dynamic_MET_{safe_folder_name}_{date}.csv"
df.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")