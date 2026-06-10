import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import savgol_filter
from pathlib import Path
from datetime import datetime


RAW_WEIGHT = 0.8
KF_WEIGHT = 1.0 - RAW_WEIGHT

date = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = f"output_dynamic_MET_{RAW_WEIGHT}_{date}"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

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
#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"

VIDEO_A = f"{INPUT_PATH}/CameraReader_0.mp4"


#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260530_232754\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260530_232754_with_swing.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260530_233444\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260530_233444_with_swing.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260610_014941\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260610_014941_with_swing.csv"
csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260611_031712\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260611_031712_with_swing.csv"
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
df["dz"] = df["z"].diff().fillna(0)

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
df["speed_fused_mps"] = KF_WEIGHT * df["kalman_v"] + RAW_WEIGHT * df["speed_mps"]

df["speed_fused_mps"] = savgol_filter(
    df["speed_fused_mps"],
    window_length=21,
    polyorder=2
)

df["speed_kmh"] = df["speed_fused_mps"] * 3.6

total_distance = df["dist_m"].sum()
avg_speed = df["speed_fused_mps"].mean()
peak_speed = df["speed_fused_mps"].max()

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

ETA = 0.15 # efficiency factor to convert mechanical energy to metabolic energy, 0.10 -> 0.15

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


speed_norm = df["speed_fused_mps"].clip(0, 3) / 3

df["MET_movement"] = (
    BADMINTON_BASE_MET
    +
    speed_norm * (BADMINTON_MATCH_MET - BADMINTON_BASE_MET)
)
print(df["jump"].value_counts())
df["MET"] = np.where(df["jump"],JUMP_ROPE_MOD_MET, df["speed_fused_mps"].apply(speed_to_met_compendium))
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

# =========================
# Player Load
# =========================
# =========================
# Player Load from Kalman-smoothed trajectory
# =========================

def kalman_1d(z, q=0.01, r=0.05, dt=1/30):
    """
    1D constant-velocity Kalman filter.
    state = [position, velocity]
    """
    z = pd.Series(z).interpolate().bfill().ffill().to_numpy()

    if len(z) >= 2:
        initial_vel = (z[1] - z[0]) / dt
    else:
        initial_vel = 0
    x = np.array([z[0], initial_vel])
    P = np.eye(2)

    F = np.array([
        [1, dt],
        [0, 1]
    ])

    H = np.array([[1, 0]])

    Q = q * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2]
    ])

    R = np.array([[r]])

    pos = []
    vel = []

    for zi in z:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        # update
        y = np.array([zi]) - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        pos.append(x[0])
        vel.append(x[1])

    return np.array(pos), np.array(vel)


df["x_orig"] = df["x"]
df["y_orig"] = df["y"]
# Kalman smooth x/y/z
df["x_kf"], _ = kalman_1d(
    df["x"],
    q=0.01,
    r=0.1, #0.05 -> 0.1
    dt=dt
)

df["y_kf"], _ = kalman_1d(
    df["y"],
    q=0.01,
    r=0.1,
    dt=dt
)

df["z_kf"], _ = kalman_1d(
    df["z"],
    q=0.01,
    r=0.1,
    dt=dt
)
'''
df["vx_kf"] = np.gradient(df["x_kf"], dt)
df["vy_kf"] = np.gradient(df["y_kf"], dt)
df["vz_kf"] = np.gradient(df["z_kf"], dt)'''

df["vx_diff"] = df["dx"] / dt
df["vy_diff"] = df["dy"] / dt
df["vz_diff"] = df["dz"] / dt

#df["ax_diff"] = np.gradient(df["vx_diff"], dt, edge_order=2)
#df["ay_diff"] = np.gradient(df["vy_diff"], dt, edge_order=2)
#df["az_diff"] = np.gradient(df["vz_diff"], dt, edge_order=2)
df["ax_diff"] = df["vx_diff"].diff().fillna(0) / dt
df["ay_diff"] = df["vy_diff"].diff().fillna(0) / dt
df["az_diff"] = df["vz_diff"].diff().fillna(0) / dt

df["vx_kf"] = np.gradient(df["x_kf"], dt, edge_order=2)
df["vy_kf"] = np.gradient(df["y_kf"], dt, edge_order=2)
df["vz_kf"] = np.gradient(df["z_kf"], dt, edge_order=2)

# acceleration from Kalman velocity
df["ax_kf"] = np.gradient(df["vx_kf"], dt, edge_order=2)
df["ay_kf"] = np.gradient(df["vy_kf"], dt, edge_order=2)
df["az_kf"] = np.gradient(df["vz_kf"], dt, edge_order=2)

# smooth acceleration
for c in ["ax_kf", "ay_kf", "az_kf"]:
    df[c] = savgol_filter(
        df[c],
        window_length=11,
        polyorder=2
    )

df["vx_combined"] = KF_WEIGHT * df["vx"] + RAW_WEIGHT * df["vx_diff"]
df["vy_combined"] = KF_WEIGHT * df["vy"] + RAW_WEIGHT * df["vy_diff"]
df["vz_combined"] = KF_WEIGHT * df["vz"] + RAW_WEIGHT * df["vz_diff"]
# delta acceleration
'''df["dax"] = df["ax_kf"].diff().fillna(0)
df["day"] = df["ay_kf"].diff().fillna(0)
df["daz"] = df["az_kf"].diff().fillna(0)'''

df["ax_combined"] = np.gradient(df["vx_combined"], dt, edge_order=2)
df["ay_combined"] = np.gradient(df["vy_combined"], dt, edge_order=2)
df["az_combined"] = np.gradient(df["vz_combined"], dt, edge_order=2)
for c in ["ax_combined", "ay_combined", "az_combined"]:
    df[c] = savgol_filter(
        df[c],
        window_length=21,
        polyorder=2
    )

df["dax"] = df["ax_combined"].diff().fillna(0)
df["day"] = df["ay_combined"].diff().fillna(0)
df["daz"] = df["az_combined"].diff().fillna(0)

df["ax"] = df["vx"].diff().fillna(0) / dt
df["ay"] = df["vy"].diff().fillna(0) / dt
df["az"] = df["vz"].diff().fillna(0) / dt

# Player Load per frame
# /10: scale to arbitrary units

df["PL_raw"] = (np.sqrt(
    df["ax_diff"]**2 +df["ay_diff"]**2 + df["az_diff"]**2) / 10000)

df["PL"] = (np.sqrt(df["dax"]**2 + df["day"]**2 + df["daz"]**2) / 100)

df["PL_raw"] = df["PL_raw"].clip(0, df["PL_raw"].quantile(0.95))
df["PL"] = df["PL"].clip(
    lower=0,
    upper=df["PL"].quantile(0.95)
)
df["PL"] = savgol_filter(
    df["PL"],
    window_length=21,
    polyorder=2
)
df["PL_norm"] = df["PL"] / df["PL"].quantile(0.95)

df["PL_norm"] = df["PL_norm"].clip(0, 1.0)
# remove extreme spikes
'''df["PL"] = df["PL"].clip(
    lower=0,
    upper=df["PL"].quantile(0.99)
)'''

# rolling PL/min
window_sec = 1
#window = max(3, int(window_sec * fps))
window = int(window_sec * fps)
'''
df["PL_sum"] = (
    df["PL"]
    .rolling(window, min_periods=1)
    .sum()
)

df["PL_window_sec"] = (
    df["PL"]
    .rolling(window, min_periods=1)
    .count()
    * dt
)

df["PL_per_min"] = (
    df["PL_sum"]
    / df["PL_window_sec"]
    * 60
)'''

df["PL_per_min"] = (
    df["PL"]
    .rolling(
        window,
        center=True,
        min_periods=1
    )
    .mean()
    * 60
)

df["PL_catapult_per_min"] = (
    df["PL"].rolling(
        window,
        center=True,
        min_periods=1
    ).sum()
    / window_sec
    * 60
)
total_pl = df["PL"].sum()

pl_per_min_catapult = (
    total_pl
    /
    (df["time_sec"].iloc[-1] / 60)/100
)

print(f"Total PL: {total_pl:.2f} AU")
print(f"PL/min (Catapult): {pl_per_min_catapult:.2f} AU/min")

df["PL_per_min"] = (df["PL"]
    .rolling(
        window,
        center=True,
        min_periods=1
    )
    .mean()
    / window_sec
    * 60
)
# smooth
df["PL_per_min"] = savgol_filter(
    df["PL_per_min"],
    window_length=11, #21 -> 11
    polyorder=2
)
df["PL_catapult_per_min"] = savgol_filter(
    df["PL_catapult_per_min"],
    window_length=11,
    polyorder=2
)
df["PL_catapult_per_min"] = df["PL_catapult_per_min"].clip(
    0, df["PL_catapult_per_min"].quantile(0.99))

df["PL_per_min"] = df["PL_per_min"].clip(
    lower=0,
    upper=df["PL_per_min"].quantile(0.99)
)
warmup_sec = 1
warmup_frames  = int(fps * warmup_sec)

#df.loc[:warmup_frames, "PL_per_min"] = df.loc[:warmup_frames, "PL_raw_per_min"]
ignore_sec = 0
valid_time_mask = df["time_sec"] >= ignore_sec

avg_pl_per_min = df["PL_per_min"].mean()
peak_pl_per_min = df["PL_per_min"].max()
#total_pl = df["PL"].sum()

peak_idx = df["PL_per_min"].idxmax()
peak_idx_info = df.loc[peak_idx, ["frame_id", "time_sec", "PL_per_min", "MET"]]


#print(f"Average PL/min: {avg_pl_per_min:.2f} AU/min")
#print(f"Peak PL/min: {peak_pl_per_min:.2f} AU/min")
#print(f"Total PL: {total_pl:.2f} AU")

#avg_pl_catapult_per_min = df["PL_catapult_per_min"].mean()
#peak_pl_catapult_per_min = df["PL_catapult_per_min"].max()
total_pl_catapult = df["PL_catapult_per_min"].sum()

peak_idx = df["PL_catapult_per_min"].idxmax()
peak_idx_info = df.loc[peak_idx, ["frame_id", "time_sec", "PL_catapult_per_min", "MET"]]

avg_pl_catapult_per_min = df["PL"].mean()
peak_pl_catapult_per_min = df["PL"].max()

#print(f"Average PL: {avg_pl_catapult_per_min:.2f} AU")
print(f"Peak PL: {peak_pl_catapult_per_min:.2f} AU")
#print(f"Total PL: {total_pl_catapult:.2f} AU")

acc_mag = np.sqrt(
    df["ax_combined"]**2 +
    df["ay_combined"]**2 +
    df["az_combined"]**2
)

df["MAD"] = (
    acc_mag
    .rolling(
        window,
        center=True,
        min_periods=1
    )
    .apply(
        lambda x:
            np.mean(
                np.abs(
                    x - np.mean(x)
                )
            ),
        raw=True
    )
)

df["MAD_Center_False"] = (
    acc_mag
    .rolling(
        window,
        center=False,
        min_periods=1
    )
    .apply(
        lambda x:
            np.mean(
                np.abs(
                    x - np.mean(x)
                )
            ),
        raw=True
    )
)

# acceleration (m/s²)
'''
df["ax"] = df["vx"].diff().fillna(0) / dt
df["ay"] = df["vy"].diff().fillna(0) / dt
df["az"] = df["vz_calc"].diff().fillna(0) / dt

# smooth acceleration first
for c in ["ax","ay","az"]:
    df[c] = savgol_filter(
        df[c],
        window_length=11,
        polyorder=2
    )
# change in acceleration
df["dax"] = df["ax"].diff().fillna(0)
df["day"] = df["ay"].diff().fillna(0)
df["daz"] = df["az"].diff().fillna(0)

# Player Load
df["PL"] = np.sqrt(df["dax"]**2 + df["day"]**2 + df["daz"]**2)

# smooth
df["PL"] = savgol_filter(df["PL"], window_length=11, polyorder=2)

# rolling PL/min
window_sec = 5
window = int(window_sec * fps)

df["PL_sum"] = (
    df["PL"]
    .rolling(
        window,
        min_periods=1
    )
    .sum()
)

df["PL_per_min"] = (
    df["PL_sum"]
    /
    (window_sec/60)
)
df["PL_per_min"] = df["PL_per_min"].clip(
    upper=df["PL_per_min"].quantile(0.99)
)'''
print(f"FPS: {fps:.2f}")
print(f"Average MET: {avg_met:.2f}")
print(f"Total Calories: {total_kcal:.2f} kcal")
print(f"Jump frames: {df['jump'].sum()}")
print(f"Swing frames: {df['is_swing'].sum()}")
#print(f"Jump smash frames: {df['is_jump_smash'].sum()}")

# =========================================================
# HIP-based Speed / Acceleration / PL / MAD
# =========================================================

# Hip velocity
df["speed_hip_mps"] = np.sqrt(
    df["vx_hip"]**2 +
    df["vy_hip"]**2
)

df["vx_hip_diff"] = np.gradient(df["x_hip"], dt)
df["vy_hip_diff"] = np.gradient(df["y_hip"], dt)
df["vz_hip_diff"] = np.gradient(df["z_hip"], dt)

df["vx_hip_combined"] = KF_WEIGHT * df["vx_hip"] + RAW_WEIGHT * df["vx_hip_diff"]
df["vy_hip_combined"] = KF_WEIGHT * df["vy_hip"] + RAW_WEIGHT * df["vy_hip_diff"]
df["vz_hip_combined"] = KF_WEIGHT * df["vz_hip"] + RAW_WEIGHT * df["vz_hip_diff"]

# Hip acceleration
df["ax_hip"] = np.gradient(
    df["vx_hip"],
    dt,
    edge_order=2
)

df["ay_hip"] = np.gradient(
    df["vy_hip"],
    dt,
    edge_order=2
)

df["az_hip"] = np.gradient(
    df["vz_hip"],
    dt,
    edge_order=2
)

df["ax_hip_combined"] = np.gradient(
    df["vx_hip_combined"],
    dt,
    edge_order=2
)
df["ay_hip_combined"] = np.gradient(
    df["vy_hip_combined"],
    dt,
    edge_order=2
)

df["az_hip_combined"] = np.gradient(
    df["vz_hip_combined"],
    dt,
    edge_order=2
)

for c in ["ax_hip", "ay_hip", "az_hip"]:
    df[c] = savgol_filter(
        df[c],
        window_length=21,
        polyorder=2
    )

for c in ["ax_hip_combined", "ay_hip_combined", "az_hip_combined"]:
    df[c] = savgol_filter(
        df[c],
        window_length=21,
        polyorder=2
    )

# Hip jerk
df["dax_hip"] = df["ax_hip"].diff().fillna(0)
df["day_hip"] = df["ay_hip"].diff().fillna(0)
df["daz_hip"] = df["az_hip"].diff().fillna(0)

df["dax_hip_combined"] = df["ax_hip_combined"].diff().fillna(0)
df["day_hip_combined"] = df["ay_hip_combined"].diff().fillna(0)
df["daz_hip_combined"] = df["az_hip_combined"].diff().fillna(0)
# Hip Player Load
df["PL_hip"] = np.sqrt(
    df["dax_hip"]**2 +
    df["day_hip"]**2 +
    df["daz_hip"]**2
)/100

df["PL_hip_combined"] = np.sqrt(
    df["dax_hip_combined"]**2 +
    df["day_hip_combined"]**2 +
    df["daz_hip_combined"]**2
)/100


df["PL_hip"] = df["PL_hip"].clip(
    lower=0,
    upper=df["PL_hip"].quantile(0.95)
)

df["PL_hip_combined"] = df["PL_hip_combined"].clip(
    lower=0,
    upper=df["PL_hip_combined"].quantile(0.95)
)

df["PL_hip"] = savgol_filter(
    df["PL_hip"],
    window_length=21,
    polyorder=2
)


df["PL_hip_combined"] = savgol_filter(
    df["PL_hip_combined"],
    window_length=21,
    polyorder=2
)

# Hip acceleration magnitude
acc_mag_hip = np.sqrt(
    df["ax_hip"]**2 +
    df["ay_hip"]**2 +
    df["az_hip"]**2
)
#print(acc_mag_hip.describe())

# Hip MAD
df["MAD_hip"] = (
    acc_mag_hip
    .rolling(
        window,
        center=True,
        min_periods=1
    )
    .apply(
        lambda x: np.mean(
            np.abs(
                x - np.mean(x)
            )
        ),
        raw=True
    )
)

acc_combined_mag_hip = np.sqrt(
    df["ax_hip_combined"]**2 +
    df["ay_hip_combined"]**2 +
    df["az_hip_combined"]**2
)

# Hip MAD
df["MAD_hip_combined"] = (
    acc_combined_mag_hip
    .rolling(
        window,
        center=True,
        min_periods=1
    )
    .apply(
        lambda x: np.mean(
            np.abs(
                x - np.mean(x)
            )
        ),
        raw=True
    )
)

# =========================
# Plots
# =========================
plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["calories_cumsum"], linewidth=2)
plt.xlabel("Time (sec)")
plt.ylabel("Calories (kcal)")
plt.title("Estimated Calorie Burn")
plt.grid(True)
plt.savefig(f"{OUTPUT_FOLDER}/calories_vs_time_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["MET"], linewidth=2, label="Dynamic MET")
plt.xlabel("Time (sec)")
plt.ylabel("MET")
plt.title("Dynamic MET")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/met_vs_time_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["speed_fused_mps"], linewidth=2, label="Fused Speed")
plt.xlabel("Time (sec)")
plt.ylabel("Speed (m/s)")
plt.title("Player Speed")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/fused_speed_vs_time_{safe_folder_name}_{date}.png", dpi=300)
plt.show()

'''
plt.figure(figsize=(12, 4))
plt.plot(df.loc[valid_time_mask, "time_sec"], df.loc[valid_time_mask, "PL_per_min"], linewidth=2, label="Player Load/min")
plt.xlabel("Time (sec)")
plt.ylabel("PL/min (AU/min)")
plt.title("Player Load per Minute")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/playerload_per_min_{safe_folder_name}_{date}.png", dpi=300)
plt.show()'''

plt.figure(figsize=(12, 4))
plt.plot(df["time_sec"], df["MET_swing_rot"], linewidth=2, label="Swing Rotational MET")
plt.xlabel("Time (sec)")
plt.ylabel("MET")
plt.title("Swing Rotational MET from Elbow Angular Velocity")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/swing_rot_met_{safe_folder_name}_{date}.png", dpi=300)
plt.show()
'''
plt.figure(figsize=(12, 4))
plt.plot(df.loc[valid_time_mask, "time_sec"], df.loc[valid_time_mask, "PL_catapult_per_min"], linewidth=2, label="Player Load/min")
plt.xlabel("Time (sec)")  
plt.ylabel("PL_Catapult/min (AU/min)")
plt.title("Player Load per Minute")
plt.grid(True)
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/playerload_catapult_per_min_{safe_folder_name}_{date}.png", dpi=300)
plt.show()'''

plt.figure(figsize=(12,4))
plt.plot(
    df["time_sec"],
    df["PL"],
    linewidth=1
)
plt.xlabel("Time (sec)")
plt.ylabel("PL (AU)")
plt.title("Instantaneous Player Load")
plt.grid(True)
plt.savefig(
    f"{OUTPUT_FOLDER}/playerload_vs_time.png",
    dpi=300
)
plt.show()


plt.figure(figsize=(12,4))
plt.plot(
    df["time_sec"],
    df["MAD"],
    linewidth=1,
    color="blue",
    label="MAD Center True",
)
plt.plot(
    df["time_sec"],
    df["MAD_Center_False"],
    linewidth=1,
    color="orange",
    label="MAD Center False"
)
plt.xlabel("Time (sec)")
plt.ylabel("MAD (AU)")
plt.title(" Mean Absolute Deviation of Acceleration")
plt.legend()
plt.grid(True)
plt.savefig(
    f"{OUTPUT_FOLDER}/mad_vs_time.png",
    dpi=300
)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(
    df["time_sec"],
    df["PL"],
    label="Ankle PL"
)
plt.plot(
    df["time_sec"],
    df["PL_hip"],
    label="Hip PL"
)
plt.xlabel("Time (sec)")
plt.ylabel("PL")
plt.title("Ankle vs Hip Player Load")
plt.legend()
plt.grid(True)

plt.savefig(
    f"{OUTPUT_FOLDER}/pl_ankle_vs_hip.png",
    dpi=300
)
plt.show()


plt.figure(figsize=(12,4))
plt.plot(
    df["time_sec"],
    df["PL"],
    label="Ankle PL"
)
plt.plot(
    df["time_sec"],
    df["PL_hip_combined"],
    label="Hip PL Combined"
)
plt.xlabel("Time (sec)")
plt.ylabel("PL")
plt.title("Ankle vs Hip_Combined Player Load")
plt.legend()
plt.grid(True)

plt.savefig(
    f"{OUTPUT_FOLDER}/pl_ankle_vs_hip_combined.png",
    dpi=300
)
plt.show()

plt.figure(figsize=(12,4))
'''plt.plot(
    df["time_sec"],
    df["MAD"],
    label="Ankle MAD"
)'''

plt.plot(
    df["time_sec"],
    df["MAD_hip"],
    label="Hip MAD"
)

plt.xlabel("Time (sec)")
plt.ylabel("MAD")
plt.title("Hip MAD")
plt.legend()
plt.grid(True)

plt.savefig(
    f"{OUTPUT_FOLDER}/mad_hip.png",
    dpi=300
)

plt.show()



plt.figure(figsize=(12,4))

plt.plot(
    df["time_sec"],
    df["MAD_hip_combined"],
    label="Hip MAD Combined"
)

plt.xlabel("Time (sec)")
plt.ylabel("MAD")
plt.title("Hip MAD Combined")
plt.legend()
plt.grid(True)

plt.savefig(
    f"{OUTPUT_FOLDER}/mad_hip_combined.png",
    dpi=300
)

plt.show()



# =========================
# Summary Report
# =========================

summary_txt = f"{OUTPUT_FOLDER}/summary.txt"
with open(summary_txt, "w") as f:
    f.write("=====================================\n")
    f.write("Badminton Motion Analysis Summary\n")
    f.write("=====================================\n\n")

    f.write(f"FPS: {fps:.2f}\n")
    f.write(f"Average MET: {avg_met:.2f}\n")
    f.write(f"Total Calories: {total_kcal:.2f} kcal\n")
    #f.write(f"Average PL/min: {avg_pl_per_min:.2f} AU/min\n")
    #f.write(f"Peak PL/min: {peak_pl_per_min:.2f} AU/min\n")
    f.write(f"Total PL: {total_pl:.2f} AU\n")
    #f.write(f"Average PL (Catapult): {avg_pl_catapult_per_min:.2f} AU\n")
    f.write(f"Peak PL (Catapult): {peak_pl_catapult_per_min:.2f} AU\n")
    f.write(f"PL/min (Catapult): {pl_per_min_catapult:.2f} AU/min\n")
    f.write(f"Jump frames: {df['jump'].sum()}\n")
    f.write(f"Swing frames: {df['is_swing'].sum()}\n")
    f.write(f"Total Distance: {total_distance:.2f} m\n")
    f.write(f"Average Speed: {avg_speed:.2f} m/s\n")
    f.write(f"Peak Speed: {peak_speed:.2f} m/s\n")
    f.write(f"Peak PL Time: {peak_idx_info['time_sec']:.2f} sec\n")
    f.write(f"Peak MET: {peak_idx_info['MET']:.2f}\n")
    #f.write(f"Jump smash frames: {df['is_jump_smash'].sum()}\n")
    print(f"Saved: {summary_txt}")


# =========================
# Save CSV
# =========================
out_csv = f"{OUTPUT_FOLDER}/Player1_trajectory_with_dynamic_MET_{safe_folder_name}_{date}.csv"
df.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")