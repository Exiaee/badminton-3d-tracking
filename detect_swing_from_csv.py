import cv2
import numpy as np
import pandas as pd
import configparser, json
import os
from skeleton_util import BodyKpt, SKELETON_CONNECTIONS, court_3d
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from datetime import datetime
from pathlib import Path
# 4 cameras in 4 corners, id: 0, 1, 2, 3
date = datetime.now().strftime("%Y%m%d_%H%M%S")

INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"
#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"

folder_name = Path(INPUT_PATH).name
CAM_PAIRS = [(0, 2)]
num_keypoints = 17

#player_anchor = "middle"
player_anchor = "right_ankel"

fill_method = "akima"
#fill_method = "pchip"

folder_name = f"{folder_name}_{player_anchor}_{fill_method}"

SELECTED_PAIR = 0 # 0 for pair (0, 1), 1 for pair (2, 3)
VIDEO_A = f"{INPUT_PATH}/CameraReader_{CAM_PAIRS[SELECTED_PAIR][0]}.mp4"

capA = cv2.VideoCapture(VIDEO_A)
fps_a = capA.get(cv2.CAP_PROP_FPS)

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260913_202856\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260913_202856.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20261012_231118\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20261012_231118.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20261013_234536\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20261013_234536.csv"
#sv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260918_030220\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260918_030220.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260530_233444\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260530_233444.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260530_232754\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260530_232754.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260720_030005\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260720_030005.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260611_031712\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260611_031712.csv"
#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260616_005512\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260616_005512.csv"

dataset_name = Path(INPUT_PATH).name

suppoeted_dataset = {
    "2026-04-09_19-12-21",
    "2026-04-09_19-13-28"
}

if dataset_name not in suppoeted_dataset:
    raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {suppoeted_dataset}")

base_dir = Path(
    r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking"
)

csv_files =  list(base_dir.glob(
    f"badminton_motion_analysis_{dataset_name}_*"
    f"/Player1_trajectory_right_ankel_{dataset_name}"
    f"_right_ankel_akima_*.csv"
))

if not csv_files:
    raise FileNotFoundError(
        f"Cannot find trajectory CSV for {dataset_name}"
    )

csv_path = str(max(csv_files, key=lambda f: f.stat().st_mtime))

print(f"Using CSV: {csv_path}")

fps = fps_a 

df = pd.read_csv(csv_path)
ANG_VEL_LIMIT = 4000   # deg/s，可依文獻與資料品質調整
SMOOTH_WIN = 7
SMOOTH_POLY = 2

def smooth_angle(series, window=7, poly=2):
    x = series.interpolate().bfill().ffill().to_numpy()

    # window 必須是奇數，且不能大於資料長度
    if len(x) < window:
        return x

    if window % 2 == 0:
        window += 1

    return savgol_filter(x, window_length=window, polyorder=poly)


def calc_ang_vel_deg(angle_deg, fps, limit=4000):
    angle_smooth = smooth_angle(angle_deg)

    ang_vel = (
        pd.Series(angle_smooth)
        .diff()
        .abs()
        * fps
    )

    ang_vel = (
        ang_vel
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(0, limit)
    )

    return angle_smooth, ang_vel

'''df["right_elbow_ang_vel"] = (
    df["right_elbow_angle (deg)"]
    .diff()
    .abs()
    * fps
)

df["left_elbow_ang_vel"] = (
    df["left_elbow_angle (deg)"]
    .diff()
    .abs()
    * fps
)'''

df["right_elbow_angle_smooth"], df["right_elbow_ang_vel"] = calc_ang_vel_deg(
    df["right_elbow_angle (deg)"],
    #df["right_elbow_angle_smooth"],
    fps,
    limit=ANG_VEL_LIMIT
)

df["left_elbow_angle_smooth"], df["left_elbow_ang_vel"] = calc_ang_vel_deg(
    df["left_elbow_angle (deg)"],
    #df["left_elbow_angle_smooth"],
    fps,
    limit=ANG_VEL_LIMIT
)
'''
df["active_angle"] = np.where(
    df["right_elbow_ang_vel"] >= df["left_elbow_ang_vel"],
    #df["right_elbow_angle (deg)"],
    #df["left_elbow_angle (deg)"]
    df["right_elbow_angle_smooth"],
    df["left_elbow_angle_smooth"]
)

df["active_ang_vel"] = np.maximum(
    df["right_elbow_ang_vel"],
    df["left_elbow_ang_vel"]
)'''

df["active_angle"] = df["right_elbow_angle_smooth"].copy()
df["active_ang_vel"] = df["right_elbow_ang_vel"].copy()


# Frame-level swing detection
active_angle_threshold = 150  # degrees
active_ang_vel_threshold = 120  # deg/s
df["is_swing"] = (
    (df["active_angle"] < active_angle_threshold)
    #&
    &
    (df["active_ang_vel"] > active_ang_vel_threshold)
)

# Detect the start of each swing event
df["swing_start"] = df["is_swing"] & ~df["is_swing"].shift(fill_value=False)

swing_count = df["swing_start"].sum()


out_path = Path(csv_path).with_name(
    Path(csv_path).stem + "_with_swing.csv"
)

df.to_csv(out_path, index=False)

print(f"Detected swing frames: {df['is_swing'].sum()}")
print(f"Swing Count: {swing_count}")
print(f"Saved: {out_path}")