import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

from scipy.signal import savgol_filter

from pathlib import Path

from datetime import datetime

import argparse



parser = argparse.ArgumentParser(description="Badminton Dynamic MET Estimation")

parser.add_argument("--height", type=float, default=1.70, help="Height of the player in meters")

parser.add_argument("--weight", type=float, default=65.0, help="Weight of the player in kg")

args = parser.parse_args()

RAW_WEIGHT = 0.8

KF_WEIGHT = 1.0 - RAW_WEIGHT



date = datetime.now().strftime("%Y%m%d_%H%M%S")

#OUTPUT_FOLDER = f"output_dynamic_MET_{RAW_WEIGHT}_{dataset_name}_{date}"

#Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)



JUMP_ROPE_SLOW_MET = 8.3

JUMP_ROPE_MOD_MET = 11.8

JUMP_ROPE_FAST_MET = 12.3



# Jump-state reference intensity

JUMP_REFERENCE_MET = 8.9







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



weight_kg = args.weight

height_m = args.height



#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"

INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"



dataset_name = Path(INPUT_PATH).name

VIDEO_A = f"{INPUT_PATH}/CameraReader_0.mp4"



OUTPUT_FOLDER = f"output_dynamic_MET_{RAW_WEIGHT}_{dataset_name}_{date}"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)



SCRPT_PATH = Path(__file__).resolve().parent

ANNOTATION_DIR = SCRPT_PATH/"swing_annotations_two_datasets"
ANNOTATION_PATH = ANNOTATION_DIR/f"swing_annotations_{dataset_name}.csv"

if not ANNOTATION_PATH.is_file():
    raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_PATH}")


#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260530_232754\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260530_232754_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260530_233444\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260530_233444_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260610_014941\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260610_014941_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260611_031712\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260611_031712_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260616_005512\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260616_005512_with_swing.csv"

#csv_path =  r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260720_030005\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260720_030005_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260918_030220\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260918_030220_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260913_202856\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260913_202856_with_swing.csv"






import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

from scipy.signal import savgol_filter

from pathlib import Path

from datetime import datetime

import argparse



parser = argparse.ArgumentParser(description="Badminton Dynamic MET Estimation")

parser.add_argument("--height", type=float, default=1.70, help="Height of the player in meters")

parser.add_argument("--weight", type=float, default=65.0, help="Weight of the player in kg")

args = parser.parse_args()

RAW_WEIGHT = 0.8

KF_WEIGHT = 1.0 - RAW_WEIGHT



date = datetime.now().strftime("%Y%m%d_%H%M%S")

#OUTPUT_FOLDER = f"output_dynamic_MET_{RAW_WEIGHT}_{dataset_name}_{date}"

#Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)



JUMP_ROPE_SLOW_MET = 8.3

JUMP_ROPE_MOD_MET = 11.8

JUMP_ROPE_FAST_MET = 12.3



# Jump-state reference intensity

JUMP_REFERENCE_MET = 8.9







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



weight_kg = args.weight

height_m = args.height



INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"

#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"



dataset_name = Path(INPUT_PATH).name

VIDEO_A = f"{INPUT_PATH}/CameraReader_0.mp4"



OUTPUT_FOLDER = f"output_dynamic_MET_{RAW_WEIGHT}_{dataset_name}_{date}"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)



SCRPT_PATH = Path(__file__).resolve().parent

ANNOTATION_DIR = SCRPT_PATH/"swing_annotations_two_datasets"
ANNOTATION_PATH = ANNOTATION_DIR/f"swing_annotations_{dataset_name}.csv"

if not ANNOTATION_PATH.is_file():
    raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_PATH}")


#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260530_232754\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260530_232754_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260530_233444\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260530_233444_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260610_014941\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260610_014941_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260611_031712\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260611_031712_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260616_005512\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260616_005512_with_swing.csv"

#csv_path =  r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260720_030005\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260720_030005_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20260918_030220\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20260918_030220_with_swing.csv"

#csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20260913_202856\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20260913_202856_with_swing.csv"


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
    f"_right_ankel_akima_*_with_swing.csv"
))


print(csv_files)
if not csv_files:
    raise FileNotFoundError(
        f"Cannot find trajectory CSV for {dataset_name}"
    )

csv_path = str(max(csv_files, key=lambda f: f.stat().st_mtime))

print(f"Using CSV: {csv_path}")

r'''
if dataset_name == "2026-04-09_19-12-21":
    csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-12-21_20261012_231118\Player1_trajectory_right_ankel_2026-04-09_19-12-21_right_ankel_akima_20261012_231118_with_swing.csv"
elif dataset_name == "2026-04-09_19-13-28":
    csv_path = r"C:\D\NCTU_CS\Thesis\Lab_Data\Multiview_3d_Tracking\badminton_motion_analysis_2026-04-09_19-13-28_20261013_234536\Player1_trajectory_right_ankel_2026-04-09_19-13-28_right_ankel_akima_20261013_234536_with_swing.csv"
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")
folder_name = str(Path(csv_path).parent.relative_to(Path(csv_path).parents[1]))
'''

folder_name = str(Path(csv_path).parent.relative_to(Path(csv_path).parents[1]))

safe_folder_name = folder_name.replace("\\\\", "_")



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



df["dist_m"] = np.clip(df["dist_m"], 0, None)



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



df["speed_fused_mps"] = np.clip(df["speed_fused_mps"], 0, None)



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

    #center=True,

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



       # df["right_elbow_ang_vel"] = df["right_elbow_ang_vel"].clip(0, 4000)

        #df["left_elbow_ang_vel"] = df["left_elbow_ang_vel"].clip(0, 4000)



        df["active_ang_vel_raw"] = df["right_elbow_ang_vel"].copy()

        df["active_angle"] = df["right_elbow_angle_smooth"].copy()



        # Suppress short derivative spikes

        df["active_ang_vel"] = (

            df["active_ang_vel_raw"]

            .rolling(

                window=5,

                center=True,

                min_periods=1

            )

            .median()

        )

        '''df["active_ang_vel"] = np.maximum(

            df["right_elbow_ang_vel"],

            df["left_elbow_ang_vel"]

        )

        df["active_angle"] = np.where(

            df["right_elbow_ang_vel"] >= df["left_elbow_ang_vel"],

            df["right_elbow_angle_smooth"],

            df["left_elbow_angle_smooth"]

        )'''



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





'''

# =========================

# Rotational energy from active_ang_vel

# =========================

# forearm + hand mass

m_arm = 0.0223 * weight_kg



# simplified center of mass distance from elbow

L_forearm = 0.146 * height_m

L_hand = 0.108 * height_m



# Approximate hand COM at 50% of hand length from the wrist

r = 0.43 * L_forearm + 0.5 *  L_hand # 0.5 *  L_hand



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

)'''



#df["MET_swing_rot"] = df["MET_swing_rot"].clip(0, 4.0)



# =========================

# Swing MET - Event-based

# =========================



# Forearm + hand mass

m_arm = 0.0223 * weight_kg



# Segment length

L_forearm = 0.146 * height_m

L_hand = 0.108 * height_m



# Approximate COM distance from elbow

r = 0.43 * L_forearm + 0.5 * L_hand



# Moment of inertia

I_elbow = m_arm * r**2



# Mechanical → metabolic efficiency

ETA = 0.15



# Initialize Swing MET

df["MET_swing_rot"] = 0.0





# -------------------------

# Identify continuous Swing Events

# -------------------------

'''swing_start = (

    df["is_swing"]

    & ~df["is_swing"].shift(1, fill_value=False)

)



df["swing_event_id"] = swing_start.cumsum()'''



# =========================

# Merge short gaps between Swing frames

# =========================



MAX_GAP_SEC = 0.10

#MAX_GAP_FRAMES = int(MAX_GAP_SEC * fps)

MAX_GAP_FRAMES = int(np.ceil(MAX_GAP_SEC * fps))



swing = df["is_swing"].to_numpy().copy()



true_idx = np.where(swing)[0]



for i in range(len(true_idx) - 1):



    current_idx = true_idx[i]

    next_idx = true_idx[i + 1]



    gap = next_idx - current_idx - 1



    # Merge short gaps

    if 0 < gap <= MAX_GAP_FRAMES:

        swing[current_idx + 1:next_idx] = True



df["is_swing"] = swing



# =========================

# Remove very short Swing Events

# =========================

'''

MIN_SWING_SEC = 0.10

#MIN_SWING_FRAMES = int(MIN_SWING_SEC * fps)

MIN_SWING_FRAMES = int(np.ceil(MIN_SWING_SEC * fps))



swing_start = (

    df["is_swing"]

    & ~df["is_swing"].shift(1, fill_value=False)

)



swing_event_id = swing_start.cumsum()



for event_id in swing_event_id[df["is_swing"]].unique():



    mask = (

        (swing_event_id == event_id)

        & df["is_swing"]

    )



    if mask.sum() < MIN_SWING_FRAMES:

        df.loc[mask, "is_swing"] = False

    # =========================

    # Rebuild Swing Event IDs

    # after merge / filtering

    # =========================



    swing_start = (

        df["is_swing"]

        & ~df["is_swing"].shift(1, fill_value=False)

    )



    df["swing_event_id"] = swing_start.cumsum()'''



# Preserve original Swing detection

df["is_swing_raw"] = df["is_swing"].fillna(False).astype(bool)



# Merge gaps of up to 2 frames

MAX_GAP_FRAMES = 2



swing = df["is_swing_raw"].to_numpy().copy()

true_idx = np.flatnonzero(swing)



for i in range(len(true_idx) - 1):

    current_idx = true_idx[i]

    next_idx = true_idx[i + 1]



    gap = next_idx - current_idx - 1



    if 0 < gap <= MAX_GAP_FRAMES:

        swing[current_idx + 1:next_idx] = True



df["is_swing_event"] = swing



# Rebuild Swing Event ID

df["swing_start_event"] = (

    df["is_swing_event"]

    & ~df["is_swing_event"].shift(1, fill_value=False)

)



df["swing_event_id"] = df["swing_start_event"].cumsum()







# ==================================================
# Swing Event Detection from Manual Annotations
# ==================================================

'''
ANNOTATION_PATH = (
    f"swing_annotations_{dataset_name}.csv"
)'''

annotation_df = pd.read_csv(ANNOTATION_PATH)

required_columns =  {
    "sample_id", "start_frame", "end_frame", "label"
}

if not required_columns.issubset(annotation_df.columns):
    raise ValueError(f"Annotation CSV must contain columns: {required_columns}")

annoations = annotation_df.sort_values(by="start_frame").reset_index(drop=True)
# Preserve the original threshold-based detection
df["is_swing_event"] = False
df["swing_event_id"] = 0
df["swing_label"] = "non_swing"

for event_id, row in enumerate(annoations.itertuples(index=False), start=1):
    start = int(row.start_frame)
    end = int(row.end_frame)

    if end < start:
        raise ValueError(f"End frame {end} is before start frame {start} for event {event_id}")

    mask = df["frame_id"].between(start, end)
    if not mask.any():
        raise ValueError(f"No frames found in DataFrame for event {event_id} with start {start} and end {end}")

    if df.loc[mask, "is_swing_event"].any():
        raise ValueError(f"Overlapping swing events detected for event {event_id} with start {start} and end {end}")

    df.loc[mask, "is_swing_event"] = True
    df.loc[mask, "swing_event_id"] = event_id
    df.loc[mask, "swing_label"] = row.label

# Swing Start and End

df["swing_start_event"] = (
    df["is_swing_event"]
    & ~df["is_swing_event"].shift(1, fill_value=False)
)

df["swing_end_event"] = (
    df["is_swing_event"]
    & ~df["is_swing_event"].shift(-1, fill_value=False)
)

print(f"Total Swing Events Detected: {len(annoations)}")
print(f"Swing Event Labels:\n{annoations['label'].value_counts()}")
# -------------------------

# Calculate Swing MET

# event by event

# -------------------------

#for event_id in df.loc[df["is_swing"], "swing_event_id"].unique():



for event_id in df.loc[df["is_swing_event"], "swing_event_id"].unique():

    mask = (

        (df["swing_event_id"] == event_id)

        & df["is_swing_event"]

    )



    n_frames = mask.sum()



    if n_frames == 0:

        continue



    # Swing duration

    duration_s = n_frames / fps

    ang_vel = df.loc[mask, "active_ang_vel"]

    # Peak angular velocity of this Swing Event

    '''peak_ang_vel = df.loc[

        mask,

        "active_ang_vel"

    ].max()'''

    peak_ang_vel = np.percentile(

        df.loc[mask, "active_ang_vel"],

        95

    )

    # deg/s → rad/s

    omega_peak = np.deg2rad(peak_ang_vel)



    # Rotational mechanical energy

    rot_energy_J = (

        0.5

        * I_elbow

        * omega_peak**2

    )



    # Mechanical energy → metabolic energy

    metabolic_energy_J = (

        rot_energy_J / ETA

    )



    # Average metabolic power during this Swing Event

    swing_power_W = (

        metabolic_energy_J / duration_s

    )



    # Power → MET

    swing_met = (

        swing_power_W

        /

        (1.225 * weight_kg)

    )



    # Event frame range

    event_indices = df.index[mask]



    start_frame = event_indices[0]

    end_frame = event_indices[-1]



    #Frame ID





    start_frame = df.loc[start_frame, "frame_id"]

    end_frame = df.loc[end_frame, "frame_id"]



    max_idx = ang_vel.idxmax()

    max_frame_id = df.loc[max_idx, "frame_id"]



    start_time = start_frame / fps

    end_time = end_frame / fps

    max_time = max_frame_id / fps



    # Assign same event-level MET

    # to frames belonging to this Swing Event

    df.loc[

        mask,

        "MET_swing_rot"

    ] = swing_met

    print(

        f"Event {event_id:3d}: "

        f"Frame={start_frame:.0f}-{end_frame:.0f}, "

        f"Time={start_time:.3f}-{end_time:.3f}s, "

        f"frames={n_frames:3d}, "

        f"duration={duration_s:.3f}s, "

        f"max={ang_vel.max():.1f}, "

        f"P95={np.percentile(ang_vel,95):.1f}, "

        f"mean={ang_vel.mean():.1f}"

    )



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

#df["MET"] = np.where(df["jump"],JUMP_ROPE_MOD_MET, df["speed_fused_mps"].apply(speed_to_met_compendium))

df["MET"] = np.where(df["jump"],JUMP_REFERENCE_MET, df["speed_fused_mps"].apply(speed_to_met_compendium))

df["MET"] = df["MET"] + df["MET_swing_rot"]





# Jump 時 Running MET 不適用，只供繪圖使用

df["RUN_MET"] = np.where(df["jump"], np.nan, df["speed_fused_mps"].apply(speed_to_met_compendium))

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



df["vx_wrist"] = df["right_wrist_vel_x"]/dt

df["vy_wrist"] = df["right_wrist_vel_y"]/dt

df["vz_wrist"] = df["right_wrist_vel_z"]/dt



df["vx_wrist_combined"] = KF_WEIGHT * df["vx_wrist"] + RAW_WEIGHT * df["vx_diff"]

df["vy_wrist_combined"] = KF_WEIGHT * df["vy_wrist"] + RAW_WEIGHT * df["vy_diff"]

df["vz_wrist_combined"] = KF_WEIGHT * df["vz_wrist"] + RAW_WEIGHT * df["vz_diff"]



df["ax_wrist_combined"] = np.gradient(df["vx_wrist_combined"], dt, edge_order=2)

df["ay_wrist_combined"] = np.gradient(df["vy_wrist_combined"], dt, edge_order=2)

df["az_wrist_combined"] = np.gradient(df["vz_wrist_combined"], dt, edge_order=2)

for c in ["ax_wrist_combined", "ay_wrist_combined", "az_wrist_combined"]:

    df[c] = savgol_filter(

        df[c],

        window_length=21,

        polyorder=2

    )





df["dax_wrist"] = df["ax_wrist_combined"].diff().fillna(0)

df["day_wrist"] = df["ay_wrist_combined"].diff().fillna(0)

df["daz_wrist"] = df["az_wrist_combined"].diff().fillna(0)



# Player Load per frame

# /10: scale to arbitrary units



df["PL_raw"] = (np.sqrt(

    df["ax_diff"]**2 +df["ay_diff"]**2 + df["az_diff"]**2) / 10000)



df["PL"] = (np.sqrt(df["dax"]**2 + df["day"]**2 + df["daz"]**2) / 100)

df["PL_wrist"] = (np.sqrt(df["dax_wrist"]**2 + df["day_wrist"]**2 + 

                          df["daz_wrist"]**2) / 100)



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

df["PL"] = np.clip(df["PL"], 0, None)



df["PL_wrist"] = df["PL_wrist"].clip(

    lower=0,

    upper=df["PL_wrist"].quantile(0.95)

)

df["PL_wrist"] = savgol_filter(

    df["PL_wrist"],

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


'''
df["PL_per_min"] = (
    df["PL"]
    .rolling(

        window,

        center=True,

        #center=False,

        min_periods=1

    )
    .mean()
    * 60
)'''

df["PL_per_min"] = (
    df["PL"]
    .rolling(

        window = window,

        center=True,

        #center=False,

        min_periods=window

    )
    .sum()

    /(window/fps)
    
    * 60
)

df["PL_per_min_sum"] = (

    df["PL"]

    .rolling(

        window,

        center=True,

        #center=False,

        min_periods=1

    )

    .sum()

    / window_sec * 60

)



df["PL_catapult_per_min"] = (

    df["PL"].rolling(

        window,

        center=True,

        #center=False,

        min_periods=1

    ).sum()

    / window_sec

    * 60

)

total_pl = df["PL"].sum()



pl_per_min = (

    total_pl

    /

    (df["time_sec"].iloc[-1] / 60)

    #(df["time_sec"].iloc[-1] / 60)/100

)

total_pl_wrist = df["PL_wrist"].sum()



pl_wrist_per_min_catapult = (

    total_pl_wrist

    /

    (df["time_sec"].iloc[-1] / 60)/100

)



print(f"Total PL: {total_pl:.2f} AU")

print(f"PL/min : {pl_per_min:.2f} AU/min")

print(f"Total PL (Wrist): {total_pl_wrist:.2f} AU")

print(f"PL/min (Wrist, Catapult): {pl_wrist_per_min_catapult:.2f} AU/min")


'''
df["PL_per_min"] = (df["PL"]

    .rolling(

        window,

        center=True,

        #center=False,

        min_periods=1

    )

    #.mean()

    #/ window_sec

    #* 60
     .sum()

    /(window/fps)
    
    * 60
)'''



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



acc_wrist_mag = np.sqrt(

    df["ax_wrist_combined"]**2 +

    df["ay_wrist_combined"]**2 +

    df["az_wrist_combined"]**2

)



df["MAD"] = (

    acc_mag

    .rolling(

        window,

        center=True,

        #center=False,

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

mad_window_sec = 1
mad_window = int(mad_window_sec * fps)


# 每 mad_window 個 Frame 分成一組
mad_groud_id =  np.arange(len(df)) // mad_window

#acc_mag_p95 = acc_mag.quantile(0.95)

# 每秒先排除超過 P95 的加速度，再計算 MAD
def calc_mad_p95(x):
    acc_mag_p95 = x.quantile(0.95)
    x_filtered = x[x <= acc_mag_p95]

    return np.mean(np.abs(x_filtered - np.mean(x_filtered)))


mad_1s = (
    pd.DataFrame({
        "time_sec": df["time_sec"].to_numpy(),
        "acc_mag": acc_mag.to_numpy(),
        "mad_groud_id": mad_groud_id
    }).groupby("mad_groud_id").agg(
        time_sec = ("time_sec", "mean"),
        #MAD = ("acc_mag", lambda x: np.mean(np.abs(x - np.mean(x)))),
        MAD = ("acc_mag", calc_mad_p95),
        count = ("acc_mag", "count")
        )
)

# 排除不足 1 秒的最後一組
mad_1s = mad_1s[mad_1s["count"] == mad_window]



df["MAD_Wrist"] = (

    acc_wrist_mag

    .rolling(

        window,

        center=True,

        #center=False,

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

#print(f"Swing frames: {df['is_swing'].sum()}")
print(f"Annotation Swing Frames: {df['is_swing_event'].sum()}")
print(f"Annotation Swing Events: {df['swing_event_id'].max()}")

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

        #center=False,

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

        #center=False,

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

plt.show(block=False)

plt.pause(2)

plt.close()


# ==========================================
# Dynamic MET with Badminton MET References
# ==========================================

BADMINTON_MET_2000 = 4.5  # MET for badminton singles (2000 Compendium)
BADMINTON_MET_2024 = 5.5  # MET for badminton singles (2024 Compendium)

average_met = df["MET"].mean()

plt.figure(figsize=(12, 4.5))

#plt.plot(df["time_sec"], df["MET"], linewidth=2, label="Dynamic MET")

plt.plot(df["time_sec"], df["MET"], linewidth=1.5, label="Dynamic MET", color="tab:blue")

#2000 Compendium reference line
plt.axhline(y=BADMINTON_MET_2000, color="tab:orange", linestyle="--", 
            linewidth=1.5, label="Badminton MET (2000)")

#2024 Compendium reference line
plt.axhline(y=BADMINTON_MET_2024, color="tab:green", linestyle="--", 
            linewidth=1.5, label="Badminton MET (2024)")

#Dataset average MET reference line
plt.axhline(y=average_met, color="tab:red", linestyle="--", 
            linewidth=1.5, label=f"Average MET ({average_met:.2f})")

plt.ylim(0, df["MET"].max() * 1.5)  # Set y-axis limit to 50% above max MET

plt.xlabel("Time (sec)")

plt.ylabel("MET")

plt.title("Dynamic MET over Time with Badminton MET References")

plt.grid(True)

plt.legend()

plt.savefig(f"{OUTPUT_FOLDER}/met_vs_time_{safe_folder_name}_{date}.png", dpi=300)

plt.show(block=False)

plt.pause(2)

plt.close()



plt.figure(figsize=(12, 4))

plt.plot(df["time_sec"], df["speed_fused_mps"], linewidth=2,) #label="Fused Speed")

plt.xlabel("Time (sec)")

plt.ylabel("Speed (m/s)")

plt.title("Player Movement Speed")

plt.grid(True)

#plt.legend()

plt.savefig(f"{OUTPUT_FOLDER}/fused_speed_vs_time_{safe_folder_name}_{date}.png", dpi=300)

plt.show(block=False)

plt.pause(2)

plt.close()



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

plt.show(block=False)

plt.pause(2)

plt.close()



plt.figure(figsize=(12, 4))

plt.plot(df["time_sec"], df["RUN_MET"], linewidth=2, label="Running MET")

plt.xlabel("Time (sec)")

plt.ylabel("Running MET")

plt.title("Running MET over Time")

plt.grid(True)

plt.legend()

plt.savefig(f"{OUTPUT_FOLDER}/run_met_{safe_folder_name}_{date}.png", dpi=300)

plt.show(block=False)

plt.pause(2)

plt.close()


# ==================================================
# Thesis Figure: Angular Velocity / Swing Events / Swing MET
# ==================================================
# Event boundaries are obtained from manually annotated CSV frames.
# Swing MET is constant within each annotated event interval.

fig, axes = plt.subplots(
    3, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={"height_ratios": [2.0, 0.8, 1.3]}
)

# (a) Right elbow angular velocity
axes[0].plot(
    df["time_sec"], df["active_ang_vel"],
    color="tab:blue", linewidth=1.1,
    label="Right Elbow Angular Velocity"
)

# (b) Manually annotated swing events (0 or 1)
axes[1].step(
    df["time_sec"], df["is_swing_event"].astype(int),
    where="post", color="tab:green", linewidth=1.6,
    label="Annotated Swing State"
)
axes[1].legend(loc ="lower right", 
               bbox_to_anchor=(1.0, 2.0), fontsize=10, borderpad=0)
axes[1].set_yticks([0, 1])
axes[1].set_ylim(-0.15, 1.50)

# (c) Event-based Swing MET
axes[2].step(
    df["time_sec"], df["MET_swing_rot"],
    where="post", color="tab:orange", linewidth=1.6,
    label="Event-based Swing MET"
)
axes[2].set_ylim(bottom=0)

# Annotated event intervals and boundaries on angular-velocity subplot
annotated_events = (
    df.loc[df["swing_event_id"] > 0]
      .sort_values("frame_id")
      .groupby("swing_event_id", sort=True)
)
for i, (event_id, event_df) in enumerate(annotated_events):
    start_row = event_df.iloc[0]
    end_row = event_df.iloc[-1]
    t0, t1 = start_row["time_sec"], end_row["time_sec"]

    axes[0].axvspan(
        t0, t1, color="tab:green", alpha=0.13,
        label="Annotated Swing Interval" if i == 0 else None
    )
    axes[0].scatter(
        t0, start_row["active_ang_vel"], color="tab:red",
        s=35, zorder=4,
        label="Swing Start" if i == 0 else None
    )
    axes[0].scatter(
        t1, end_row["active_ang_vel"], color="tab:purple",
        marker="s", s=30, zorder=4,
        label="Swing End" if i == 0 else None
    )

axes[0].set_ylabel("Angular Velocity (deg/s)")
axes[1].set_ylabel("Swing State")
axes[2].set_ylabel("Swing MET")
axes[2].set_xlabel("Time (sec)")
axes[0].set_title(
    f"{dataset_name}: Angular Velocity, Annotated Swing Events and Swing MET"
)

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/Fig_Swing_MET_Analysis_{dataset_name}_{date}.png",
    dpi=300, bbox_inches="tight"
)
plt.show(block=False)
plt.pause(2)
plt.close(fig)




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

plt.ylabel("Player Load (AU)")

plt.title("Instantaneous Player Load")

plt.ylim(bottom=0)

plt.grid(True)

plt.savefig(

    f"{OUTPUT_FOLDER}/playerload_vs_time.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()





plt.figure(figsize=(12,4))

plt.plot(

    df["time_sec"],

    df["PL_wrist"],

    linewidth=1

)

plt.xlabel("Time (sec)")

plt.ylabel("PL_Wrist (AU)")

plt.title("Instantaneous Player Load from Wrist")

plt.grid(True)

plt.savefig(

    f"{OUTPUT_FOLDER}/playerload_wrist_vs_time.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()





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

plt.ylabel("MAD (m/s²)")

plt.title(" Mean Absolute Deviation of Acceleration")

plt.legend()

plt.grid(True)

plt.savefig(

    f"{OUTPUT_FOLDER}/mad_true_false_vs_time.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()







plt.figure(figsize=(12,4))

plt.plot(

    df["time_sec"],

    df["MAD"],

    linewidth=1,

    color="blue",

    label="MAD",

)

plt.xlabel("Time (sec)")

plt.ylabel("MAD (m/s²)")

plt.title(" Mean Absolute Deviation of Acceleration")

plt.legend()

plt.grid(True)

plt.savefig(

    f"{OUTPUT_FOLDER}/mad_vs_time.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()





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

plt.show(block=False)

plt.pause(2)

plt.close()





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

plt.show(block=False)

plt.pause(2)

plt.close()





plt.figure(figsize=(12,4))

plt.plot(

    df["time_sec"],

    df["PL"],

    label="Ankle PL"

)

plt.plot(

    df["time_sec"],

    df["PL_wrist"],

    label="Wrist PL"

)

plt.xlabel("Time (sec)")

plt.ylabel("PL")

plt.title("Ankle vs Wrist Player Load")

plt.legend()

plt.grid(True)



plt.savefig(

    f"{OUTPUT_FOLDER}/pl_ankle_vs_wrist.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()



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



plt.show(block=False)

plt.pause(2)

plt.close()





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

plt.show(block=False)

plt.pause(2)

plt.close()





plt.figure(figsize=(12,4))



plt.plot(

    df["time_sec"],

    df["MAD_Wrist"],

    label="Wrist MAD"

)



plt.xlabel("Time (sec)")

plt.ylabel("MAD")

plt.title("Wrist MAD")

plt.legend()

plt.grid(True)



plt.savefig(

    f"{OUTPUT_FOLDER}/mad_wrist.png",

    dpi=300

)



plt.show(block=False)

plt.pause(2)

plt.close()



plt.figure(figsize=(12,4))

plt.plot(

    df["time_sec"],

    df["MAD"],

    label="Ankle MAD"

)

plt.plot(

    df["time_sec"],

    df["MAD_Wrist"],

    label="Wrist MAD"

)

plt.xlabel("Time (sec)")

plt.ylabel("MAD")

plt.title("Ankle vs Wrist Mean Absolute Deviation of Acceleration")

plt.legend()

plt.grid(True)



plt.savefig(

    f"{OUTPUT_FOLDER}/mad_ankle_vs_wrist.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()





swing_raw_start = (

    df["is_swing"].astype(bool) &

    ~df["is_swing"].astype(bool).shift(1, fill_value=False)

)



min_swing_interval_sec = 0.3

min_swing_interval_frames = int(min_swing_interval_sec * fps)



start_indices = np.where(swing_raw_start)[0]

filtered_swing_indices = []

last_index = -min_swing_interval_frames



for idx in start_indices:

    if idx - last_index >= min_swing_interval_frames:

        filtered_swing_indices.append(idx)

        last_index = idx


swing_start = (

    df["is_swing"].astype(bool)

    & ~df["is_swing"].astype(bool).shift(1, fill_value=False)

)

swing_start = pd.Series(False, index=df.index)

swing_start.iloc[filtered_swing_indices] = True



plt.figure(figsize=(12,4))

plt.plot(

    df["time_sec"],

    df["active_ang_vel"],

    linewidth=1.2,

    color="tab:blue",

    label="Active Angular Velocity"

)



plt.axhline(

    y = 80,

    linestyle="--",

    linewidth=1.5,

    color="tab:orange",

    label="Angular Velocity Threshold (80 deg/s)",

)



swing_mask = df["is_swing"].astype(bool)



plt.scatter(

    #df.loc[swing_mask, "time_sec"],

    #df.loc[swing_mask, "active_ang_vel"],

    df.loc[swing_start, "time_sec"],

    df.loc[swing_start, "active_ang_vel"],

    s=40,

    color="tab:red",

    label="Swing Start",

    zorder=3

    #label="Swing Frame"

)



plt.xlabel("Time (sec)")

plt.ylabel("Active Angular Velocity (deg/s)")

plt.title("Active Angular Velocity and Swing Detection")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    f"{OUTPUT_FOLDER}/active_ang_vel_swing_detection_{safe_folder_name}_{date}.png",

    dpi=300

)

plt.show(block=False)

plt.pause(2)

plt.close()


# ==================================================
# Active Angular Velocity and Annotated Swing Events
# ==================================================

fig, ax = plt.subplots(figsize=(12, 4))
#Active Angular Velocity
ax.plot(
    df["time_sec"],
    df["active_ang_vel"],
    linewidth=1.2,
    color="tab:blue",
    label="Active Angular Velocity"
)
# Manual annotation event intervals
events = df.loc[df["swing_event_id"] > 0].groupby("swing_event_id",sort=False)

for i, (event_id, event) in enumerate(events):
    start_row = event.iloc[0]
    end_row = event.iloc[-1]
    ax.axvspan(
        start_row["time_sec"],
        end_row["time_sec"],
        color="tab:green",
        alpha=0.15,
        label="Annotated Swing Event" if i == 0 else None
    )
    #Swing Start
    ax.scatter(
        start_row["time_sec"],
        start_row["active_ang_vel"],
        s=45,
        color="tab:red",
        label="Annotated Swing Start" if i == 0 else None,
        zorder=5
    )
    #Swing End
    ax.scatter(
        end_row["time_sec"],
        end_row["active_ang_vel"],
        s=40,
        color="tab:purple",
        label="Annotated Swing End" if i == 0 else None,
        zorder=5
    )
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Active Angular Velocity (deg/s)")
ax.set_title("Active Angular Velocity and Annotated Swing Events")

ax.grid(True)
#ax.legend()
ax.legend(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.1),
    ncol=2,
    fontsize=10,
    frameon=True,
)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_FOLDER}/active_ang_vel_annotated_swing_events_{safe_folder_name}_{date}.png",
    dpi=300
)
plt.show(block=False)
plt.pause(2)
plt.close()
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

    f.write(f"PL/min : {pl_per_min:.2f} AU/min\n")

    f.write(f"Total PL (Wrist): {total_pl_wrist:.2f} AU\n")

    f.write(f"PL/min (Wrist, Catapult): {pl_wrist_per_min_catapult:.2f} AU/min\n")

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

# =========================================================
# Figure 4-9: Ankle vertical position / velocity / Jump State
# =========================================================
# Uses the existing CSV jump labels; does NOT recalculate jump detection.
# The plotted velocity is derived from ankle Z coordinates, not the
# independent right-ankle velocity column.

def plot_figure_4_9(data, fps, output_folder):
    required = ["z_left_ankle", "z_right_ankle", "jump"]
    missing = [name for name in required if name not in data.columns]
    if missing:
        print(f"Skipping Figure 4-9: missing columns {missing}")
        return

    time = (data["frame_id"].to_numpy(dtype=float) / fps
            if "frame_id" in data.columns
            else np.arange(len(data), dtype=float) / fps)
    jump = data["jump"].fillna(False)
    if jump.dtype == object:
        jump = jump.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    jump = jump.to_numpy(dtype=bool)

    def ankle_signal(column):
        z = pd.to_numeric(data[column], errors="coerce")
        z = z.interpolate(limit_direction="both")
        invalid_mask = ~np.isfinite(z) | (z < 0) | (z > 1.5)
        z[invalid_mask] = np.nan
        #z = z.interpolate(limit_direction="both")
        if z.isna().all():
            return np.full(len(data), np.nan), np.full(len(data), np.nan)
        z = z.to_numpy(dtype=float)
        z = z - 0.7
        z= np.clip(z, 0.0, 1.0)
        window = min(11, len(z) if len(z) % 2 else len(z) - 1)
        # 暫時補齊 NaN 以避免平滑運算受影響
        z_for_filter = pd.Series(z).interpolate(
            limit_direction="both"
        ).to_numpy()
        #z_smooth = savgol_filter(z, window, 2) if window >= 3 else z
        z_smooth = savgol_filter(z_for_filter, window, 2) if window >= 3 else z_for_filter
        vz = np.gradient(z_smooth, 1.0 / fps) if len(z) >= 2 else np.zeros_like(z)
        #vz = vz - 3.0
        vz = np.clip(vz, 0.0, 3.5)  # 垂直速度最小限制為 0 m/s
        return z, vz

    z_left, vz_left = ankle_signal("z_left_ankle")
    z_right, vz_right = ankle_signal("z_right_ankle")

    #fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True,
    #                         gridspec_kw={"height_ratios": [1, 1, 0.65]})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 0.65]})
    axes[0].plot(time, z_left, label="Left ankle", color="tab:blue", lw=1.1)
    axes[0].plot(time, z_right, label="Right ankle", color="tab:orange", lw=1.1)
    axes[0].set_ylabel("Vertical position (m)")
    axes[0].set_title("(a) Ankle vertical position")
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, 1.5)
    '''
    axes[1].plot(time, vz_left, color="tab:blue", lw=1.0, label="Left ankle")
    axes[1].plot(time, vz_right, color="tab:orange", lw=1.0, label="Right ankle")
    axes[1].set_ylabel("Vertical velocity (m/s)")
    axes[1].set_title("(b) Ankle vertical velocity")
    axes[1].set_ylim(bottom=0)

    axes[2].step(time, jump.astype(int), where="post", color="tab:purple", lw=1.4)
    axes[2].set_ylim(-0.1, 1.15)
    axes[2].set_yticks([0, 1])
    axes[2].set_ylabel("Jump state")
    axes[2].set_title("(c) Detected jump state")
    axes[2].set_xlabel("Time (s)")'''
    axes[1].step(time, jump.astype(int), where="post", color="tab:purple", lw=1.4)
    axes[1].set_ylim(-0.1, 1.15)
    axes[1].set_yticks([0, 1])
    axes[1].set_ylabel("Jump state")
    axes[1].set_title("(b) Detected jump state")
    axes[1].set_xlabel("Time (s)")

    # Highlight contiguous jump intervals, including single-frame events.
    boundaries = np.diff(np.r_[False, jump, False].astype(int))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    for start, end in zip(starts, ends):
        t0 = time[start]
        t1 = time[end] if end < len(time) else time[-1] + 1 / fps
        for ax in axes:
            ax.axvspan(t0, t1, color="mediumpurple", alpha=0.15, lw=0)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout(h_pad=1.25)
    target = Path(output_folder)
    target.mkdir(parents=True, exist_ok=True)
    png = target / "Fig4-9_Jump_Detection.png"
    pdf = target / "Fig4-9_Jump_Detection.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Jump frames: {jump.sum()}; jump intervals: {len(starts)}")


plot_figure_4_9(df, fps, OUTPUT_FOLDER)

plt.figure(figsize=(12, 4))
'''
plt.plot(
    mad_1s["time_sec"],
    mad_1s["MAD"],
    marker="o",
    markersize=4,
    linewidth=1.5,
    color="tab:blue",
    label="Non-overlapping MAD (1 s)"
)'''


plt.step(
    mad_1s["time_sec"],
    mad_1s["MAD"],
    where="mid",
    color="tab:blue",
    linewidth=1.8,
    label="Non-overlapping MAD (1 s)"
)

plt.scatter(
    mad_1s["time_sec"],
    mad_1s["MAD"],
    color="tab:blue",
    s=25
)

plt.ylim(bottom=0, top=mad_1s["MAD"].max() * 1.2)

plt.xlabel("Time (sec)")
plt.ylabel("MAD (m/s²)")
plt.title("1-second Non-overlapping MAD")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_FOLDER}/mad_nonoverlap_1s_{safe_folder_name}_{date}.png",
    dpi=300
)

plt.show(block=False)
plt.pause(2)
plt.close()