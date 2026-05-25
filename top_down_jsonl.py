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

#INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-12-21"
INPUT_PATH = r"C:\D\NCTU_CS\Thesis\Lab_Data\dataset\dataset\2026-04-09_19-13-28"
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
VIDEO_B = f"{INPUT_PATH}/CameraReader_{CAM_PAIRS[SELECTED_PAIR][1]}.mp4"
capA = cv2.VideoCapture(VIDEO_A)
capB = cv2.VideoCapture(VIDEO_B)
fps_a = capA.get(cv2.CAP_PROP_FPS)

COLOR_R = (0, 0, 255)
COLOR_G = (0, 255, 0)
COLOR_B = (255, 0, 0)
COLOR_Y = (0, 255, 255)
COLOR_W = (255, 255, 255)
COLOR_K = (0, 0, 0)
COLOR_TRAJ = (255, 255, 0)
COLOR_P = (255,0,255)

# Projection Matrices
cfg_files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".cfg")]
cfg_files.sort()
projMtxs = []
H_inv_Mtxs = []
parser = configparser.ConfigParser()
for cfg_file in cfg_files:
    parser.read(os.path.join(INPUT_PATH, cfg_file))
    mtx_str = parser["Other"]["projection_mat"]
    K = parser["Other"]["newcameramtx"]
    Rt = parser["Other"]["extrinsic_mat"]
    
    mtx = np.array(json.loads(mtx_str))
    # print(f"Projection matrix: {mtx.shape}\n{mtx}")
    H = mtx[:, [0, 1, 3]]  # Use the first two columns and the last column for homography
    # K = np.array(json.loads(K))
    # Rt = np.array(json.loads(Rt))
    # R = Rt[:3, :3]
    # t = Rt[:3, 3].reshape(3, 1)
    # H = K @ np.column_stack((R[:, 0], R[:, 1], t.flatten()))
    # print(f"{mtx}\n")
    projMtxs.append(mtx)
    H_inv_Mtxs.append(np.linalg.inv(H))
    
print(f"Done reading {len(projMtxs)} camera configs.")
# Resolution factors for denormalization
# res_string = parser["Camera"]["RecordResolution"]
# factors = res_string[1:-1].split(",")
factors = [640, 640]  # hardcoded for now since we know the resolution, can be read from cfg if needed
normalize_factor_x = float(factors[0])
normalize_factor_y = float(factors[1])
y_factor = 480 / 640
print(f"Resolution: {normalize_factor_x} x {normalize_factor_y}")

def read_jsonl(file_path):
    """Generator yielding one frame's data at a time."""
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            yield json.loads(line)
            
def akima_fill(series, frame_ids):
    valid = series.notna()
    interp = Akima1DInterpolator(
        frame_ids[valid],
        series[valid]
    )

    filled = pd.Series(
        interp(frame_ids),
        index=series.index
    )
    return filled.interpolate(method="linear", limit_direction="both")

def pchip_fill(series, frame_ids):

    valid = series.notna()

    interp = PchipInterpolator(
        frame_ids[valid],
        series[valid]
    )

    filled = pd.Series(
        interp(frame_ids),
        index=series.index
    )

    return filled.interpolate(method="linear", limit_direction="both")

jsonl_files = [f for f in os.listdir(INPUT_PATH) if (f.startswith("Pose_") and f.endswith(".jsonl"))]
jsonl_files.sort()
pose_jsonls = []


'''
for jsonl_file in jsonl_files:
    pose_tmp = []
    # pose_tmp = pd.DataFrame(columns=["frame_id"] + [f"keypoint_{i}_{axis}" for i in range(num_keypoints) for axis in ["x", "y"]])
    for line in read_jsonl(os.path.join(INPUT_PATH, jsonl_file)):
        line_data = {"frame_id": line["frame_id"], "timestamp": line["timestamp"]}
        
        if "detection" in line and len(line["detection"]) > 0:    
            
            kpts = line["detection"][0]["kpts"]
            for i in range(num_keypoints):
                line_data[f"kpts_{i}_x"] = kpts[2*i]
                line_data[f"kpts_{i}_y"] = kpts[2*i+1]
                
            bbox = line["detection"][0]["bbox"]
            bbox_x, bbox_y = bbox[0], bbox[1]
            line_data["bbox_x"] = bbox_x
            line_data["bbox_y"] = bbox_y
            
        else:
            for i in range(num_keypoints):
                line_data[f"kpts_{i}_x"] = np.nan
                line_data[f"kpts_{i}_y"] = np.nan
            line_data["bbox_x"] = np.nan
            line_data["bbox_y"] = np.nan
                
        pose_tmp.append(line_data)
    # format: frame_id | timestamp | kpts_0_x | kpts_0_y | ... | kpts_16_x | kpts_16_y | bbox_x | bbox_y
    pose_tmp = pd.DataFrame(pose_tmp)
    pose_tmp = pose_tmp.sort_values("frame_id")
    pose_tmp = pose_tmp.drop_duplicates(subset="frame_id", keep="first")
    print(f"Loaded {pose_tmp.shape} frames from {jsonl_file}")
    # Forward fill NaN values
    full_index = np.arange(
        pose_tmp["frame_id"].min(),
        pose_tmp["frame_id"].max() + 1
    )
    pose_tmp = pose_tmp.set_index("frame_id")
    pose_tmp = pose_tmp.reindex(full_index)
    pose_tmp["timestamp"] = pose_tmp["timestamp"].interpolate(
        method="linear", limit_direction="both")
    for i in range(num_keypoints):
        pose_tmp[f"kpts_{i}_x"] = pose_tmp[f"kpts_{i}_x"].interpolate(
        method="linear",
        limit=12,
        limit_direction="both"
        )

        pose_tmp[f"kpts_{i}_y"] = pose_tmp[f"kpts_{i}_y"].interpolate(
            method="linear",
            limit=12,
            limit_direction="both"
        )
        pose_tmp[f"kpts_{i}_x"] = pose_tmp[f"kpts_{i}_x"].interpolate(
        method="linear",
        limit_direction="both"
        )
        pose_tmp[f"kpts_{i}_y"] = pose_tmp[f"kpts_{i}_y"].interpolate(
        method="linear",
        limit_direction="both"
        )

        pose_tmp[f"kpts_{i}_x"] = akima_fill(
            pose_tmp[f"kpts_{i}_x"],
            pose_tmp.index
        )
        
        pose_tmp[f"kpts_{i}_y"] = akima_fill(
            pose_tmp[f"kpts_{i}_y"],
            pose_tmp.index
        )
        
        pose_tmp[f"kpts_{i}_x"] = pchip_fill(
            pose_tmp[f"kpts_{i}_x"],
            pose_tmp.index
        )
        pose_tmp[f"kpts_{i}_y"] = pchip_fill(
            pose_tmp[f"kpts_{i}_y"],
            pose_tmp.index
        )

    pose_tmp["bbox_x"] =  pose_tmp["bbox_x"].interpolate(
        method="linear",
        limit_direction="both"
    )

    pose_tmp["bbox_y"] = pose_tmp["bbox_y"].interpolate(
        method="linear",
        limit_direction="both"
    )
    pose_tmp["bbox_x"] = akima_fill(
    pose_tmp["bbox_x"],
    pose_tmp.index )

    pose_tmp["bbox_y"] = akima_fill(
    pose_tmp["bbox_y"],
    pose_tmp.index )
    
    pose_tmp["bbox_x"] = pchip_fill(
    pose_tmp["bbox_x"],
    pose_tmp.index )
    
    pose_tmp["bbox_y"] = pchip_fill(
    pose_tmp["bbox_y"],
    pose_tmp.index )
    pose_tmp["bbox_x"] = pose_tmp["bbox_x"].interpolate(
    method="linear",
    limit=12,
    limit_direction="both"
    )

    pose_tmp["bbox_y"] = pose_tmp["bbox_y"].interpolate(
        method="linear",
        limit=12,
        limit_direction="both"
    )

    #pose_tmp.ffill(inplace=True)
    # Backward fill any remaining NaN values (if the first few rows are NaN)
    #pose_tmp.bfill(inplace=True)
    pose_tmp = pose_tmp.reset_index()
    # pose_tmp.fillna(0, inplace=True)  # fill NaN with 0 for now, can be improved with interpolation if needed
    # vectorize denormalization of 2D keypoints
    pose_tmp.iloc[:, 7::2] *= normalize_factor_x
    pose_tmp.iloc[:, 8::2] *= normalize_factor_y
    pose_jsonls.append(pose_tmp)
    
print(f"Done reading {len(pose_jsonls)} pose JSONL files.")'''


jsonl_files = [f for f in os.listdir(INPUT_PATH)
               if f.startswith("Pose_") and f.endswith(".jsonl")]
jsonl_files.sort()

pose_jsonls = []

for jsonl_file in jsonl_files:
    pose_tmp = []

    for line in read_jsonl(os.path.join(INPUT_PATH, jsonl_file)):
        line_data = {
            "frame_id": line["frame_id"],
            "timestamp": line["timestamp"]
        }

        if "detection" in line and len(line["detection"]) > 0:
            kpts = line["detection"][0]["kpts"]

            for i in range(num_keypoints):
                line_data[f"kpts_{i}_x"] = kpts[2 * i]
                line_data[f"kpts_{i}_y"] = kpts[2 * i + 1]

            bbox = line["detection"][0]["bbox"]
            line_data["bbox_x"] = bbox[0]
            line_data["bbox_y"] = bbox[1]

        else:
            for i in range(num_keypoints):
                line_data[f"kpts_{i}_x"] = np.nan
                line_data[f"kpts_{i}_y"] = np.nan

            line_data["bbox_x"] = np.nan
            line_data["bbox_y"] = np.nan

        pose_tmp.append(line_data)

    pose_tmp = pd.DataFrame(pose_tmp)
    pose_tmp = pose_tmp.sort_values("frame_id")
    pose_tmp = pose_tmp.drop_duplicates(subset="frame_id", keep="first")

    print(f"Loaded {pose_tmp.shape} frames from {jsonl_file}")

    # Step 1: 先用 ffill/bfill 補原本 jsonl 裡面的 NaN
    pose_tmp.ffill(inplace=True)
    pose_tmp.bfill(inplace=True)

    # Step 2: 再把每 4 frame 補成每 1 frame
    full_index = np.arange(
        pose_tmp["frame_id"].min(),
        pose_tmp["frame_id"].max() + 1
    )

    pose_tmp = pose_tmp.set_index("frame_id")
    pose_tmp = pose_tmp.reindex(full_index)

    pose_tmp["timestamp"] = pose_tmp["timestamp"].interpolate(
        method="linear",
        limit_direction="both"
    )

    # Step 3: Akima 補中間 frame
    if fill_method == "akima":
        for i in range(num_keypoints):
            pose_tmp[f"kpts_{i}_x"] = akima_fill(
                pose_tmp[f"kpts_{i}_x"],
                pose_tmp.index
            )

            pose_tmp[f"kpts_{i}_y"] = akima_fill(
                pose_tmp[f"kpts_{i}_y"],
                pose_tmp.index
            )

        pose_tmp["bbox_x"] = akima_fill(
            pose_tmp["bbox_x"],
            pose_tmp.index
        )

        pose_tmp["bbox_y"] = akima_fill(
            pose_tmp["bbox_y"],
            pose_tmp.index
        )
    elif fill_method == "pchip":
        for i in range(num_keypoints):
            pose_tmp[f"kpts_{i}_x"] = pchip_fill(
                pose_tmp[f"kpts_{i}_x"],
                pose_tmp.index
            )

            pose_tmp[f"kpts_{i}_y"] = pchip_fill(
                pose_tmp[f"kpts_{i}_y"],
                pose_tmp.index
            )

        pose_tmp["bbox_x"] = pchip_fill(
            pose_tmp["bbox_x"],
            pose_tmp.index
        )
        pose_tmp["bbox_y"] = pchip_fill(
            pose_tmp["bbox_y"],
            pose_tmp.index
        )
    pose_tmp = pose_tmp.reset_index()
    pose_tmp = pose_tmp.rename(columns={"index": "frame_id"})

    # Step 4: denormalize
    pose_tmp.iloc[:, 2::2] *= normalize_factor_x
    pose_tmp.iloc[:, 3::2] *= normalize_factor_y

    pose_jsonls.append(pose_tmp)
    # ============================
    # Save filled pose CSV
    # ============================

    save_dir = Path( INPUT_PATH) / f"filled_pose_{fill_method}"

    save_dir.mkdir(parents=True, exist_ok=True)

    pose_name = Path(jsonl_file).stem

    save_path = save_dir / (
        f"{pose_name}_filled_{date}.csv"
    )

    pose_tmp.to_csv(save_path, index=False)

    print(f"[SAVE] {save_path}")

    #pose_jsonls.append(pose_tmp)

    print(f"Done reading {len(pose_jsonls)} pose JSONL files.")


# null_point = np.ones((4, 12))
def rule_base_filter(points3d):
    is_valid = 1
    com = points3d[:3, -1]  # using the homogeneous coordinate as reference for center of mass (bbox center)
    # Rule 1: not underground or floating.
    if com[2] < -1 or com[2] > 3:
        is_valid = 0
    # Rule 2: in top-down, should be near (<= 2m) center of bounding box.
    proj_xy = points3d[:2, BodyKpt.Left_Shoulder:BodyKpt.Bbox_Center]  # exclude the homogeneous coordinate
    dist_xy = np.linalg.norm(proj_xy - com[:2, None], axis=0)
    if np.any(dist_xy > 2):
        is_valid = 0
    return is_valid


def draw_top_trajectory(canvas, trajectory, color=(255, 255, 0)): 
    # Green: (0, 255, 0), Green: (255,255,0)
    scale_vis = 40
    bg_h, bg_w = canvas.shape[:2]
    sx, sy = 400, 800
    offset_x = (bg_w - sx) //2
    offset_y = (bg_h - sy) //2
    #cx, cy = sx//2, sy//2
    cx = offset_x + sx//2
    cy = offset_y + sy//2
    if len(trajectory) < 2:
        return canvas
    
    pts = []
    #for x, y in trajectory:
    for item in trajectory:
        x , y = item["pos"]
        is_jump = item["jump"]
        if np.isnan(x) or np.isnan(y):
            continue

        px = int(cx - x * scale_vis)
        py = int(cy + y * scale_vis)
        pts.append((px, py, is_jump))
    
    for i in range(1, len(pts)):
        p1 = pts[i-1][:2]
        p2 = pts[i][:2]
        cv2.line(canvas,p1, p2, color, 2)
        curr_jump = pts[i][2]
        prev_jump = pts[i-1][2]
        if curr_jump and not prev_jump:
            px, py = p2
    # pulse animation
            pulse = int(
                10
                + 8 * abs(np.sin(i * 0.5))
            )

            cv2.circle(
                canvas,
                p2,
                #pulse,
                6,
                (0,0,255),
                -1
            )

            '''cv2.putText(
                canvas,
                "J",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.2,
                (0,0,255),
                2
            )'''

    return canvas
# ===== Helper: draw top view =====
def draw_mini_court(height=800):

    court_ratio = 6.1 / 13.4

    width = int(height * court_ratio)

    court = np.zeros((height, width, 3), dtype=np.uint8)

    # green background
    court[:] = (80, 160, 80)

    line_color = (220,255,220)
    thickness = 2

    margin = 20

    left = margin
    right = width - margin

    top = margin
    bottom = height - margin

    court_w = right - left
    court_h = bottom - top

    # ===== real badminton dimensions =====

    doubles_width = 6.10
    singles_width = 5.18

    court_length = 13.40

    short_service = 1.98
    long_service_double = 0.76

    center_x = width // 2
    net_y = height // 2

    scale = court_h / court_length

    # outer court
    cv2.rectangle(
        court,
        (left, top),
        (right, bottom),
        line_color,
        thickness
    )

    # singles sidelines
    single_margin = int(
        ((doubles_width - singles_width)/2)
        * scale
    )

    cv2.line(
        court,
        (left + single_margin, top),
        (left + single_margin, bottom),
        line_color,
        thickness
    )

    cv2.line(
        court,
        (right - single_margin, top),
        (right - single_margin, bottom),
        line_color,
        thickness
    )

    # net
    cv2.line(
        court,
        (left, net_y),
        (right, net_y),
        line_color,
        thickness
    )

    # short service line
    short_offset = int(short_service * scale)

    cv2.line(
        court,
        (left, net_y - short_offset),
        (right, net_y - short_offset),
        line_color,
        thickness
    )

    cv2.line(
        court,
        (left, net_y + short_offset),
        (right, net_y + short_offset),
        line_color,
        thickness
    )

    # center line
    cv2.line(
        court,
        (center_x, top),
        (center_x, bottom),
        line_color,
        thickness
    )

    # long service line for doubles
    long_offset = int(long_service_double * scale)

    cv2.line(
        court,
        (left, top + long_offset),
        (right, top + long_offset),
        line_color,
        thickness
    )

    cv2.line(
        court,
        (left, bottom - long_offset),
        (right, bottom - long_offset),
        line_color,
        thickness
    )

    return court
def draw_top_view(points3dP1=None, points3dP2=None, extra_info=None):

    scale_vis = 40
    scale_real = 100  # meter → pixel
    sx, sy = 400, 800
    cx, cy = sx//2, sy//2
    #canvas = np.zeros((sy, sx, 3), dtype=np.uint8)
    canvas = np.full(
        (sy, sx, 3),
        (40, 100, 40),
        dtype = np.uint8
    )
    mid_margin = 30
    mid_canvas = np.full(
        (sy+mid_margin, sx+mid_margin, 3),
        (40, 100, 40),
        dtype = np.uint8
    )
    court_w_m = 6.10
    court_h_m = 13.40
    single_w_m = 5.18

    scale_x = sx / court_w_m
    scale_y = sy / court_h_m
      
    #canvas = draw_mini_court(height=sy)
    SERVE_AREA_A, SERVE_AREA_B = cy-90, cy+90
    
    single_x_offset = int(((court_w_m - single_w_m) / 2) * scale_x)
    single_y_offset = int(0.76 * scale_y)
    canvas = cv2.line(canvas, (cx, 0), (cx, SERVE_AREA_A), COLOR_Y, 2)
    canvas = cv2.line(canvas, (cx, SERVE_AREA_B), (cx, sy), COLOR_Y, 2)
    canvas = cv2.line(canvas, (0, cy), (sx, cy), COLOR_W, 2)
    canvas = cv2.line(canvas, (0, 0), (sx, 0), COLOR_W, 2)
    canvas = cv2.line(canvas, (0, sy), (sx, sy), COLOR_W, 2)
    canvas = cv2.line(canvas, (sx, 0), (sx, sy), COLOR_W, 2)
    canvas = cv2.line(canvas, (0, 0), (0, sy), COLOR_W, 2)
    
    canvas = cv2.line(canvas, (single_x_offset, 0), (single_x_offset, sy), COLOR_W, 2)
    canvas = cv2.line(canvas, (sx-single_x_offset, 0), (sx-single_x_offset, sy), COLOR_W, 2)

    canvas = cv2.line(canvas, (0, single_y_offset), (sx, single_y_offset), COLOR_W, 2)
    canvas = cv2.line(canvas, (0, sy - single_y_offset), (sx, sy - single_y_offset), COLOR_W, 2)
    # serve line
    canvas = cv2.line(canvas, (0, SERVE_AREA_A), (sx, SERVE_AREA_A), COLOR_Y, 2)
    canvas = cv2.line(canvas, (0, SERVE_AREA_B), (sx, SERVE_AREA_B), COLOR_Y, 2)
    draw_kpts = [
        BodyKpt.Left_Ankle,
        BodyKpt.Right_Ankle,
    ]
    bg_h = 950 
    bg_w = 500
    '''big_canvas = np.full(
        (bg_h, bg_w, 3),
        (40, 100, 40),
        dtype = np.uint8
    )'''
    big_canvas = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
      

    court_h, court_w = canvas.shape[:2]
    offset_x = (bg_w - court_w) // 2
    offset_y = (bg_h - court_h) // 2
    cx = offset_x + sx//2
    cy = offset_y + sy//2

    mid_court_h, mid_court_w = mid_canvas.shape[:2]
    offset_x_mid = (bg_w - mid_court_w) // 2
    offset_y_mid = (bg_h - mid_court_h) // 2
    big_canvas[
        offset_y_mid-10:offset_y_mid+mid_court_h-10, # 50 -> 20
        offset_x_mid:offset_x_mid+mid_court_w
    ] = mid_canvas
    big_canvas[
        offset_y-5:offset_y+court_h-5, # 40 -> 10
        offset_x:offset_x+court_w
    ] = canvas
    for i, points3d in enumerate([points3dP1, points3dP2]):
        if points3d is None:
            continue
        proj_xy = points3d[BodyKpt.Left_Shoulder:BodyKpt.Bbox_Center, :2]
        bbox_center = points3d[BodyKpt.Bbox_Center]
        centorid = np.mean(proj_xy, axis=0)
        #cv2.circle(canvas, (int(cx - centorid[0] * scale_vis), int(cy + centorid[1] * scale_vis)), 36, COLOR_G, 2)
        #cv2.circle(canvas, (int(cx - bbox_center[0] * scale_vis), int(cy + bbox_center[1] * scale_vis)), 5, COLOR_P, -1)
        #cv2.circle(canvas, (int(cx - bbox_center[0] * scale_vis), int(cy + bbox_center[1] * scale_vis)), 36, COLOR_R, 2)
        for j, point3d in enumerate(points3d):
            if j < BodyKpt.Left_Shoulder:
                continue
            if j not in draw_kpts:
                continue
            #x, y = point3d[:2]
            x, y, z = point3d
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue

            px = int(cx - x * scale_vis)
            py = int(cy + y * scale_vis)
            cv2.circle(big_canvas, (px, py), 3, COLOR_G if i == 0 else COLOR_R, -1)
            if j == BodyKpt.Right_Ankle:
                cv2.putText(big_canvas, f"x:{x * scale_real:.2f}, y:{y * scale_real:.2f},  z:{z * scale_real:.2f}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_W, 1)
            # cv2.putText(canvas, f"{i}", (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_W, 1)
    
   
    '''for i, point in enumerate(extra_info):
        x, y = point
        px = int(cx - x * scale_vis)
        py = int(cy + y * scale_vis)
        if i % 2:
            cv2.circle(canvas, (px, py), 5, COLOR_TRAJ, -1)
        # elif i % 3 == 1:
        #     cv2.circle(canvas, (px, py), 5, COLOR_B, -1)
        else:
            cv2.circle(canvas, (px, py), 5, COLOR_Y, -1)'''
        
       
    return big_canvas
'''
def draw_top_view(points3dP1=None, points3dP2=None, extra_info=None):
    scale_vis = 45
    sx, sy = 400, 800
    cx, cy = sx // 2, sy // 2
    canvas = np.zeros((sy, sx, 3), dtype=np.uint8)

    COURT_W = 6.10
    COURT_H = 13.40

    left   = int(cx - COURT_W / 2 * scale_vis)
    right  = int(cx + COURT_W / 2 * scale_vis)
    top    = int(cy - COURT_H / 2 * scale_vis)
    bottom = int(cy + COURT_H / 2 * scale_vis)

    # outer court
    cv2.rectangle(canvas, (left, top), (right, bottom), COLOR_W, 2)

    # net
    cv2.line(canvas, (left, cy), (right, cy), COLOR_Y, 2)

    # center line
    cv2.line(canvas, (cx, top), (cx, bottom), COLOR_Y, 1)

    # short service lines
    service_dist = 1.98
    cv2.line(canvas, (left, int(cy - service_dist * scale_vis)),
             (right, int(cy - service_dist * scale_vis)), COLOR_Y, 1)
    cv2.line(canvas, (left, int(cy + service_dist * scale_vis)),
             (right, int(cy + service_dist * scale_vis)), COLOR_Y, 1)

    # player skeleton points
    for i, points3d in enumerate([points3dP1, points3dP2]):
        if points3d is None:
            continue

        for j, point3d in enumerate(points3d):
            if j < BodyKpt.Left_Shoulder:
                continue

            x, y = point3d[:2]
            if np.isnan(x) or np.isnan(y):
                continue

            px = int(cx + x * scale_vis)
            py = int(cy + y * scale_vis)

            color = COLOR_G if i == 0 else COLOR_R
            cv2.circle(canvas, (px, py), 4, color, -1)

    return canvas
'''
# ===== Helper: draw front/back view =====
def draw_front_back_view(points3dP1=None, points3dP2=None):
    sx, sz = 400, 400
    cx, cz = sx//2, sz//2
    canvas = np.zeros((sz, sx, 3), dtype=np.uint8)
    scale_vis = 50
    scale_real = 100  # meter → pixel
    
    for bone in SKELETON_CONNECTIONS:
        x1, y1, z1 = points3dP1[bone[0]]
        x2, y2, z2 = points3dP1[bone[1]]
        px1 = int(cx - x1 * scale_vis)
        px2 = int(cx - x2 * scale_vis)
        pz1 = int(cz - z1 * scale_vis)
        pz2 = int(cz - z2 * scale_vis)
        cv2.line(canvas, (px1, pz1), (px2, pz2), COLOR_Y, 2)

    for i, points3d in enumerate([points3dP1, points3dP2]):
        if points3d is None:
            continue
        bbox_center = points3d[BodyKpt.Bbox_Center]
        # cv2.circle(canvas, (int(cx - bbox_center[0] * scale_vis), int(cz - bbox_center[2] * scale_vis)), 5, COLOR_R, -1)
        for j, point_xz in enumerate(points3d[:, [0, 2]]):
            if j < BodyKpt.Left_Shoulder:
                continue
            x, z = point_xz
            
            px = int(cx - x * scale_vis)
            pz = int(cz - z * scale_vis)
            # homogeneous coordinate for bbox center, should be around (cx, cz) if the points are valid
            point_color = COLOR_G if i == 0 else COLOR_R
            if j == BodyKpt.Right_Ankle:
                # cv2.circle(canvas, (px, pz), 5, COLOR_R, -1)
                cv2.putText(canvas, f"x:{x * scale_real:.2f}, z:{z * scale_real:.2f}", (px + 5, pz - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_W, 1)
            # else:
            if j == BodyKpt.Right_Ankle:
                point_color = (0,0,255)      # red

            elif j == BodyKpt.Right_Shoulder:
                point_color = (255,0,0)      # blue

            cv2.circle(
                canvas,
                (px,pz),
                5 if j in [
                    BodyKpt.Right_Ankle,
                    BodyKpt.Right_Shoulder
                ] else 3,
                point_color,
                -1
            )
            cv2.circle(canvas, (px, pz), 3, COLOR_G if i == 0 else COLOR_R, -1)
    
    return canvas

# Function to draw the virtual badminton court
def draw_virtual_court(frame, projMtx):
    for i in range(0, len(court_3d), 2):
        # Project the 3D points to 2D
        pt1_3d = np.array(court_3d[i] + [1])  # Homogeneous coordinates
        pt2_3d = np.array(court_3d[i + 1] + [1])

        pt1_2d = projMtx @ pt1_3d
        pt2_2d = projMtx @ pt2_3d

        # Normalize homogeneous coordinates
        pt1_2d /= pt1_2d[2]
        pt2_2d /= pt2_2d[2]

        # Convert to integer pixel values
        pt1 = (int(pt1_2d[0]), int(pt1_2d[1]))
        pt2 = (int(pt2_2d[0]), int(pt2_2d[1]))

        # Draw dotted yellow line
        for j in range(0, 101, 5):
            alpha = j / 100
            inter_pt = (
                int(pt1[0] * (1 - alpha) + pt2[0] * alpha),
                int(pt1[1] * (1 - alpha) + pt2[1] * alpha)
            )
            cv2.circle(frame, inter_pt, 1, COLOR_Y, -1)


class Kalman3D:
    def __init__(self, dt = 1/30, process_noise = 0.01, measurement_noise = 0.05): # process_noise = 0.03, measurement_noise = 0.05)
        self.initialized = False
        self.frame_count = 0
        self.x = np.zeros((6, 1), dtype=np.float32)  # [x,y,z,vx,vy,vz]
        self.prev_z = None
        self.dt = dt 
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        self.P = np.eye(6, dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        self.R = np.eye(3, dtype=np.float32) * measurement_noise
    
    def predict(self):
        if not self.initialized:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @self.F.T + self.Q
        return self.x[:3].flatten()

    def update(self, z):
        '''z = np.asarray(z, dtype=np.float32).reshape(3,1)
        if np.any(np.isnan(z)):
            return self.predict()
        
        if not self.initialized:
            self.x[:3] = z
            self.initialized = True
            return self.x[:3].flatten()
        
        self.predict()

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float32) - K @ self.H) @ self.P

        return self.x[:3].flatten()'''

        z = np.asarray(z, dtype=np.float32).reshape(3,1)

        if np.any(np.isnan(z)):
            return self.predict()

        # count frame
        self.frame_count += 1

        # initialize
        if not self.initialized:
            self.x[:3] = z
            self.prev_z = z.copy()
            self.initialized = True
            return self.x[:3].flatten()

        # first 10 frames:
        # directly use measurement
        #if self.frame_count < 10:
        #    self.x[:3] = z
        #    return self.x[:3].flatten()
        meas_v = (z - self.prev_z) / self.dt
        self.prev_z = z.copy()
        self.predict()

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        self.P = (
            np.eye(6, dtype=np.float32)
            - K @ self.H
        ) @ self.P

        return self.x[:3].flatten()
    def get_velocity(self):
        if not self.initialized:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        
        return self.x[3:].flatten()

def apply_kalman_to_skeleton(points_3d, kalman_filters):
    filtered = np.zeros_like(points_3d, dtype=np.float32)
    velocities = np.zeros_like(points_3d, dtype=np.float32)

    for k in range(points_3d.shape[0]):
        filtered[k] = kalman_filters[k].update(points_3d[k])
        velocities[k] = kalman_filters[k].get_velocity()

    return filtered, velocities


def angle_3d(a, b, c):
    """Calculate the angle at point b formed by points a, b, c in 3D space."""
    ba = a - b
    bc = c - b

    # Normalize the vectors
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
        return np.nan
    
    ba_unit = ba / ba_norm
    bc_unit = bc / bc_norm

    # Calculate the cosine of the angle using the dot product
    cos_angle = np.clip(np.dot(ba_unit, bc_unit), -1.0, 1.0)

    # Return the angle in degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

pose2D_projMtx_P1 = (pose_jsonls[CAM_PAIRS[0][0]], pose_jsonls[CAM_PAIRS[0][1]], projMtxs[CAM_PAIRS[0][0]], projMtxs[CAM_PAIRS[0][1]])
# pose2D_projMtx_P2 = (pose_jsonls[CAM_PAIRS[1][0]], pose_jsonls[CAM_PAIRS[1][1]], projMtxs[CAM_PAIRS[1][0]], projMtxs[CAM_PAIRS[1][1]])

if SELECTED_PAIR == 0:
    debug_projMtxs = (projMtxs[CAM_PAIRS[0][0]], projMtxs[CAM_PAIRS[0][1]])
else:
    debug_projMtxs = (projMtxs[CAM_PAIRS[1][0]], projMtxs[CAM_PAIRS[1][1]])
    
print(pose2D_projMtx_P1[0].shape, pose2D_projMtx_P1[1].shape)  # (num_frames, num_keypoints, 2)
# Define a buffer to store the last N frames of 3D points
BUFFER_SIZE = 3  # Window size for smoothing
buffer_P1 = []
buffer_P2 = []

kalman_P1 = [Kalman3D(dt = 1/fps_a) for _ in range(num_keypoints + 1)]
kalman_P2 = [Kalman3D(dt = 1/fps_a) for _ in range(num_keypoints + 1)]


def smooth_points(buffer, new_points):
    """Smooth 3D points using a moving average filter."""
    buffer.append(new_points)
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)  # Remove the oldest frame

    # Compute the moving average
    smoothed_points = np.mean(buffer, axis=0)
    return smoothed_points

# Add a buffer to store the last few positions of keypoints for fading footsteps
FOOTSTEP_BUFFER_SIZE = 20  # Number of frames to keep for fading footsteps
footstep_buffer_A = []  # Buffer for Camera A
footstep_buffer_B = []  # Buffer for Camera B

# Function to draw fading footsteps
def draw_fading_footsteps(frame, footstep_buffer, color):
    for i, keypoints in enumerate(footstep_buffer):
        alpha = (i + 1) / len(footstep_buffer)  # Gradual fading effect
        faded_color = tuple(int(c * alpha) for c in color)
        for i, (x, y) in enumerate(keypoints):
            y *= y_factor  # adjust for padding
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame, (int(x), int(y)), 2, faded_color, -1)

def homography_approx(ankle_points_2d, H_inv_Mtx):
    """Calculate approx human position from ankle keypoints and homography matrix."""
    
    ankle_points_2d_hom = np.vstack((ankle_points_2d.T, np.ones((1, ankle_points_2d.shape[0]))))  # Shape: (3, 2)
    # print(f"Ankle points 2D (homogeneous):\n{ankle_points_2d_hom}")
    ankle_points_3d_hom = H_inv_Mtx @ ankle_points_2d_hom  # Shape: (3, 2)
    ankle_points_3d = ankle_points_3d_hom[:3] / ankle_points_3d_hom[2]  # Normalize homogeneous coordinates
    return ankle_points_3d[:2].T  # Return x, y position in real-world coordinates


top_traj_P1 = []

# ===== Main loop =====
def main():
    # State
    trajectory_P1 = []
    trajectory_P2 = []
    frame_id = 0
    paused = False  # State to track if the program is paused
    
    retAS, frameAS = capA.read()
    retBS, frameBS = capB.read()
    if not retAS or not retBS:
        return

    capA.set(cv2.CAP_PROP_POS_FRAMES, 0)
    capB.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.namedWindow("Camera A")
    cv2.namedWindow("Camera B")
    cv2.namedWindow("Top View")
    cv2.namedWindow("Front/Back View")
    last_valid_points = [np.ones((4, num_keypoints + 1)), np.ones((4, num_keypoints + 1))]  # To store the last valid 3D points for each pair
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_out = 30
    writer_camA = cv2.VideoWriter(
        f"CameraA_{player_anchor}_{folder_name}_{date}.mp4",
        fourcc,
        fps_out,
        (frameAS.shape[1], frameAS.shape[0])
    )

    writer_camB = cv2.VideoWriter(
        f"CameraB_{player_anchor}_{folder_name}_{date}.mp4",
        fourcc,
        fps_out,
        (frameBS.shape[1], frameBS.shape[0])
    )

    writer_top = cv2.VideoWriter(
        f"TopView_{player_anchor}_{folder_name}_{date}.mp4",
        fourcc,
        fps_out,
        #(400, 800)
        (500, 950)
    )

    writer_front = cv2.VideoWriter(
        f"FrontBackView_{player_anchor}_{folder_name}_{date}.mp4",
        fourcc,
        fps_out,
        (400, 400)
    )

    while True:
    

        if paused:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused  # Toggle pause state
            continue  # Skip the rest of the loop if paused
        
        frame_id += 1
        #if frame_id % 4 != 0:
        #    continue
        
        capA.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        capB.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        retA, frameA = capA.read()
        retB, frameB = capB.read()
     
        # padding to 640 x 640
        # padding_y = (640 - 480) // 2
        # frameA = cv2.copyMakeBorder(frameA, padding_y, padding_y, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_K)
        # frameB = cv2.copyMakeBorder(frameB, padding_y, padding_y, 0, 0, cv2.BORDER_CONSTANT, value=COLOR_K)

        if not retA or not retB:
            break
        
        points_3d_P1 = []
        points_3d_P2 = []
        is_valid = [False, False]
        for i, (pose2D_A, pose2D_B, projMtx_A, projMtx_B) in enumerate([pose2D_projMtx_P1]):
            points_3d = np.ones((4, num_keypoints + 1))  # Homogeneous coordinates for triangulation
            # Extract the row corresponding to the current frame_id
            row_A = pose2D_A[pose2D_A["frame_id"] == frame_id]
            row_B = pose2D_B[pose2D_B["frame_id"] == frame_id]
            if row_A.empty or row_B.empty:
                continue
            # Extract 2D keypoints for both cameras
            points_2d_A = row_A.iloc[0][2:].values.reshape(-1, 2)
            points_2d_B = row_B.iloc[0][2:].values.reshape(-1, 2)
            
            ap_A = homography_approx(points_2d_A[[BodyKpt.Left_Ankle, BodyKpt.Right_Ankle]], H_inv_Mtxs[CAM_PAIRS[i][0]])
            ap_B = homography_approx(points_2d_B[[BodyKpt.Left_Ankle, BodyKpt.Right_Ankle]], H_inv_Mtxs[CAM_PAIRS[i][1]])
            # Add the current keypoints to the footstep buffer
            footstep_buffer_A.append(points_2d_A[[BodyKpt.Left_Ankle, BodyKpt.Right_Ankle]])
            footstep_buffer_B.append(points_2d_B[[BodyKpt.Left_Ankle, BodyKpt.Right_Ankle]])

            # Ensure the buffer size does not exceed the limit
            if len(footstep_buffer_A) > FOOTSTEP_BUFFER_SIZE:
                footstep_buffer_A.pop(0)
            if len(footstep_buffer_B) > FOOTSTEP_BUFFER_SIZE:
                footstep_buffer_B.pop(0)

            # Draw fading footsteps on the frames
            draw_fading_footsteps(frameA, footstep_buffer_A, COLOR_G)
            draw_fading_footsteps(frameB, footstep_buffer_B, COLOR_R)

            points_3d = cv2.triangulatePoints(
                projMtx_A, projMtx_B, points_2d_A.T, points_2d_B.T
            )
            points_3d /= points_3d[3]  # Normalize homogeneous coordinates
            bbox_hg = points_3d[:, -1]
            # Apply rule-based filtering to the triangulated 3D points
            is_valid[i] = rule_base_filter(points_3d)
            # use last valid points to replace current invalid points for better visualization
            points_3d = points_3d if is_valid[i] else last_valid_points[i]
            '''if is_valid[i]:
                last_valid_points[i] = points_3d'''
            if not is_valid[i]:
                points_3d[:] = np.nan
            # Project back to 2D for visualization
            # projected_2d_A = debug_projMtxs[0] @ points_3d
            # projected_2d_A /= projected_2d_A[2]  # Normalize homogeneous coordinates
            # projected_2d_B = debug_projMtxs[1] @ points_3d
            # projected_2d_B /= projected_2d_B[2]  # Normalize homogeneous coordinates
            points_3d = points_3d[:3].T  # Convert to Nx3
            # # padding adjustment for visualization
            
            # # # Original keypoints from camera A in green
            # # # Projected point after triangulation from camera B in red
            
            # projected_2d_A = projected_2d_A[:2].T
            # projected_2d_B = projected_2d_B[:2].T
            # kpt_id = 0
            # for (x, y), (px, py) in zip(points_2d_A, projected_2d_A):
            #     # adjust for padding
            #     y *= y_factor
            #     py *= y_factor
            #     if SELECTED_PAIR == i:
            #         cv2.circle(frameA, (int(x), int(y)), 2, COLOR_Y, -1)
            #     cv2.circle(frameA, (int(px), int(py)), 2, COLOR_G, -1)
            #     cv2.putText(frameA, f"{kpt_id}", (int(px)+8, int(py)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_W, 1)
            #     kpt_id += 1
                
            # # Draw skeleton connections for camera A
            # for idx1, idx2 in SKELETON_CONNECTIONS:
            #     if idx1 < num_keypoints and idx2 < num_keypoints:
            #         x1, y1 = points_2d_A[idx1]
            #         x2, y2 = points_2d_A[idx2]
            #         # adjust for padding
            #         y1 *= y_factor
            #         y2 *= y_factor
            #         cv2.line(frameA, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_Y, 1)
                    
            # # Original keypoints from camera B in green
            # # Projected point after triangulation from camera B in red
            # kpt_id = 0
            # for (x, y), (px, py) in zip(points_2d_B, projected_2d_B):
            #     y *= y_factor
            #     py *= y_factor
            #     if SELECTED_PAIR == i:
            #         cv2.circle(frameB, (int(x), int(y)), 2, COLOR_Y, -1)
            #     cv2.circle(frameB, (int(px), int(py)), 2, COLOR_R, -1)
            #     cv2.putText(frameB, f"{kpt_id}", (int(px)+8, int(py)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_W, 1)
            #     kpt_id += 1

            # # Draw skeleton connections for camera B
            # for idx1, idx2 in SKELETON_CONNECTIONS:
            #     if idx1 < num_keypoints and idx2 < num_keypoints:
            #         x1, y1 = points_2d_B[idx1]
            #         x2, y2 = points_2d_B[idx2]
            #         y1 *= y_factor
            #         y2 *= y_factor
            #         cv2.line(frameB, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_Y, 1)
            
            '''if i == 0:
                smoothed_points = smooth_points(buffer_P1, points_3d)
                # print(smoothed_points.shape)
                points_3d_P1.append(smoothed_points)
            else:
                smoothed_points = smooth_points(buffer_P2, points_3d)
                points_3d_P2.append(smoothed_points)'''
            if i == 0 :
                filtered_points,  velocities = apply_kalman_to_skeleton(points_3d, kalman_P1)
                points_3d_P1.append(filtered_points)
            else:
                filtered_points, velocities = apply_kalman_to_skeleton(points_3d, kalman_P2)
                points_3d_P2.append(filtered_points)

            # ==========================
            # Right arm biomechanics
            # ==========================

            right_shoulder = filtered_points[BodyKpt.Right_Shoulder]
            right_elbow = filtered_points[BodyKpt.Right_Elbow]
            right_wrist = filtered_points[BodyKpt.Right_Wrist]
            right_elbow_angle = angle_3d(right_shoulder, right_elbow, right_wrist)

            left_shoulder = filtered_points[BodyKpt.Left_Shoulder]
            left_elbow = filtered_points[BodyKpt.Left_Elbow]
            left_wrist = filtered_points[BodyKpt.Left_Wrist]
            left_elbow_angle = angle_3d(left_shoulder, left_elbow, left_wrist)

       
            left_ankle = filtered_points[BodyKpt.Left_Ankle]
            right_ankle = filtered_points[BodyKpt.Right_Ankle]
            #left_ankle = smoothed_points[BodyKpt.Left_Ankle]
            #right_ankle = smoothed_points[BodyKpt.Right_Ankle]
            player_pos = np.nanmean(
                np.vstack([left_ankle, right_ankle]),
                axis=0
            )
            player_pos_right_ankle = np.nanmean(
                np.vstack([right_ankle, right_ankle]),
                axis=0
            )
            left_v = velocities[BodyKpt.Left_Ankle]
            right_v = velocities[BodyKpt.Right_Ankle]
            left_z = left_ankle[2]
            right_z = right_ankle[2]

            left_vz = left_v[2]
            right_vz = right_v[2]
            player_v = np.nanmean(
            np.vstack([left_v, right_v]),
            axis=0)

            speed_mps = np.linalg.norm(player_v[:2])
            speed_kmh = speed_mps * 3.6
            
            is_jump =(
                left_z > 0.90 and
                right_z > 0.90 and
                abs(left_vz) > 0.3 and
                abs(right_vz) > 0.3
            )
            if is_jump:
                print(f"left_z: {left_z}, right_z: {right_z}")
            if player_anchor == "middle":
                if not np.any(np.isnan(player_pos[:2])):
                    top_traj_P1.append({"pos": player_pos[:2].copy(),
                                    "jump": is_jump})
            if player_anchor == "right_ankel":
                if not np.any(np.isnan(player_pos_right_ankle[:2])):
                    top_traj_P1.append({"pos": player_pos_right_ankle[:2].copy(),
                                    "jump": is_jump})
                

            data = {
                "frame_id": frame_id,
                "x": player_pos[0],
                "y": player_pos[1],
                "z": player_pos[2],
                "vx": player_v[0],
                "vy": player_v[1],
                "vz": player_v[2],
                "right_elbow_angle (deg)": right_elbow_angle,
                "left_elbow_angle (deg)": left_elbow_angle,
                "jump": is_jump,
                "speed_mps": speed_mps,
                "speed_kmh": speed_kmh
            }

            if i == 0:
                trajectory_P1.append(data)
            else:
                trajectory_P2.append(data)
        if len(points_3d_P1) == 0:
            print(f"Skip frame {frame_id}: no valid 3D points")
            continue        
        all_homography_points = np.vstack((ap_A, ap_B))
        # print(all_homography_points)

        top_view = draw_top_view(points_3d_P1[0], None, all_homography_points)
        top_view = draw_top_trajectory(top_view, top_traj_P1)
        # testing homography approximation for human position
        # ap_Ax, ap_Ay = ap_A
        # ap_Bx, ap_By = ap_B
        # cv2.circle(top_view, (int(ap_Ax), int(ap_Ay)), 5, COLOR_G, -1)
        # cv2.circle(top_view, (int(ap_Bx), int(ap_By)), 5, COLOR_R, -1)
        # Calculate human positions for Camera A and Camera B
        # human_positions_A = calculate_human_position(pose2D_projMtx_P1[0])
        # human_positions_B = calculate_human_position(pose2D_projMtx_P1[1])

        # # Plot human positions on the top-down view
        # plot_human_positions(top_view, human_positions_A, COLOR_G)
        # plot_human_positions(top_view, human_positions_B, COLOR_R)
        # show images
        # Draw the virtual court on both camera views
        draw_virtual_court(frameA, debug_projMtxs[0])
        draw_virtual_court(frameB, debug_projMtxs[1])
        front_back_view = draw_front_back_view(points_3d_P1[0])
        writer_camA.write(frameA)
        writer_camB.write(frameB)
        writer_top.write(top_view)
        writer_front.write(front_back_view)

        cv2.imshow("Camera A", frameA)
        cv2.imshow("Camera B", frameB)
        cv2.imshow("Top View", top_view)
      

        # Show images
        cv2.imshow("Front/Back View", front_back_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused  # Toggle pause state
    writer_camA.release()
    writer_camB.release()
    writer_top.release()
    writer_front.release()
    df_P1 = pd.DataFrame(trajectory_P1)
    df_P2 = pd.DataFrame(trajectory_P2)

    df_P1.to_csv(f"Player1_trajectory_{player_anchor}_{folder_name}_{date}.csv", index=False)
    df_P2.to_csv(f"Player2_trajectory_{player_anchor}_{folder_name}_{date}.csv", index=False)

    print(f"Saved Player1_trajectory_{date}.csv")
    print(f"Saved Player2_trajectory_{date}.csv")
    capA.release()
    capB.release()
    # capB.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
