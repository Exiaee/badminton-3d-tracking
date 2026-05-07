from enum import Enum, IntEnum
import numpy as np

class BodyKpt(IntEnum):
    Nose = 0,
    Left_Eye = 1,
    Right_Eye = 2,
    Left_Ear = 3,
    Right_Ear = 4,
    Left_Shoulder = 5,
    Right_Shoulder = 6,
    Left_Elbow = 7,
    Right_Elbow = 8,
    Left_Wrist = 9,
    Right_Wrist = 10,
    Left_Hip = 11,
    Right_Hip = 12,
    Left_Knee = 13,
    Right_Knee = 14,
    Left_Ankle = 15,
    Right_Ankle = 16,
    Bbox_Center = 17,  # added for convenience, not part of original COCO keypoints

# Keypoint connections for drawing skeletons (using COCO format as reference)
SKELETON_CONNECTIONS = [
    (BodyKpt.Left_Shoulder, BodyKpt.Right_Shoulder), (BodyKpt.Left_Hip, BodyKpt.Right_Hip), (BodyKpt.Left_Shoulder, BodyKpt.Left_Hip), (BodyKpt.Right_Shoulder, BodyKpt.Right_Hip), # Torso
    (BodyKpt.Right_Shoulder, BodyKpt.Right_Elbow), (BodyKpt.Right_Elbow, BodyKpt.Right_Wrist), # Right_arm
    (BodyKpt.Left_Shoulder, BodyKpt.Left_Elbow), (BodyKpt.Left_Elbow, BodyKpt.Left_Wrist), # Left_arm
    (BodyKpt.Right_Hip, BodyKpt.Right_Knee), (BodyKpt.Right_Knee, BodyKpt.Right_Ankle), # Right_leg
    (BodyKpt.Left_Hip, BodyKpt.Left_Knee), (BodyKpt.Left_Knee, BodyKpt.Left_Ankle), # Left_leg
]

# Define the 3D coordinates of the badminton court in real-world space
# Court dimensions (in meters): 13.4m x 6.1m
# Assuming the court is centered at (0, 0, 0) in the 3D space
court_3d = [
    [-3.1, -6.7, 0], [+3.1, -6.7, 0],  # Bottom line
    [+3.1, -6.7, 0], [+3.1, +6.7, 0],  # Right line
    [+3.1, +6.7, 0], [-3.1, +6.7, 0],  # Top line
    [-3.1, +6.7, 0], [-3.1, -6.7, 0],  # Left line
    [0, -6.7, 0], [0, +6.7, 0],        # Center line
    [-3.1, 0, 0], [+3.1, 0, 0],         # Net line
    [-3.1, -2, 0], [+3.1, -2, 0],   # Service line A
    [-3.1, +2, 0], [+3.1, +2, 0]    # Service line B
]