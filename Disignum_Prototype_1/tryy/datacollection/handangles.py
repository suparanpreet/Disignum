import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import json
import time
from matplotlib import pyplot as plt
import math
from collections import defaultdict, deque

# More focused joint list for better visualization (most important angles)
joint_list = [
    # Thumb joints
    [2, 1, 0],  # Thumb CMC joint (connecting to wrist)
    [3, 2, 1],  # Thumb MCP joint
    [4, 3, 2],  # Thumb IP joint

    # Index finger joints
    [6, 5, 0],  # Index MCP joint (connecting to wrist)
    [7, 6, 5],  # Index PIP joint
    [8, 7, 6],  # Index DIP joint

    # Middle finger joints
    [10, 9, 0],  # Middle MCP joint (connecting to wrist)
    [11, 10, 9],  # Middle PIP joint
    [12, 11, 10],  # Middle DIP joint

    # Ring finger joints
    [14, 13, 0],  # Ring MCP joint (connecting to wrist)
    [15, 14, 13],  # Ring PIP joint
    [16, 15, 14],  # Ring DIP joint

    # Pinky finger joints
    [18, 17, 0],  # Pinky MCP joint (connecting to wrist)
    [19, 18, 17],  # Pinky PIP joint
    [20, 19, 18],  # Pinky DIP joint

    # Additional inter-finger angles (optional)
    [5, 0, 9],  # Angle between index and middle at wrist
    [9, 0, 13],  # Angle between middle and ring at wrist
    [13, 0, 17],  # Angle between ring and pinky at wrist
]

# Global variable to store angle data
angle_data_buffer = []
recording = False
start_time = None

# One Euro Filter for jitter reduction
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.last_time = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class LandmarksSmoother:
    """Smoothing wrapper for MediaPipe landmarks"""
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.filters = defaultdict(lambda: {
            'x': OneEuroFilter(freq, min_cutoff, beta, d_cutoff),
            'y': OneEuroFilter(freq, min_cutoff, beta, d_cutoff),
            'z': OneEuroFilter(freq, min_cutoff, beta, d_cutoff),
            'v': OneEuroFilter(freq, min_cutoff, beta, d_cutoff),
        })

    def smooth_pose(self, pose_results):
        if not pose_results or not pose_results.pose_landmarks:
            return pose_results
        for i, lm in enumerate(pose_results.pose_landmarks.landmark):
            f = self.filters[f'pose_{i}']
            lm.x = float(f['x'].filter(lm.x))
            lm.y = float(f['y'].filter(lm.y))
            # MediaPipe Pose may not always have meaningful z/visibility; still filter if present
            if hasattr(lm, 'z'):
                lm.z = float(f['z'].filter(lm.z))
            if hasattr(lm, 'visibility'):
                lm.visibility = float(f['v'].filter(lm.visibility))
        return pose_results

    def smooth_hands(self, hand_results):
        if not hand_results or not hand_results.multi_hand_landmarks:
            return hand_results
        for hand_idx, hand in enumerate(hand_results.multi_hand_landmarks):
            for i, lm in enumerate(hand.landmark):
                f = self.filters[f'hand_{hand_idx}_{i}']
                lm.x = float(f['x'].filter(lm.x))
                lm.y = float(f['y'].filter(lm.y))
                if hasattr(lm, 'z'):
                    lm.z = float(f['z'].filter(lm.z))
        return hand_results


# REQUIRED KEYS for schema normalization (matches the runtime player expectations)
REQUIRED_KEYS = [
    "duration", "head_y",
    "left_arm_x", "left_arm_y", "left_arm_z", "left_elbow_x", "left_elbow_z",
    "left_hand_x", "left_hand_y", "left_hand_z",
    "right_arm_x", "right_arm_y", "right_arm_z", "right_elbow_x", "right_elbow_z",
    "right_hand_x", "right_hand_y", "right_hand_z",
    "left_thumb_1_x", "left_thumb_2_x", "left_thumb_3_x",
    "left_index_1_z", "left_index_2_z", "left_index_3_z",
    "left_middle_1_z", "left_middle_2_z", "left_middle_3_z",
    "left_ring_1_z", "left_ring_2_z", "left_ring_3_z",
    "left_pinky_1_z", "left_pinky_2_z", "left_pinky_3_z",
    "right_thumb_1_x", "right_thumb_2_x", "right_thumb_3_x",
    "right_index_1_z", "right_index_2_z", "right_index_3_z",
    "right_middle_1_z", "right_middle_2_z", "right_middle_3_z",
    "right_ring_1_z", "right_ring_2_z", "right_ring_3_z",
    "right_pinky_1_z", "right_pinky_2_z", "right_pinky_3_z",
]


def normalize_frame(frame: dict, default_duration: int) -> dict:
    """Ensure all keys exist and clamp to safe anatomical limits."""
    out = dict(frame)  # copy
    out["duration"] = int(max(default_duration, 800))

    def clamp(v, lo, hi):
        try:
            return float(np.clip(v, lo, hi))
        except Exception:
            return float(lo)

    # Global
    out["head_y"] = clamp(out.get("head_y", 0.0), -90.0, 90.0)

    # Left shoulder/arm rules
    out["left_arm_x"] = clamp(out.get("left_arm_x", 0.0), -90.0, 0.0)
    out["left_arm_y"] = clamp(out.get("left_arm_y", 0.0), -90.0, 90.0)
    out["left_arm_z"] = clamp(out.get("left_arm_z", 0.0), -90.0, 90.0)
    out["left_elbow_x"] = clamp(out.get("left_elbow_x", 0.0), -145.0, 0.0)
    out["left_elbow_z"] = clamp(out.get("left_elbow_z", 0.0), -45.0, 45.0)
    out["left_hand_x"] = clamp(out.get("left_hand_x", 0.0), -90.0, 0.0)
    out["left_hand_y"] = clamp(out.get("left_hand_y", 0.0), -90.0, 90.0)
    out["left_hand_z"] = clamp(out.get("left_hand_z", 0.0), -90.0, 90.0)

    # Right shoulder/arm rules
    out["right_arm_x"] = clamp(out.get("right_arm_x", 0.0), 0.0, 90.0)
    out["right_arm_y"] = clamp(out.get("right_arm_y", 0.0), -90.0, 90.0)
    out["right_arm_z"] = clamp(out.get("right_arm_z", 0.0), -90.0, 90.0)
    out["right_elbow_x"] = clamp(out.get("right_elbow_x", 0.0), 0.0, 145.0)
    out["right_elbow_z"] = clamp(out.get("right_elbow_z", 0.0), -45.0, 45.0)
    out["right_hand_x"] = clamp(out.get("right_hand_x", 0.0), 0.0, 90.0)
    out["right_hand_y"] = clamp(out.get("right_hand_y", 0.0), 0.0, 90.0)
    out["right_hand_z"] = clamp(out.get("right_hand_z", 0.0), -90.0, 90.0)

    # Fingers bend 0..120, Right thumb negative (closing) by your spec
    for side in ["left", "right"]:
        if side == "left":
            thumb_sign = 1.0
        else:
            thumb_sign = -1.0
        for i, key in enumerate(["thumb_1_x", "thumb_2_x", "thumb_3_x"]):
            val = out.get(f"{side}_{key}", 0.0)
            # MCP can swing ±60; PIP/DIP 0..90
            if i == 0:
                out[f"{side}_{key}"] = clamp(thumb_sign * val, -60.0 if side == "right" else 0.0, 60.0 if side == "left" else 0.0)
            elif i == 1:
                out[f"{side}_{key}"] = clamp(thumb_sign * val, -80.0 if side == "right" else 0.0, 80.0 if side == "left" else 0.0)
            else:
                out[f"{side}_{key}"] = clamp(thumb_sign * val, -90.0 if side == "right" else 0.0, 90.0 if side == "left" else 0.0)

        for finger in ["index", "middle", "ring", "pinky"]:
            for j in [1, 2, 3]:
                key = f"{side}_{finger}_{j}_z"
                out[key] = clamp(out.get(key, 0.0), 0.0, 120.0)

    # Ensure missing keys exist
    for k in REQUIRED_KEYS:
        if k not in out:
            out[k] = 0.0 if k != "duration" else int(max(default_duration, 800))

    # Final rounding
    for k, v in out.items():
        if k != "duration":
            try:
                out[k] = round(float(v), 2)
            except Exception:
                pass
    return out


def _np(p):  # helper to numpy vec
    return np.array([p[0], p[1], p[2] if len(p) > 2 else 0.0], dtype=float)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _angle(a, b):
    a = _norm(a)
    b = _norm(b)
    d = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(d)))


def compute_torso_axes(pose_landmarks):
    """Build a stable torso-aligned coordinate frame."""
    LSH = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    RSH = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    LHP = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    RHP = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    lsh = np.array([LSH.x, LSH.y, getattr(LSH, "z", 0.0)], float)
    rsh = np.array([RSH.x, RSH.y, getattr(RSH, "z", 0.0)], float)
    lhp = np.array([LHP.x, LHP.y, getattr(LHP, "z", 0.0)], float)
    rhp = np.array([RHP.x, RHP.y, getattr(RHP, "z", 0.0)], float)

    x_axis = _norm(rsh - lsh)             # left->right
    y_axis = _norm(((lsh + rsh) / 2.0) - ((lhp + rhp) / 2.0))  # hips->shoulders (up)
    z_axis = _norm(np.cross(x_axis, y_axis))  # forward/back

    # Re-orthogonalize to ensure right-handed frame
    y_axis = _norm(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def compute_shoulder_angles(pose_landmarks, side: str):
    """Decompose shoulder motion into flexion/extension (x), ab/adduction (y), rotation (z)
    using a torso-aligned coordinate frame for stable, signed angles."""
    shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER if side == 'left' else mp_pose.PoseLandmark.RIGHT_SHOULDER
    elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW if side == 'left' else mp_pose.PoseLandmark.RIGHT_ELBOW
    wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST if side == 'left' else mp_pose.PoseLandmark.RIGHT_WRIST

    SH = pose_landmarks[shoulder_idx.value]
    EL = pose_landmarks[elbow_idx.value]
    WR = pose_landmarks[wrist_idx.value]

    sh = np.array([SH.x, SH.y, getattr(SH, "z", 0.0)], float)
    el = np.array([EL.x, EL.y, getattr(EL, "z", 0.0)], float)
    wr = np.array([WR.x, WR.y, getattr(WR, "z", 0.0)], float)

    x_axis, y_axis, z_axis = compute_torso_axes(pose_landmarks)
    upper = _norm(el - sh)
    forearm = _norm(wr - el)

    # Flexion/extension (x): angle in sagittal plane (spanned by y,z) w.r.txt.t. +y (up)
    upper_sag = _norm(upper - np.dot(upper, x_axis) * x_axis)  # remove x component
    flex = _angle(upper_sag, y_axis)
    flex_sign = np.sign(np.dot(upper_sag, z_axis))  # forward -> + sign
    arm_x = flex_sign * flex
    # convention: left forward negative, right forward positive
    if side == 'left':
        arm_x = -abs(arm_x)
    else:
        arm_x = abs(arm_x)
    arm_x = float(np.clip(arm_x, -90.0, 90.0))

    # Ab/adduction (y): angle in frontal plane (spanned by x,y) w.r.txt.t. +y (up)
    upper_front = _norm(upper - np.dot(upper, z_axis) * z_axis)  # remove z component
    abd = _angle(upper_front, y_axis)
    # up (dot with y positive) should be negative by your spec
    abd_sign = -1.0 if np.dot(upper, y_axis) > 0 else 1.0
    arm_y = float(np.clip(abd_sign * abd, -90.0, 90.0))

    # Internal/external rotation (z): twist of forearm around upper arm
    # Compare plane normals formed by (upper, reference) and (upper, forearm)
    # Use torso x-axis as reference to avoid degeneracy when arms are down
    n_ref = _norm(np.cross(upper, x_axis))
    n_fa = _norm(np.cross(upper, forearm))
    rot = _angle(n_ref, n_fa)
    twist_sign = np.sign(np.dot(np.cross(n_ref, n_fa), upper))
    arm_z = float(np.clip(twist_sign * rot, -90.0, 90.0))

    # Elbow flexion: 0 straight -> 145 flexed
    elbow_angle = _angle(-upper, forearm)
    elbow_angle = float(np.clip(elbow_angle, 0.0, 145.0))
    elbow_x = -elbow_angle if side == 'left' else elbow_angle

    return arm_x, arm_y, arm_z, elbow_x


def compute_wrist_angles(hand_landmarks, handed_label: str, elbow_vec: np.ndarray):
    """Compute wrist flexion/extension (x), ulnar/radial deviation (y), and pronation/supination (z)
    using a forearm-aligned local basis for stable signed angles."""
    # Landmarks
    w = hand_landmarks.landmark[0]   # wrist
    i5 = hand_landmarks.landmark[5]  # index MCP
    m9 = hand_landmarks.landmark[9]  # middle MCP
    p17 = hand_landmarks.landmark[17]  # pinky MCP

    wv = np.array([w.x, w.y, getattr(w, "z", 0.0)], float)
    i5v = np.array([i5.x, i5.y, getattr(i5, "z", 0.0)], float)
    m9v = np.array([m9.x, m9.y, getattr(m9, "z", 0.0)], float)
    p17v = np.array([p17.x, p17.y, getattr(p17, "z", 0.0)], float)

    # Forearm axis (elbow->wrist)
    u = _norm(elbow_vec)  # main axis
    # Hand long axis (palm to middle finger)
    v = _norm(m9v - wv)
    # Hand side axis (across palm width)
    s = _norm(i5v - p17v)

    # Build orthonormal basis around forearm: (u, r.txt, n)
    # r.txt is perpendicular to (u, s); n completes the right-handed frame
    r = _norm(np.cross(u, s))
    if np.linalg.norm(r) < 1e-6:
        # fall back to cross with v if s is degenerate
        r = _norm(np.cross(u, v))
    n = _norm(np.cross(r, u))

    def signed_angle(a, b, axis):
        # atan2 of the magnitude of rotation around axis
        a = _norm(a); b = _norm(b); axis = _norm(axis)
        ang = np.degrees(np.arctan2(np.dot(np.cross(a, b), axis), np.dot(a, b)))
        return ang

    # Flexion/extension: angle between u and v in (u,n) plane, sign by n
    v_un = _norm(v - np.dot(v, r) * r)  # remove r.txt component
    flex = signed_angle(u, v_un, r)  # + when v rotates towards +n from u
    # Convention: left flexion negative, right flexion positive
    wrist_x = -flex if handed_label == "left" else flex
    wrist_x = float(np.clip(wrist_x, -90.0 if handed_label == "left" else 0.0,
                            0.0 if handed_label == "left" else 90.0))

    # Ulnar/radial deviation: angle between u and v in (u,r.txt) plane, sign by n
    v_ur = _norm(v - np.dot(v, n) * n)  # remove n component
    dev = signed_angle(u, v_ur, n)  # + when v rotates towards +r.txt from u
    # Convention: make left deviation negative, right positive
    wrist_y = -dev if handed_label == "left" else dev
    wrist_y = float(np.clip(wrist_y, -90.0 if handed_label == "left" else 0.0,
                            0.0 if handed_label == "left" else 90.0))

    # Pronation/supination: rotation of hand side axis s around forearm axis u, using r.txt as baseline
    s_proj = _norm(s - np.dot(s, u) * u)  # project onto plane perpendicular to forearm
    z_rot = signed_angle(r, s_proj, u)
    wrist_z = float(np.clip(z_rot, -90.0, 90.0))

    return wrist_x, wrist_y, wrist_z


# Smoothers and calibration baseline
smoother = LandmarksSmoother(freq=30.0, min_cutoff=1.2, beta=0.02, d_cutoff=1.0)
baseline_offsets = None  # will store a movement frame to subtract as neutral if set

MIRRORED_VIEW = True


def calculate_angle_accurate(a, b, c):
    """
    Calculate the angle at point b formed by points a, b, c
    More accurate method using dot product and cross product
    """
    # Convert to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Create vectors
    ba = a - b  # Vector from b to a
    bc = c - b  # Vector from b to c

    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Handle numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in degrees
    angle = np.arccos(cosine_angle) * 180.0 / np.pi

    return angle


def calculate_anatomical_angle(a, b, c, joint_type, side='left'):
    """
    Calculate anatomical angles with proper constraints and directions
    """
    # Convert to numpy arrays with z coordinates
    a = np.array([a[0], a[1], getattr(a, 'z', 0.0) if hasattr(a, 'z') else 0.0])
    b = np.array([b[0], b[1], getattr(b, 'z', 0.0) if hasattr(b, 'z') else 0.0])
    c = np.array([c[0], c[1], getattr(c, 'z', 0.0) if hasattr(c, 'z') else 0.0])

    # Create vectors
    ba = a - b
    bc = c - b

    # Calculate basic angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    base_angle = np.arccos(cosine_angle) * 180.0 / np.pi

    # Apply anatomical constraints and sign conventions
    if joint_type == 'arm_x':  # Arm forward/backward (shoulder flexion/extension)
        if side == 'left':
            # Left arm forward: negative angles (0 to -90)
            angle = -(base_angle / 2.0) if base_angle > 90 else -(base_angle / 2.0)
            return np.clip(angle, -90, 0)
        else:  # right
            # Right arm forward: positive angles (0 to 90)
            angle = (base_angle / 2.0) if base_angle > 90 else (base_angle / 2.0)
            return np.clip(angle, 0, 90)

    elif joint_type == 'arm_y':  # Arm up/down (shoulder abduction/adduction)
        if side == 'left':
            # Up: negative (0 to -90), Down: positive (0 to 90)
            if ba[1] < 0:  # Moving up
                angle = -(base_angle / 2.0)
                return np.clip(angle, -90, 0)
            else:  # Moving down
                angle = (base_angle / 2.0)
                return np.clip(angle, 0, 90)
        else:  # right
            # Same logic for right arm
            if ba[1] < 0:  # Moving up
                angle = -(base_angle / 2.0)
                return np.clip(angle, -90, 0)
            else:  # Moving down
                angle = (base_angle / 2.0)
                return np.clip(angle, 0, 90)

    elif joint_type == 'arm_z':  # Arm rotation (shoulder internal/external rotation)
        if side == 'left':
            # Internal rotation: negative (0 to -90)
            angle = -(base_angle / 2.0)
            return np.clip(angle, -90, 0)
        else:  # right
            # Internal rotation: positive angles (0 to 90)
            angle = (base_angle / 2.0)
            return np.clip(angle, 0, 90)

    elif joint_type == 'elbow_x':  # Elbow flexion
        if side == 'left':
            # Left elbow: negative angles (0 to -145)
            angle = -(base_angle * 145.0 / 180.0)
            return np.clip(angle, -145, 0)
        else:  # right
            # Right elbow: positive angles (0 to 145)
            angle = (base_angle * 145.0 / 180.0)
            return np.clip(angle, 0, 145)

    elif joint_type == 'elbow_z':  # Elbow up/down
        if side == 'left':
            # Left elbow up: positive angles (0 to 90)
            angle = base_angle * 90.0 / 180.0
            return np.clip(angle, 0, 90)
        else:  # right
            # Right elbow up: negative angles (0 to -90)
            angle = -(base_angle * 90.0 / 180.0)
            return np.clip(angle, -90, 0)

    elif joint_type == 'hand_x' or joint_type == 'hand_y':  # Wrist flexion/extension
        if side == 'left':
            # Left wrist: negative angles (0 to -90)
            angle = -(base_angle / 2.0)
            return np.clip(angle, -90, 0)
        else:  # right
            # Right wrist: positive angles (0 to 90)
            angle = (base_angle / 2.0)
            return np.clip(angle, 0, 90)

    elif joint_type == 'hand_z':  # Wrist up/down rotation
        # Both hands: -90 (up) to +90 (down)
        if ba[1] < 0:  # Moving up
            angle = -90.0 * (base_angle / 180.0)
        else:  # Moving down
            angle = 90.0 * (base_angle / 180.0)
        return np.clip(angle, -90, 90)

    elif joint_type == 'finger':  # Finger joints
        # All fingers bend towards palm: positive angles (0 to 120)
        angle = base_angle * 120.0 / 180.0
        return np.clip(angle, 0, 120)

    elif joint_type == 'thumb':  # Thumb joints
        if side == 'left':
            # Left thumb: positive angles (0 to 120)
            angle = base_angle * 120.0 / 180.0
            return np.clip(angle, 0, 120)
        else:  # right
            # Right thumb: negative angles (0 to -120)
            angle = -(base_angle * 120.0 / 180.0)
            return np.clip(angle, -120, 0)

    return base_angle


def extract_angles_data(pose_results, hand_results):
    """
    Extract all angles and format them according to the JSON structure with anatomical constraints
    Now uses torso-aligned axes for shoulder/elbow and smoothed landmarks for stability.
    """
    global baseline_offsets

    angles_dict = {
        "duration": 1000,
        "head_y": 0.0,
        "left_arm_x": 0.0, "left_arm_y": 0.0, "left_arm_z": 0.0,
        "left_elbow_x": 0.0, "left_elbow_z": 0.0,  # add left_elbow_z
        "left_hand_x": 0.0, "left_hand_y": 0.0, "left_hand_z": 0.0,
        "right_arm_x": 0.0, "right_arm_y": 0.0, "right_arm_z": 0.0,
        "right_elbow_x": 0.0, "right_elbow_z": 0.0,  # add right_elbow_z
        "right_hand_x": 0.0, "right_hand_y": 0.0, "right_hand_z": 0.0,
        "left_thumb_1_x": 0.0, "left_thumb_2_x": 0.0, "left_thumb_3_x": 0.0,
        "left_index_1_z": 0.0, "left_index_2_z": 0.0, "left_index_3_z": 0.0,
        "left_middle_1_z": 0.0, "left_middle_2_z": 0.0, "left_middle_3_z": 0.0,
        "left_ring_1_z": 0.0, "left_ring_2_z": 0.0, "left_ring_3_z": 0.0,
        "left_pinky_1_z": 0.0, "left_pinky_2_z": 0.0, "left_pinky_3_z": 0.0,
        "right_thumb_1_x": 0.0, "right_thumb_2_x": 0.0, "right_thumb_3_x": 0.0,
        "right_index_1_z": 0.0, "right_index_2_z": 0.0, "right_index_3_z": 0.0,
        "right_middle_1_z": 0.0, "right_middle_2_z": 0.0, "right_middle_3_z": 0.0,
        "right_ring_1_z": 0.0, "right_ring_2_z": 0.0, "right_ring_3_z": 0.0,
        "right_pinky_1_z": 0.0, "right_pinky_2_z": 0.0, "right_pinky_3_z": 0.0
    }

    # Skip if torso landmarks missing or low confidence
    if pose_results and pose_results.pose_landmarks:
        vis_ok = True
        for idx in [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value,
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.RIGHT_HIP.value]:
            lm = pose_results.pose_landmarks.landmark[idx]
            if hasattr(lm, "visibility") and lm.visibility < 0.5:
                vis_ok = False
                break
        if vis_ok:
            try:
                l_ax, l_ay, l_az, l_ex = compute_shoulder_angles(pose_results.pose_landmarks.landmark, 'left')
                r_ax, r_ay, r_az, r_ex = compute_shoulder_angles(pose_results.pose_landmarks.landmark, 'right')
                l_ez = compute_elbow_updown(pose_results.pose_landmarks.landmark, 'left')
                r_ez = compute_elbow_updown(pose_results.pose_landmarks.landmark, 'right')

                angles_dict.update({
                    "left_arm_x": l_ax, "left_arm_y": l_ay, "left_arm_z": l_az,
                    "left_elbow_x": l_ex, "left_elbow_z": l_ez,
                    "right_arm_x": r_ax, "right_arm_y": r_ay, "right_arm_z": r_az,
                    "right_elbow_x": r_ex, "right_elbow_z": r_ez
                })
            except Exception as e:
                print(f"Error computing shoulder/elbow: {e}")

    # Wrist and fingers
    if hand_results and hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(
                zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
            hand_label = handedness.classification[0].label.lower()  # 'left' or 'right'
            if MIRRORED_VIEW:
                hand_label = 'left' if hand_label == 'right' else 'right'
            try:
                # Approximate forearm vector from pose (elbow->wrist)
                if pose_results and pose_results.pose_landmarks:
                    if hand_label == 'left':
                        EL = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        WR = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    else:
                        EL = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                        WR = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    elbow_vec = np.array([WR.x - EL.x, WR.y - EL.y, getattr(WR, "z", 0.0) - getattr(EL, "z", 0.0)], float)
                else:
                    elbow_vec = np.array([0.0, -1.0, 0.0], float)

                wx, wy, wz = compute_wrist_angles(hand_landmarks, hand_label, elbow_vec)
                angles_dict[f"{hand_label}_hand_x"] = wx
                angles_dict[f"{hand_label}_hand_y"] = wy
                angles_dict[f"{hand_label}_hand_z"] = wz

                # Fingers: bend towards palm (positive), thumb sign handled in normalize_frame
                finger_map = {
                    'thumb': {'joints': [[2, 1, 0], [3, 2, 1], [4, 3, 2]], 'names': ['thumb_1_x', 'thumb_2_x', 'thumb_3_x']},
                    'index': {'joints': [[6, 5, 0], [7, 6, 5], [8, 7, 6]], 'names': ['index_1_z', 'index_2_z', 'index_3_z']},
                    'middle': {'joints': [[10, 9, 0], [11, 10, 9], [12, 11, 10]], 'names': ['middle_1_z', 'middle_2_z', 'middle_3_z']},
                    'ring': {'joints': [[14, 13, 0], [15, 14, 13], [16, 15, 14]], 'names': ['ring_1_z', 'ring_2_z', 'ring_3_z']},
                    'pinky': {'joints': [[18, 17, 0], [19, 18, 17], [20, 19, 18]], 'names': ['pinky_1_z', 'pinky_2_z', 'pinky_3_z']},
                }
                for finger, config in finger_map.items():
                    for (a_i, b_i, c_i), name in zip(config['joints'], config['names']):
                        a = hand_landmarks.landmark[a_i]
                        b = hand_landmarks.landmark[b_i]
                        c = hand_landmarks.landmark[c_i]
                        a_coords = [a.x, a.y, getattr(a, 'z', 0.0)]
                        b_coords = [b.x, b.y, getattr(b, 'z', 0.0)]
                        c_coords = [c.x, c.y, getattr(c, 'z', 0.0)]
                        ang = calculate_angle_accurate(a_coords, b_coords, c_coords)
                        if finger == 'thumb':
                            angles_dict[f"{hand_label}_{name}"] = ang
                        else:
                            angles_dict[f"{hand_label}_{name}"] = ang
            except Exception as e:
                print(f"Error processing {hand_label} hand: {e}")

    # Apply baseline neutral offsets if set
    if baseline_offsets:
        for k in angles_dict.keys():
            if k in baseline_offsets and k != "duration":
                angles_dict[k] = angles_dict[k] - baseline_offsets[k]

    return angles_dict


def save_angles_to_try_json(angle_data_list, animation_name="recorded_motion", filename="try.json"):
    """
    Merge-save angle data to try.json under animations[animation_name].
    Ensures proper schema for every frame.
    """
    # Normalize frames
    normalized = [normalize_frame(frame, default_duration=frame.get("duration", 1000)) for frame in angle_data_list]

    payload = {
        "text": animation_name.replace("_", " "),
        "movements": normalized,
    }

    try:
        if os.path.exists(filename):
            with open(filename, 'r.txt', encoding='utf-8') as f:
                db = json.load(f)
        else:
            db = {"animations": {}}

        if "animations" not in db or not isinstance(db["animations"], dict):
            db["animations"] = {}

        db["animations"][animation_name.lower().strip()] = payload

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        print(f"Saved '{animation_name}' with {len(normalized)} frames to {filename}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")


def save_angles_to_json(angle_data_list, filename="hand_arm_angles.json", animation_name="recorded_motion"):
    """
    Save angle data to JSON file in the specified format
    """
    json_structure = {
        "animations": {
            animation_name: {
                "text": animation_name.replace("_", " "),
                "movements": angle_data_list
            }
        }
    }

    try:
        with open(filename, 'w') as f:
            json.dump(json_structure, f, indent=2)
        print(f"Angles saved to {filename}")
        print(f"Total frames recorded: {len(angle_data_list)}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")


def draw_finger_angles(image, results, joint_list):
    """
    Draw finger angles on the image with improved accuracy and readability
    """
    height, width, _ = image.shape

    # Loop through hands
    for hand in results.multi_hand_landmarks:
        # Loop through joint sets
        for i, joint in enumerate(joint_list):
            try:
                # Get landmark coordinates (convert to pixel coordinates)
                a = [hand.landmark[joint[0]].x * width, hand.landmark[joint[0]].y * height]
                b = [hand.landmark[joint[1]].x * width, hand.landmark[joint[1]].y * height]
                c = [hand.landmark[joint[2]].x * width, hand.landmark[joint[2]].y * height]

                # Calculate accurate angle
                angle = calculate_angle_accurate(a, b, c)

                # Position text with offset to avoid overlap
                text_offset_x = (i % 4) * 30 - 60  # Spread horizontally
                text_offset_y = (i // 4) * 20 - 40  # Stack vertically

                text_pos = (int(b[0] + text_offset_x), int(b[1] + text_offset_y))

                # Draw the angle text with background for better readability
                text = f"{round(angle, 1)}"

                # Add background rectangle
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(image,
                              (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                              (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                              (0, 0, 0), -1)

                # Draw text
                cv2.putText(image, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            except Exception as e:
                print(f"Error calculating angle for joint {joint}: {e}")
                continue

    return image


def draw_arm_angles(image, pose_results, hand_results):
    """
    Draw arm angles (shoulder, elbow, wrist) for both left and right arms
    Uses pose landmarks for shoulder/elbow and hand landmarks for better wrist calculation
    """
    if not pose_results.pose_landmarks:
        return image

    pose_landmarks = pose_results.pose_landmarks.landmark
    height, width, _ = image.shape

    # Define arm joints for both sides
    arm_joints = {
        'left': {
            'shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            'wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            'hip': mp.solutions.pose.PoseLandmark.LEFT_HIP
        },
        'right': {
            'shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            'elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            'wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            'hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP
        }
    }

    # Get hand landmarks for wrist calculations
    hand_wrist_coords = {}
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(
                zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
            # Get hand label (Left/Right)
            hand_label = handedness.classification[0].label.lower()

            # Get wrist and finger points for wrist angle calculation
            wrist_coords = [hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y]  # Wrist
            middle_mcp = [hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y]  # Middle finger MCP
            index_mcp = [hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y]  # Index finger MCP

            hand_wrist_coords[hand_label] = {
                'wrist': wrist_coords,
                'middle_mcp': middle_mcp,
                'index_mcp': index_mcp
            }

    for side, joints in arm_joints.items():
        try:
            # Get coordinates for pose joints
            shoulder_coords = [pose_landmarks[joints['shoulder'].value].x,
                               pose_landmarks[joints['shoulder'].value].y]
            elbow_coords = [pose_landmarks[joints['elbow'].value].x,
                            pose_landmarks[joints['elbow'].value].y]
            pose_wrist_coords = [pose_landmarks[joints['wrist'].value].x,
                                 pose_landmarks[joints['wrist'].value].y]
            hip_coords = [pose_landmarks[joints['hip'].value].x,
                          pose_landmarks[joints['hip'].value].y]

            # Calculate shoulder angle (hip-shoulder-elbow)
            shoulder_angle = calculate_angle_accurate(hip_coords, shoulder_coords, elbow_coords)

            # Calculate elbow angle (shoulder-elbow-wrist)
            elbow_angle = calculate_angle_accurate(shoulder_coords, elbow_coords, pose_wrist_coords)

            # Calculate wrist angle using hand landmarks if available
            wrist_angle = None
            if side in hand_wrist_coords:
                hand_data = hand_wrist_coords[side]
                # Wrist angle: elbow-wrist-middle_finger_direction
                # Use the direction from wrist to middle finger MCP as the "hand direction"
                wrist_angle = calculate_angle_accurate(elbow_coords, hand_data['wrist'], hand_data['middle_mcp'])

            # Convert to pixel coordinates for display
            shoulder_pixel = tuple(np.multiply(shoulder_coords, [width, height]).astype(int))
            elbow_pixel = tuple(np.multiply(elbow_coords, [width, height]).astype(int))
            wrist_pixel = tuple(np.multiply(pose_wrist_coords, [width, height]).astype(int))

            # Position offsets to avoid overlap
            y_offset = 0 if side == 'left' else 0
            x_offset = -120 if side == 'left' else 20

            # Display shoulder angle
            shoulder_text = f"{side.capitalize()} Shoulder: {round(shoulder_angle, 1)}°"
            shoulder_pos = (shoulder_pixel[0] + x_offset, shoulder_pixel[1] + y_offset - 40)

            # Add background for shoulder text
            text_size = cv2.getTextSize(shoulder_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image,
                          (shoulder_pos[0] - 2, shoulder_pos[1] - text_size[1] - 2),
                          (shoulder_pos[0] + text_size[0] + 2, shoulder_pos[1] + 2),
                          (0, 0, 0), -1)

            cv2.putText(image, shoulder_text, shoulder_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

            # Display elbow angle
            elbow_text = f"{side.capitalize()} Elbow: {round(elbow_angle, 1)}°"
            elbow_pos = (elbow_pixel[0] + x_offset, elbow_pixel[1] + y_offset - 20)

            # Add background for elbow text
            text_size = cv2.getTextSize(elbow_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image,
                          (elbow_pos[0] - 2, elbow_pos[1] - text_size[1] - 2),
                          (elbow_pos[0] + text_size[0] + 2, elbow_pos[1] + 2),
                          (0, 0, 0), -1)

            cv2.putText(image, elbow_text, elbow_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            # Display wrist angle if calculated
            if wrist_angle is not None:
                wrist_text = f"{side.capitalize()} Wrist: {round(wrist_angle, 1)}°"
                wrist_pos = (wrist_pixel[0] + x_offset, wrist_pixel[1] + y_offset)

                # Add background for wrist text
                text_size = cv2.getTextSize(wrist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image,
                              (wrist_pos[0] - 2, wrist_pos[1] - text_size[1] - 2),
                              (wrist_pos[0] + text_size[0] + 2, wrist_pos[1] + 2),
                              (0, 0, 0), -1)

                cv2.putText(image, wrist_text, wrist_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

            # Draw lines to visualize the angles (optional)
            cv2.line(image, shoulder_pixel, elbow_pixel, (255, 0, 0), 2)
            cv2.line(image, elbow_pixel, wrist_pixel, (255, 0, 0), 2)

        except Exception as e:
            print(f"Error calculating {side} arm angles: {e}")
            continue

    return image


def get_label(index, hand, results):
    """
    Fixed function to properly get hand labels (Left/Right)
    """
    output = None

    # Check if we have handedness results and the index is valid
    if results.multi_handedness and index < len(results.multi_handedness):
        classification = results.multi_handedness[index]

        # Get label and score
        label = classification.classification[0].label
        score = classification.classification[0].score
        text = '{} {}'.format(label, round(score, 2))

        # Extract Coordinates for wrist position
        coords = tuple(np.multiply(
            np.array((hand.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                      hand.landmark[mp.solutions.hands.HandLandmark.WRIST].y)),
            [640, 480]).astype(int))

        output = text, coords

    return output


# Initialize MediaPipe
mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Main execution
cap = cv2.VideoCapture(0)

print("Controls:")
print("Press 'r.txt' to start/stop recording angles")
print("Press 's' to save recorded angles to try.json (you will be prompted for the animation name)")
print("Press 'd' to set frame duration (minimum 800ms)")
print("Press 'n' to capture current neutral baseline (calibration)")
print("Press 'c' to clear current buffer")
print("Press 'q' to quit")
print("\nAnatomical Angle Constraints:")
print("- Left arm forward: 0 to -90°")
print("- Right arm forward: 0 to 90°")
print("- Elbows - Left: 0 to -145°, Right: 0 to 145°")
print("- Wrist rotation: -90° to +90°")
print("- Fingers bend: 0° to 120°")
print("- Right thumb: 0° to -120°")

# Duration setting
frame_duration = 1000  # Default 1000ms

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands:
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False

            # Raw detections
            raw_hand_results = hands.process(image)
            raw_pose_results = pose.process(image)

            # Smooth detections
            pose_results = smoother.smooth_pose(raw_pose_results)
            results = smoother.smooth_hands(raw_hand_results)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract and store angle data if recording
            if recording:
                angles_data = extract_angles_data(pose_results, results)
                angles_data["duration"] = max(frame_duration, 800)
                angle_data_buffer.append(angles_data)

                cv2.putText(image, "RECORDING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Frames: {len(angle_data_buffer)} | Duration: {frame_duration}ms", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw pose landmarks
            if pose_results and pose_results.pose_landmarks:
                mp_drawings.draw_landmarks(
                    image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawings.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawings.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                # Show arm angles overlay (kept as before)
                try:
                    draw_arm_angles(image, pose_results, results)
                except Exception:
                    pass

            # Rendering hand results
            if results and results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawings.draw_landmarks(
                        image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawings.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawings.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
                    label_result = get_label(num, hand, results)
                    if label_result:
                        text, coord = label_result
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                draw_finger_angles(image, results, joint_list)

            cv2.putText(image, "Press 'r.txt':record 's':save 'd':duration 'n':neutral 'c':clear 'q':quit",
                        (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Hand and Arm Tracking - Accurate Angles', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r.txt'):
                if not recording:
                    recording = True
                    angle_data_buffer = []
                    start_time = time.time()
                    print("Recording started...")
                else:
                    recording = False
                    print(f"Recording stopped. Captured {len(angle_data_buffer)} frames")
            elif key == ord('s'):
                if angle_data_buffer:
                    try:
                        # Ask for the animation name to save under (this is the SAME text you'll type later)
                        animation_name = input("\nEnter animation name (text key for try.json): ").strip()
                        if not animation_name:
                            animation_name = f"recorded_motion_{int(time.time())}"
                        save_angles_to_try_json(angle_data_buffer, animation_name=animation_name, filename="try.json")
                    except Exception as e:
                        print(f"Error while saving: {e}")
                else:
                    print("No data to save. Record some angles first.")
            elif key == ord('d'):
                try:
                    duration_input = input("\nEnter frame duration in milliseconds (minimum 800): ")
                    new_duration = int(duration_input)
                    if new_duration >= 800:
                        frame_duration = new_duration
                        print(f"Duration set to {frame_duration}ms")
                    else:
                        print("Duration must be at least 800ms. Using 800ms.")
                        frame_duration = 800
                except ValueError:
                    print("Invalid input. Duration unchanged.")
                except:
                    pass
            elif key == ord('n'):
                # Capture current neutral baseline (subtract from subsequent frames)
                if pose_results or (results and results.multi_hand_landmarks):
                    neutral = extract_angles_data(pose_results, results)
                    neutral["duration"] = 0
                    baseline_offsets = neutral
                    print("Neutral baseline captured. Future frames will be offset by this baseline.")
                else:
                    print("Cannot capture neutral baseline: no landmarks detected.")
            elif key == ord('c'):
                angle_data_buffer = []
                print("Cleared current recording buffer.")

cap.release()
cv2.destroyAllWindows()


def compute_elbow_updown(pose_landmarks, side: str) -> float:
    """
    Signed 'up/down' deviation of the forearm in the frontal plane (x-y) at the elbow.
    Positive means lateral (toward +x), negative toward -x. Clamped later in normalize.
    """
    shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER if side == 'left' else mp_pose.PoseLandmark.RIGHT_SHOULDER
    elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW if side == 'left' else mp_pose.PoseLandmark.RIGHT_ELBOW
    wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST if side == 'left' else mp_pose.PoseLandmark.RIGHT_WRIST

    SH = pose_landmarks[shoulder_idx.value]
    EL = pose_landmarks[elbow_idx.value]
    WR = pose_landmarks[wrist_idx.value]

    sh = np.array([SH.x, SH.y, getattr(SH, "z", 0.0)], float)
    el = np.array([EL.x, EL.y, getattr(EL, "z", 0.0)], float)
    wr = np.array([WR.x, WR.y, getattr(WR, "z", 0.0)], float)

    x_axis, y_axis, z_axis = compute_torso_axes(pose_landmarks)
    upper = _norm(el - sh)
    forearm = _norm(wr - el)

    # Project forearm to frontal plane (remove forward/back component)
    forearm_front = _norm(forearm - np.dot(forearm, z_axis) * z_axis)
    # Measure angle from vertical (+y) with sign by x side
    angle = _angle(forearm_front, y_axis)
    sign = np.sign(np.dot(forearm_front, x_axis))  # rightwards positive
    elbow_z = float(sign * angle)

    # For left/right rigs, keep sign natural; map later if needed
    return elbow_z
