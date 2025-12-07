# Utils.py - Comprehensive Documentation

## Overview
`utils.py` is a comprehensive pose estimation and joint angle calculation module designed for exercise tracking applications. It uses the YOLO11 pose estimation model to detect human keypoints and calculates various biomechanical angles for exercise analysis.

---

## Table of Contents
1. [Core Functionality](#core-functionality)
2. [Key Components](#key-components)
3. [Supported Exercises](#supported-exercises)
4. [Keypoint System](#keypoint-system)
5. [Important Methods](#important-methods)
6. [Angle Calculations](#angle-calculations)
7. [Usage Examples](#usage-examples)

---

## Core Functionality

### What This Module Does:
- **Pose Detection**: Detects 17 human body keypoints using YOLO11
- **Angle Calculation**: Computes joint angles (elbows, knees, hips, shoulders)
- **Exercise-Specific Metrics**: Calculates specialized measurements for 8 different exercises
- **Visual Feedback**: Draws skeleton, keypoints, and angles on video frames
- **Calibration System**: Optional calibration phase for pose detection

---

## Key Components

### 1. **PoseCalibrator Class**
The main class that handles all pose estimation and angle calculations.

```python
calibrator = PoseCalibrator(model_path='yolo11n-pose.pt', calibration_time=0)
```

**Parameters:**
- `model_path`: Path to YOLO11 pose model file (default: `'yolo11n-pose.pt'`)
- `calibration_time`: Seconds for calibration phase (default: 0 = skip calibration)

---

## Keypoint System

### 17 COCO Keypoints Used:
```
0:  nose          1:  left_eye       2:  right_eye
3:  left_ear      4:  right_ear      5:  left_shoulder
6:  right_shoulder 7:  left_elbow    8:  right_elbow
9:  left_wrist    10: right_wrist    11: left_hip
12: right_hip     13: left_knee      14: right_knee
15: left_ankle    16: right_ankle
```

### Skeleton Connections:
The module draws 16 connections between keypoints to form a human skeleton:
- Face: nose → eyes → ears
- Upper body: shoulders → elbows → wrists
- Torso: shoulders → hips
- Lower body: hips → knees → ankles

---

## Supported Exercises

The module calculates specialized angles and metrics for:

1. **Pushups** - Elbow angles, hip alignment
2. **Squats** - Knee angles, torso alignment, shin angle
3. **Situps** - Torso inclination, hip flexion
4. **Sit & Reach** - Reach distance, hip angle, back angle, knee straightness
5. **Skipping** - Back posture, knee angles, jump detection
6. **Jumping Jacks** - Arm elevation, back posture, leg spread
7. **Vertical Jump** - Countermovement depth, arm swing, landing mechanics
8. **Broad Jump** - Countermovement, arm swing, takeoff angle

---

## Important Methods

### 1. **detect_pose(frame)**
Detects human pose in a single frame using YOLO11.

**Input:** 
- `frame`: BGR image (numpy array)

**Output:**
- `keypoints`: Array of shape (17, 3) containing [x, y, confidence] for each keypoint
- Returns `None` if no person detected

**Example:**
```python
keypoints = calibrator.detect_pose(frame)
if keypoints is not None:
    print(f"Detected {len(keypoints)} keypoints")
```

---

### 2. **calculate_angle(pt1, pt2, pt3)**
Calculates the angle formed by three points (vertex at pt2).

**Mathematics:**
Uses the dot product formula:
```
angle = arccos((v1 · v2) / (|v1| × |v2|))
where v1 = pt1 - pt2, v2 = pt3 - pt2
```

**Input:**
- `pt1`, `pt2`, `pt3`: Points as (x, y) tuples or arrays
- `pt2` is the vertex where angle is measured

**Output:**
- Angle in degrees (0-180°)

**Example:**
```python
# Calculate elbow angle
shoulder = (100, 100)
elbow = (150, 150)  # vertex
wrist = (200, 200)
angle = calibrator.calculate_angle(shoulder, elbow, wrist)
print(f"Elbow angle: {angle}°")
```

---

### 3. **get_all_joint_angles(keypoints)**
The master method that calculates ALL angles for detected pose.

**Calculates:**
- **Basic Joint Angles**: 8 major joints (elbows, shoulders, hips, knees)
- **Torso Angle**: Inclination relative to vertical
- **Shin Angles**: Left and right leg shin angles
- **Situp Metrics**: Torso inclination, hip flexion
- **Sit & Reach Metrics**: 6 specialized measurements
- **Skipping Metrics**: Back angle, knee angles
- **Jumping Jacks Metrics**: Arm elevation, back posture
- **Jump Metrics**: Countermovement, arm swing, landing angles

**Input:**
- `keypoints`: Array from `detect_pose()`

**Output:**
- Dictionary with all calculated angles
- Returns `None` for angles that couldn't be calculated

**Example:**
```python
angles = calibrator.get_all_joint_angles(keypoints)
print(f"Left elbow: {angles['left_elbow']}°")
print(f"Right knee: {angles['right_knee']}°")
print(f"Torso angle: {angles['torso_angle']}°")
```

---

### 4. **process_frame(frame, show_angles_panel=True)**
Main processing pipeline - detects pose, calculates angles, and draws visualization.

**Input:**
- `frame`: Video frame (BGR image)
- `show_angles_panel`: Whether to show angles panel (default: True)

**Output:**
- `frame`: Annotated frame with skeleton and angles drawn
- `keypoints`: Detected keypoints
- `angles`: Dictionary of all calculated angles

**Example:**
```python
frame, keypoints, angles = calibrator.process_frame(frame)
cv2.imshow('Exercise Tracking', frame)
```

---

## Angle Calculations (Detailed)

### Basic Joint Angles

#### 1. **Elbow Angle**
```python
'left_elbow': (shoulder, elbow, wrist)
'right_elbow': (shoulder, elbow, wrist)
```
- **Range**: 0° (fully extended) to 180° (fully flexed)
- **Used for**: Pushups, bicep curls, arm exercises

#### 2. **Knee Angle**
```python
'left_knee': (hip, knee, ankle)
'right_knee': (hip, knee, ankle)
```
- **Range**: 0° (fully flexed) to 180° (fully extended)
- **Used for**: Squats, lunges, leg exercises

#### 3. **Hip Angle**
```python
'left_hip': (shoulder, hip, knee)
'right_hip': (shoulder, hip, knee)
```
- **Range**: 0° to 180°
- **Used for**: Squats, situps, body alignment

#### 4. **Shoulder Angle**
```python
'left_shoulder': (elbow, shoulder, hip)
'right_shoulder': (elbow, shoulder, hip)
```
- **Range**: 0° to 180°
- **Used for**: Arm raises, overhead movements

---

### Exercise-Specific Angles

#### **SQUATS**

##### 1. `torso_angle`
Measures torso inclination relative to vertical axis.

**Calculation:**
- Midpoint of shoulders → Midpoint of hips
- Angle between this vector and vertical (0°, -1)

**Interpretation:**
- 0° = Perfectly upright
- 30° = Slight forward lean (acceptable)
- 60°+ = Excessive forward lean (poor form)

**Code:**
```python
def calculate_torso_angle(self, keypoints):
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    torso_vector = shoulder_mid - hip_mid
    angle = arccos(torso_vector · vertical_vector)
```

##### 2. `shin_angle_left` / `shin_angle_right`
Measures shin angle relative to vertical (knee tracking).

**Calculation:**
- Ankle → Knee vector
- Angle with vertical axis

**Interpretation:**
- 0° = Shin perfectly vertical (knees not forward)
- 15-20° = Good squat depth (knees forward)
- 30°+ = Knees too far forward

---

#### **SITUPS**

##### 1. `torso_inclination_horizontal`
Measures torso angle relative to horizontal (lying = 0°, sitting = 90°).

**Calculation:**
```python
dy = shoulder_mid[y] - hip_mid[y]
dx = shoulder_mid[x] - hip_mid[x]
angle = abs(arctan2(-dy, dx))
```

**Interpretation:**
- 0-20° = Lying down (rest position)
- 40-70° = Crunch position (rep in progress)
- 70-90° = Full situp (peak position)

##### 2. `hip_flexion_angle`
Measures hip bending (shoulder-hip-knee angle).

**Interpretation:**
- 180° = Straight body (lying)
- 90° = Sitting position
- <60° = Deep crunch

---

#### **SIT & REACH**

##### 1. `reach_distance`
Forward reach distance in pixels (wrist_x - ankle_x).

**Calculation:**
```python
avg_wrist_x = (left_wrist[x] + right_wrist[x]) / 2
avg_ankle_x = (left_ankle[x] + right_ankle[x]) / 2
reach_distance = avg_wrist_x - avg_ankle_x
```

**Usage:**
- Track maximum reach during test
- Positive = reaching past feet
- Negative = not reaching feet

##### 2. `arm_length`
Distance from shoulder to wrist (normalization factor).

##### 3. `sitnreach_hip_angle`
Hip flexibility angle (shoulder-hip-knee).

**Interpretation:**
- 60° or less = Excellent flexibility
- 60-80° = Good flexibility
- 80-100° = Average flexibility
- 100°+ = Poor flexibility

##### 4. `sitnreach_knee_angle`
Validity check - knees should remain straight.

**Interpretation:**
- 165°+ = Valid (straight legs)
- <165° = Invalid (bent knees)

##### 5. `reach_symmetry`
Left vs right wrist position difference.

**Interpretation:**
- <20 pixels = Good symmetry
- 20-50 pixels = Acceptable
- >50 pixels = Asymmetric reach

---

#### **SKIPPING**

##### 1. `skip_back_angle`
Back posture during skipping (shoulder-hip-knee).

**Interpretation:**
- 150-180° = Good upright posture
- 120-150° = Slight lean (acceptable)
- <120° = Poor posture (hunched)

##### 2. `skip_knee_angle`
Knee bend during jump phase.

---

#### **JUMPING JACKS**

##### 1. `jj_arm_angle`
Arm elevation angle (hip-shoulder-wrist).

**Calculation:**
Angle at shoulder joint between torso direction and arm direction.

**Interpretation:**
- 0-30° = Arms down (closed position) ✓
- 90° = Arms horizontal (transition)
- 150-180° = Arms overhead (open position) ✓

**State Detection:**
```python
if arm_angle < 60:
    state = "closed"
elif arm_angle > 120:
    state = "open"
```

##### 2. `jj_back_angle`
Back alignment during movement.

---

#### **VERTICAL JUMP**

##### 1. `vjump_countermovement_angle`
Knee angle during squat phase before jump.

**Interpretation:**
- 180° = Standing straight
- 90-110° = Good countermovement ✓
- <90° = Too deep (inefficient)

##### 2. `vjump_arm_swing_angle`
Elbow angle during arm swing.

##### 3. `vjump_landing_knee_angle`
Knee angle on landing (shock absorption).

**Interpretation:**
- 90-120° = Good absorption ✓
- <90° = Too deep (risk)
- >150° = Stiff landing (risk)

---

#### **BROAD JUMP**

Uses same metrics as vertical jump:
- `bjump_countermovement_angle`
- `bjump_arm_swing_angle`

---

### Visual Methods

#### **draw_keypoints(frame, keypoints, min_confidence=0.5)**
Draws circular markers at each detected keypoint.

**Visual Elements:**
- Green filled circles (radius 6)
- Magenta borders (radius 9)
- Only draws if confidence > 0.5

#### **draw_skeleton(frame, keypoints, min_confidence=0.5)**
Draws lines connecting keypoints to form skeleton.

**Visual Elements:**
- White lines (thickness 3)
- 16 connection lines
- Only draws connections where both keypoints are confident

#### **draw_joint_angles(frame, keypoints, angles)**
Displays angle values next to joints.

**Visual Elements:**
- Cyan text with black background
- Positioned near joint vertices
- Only shows traditional joint angles (excludes torso, shin, etc.)

#### **draw_calibration_status(frame, keypoints)**
Shows calibration progress banner at top of frame.

**States:**
- "Please stand in front of camera" (red) - No person detected
- "Calibrating... 50% (3s remaining)" (yellow) - In progress
- "CALIBRATION SUCCESSFUL!" (green) - Complete

---

