# Test.py - Comprehensive Documentation

## Overview
`test.py` is the main application interface for the AI Exercise Trainer system. It provides an interactive CLI for analyzing 8 different exercises using real-time pose estimation, biomechanical tracking, and performance evaluation. The module integrates `utils.py` (pose detection) and `metrics.py` (performance tracking) to deliver comprehensive exercise analysis.

---

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [Exercise Detection Logic](#exercise-detection-logic)
3. [Formulas & Calculations](#formulas--calculations)
4. [State Machines](#state-machines)
5. [Thresholds & Scoring](#thresholds--scoring)
6. [Important Functions](#important-functions)
7. [Exercise-Specific Logic](#exercise-specific-logic)
8. [Visual Feedback System](#visual-feedback-system)
9. [Usage Examples](#usage-examples)

---

## Core Architecture

### Class: ExerciseEvaluator

The main class that orchestrates exercise tracking and analysis.

**Key Components:**
```python
class ExerciseEvaluator:
    - calibrator: PoseCalibrator  # Pose detection engine
    - metrics: PerformanceMetrics  # Performance tracking
    - current_exercise: str        # Active exercise type
    - counter: int                 # Rep counter
    - stage: str                   # Current exercise phase
    - feedback: str                # Real-time feedback text
    - start_time: float            # Session start timestamp
```

**State Variables:**
- `counter`: Tracks completed repetitions
- `stage`: Current phase (e.g., "UP", "DOWN", "AIR", "GROUND")
- `feedback`: Real-time form feedback shown to user
- `last_knee_angles`: Smoothing buffer for angle stability
- `last_elbow_angles`: Smoothing buffer for elbow tracking

---

## Exercise Detection Logic

### 1. PUSHUPS

#### State Machine:
```
UP (elbow > 140°) → DOWN (elbow < 90°) → UP = 1 REP
```

#### Detection Formula:
```python
# Angle Smoothing (3-frame moving average)
elbow_smooth = mean([elbow_t-2, elbow_t-1, elbow_t])

# State Transition Logic
if elbow_smooth > 140° and stage == "DOWN":
    counter += 1
    stage = "UP"
elif elbow_smooth < 90°:
    stage = "DOWN"
```

#### Form Validation:
```python
# Hip alignment check
hip_threshold = 130°  # Minimum acceptable hip angle
if hip_angle < (hip_threshold - 20):
    feedback = "Fix Back!"  # Body sagging
else:
    feedback = "Good Form"
```

#### Tracked Metrics:
- **Elbow angles** (left/right): 0-180° (bent to extended)
- **Hip angles**: Check for body sagging (should stay > 130°)
- **Shoulder angles**: Verify proper alignment
- **Rep quality**: Good form vs bad form reps

#### Scoring Formula:
```python
# Rep Quality Score
if hip_angle >= 130 and elbow_min < 95 and elbow_max > 140:
    rep_quality = "GOOD"
    good_reps += 1
else:
    rep_quality = "BAD"
    bad_reps += 1

# Overall Score (0-100)
form_score = (good_reps / total_reps) × 100
range_score = ((180 - mean_elbow_range) / 180) × 100
final_score = (form_score × 0.6) + (range_score × 0.4)
```

---

### 2. SQUATS

#### State Machine:
```
UP (knee > 155°) → DOWN (knee < 135°) → UP = 1 REP
```

#### Detection Formula:
```python
# 3-frame moving average for stability
knee_smooth = mean([knee_t-2, knee_t-1, knee_t])

# Phase Detection
if knee_smooth > 155°:
    if stage == "DOWN":
        counter += 1
        # Record eccentric time
        eccentric_time = current_time - rep_bottom_time
    stage = "UP"
    current_phase = "standing"
    
elif knee_smooth < 135°:
    if stage == "UP":
        # Record descent (eccentric phase)
        eccentric_time = current_time - rep_start_time
        rep_bottom_time = current_time
    stage = "DOWN"
    current_phase = "bottom"
```

#### Depth Scoring:
```python
# Depth Categories
PARALLEL = 135°      # Thigh parallel to ground
DEEP = 90°           # Full squat

# Depth Score Formula
if knee_angle < 90°:
    depth_quality = "EXCELLENT"
    depth_score = 100
elif knee_angle < 120°:
    depth_quality = "GOOD"
    depth_score = 80
elif knee_angle < 135°:
    depth_quality = "PARALLEL"
    depth_score = 60
else:
    depth_quality = "SHALLOW"
    depth_score = 30
```

#### Form Validation:
```python
# Torso Lean Check
torso_angle = calculate_torso_angle(shoulder_mid, hip_mid)
if torso_angle > 45°:
    feedback = "Too Much Lean!"
    form_penalty = -20

# Knee Tracking (Shin Angle)
shin_angle = angle_between(ankle, knee, vertical)
if shin_angle > 30°:
    feedback = "Knees Too Far Forward!"
    form_penalty = -15
```

#### Velocity Tracking:
```python
# Angular Velocity (degrees per second)
angular_velocity = (knee_angle_t - knee_angle_t-1) / time_delta

# Sticking Point Detection
if angular_velocity < min_velocity:
    min_velocity = angular_velocity
    sticking_point = knee_angle
    # Sticking point indicates weak range
```

#### Tempo Analysis:
```python
# Rep Phases Timing
eccentric_phase = time_from_top_to_bottom  # Descent
isometric_phase = time_at_bottom           # Pause at bottom
concentric_phase = time_from_bottom_to_top # Ascent

# Tempo Ratio (ideal: 2:1:2)
tempo_ratio = eccentric : isometric : concentric

# Tempo Score
if tempo_ratio ≈ [2, 1, 2]:
    tempo_score = 100
else:
    tempo_score = 100 - abs(deviation) × 10
```

#### Overall Squat Score:
```python
# Weighted Scoring System
depth_weight = 0.35
form_weight = 0.30
tempo_weight = 0.20
consistency_weight = 0.15

squat_score = (
    depth_score × depth_weight +
    form_score × form_weight +
    tempo_score × tempo_weight +
    consistency_score × consistency_weight
)
```

---

### 3. SITUPS

#### State Machine:
```
REST (torso < 20°) → ASCENDING → PEAK (torso > 70°) → DESCENDING → REST = 1 REP
```

#### Detection Formula:
```python
# Torso Inclination (relative to horizontal)
torso_vector = shoulder_mid - hip_mid
torso_angle = arctan2(-dy, dx) × (180/π)

# Hip Flexion Angle (shoulder-hip-knee)
hip_flexion = angle_at_hip(shoulder, hip, knee)

# State Transitions
if torso_angle <= 20°:
    if state == "descending":
        counter += 1
        # Calculate momentum score
        momentum = calculate_momentum()
    state = "rest"
    
elif torso_angle >= 70° or hip_flexion <= 50°:
    if state in ["rest", "ascending"]:
        # Peak reached
        record_peak_metrics()
    state = "peak"
    
else:
    if state == "rest":
        state = "ascending"
    elif state == "peak":
        state = "descending"
```

#### Range of Motion Scoring:
```python
# ROM Categories
FULL_ROM = 70°       # Full sit-up
GOOD_CRUNCH = 50°    # Crunch with hip flexion
PARTIAL = 30°        # Partial range

# ROM Score
if torso_angle >= 70° and hip_flexion <= 50°:
    rom_quality = "PERFECT"
    rom_score = 100
elif torso_angle >= 70°:
    rom_quality = "GOOD"
    rom_score = 85
elif torso_angle >= 50°:
    rom_quality = "MODERATE"
    rom_score = 60
else:
    rom_quality = "INSUFFICIENT"
    rom_score = 30
```

#### Form Violations:
```python
# Foot Lift Detection
ankle_velocity_y = (ankle_y_t - ankle_y_t-1) / time_delta
if abs(ankle_velocity_y) > LIFT_THRESHOLD:
    foot_lifted = True
    violation_penalty = -25

# Neck Strain Detection
nose_shoulder_dist = distance(nose, shoulder_mid)
if nose_shoulder_dist < NECK_STRAIN_THRESHOLD:
    neck_strain = True
    violation_penalty = -15

# Momentum Detection
shoulder_velocity = velocity(shoulder_mid)
if shoulder_velocity > MOMENTUM_THRESHOLD:
    using_momentum = True
    quality_penalty = -10
```

#### Momentum Score Calculation:
```python
# Momentum Score (0-100, lower is better control)
shoulder_velocities = [v1, v2, ..., vn]
avg_velocity = mean(shoulder_velocities)
max_velocity = max(shoulder_velocities)

# Smooth = low momentum, jerky = high momentum
momentum_score = 100 - (max_velocity / (avg_velocity + 1)) × 10
momentum_score = clip(momentum_score, 0, 100)

# Interpretation:
# 90-100: Controlled, minimal momentum
# 70-89: Moderate control
# 50-69: Using significant momentum
# <50: Excessive momentum (poor form)
```

#### Overall Situp Score:
```python
situp_score = (
    rom_score × 0.40 +
    form_score × 0.35 +
    tempo_score × 0.15 +
    (100 - momentum_score) × 0.10
) - violation_penalties
```

---

### 4. SIT-AND-REACH (Flexibility Test)

#### Measurement Formula:
```python
# Reach Distance (in pixels)
avg_wrist_x = (left_wrist.x + right_wrist.x) / 2
avg_ankle_x = (left_ankle.x + right_ankle.x) / 2
reach_distance = avg_wrist_x - avg_ankle_x

# Normalized Reach (relative to arm length)
arm_length = distance(shoulder, wrist)
normalized_reach = reach_distance / arm_length

# Reach Score = reach_distance / arm_length × 100
```

#### Flexibility Score:
```python
# Hip Angle Scoring
hip_angle = angle(shoulder, hip, knee)

if hip_angle < 60°:
    flexibility = "EXCELLENT"
    hip_score = 100
elif hip_angle < 80°:
    flexibility = "GOOD"
    hip_score = 80
elif hip_angle < 100°:
    flexibility = "AVERAGE"
    hip_score = 60
else:
    flexibility = "POOR"
    hip_score = 30
```

#### Validity Checks:
```python
# Knee Straightness
knee_angle = angle(hip, knee, ankle)
if knee_angle < 165°:
    valid = False
    feedback = "Straighten legs!"
    validity_penalty = -50

# Symmetry Check
wrist_difference = abs(left_wrist.x - right_wrist.x)
if wrist_difference > 50:  # pixels
    symmetry_score = 100 - (wrist_difference / 2)
else:
    symmetry_score = 100
```

#### Back Alignment Score:
```python
# Back Angle (shoulder-hip-knee)
back_angle = angle(shoulder, hip, knee)

# Ideal: Straight back = high angle (>140°)
if back_angle > 140°:
    back_score = 100
elif back_angle > 120°:
    back_score = 80
else:
    back_score = 50
```

#### Overall Flexibility Score:
```python
flexibility_score = (
    normalized_reach × 0.40 +
    hip_score × 0.30 +
    symmetry_score × 0.15 +
    back_score × 0.15
) × validity_multiplier

# validity_multiplier = 1.0 if valid else 0.5
```

---

### 5. SKIPPING (JUMP ROPE)

#### Jump Detection Formula:
```python
# Ankle Tracking
avg_ankle_y = (left_ankle.y + right_ankle.y) / 2

# Vertical Velocity
ankle_velocity_y = (ankle_y_t - ankle_y_t-1) / time_delta

# Jump Detection
JUMP_THRESHOLD = -30  # pixels/frame (negative = upward)
if ankle_velocity_y < JUMP_THRESHOLD and state == "ground":
    state = "air"
    takeoff_time = current_time
    
elif ankle_velocity_y > -5 and state == "air":
    state = "ground"
    landing_time = current_time
    flight_time = landing_time - takeoff_time
    jump_count += 1
```

#### Jump Height Calculation:
```python
# Height = Distance traveled in air
jump_height = ground_y - min_ankle_y_during_flight

# Physics-based height (if FPS known)
# h = 0.5 × g × (flight_time/2)²
# where g = 9.81 m/s²
physics_height = 0.5 × 9.81 × (flight_time / 2)²
```

#### Frequency Calculation:
```python
# Skips per Second
time_elapsed = current_time - start_time
skip_frequency = jump_count / time_elapsed

# Categorization
if frequency > 3.0:
    rating = "EXCELLENT"  # 180+ skips/min
elif frequency > 2.5:
    rating = "GOOD"       # 150+ skips/min
elif frequency > 2.0:
    rating = "AVERAGE"    # 120+ skips/min
else:
    rating = "SLOW"       # <120 skips/min
```

#### Form Analysis:
```python
# Back Posture (should stay upright)
back_angle = angle(shoulder, hip, knee)
posture_score = 100 - abs(180 - back_angle)

# Knee Bend (should be minimal)
knee_angle = angle(hip, knee, ankle)
if knee_angle > 150°:
    knee_score = 100  # Good, minimal bend
else:
    knee_score = (knee_angle / 150) × 100

# Consistency (flight time variance)
flight_time_std = std_dev(flight_times)
consistency_score = 100 - (flight_time_std × 10)
```

#### Overall Skipping Score:
```python
skipping_score = (
    frequency_score × 0.40 +
    height_score × 0.25 +
    posture_score × 0.20 +
    consistency_score × 0.15
)
```

---

### 6. JUMPING JACKS

#### State Machine:
```
CLOSED (arms down, legs together) ↔ OPEN (arms up, legs spread) = 1 REP
```

#### Arm Detection Formula:
```python
# Arm Angle at Shoulder (hip-shoulder-wrist)
arm_angle = angle(hip, shoulder, wrist)

# State Detection
ARM_OPEN_THRESHOLD = 120°   # Arms raised
ARM_CLOSED_THRESHOLD = 60°  # Arms down

if arm_angle > ARM_OPEN_THRESHOLD and state == "closed":
    state = "open"
elif arm_angle < ARM_CLOSED_THRESHOLD and state == "open":
    state = "closed"
    rep_count += 1
```

#### Arm Spread Calculation:
```python
# Horizontal Distance Between Wrists
arm_spread = abs(left_wrist.x - right_wrist.x)

# Spread Score
if arm_spread > 180:  # pixels (wide spread)
    arm_spread_score = 100
elif arm_spread > 140:
    arm_spread_score = 80
else:
    arm_spread_score = 50
```

#### Leg Spread Calculation:
```python
# Distance Between Ankles
leg_spread = abs(left_ankle.x - right_ankle.x)

# Synchronization with Arms
if arm_angle > 120° and leg_spread > 120:
    sync_score = 100  # Good synchronization
elif arm_angle < 60° and leg_spread < 100:
    sync_score = 100  # Good synchronization
else:
    sync_score = 50   # Poor synchronization
```

#### Symmetry Analysis:
```python
# Left vs Right Arm Height
left_arm_height = shoulder.y - left_wrist.y
right_arm_height = shoulder.y - right_wrist.y
arm_symmetry_error = abs(left_arm_height - right_arm_height)

symmetry_score = 100 - (arm_symmetry_error / 2)
```

#### Tempo Tracking:
```python
# Open/Close Cycle Time
cycle_times = [t1, t2, ..., tn]
avg_cycle_time = mean(cycle_times)
tempo_consistency = 100 - (std_dev(cycle_times) × 20)

# Ideal tempo: ~1 rep per second
if 0.8 < avg_cycle_time < 1.2:
    tempo_score = 100
else:
    tempo_score = 100 - abs(1.0 - avg_cycle_time) × 30
```

#### Overall Jumping Jacks Score:
```python
jj_score = (
    arm_spread_score × 0.25 +
    leg_spread_score × 0.20 +
    sync_score × 0.25 +
    symmetry_score × 0.15 +
    tempo_score × 0.15
)
```

---

### 7. VERTICAL JUMP

#### State Machine:
```
STANDING → PREPARING (squat) → AIRBORNE → LANDING → STANDING
```

#### Jump Detection Formula:
```python
# Ankle Tracking for Jump Detection
avg_ankle_y = (left_ankle.y + right_ankle.y) / 2

# State Transitions
if state == "standing":
    if avg_ankle_y > baseline_y + 20:
        state = "preparing"
        squat_start_y = avg_ankle_y
        
elif state == "preparing":
    knee_angle = angle(hip, knee, ankle)
    countermovement_depth = knee_angle
    if avg_ankle_y < baseline_y - 10:
        state = "airborne"
        takeoff_time = current_time
        min_ankle_y = avg_ankle_y
        
elif state == "airborne":
    min_ankle_y = min(min_ankle_y, avg_ankle_y)
    if avg_ankle_y > baseline_y - 10:
        state = "landing"
        landing_time = current_time
        flight_time = landing_time - takeoff_time
        
elif state == "landing":
    if avg_ankle_y >= baseline_y:
        state = "standing"
        # Record jump metrics
        jump_height = baseline_y - min_ankle_y
        jump_count += 1
```

#### Jump Height Formula:
```python
# Pixel-based Height
jump_height_px = ground_y - min_ankle_y_in_air

# Physics-based Height (from flight time)
# h = ½gt² where t = flight_time/2
g = 9.81  # m/s²
jump_height_m = 0.5 × g × (flight_time / 2)²

# Conversion (if pixel_to_meter ratio known)
jump_height_cm = jump_height_m × 100
```

#### Countermovement Analysis:
```python
# Squat Depth Before Jump
countermovement_angle = knee_angle_at_lowest_point

# Optimal Range: 90-120°
if 90 <= countermovement_angle <= 120:
    countermovement_score = 100
elif 80 <= countermovement_angle < 90:
    countermovement_score = 90  # Too deep
elif 120 < countermovement_angle <= 130:
    countermovement_score = 85  # Shallow
else:
    countermovement_score = 60  # Suboptimal
```

#### Arm Swing Analysis:
```python
# Arm Swing Contribution
arm_swing_angle = angle(shoulder, elbow, wrist)

# Full arm swing: ~180° (straight overhead)
if arm_swing_angle > 160°:
    arm_contribution = 100
else:
    arm_contribution = (arm_swing_angle / 160) × 100
```

#### Landing Mechanics:
```python
# Landing Knee Angle (shock absorption)
landing_knee = knee_angle_at_landing

# Optimal: 90-120° (good flexion)
if 90 <= landing_knee <= 120:
    landing_score = 100
    feedback = "Good landing!"
elif landing_knee < 90:
    landing_score = 70
    feedback = "Too much bend!"
else:
    landing_score = 60
    feedback = "Land softer!"
```

#### Overall Vertical Jump Score:
```python
vjump_score = (
    jump_height_score × 0.40 +
    countermovement_score × 0.25 +
    arm_swing_score × 0.20 +
    landing_score × 0.15
)

# Height Score Calculation
max_height = max(all_jump_heights)
height_score = min(100, (max_height / expected_height) × 100)
```

---

### 8. BROAD JUMP (HORIZONTAL JUMP)

#### State Machine:
```
STANDING → AIRBORNE → LANDING → STANDING
```

#### Distance Calculation:
```python
# Horizontal Distance Tracking
start_x = avg_ankle_x_at_takeoff
end_x = avg_ankle_x_at_landing
jump_distance = abs(end_x - start_x)

# State Machine
if state == "standing":
    if ankle_y < ground_y - 30:
        state = "airborne"
        start_x = avg_ankle_x
        min_y_during_flight = ankle_y
        
elif state == "airborne":
    max_x = max(max_x, avg_ankle_x)
    if ankle_y > ground_y - 20:
        state = "landing"
        end_x = avg_ankle_x
        
elif state == "landing":
    if ankle_y >= ground_y:
        state = "standing"
        jump_distance = abs(end_x - start_x)
        jump_count += 1
```

#### Takeoff Angle:
```python
# Takeoff Angle (optimal: 45°)
takeoff_velocity_x = (ankle_x_t - ankle_x_t-1) / dt
takeoff_velocity_y = (ankle_y_t - ankle_y_t-1) / dt
takeoff_angle = arctan(velocity_y / velocity_x) × (180/π)

# Angle Score
if 40° <= takeoff_angle <= 50°:
    angle_score = 100
else:
    angle_score = 100 - abs(45 - takeoff_angle) × 2
```

#### Countermovement (Same as Vertical Jump):
```python
countermovement_angle = knee_angle_before_jump

if 90 <= countermovement_angle <= 120:
    cm_score = 100
else:
    cm_score = 100 - abs(105 - countermovement_angle)
```

#### Landing Distance Score:
```python
# Distance Scoring
max_distance = max(all_jump_distances)
avg_distance = mean(all_jump_distances)

distance_score = (max_distance / expected_distance) × 100
consistency = 100 - (std_dev(jump_distances) / avg_distance) × 100
```

#### Overall Broad Jump Score:
```python
bjump_score = (
    distance_score × 0.45 +
    countermovement_score × 0.25 +
    angle_score × 0.20 +
    consistency_score × 0.10
)
```

---

## Thresholds & Scoring

### Global Thresholds Dictionary:
```python
thresholds = {
    'pushup': {
        'down': 90,           # Elbow bent (chest to ground)
        'up': 140,            # Elbow extended (top position)
        'form_hip_min': 130   # Minimum hip angle (body alignment)
    },
    'squat': {
        'down': 135,          # Squatting position (thigh parallel)
        'up': 155,            # Standing position
        'deep': 90            # Deep squat threshold
    },
    'situp': {
        'up': 70,             # Peak torso angle
        'down': 20,           # Rest position
        'good_crunch': 50     # Hip flexion for crunch
    },
    'sitnreach': {
        'excellent_hip': 60,  # Excellent flexibility
        'average_hip': 80,    # Average flexibility
        'knee_valid': 165     # Knee straightness validity
    },
    'skipping': {
        'jump_threshold': 30, # Vertical velocity threshold
        'min_height': 20      # Minimum jump height
    },
    'jumpingjacks': {
        'arm_open': 150,      # Arms raised threshold
        'leg_open': 150       # Legs spread threshold
    },
    'vjump': {
        'min_height': 30,              # Minimum valid jump
        'good_countermovement': 110    # Optimal squat depth
    },
    'bjump': {
        'min_distance': 50,            # Minimum horizontal distance
        'good_countermovement': 110    # Optimal squat depth
    }
}
```

---

## Important Functions

### 1. **_smooth_knee_angle(knee_angle)**
Applies 3-frame moving average to reduce noise in knee angle measurements.

**Formula:**
```python
smoothed_angle = (angle_t-2 + angle_t-1 + angle_t) / 3
```

**Purpose:**
- Eliminates jitter from pose detection
- Prevents false state transitions
- Improves counting accuracy

**Example:**
```python
Raw angles:     [145, 148, 152, 149, 151]
Smoothed:       [145, 148, 150, 151, 151]
```

---

### 2. **_smooth_elbow_angle(elbow_angle)**
Same as knee smoothing but for elbow tracking.

**Usage:**
```python
raw_elbow = 135
smoothed_elbow = self._smooth_elbow_angle(raw_elbow)
```

---

### 3. **_draw_dashboard(frame, exercise_name)**
Renders real-time statistics overlay on video frame.

**Dashboard Components:**

#### Standard Dashboard (Pushups, Squats, Situps):
```
┌─────────────────────────┐
│ REPS: 12  STAGE: DOWN   │
│ Good Form               │
└─────────────────────────┘
```

#### Sit-and-Reach Dashboard:
```
┌─────────────┬─────────────┐
│ MAX REACH   │ CURRENT     │
│    245 px   │   240 px    │
└─────────────┴─────────────┘
│ VALID - Good flexibility  │
└───────────────────────────┘
```

#### Skipping Dashboard:
```
┌───────────┬────────────┐
│ JUMPS: 45 │ FREQ: 2.5  │
└───────────┴────────────┘
│ STATE: AIR             │
└────────────────────────┘
```

#### Jumping Jacks Dashboard:
```
┌───────────┬────────────┐
│ REPS: 20  │ STATE: OPEN│
└───────────┴────────────┘
│ Angle:145° ✓ Spread:220px ✓│
└────────────────────────┘
```

**Drawing Code:**
```python
# Box rendering
cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

# Text overlay
cv2.putText(frame, text, (x, y), font, scale, color, thickness)
```

---

### 4. **process_pushup(angles, keypoints)**
Executes pushup detection logic with form validation.

**Algorithm:**
```
1. Extract elbow and hip angles from both sides
2. Apply smoothing to elbow angle
3. Check hip alignment for form validation
4. Update metrics with all joint angles
5. State machine:
   - If elbow > 140° and stage=="DOWN": count rep, stage="UP"
   - If elbow < 90°: stage="DOWN"
6. Provide real-time feedback
```

**Keypoint Usage:**
- Left/Right Elbow (7, 8)
- Left/Right Hip (11, 12)
- Left/Right Shoulder (5, 6)
- Left/Right Knee (13, 14)
- Left/Right Ankle (15, 16)
- Left/Right Wrist (9, 10)

---

### 5. **process_squat(angles, keypoints)**
Executes squat detection with comprehensive biomechanical tracking.

**Algorithm:**
```
1. Extract knee angle (use most visible side)
2. Apply 3-frame smoothing
3. Get torso angle and shin angle
4. Update squat-specific metrics
5. Phase tracking:
   - STANDING (knee > 155°): Start eccentric timer
   - DESCENDING: Track velocity for sticking points
   - BOTTOM (knee < 135°): Record depth, count rep
   - ASCENDING: Calculate concentric time
6. Form evaluation (depth, torso lean, shin angle)
```

**Velocity Tracking:**
```python
angular_velocity = (knee_angle_t - knee_angle_t-1) / time_delta

if angular_velocity < min_velocity:
    sticking_point = current_knee_angle
    # Indicates weak range of motion
```

---

### 6. **process_situp(angles, keypoints)**
Situp detection with momentum and form violation tracking.

**Algorithm:**
```
1. Extract torso inclination and hip flexion
2. Update situp metrics (shoulder tracking, foot lift)
3. Detect violations:
   - Foot lift (ankle velocity > threshold)
   - Neck strain (nose-shoulder distance)
   - Momentum (shoulder velocity spikes)
4. State machine:
   - REST (torso < 20°): Ready for next rep
   - ASCENDING: Moving up
   - PEAK (torso > 70° or hip flexion < 50°): Count rep
   - DESCENDING: Controlled return
5. Calculate momentum score
```

**Momentum Calculation:**
```python
# Track shoulder position changes
shoulder_velocities = []
for frame in rep:
    velocity = distance(shoulder_t, shoulder_t-1) / dt
    shoulder_velocities.append(velocity)

# High variance = jerky motion (momentum)
momentum_score = 100 - (std_dev(velocities) × 10)
```

---

### 7. **process_sitnreach(angles, keypoints)**
Flexibility test with validity checks.

**Algorithm:**
```
1. Calculate reach distance (wrist - ankle)
2. Normalize by arm length
3. Track hip angle, back angle, knee angle
4. Validity checks:
   - Knees straight (angle > 165°)
   - Symmetry (wrist positions balanced)
5. Update max reach continuously
6. Real-time feedback on form
```

---

### 8. **process_skipping(angles, keypoints)**
Jump rope counting with frequency analysis.

**Algorithm:**
```
1. Track ankle vertical position
2. Calculate vertical velocity
3. Jump detection:
   - Takeoff: velocity < -30 px/frame
   - Landing: return to baseline
4. Measure flight time and jump height
5. Calculate skip frequency (jumps/second)
6. Check posture (back angle, knee bend)
```

---

### 9. **process_jumpingjacks(angles, keypoints)**
Jumping jacks with arm/leg synchronization tracking.

**Algorithm:**
```
1. Calculate arm angle at shoulder (hip-shoulder-wrist)
2. Measure arm spread (wrist distance)
3. Measure leg spread (ankle distance)
4. State detection:
   - OPEN: arm_angle > 120° and leg_spread > 120
   - CLOSED: arm_angle < 60° and leg_spread < 100
5. Check synchronization (arms and legs move together)
6. Track symmetry (left vs right)
```

---

### 10. **process_vjump(angles, keypoints)**
Vertical jump with countermovement and landing analysis.

**Algorithm:**
```
1. Track ankle Y position for height
2. State machine:
   - STANDING: baseline position
   - PREPARING: squat phase (track depth)
   - AIRBORNE: measure flight time and max height
   - LANDING: check landing mechanics
3. Measure countermovement depth (knee angle)
4. Track arm swing contribution
5. Evaluate landing knee flexion
```

---

### 11. **process_bjump(angles, keypoints)**
Broad jump with horizontal distance measurement.

**Algorithm:**
```
1. Track ankle X position (horizontal)
2. Track ankle Y position (vertical for airborne detection)
3. State machine:
   - STANDING: record start position
   - AIRBORNE: track max X position
   - LANDING: calculate distance
4. Measure countermovement
5. Calculate takeoff angle (optimal: 45°)
```

---

### 12. **_resize_for_display(frame, max_width=1280, max_height=720)**
Resizes frame for optimal display while maintaining aspect ratio.

**Formula:**
```python
scale = min(max_width / width, max_height / height)
new_width = int(width × scale)
new_height = int(height × scale)
```

---

### 13. **run(exercise_type, source, save_output)**
Main execution loop that orchestrates the entire analysis.

**Pipeline:**
```
1. Open video source (webcam or file)
2. Initialize video writer (if saving)
3. For each frame:
   a. Detect pose (calibrator.process_frame)
   b. Calculate angles
   c. Run exercise-specific logic
   d. Draw dashboard overlay
   e. Display frame
   f. Save frame (if enabled)
4. Generate performance report
5. Save metrics to JSON file
```

---

## Visual Feedback System

### Color Coding:
```python
colors = {
    'good_form': (0, 255, 0),      # Green
    'bad_form': (0, 0, 255),        # Red
    'warning': (0, 165, 255),       # Orange
    'neutral': (200, 200, 200),     # Gray
    'excellent': (0, 255, 255)      # Yellow
}
```

### Feedback Messages:

#### Pushups:
- "Good Form" - Hip angle > 130°
- "Fix Back!" - Body sagging
- "⚠ Position yourself so arms are visible"

#### Squats:
- "Great Depth!" - Knee < 90°
- "Go Lower" - Shallow squat
- "Too Much Lean!" - Torso angle > 45°
- "⚠ Move back - show full body"

#### Situps:
- "Perfect Rep!" - Full ROM + good crunch
- "Good - Crunch Tighter" - Full ROM, weak hip flexion
- "Go Higher!" - Insufficient ROM
- "Feet Lifted!" - Foot lift violation

#### Sit-and-Reach:
- "MAX REACH!" - At or near maximum
- "Straighten Legs!" - Knees bent
- "Balance Both Sides" - Asymmetric reach
- "Excellent Flex!" - Hip angle < 60°

#### Skipping:
- "IN AIR" - Airborne state
- "Stand Upright!" - Poor posture
- "Excellent Speed!" - Frequency > 3/sec

#### Jumping Jacks:
- "STATE: OPEN" / "STATE: CLOSED"
- "↑ RAISE ARMS!" - Arms not raised enough
- "↓ LOWER ARMS!" - Arms not down
- "→← FEET TOGETHER!" - Legs not together

#### Vertical/Broad Jump:
- "Ready to jump" - Standing state
- "Swing arms up!" - Preparing phase
- "In the air!" - Airborne
- "Good landing!" - Proper knee flexion

---

## Usage Examples

### Example 1: Analyze Video File
```python
trainer = ExerciseEvaluator()
trainer.run(
    exercise_type='squat',
    source='workout_video.mp4',
    save_output=True
)
```

### Example 2: Live Webcam Analysis
```python
trainer = ExerciseEvaluator()
trainer.run(
    exercise_type='pushup',
    source='0',  # Webcam
    save_output=False
)
```

### Example 3: Custom Thresholds
```python
trainer = ExerciseEvaluator()
# Modify thresholds for specific needs
trainer.thresholds['squat']['deep'] = 80  # Require deeper squats
trainer.run(exercise_type='squat', source='0', save_output=False)
```

### Example 4: CLI Interactive Mode
```bash
python test.py
# Menu appears:
# Select Exercise: 2 (Squats)
# Select Source: 2 (Video File)
# Enter path: my_squat_video.mp4
# Save output: y
```

---

## Performance Metrics Output

### JSON Structure:
```json
{
  "exercise": "squat",
  "timestamp": "2025-12-06 15:30:00",
  "reps": {
    "total": 15,
    "good_form": 12,
    "bad_form": 3
  },
  "depth_analysis": {
    "max_depth": 85,
    "avg_depth": 95,
    "depths": [90, 95, 85, ...]
  },
  "tempo": {
    "avg_eccentric": 2.1,
    "avg_concentric": 1.8,
    "eccentric_times": [2.0, 2.2, 2.1, ...],
    "concentric_times": [1.8, 1.9, 1.7, ...]
  },
  "form_violations": {
    "torso_lean_count": 2,
    "knee_cave_count": 1
  },
  "overall_score": 87.5
}
```

---

## Technical Implementation Details

### Frame Processing Pipeline:
```
Input Frame (BGR)
    ↓
YOLO11 Pose Detection (utils.py)
    ↓
17 Keypoints + Confidence Scores
    ↓
Angle Calculation (utils.py)
    ↓
30+ Biomechanical Angles
    ↓
Smoothing (3-frame moving average)
    ↓
Exercise-Specific Logic (test.py)
    ↓
State Machine Updates
    ↓
Metrics Recording (metrics.py)
    ↓
Dashboard Overlay
    ↓
Output Frame with Visualizations
```

### Real-time Processing:
- **FPS**: 15-30 depending on hardware
- **Latency**: <50ms per frame
- **Memory**: ~500MB for model + video buffer

---

## Error Handling

### Robust Detection:
```python
# Keypoint validation
if keypoints is None or len(keypoints) < 17:
    feedback = "Body not detected"
    return

# Angle validation
if angle is None:
    feedback = "Position not visible"
    return

# Side selection (use most visible side)
if left_elbow and left_hip:
    use_left = True
elif right_elbow and right_hip:
    use_right = True
else:
    feedback = "Move to show arms"
```

---

## Summary

`test.py` is a comprehensive exercise analysis application that:

✅ **Tracks 8 Different Exercises** with specialized logic for each
✅ **Uses State Machines** for accurate rep counting
✅ **Applies Biomechanical Formulas** for form evaluation
✅ **Provides Real-time Feedback** with visual overlays
✅ **Calculates Performance Scores** using weighted metrics
✅ **Detects Form Violations** (momentum, foot lifts, neck strain)
✅ **Tracks Tempo & Velocity** for advanced analysis
✅ **Outputs Detailed Reports** in JSON format
✅ **Supports Live & Recorded Video** with optional saving
✅ **Handles Edge Cases** with robust error checking

**Key Innovation**: Integration of computer vision (YOLO11 pose), biomechanics (joint angles), and exercise science (rep counting, form validation) into a single unified system.
