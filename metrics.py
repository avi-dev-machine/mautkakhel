#Test wise function containing individual metrics 

import numpy as np
from collections import deque
import statistics

class PerformanceMetrics:
    """
    Comprehensive performance metrics.
    Separates reporting logic for Pushups and Squats.
    """
    
    def __init__(self):
        self.exercise = None
        self.good_reps = 0
        self.bad_reps = 0
        self.bad_form_count = 0
        
        # Angle tracking per rep
        self.rep_angles = []  # List of (max_angle, min_angle) tuples per rep
        self.back_angles = deque(maxlen=100)
        self.elbow_angles = deque(maxlen=100)
        
        self.rep_durations = []  # Time in seconds for each rep
        
        # Squat-specific tracking
        self.squat_depths = []  # Min knee angle per rep
        self.torso_angles = deque(maxlen=100)  # Back inclination angles
        self.shin_angles = deque(maxlen=100)  # Shin angles relative to vertical
        self.knee_positions = deque(maxlen=100)  # For knee stability (valgus/varus)
        self.hip_velocities = []  # Velocity during concentric phase
        self.eccentric_times = []  # Descent time per rep
        self.concentric_times = []  # Ascent time per rep
        self.sticking_points = []  # Knee angle at minimum velocity
        self.rep_start_time = None  # type: float | None
        self.rep_bottom_time = None  # type: float | None
        self.last_hip_y = None  # type: float | None
        self.last_frame_time = None  # type: float | None
        self.current_phase = 'standing'  # standing, descending, bottom, ascending
        self.min_velocity_angle = None  # type: int | None
        self.min_velocity = float('inf')
        
        # Situp-specific tracking
        self.situp_torso_inclinations = []  # Peak torso angles per rep
        self.situp_hip_flexions = []  # Minimum hip flexion per rep
        self.situp_foot_lifts = []  # Foot lift violations per rep
        self.situp_neck_strains = []  # Neck strain detection per rep
        self.situp_concentric_times = []  # Up phase time
        self.situp_eccentric_times = []  # Down phase time
        self.situp_momentum_scores = []  # Jerk/momentum per rep
        self.situp_short_rom_count = 0  # Reps with incomplete ROM
        self.situp_valid_reps = 0  # Strict reps only
        self.ankle_baseline_y = None  # type: float | None
        self.knee_baseline_y = None  # type: float | None
        self.shoulder_positions = deque(maxlen=10)  # For acceleration calc
        self.situp_rep_start_time = None  # type: float | None
        self.situp_peak_time = None  # type: float | None
        self.situp_state = 'rest'  # rest, ascending, peak, descending
        self.max_torso_inclination = 0
        self.min_hip_flexion = 180
        
        # Sit-and-Reach specific tracking
        self.reach_distances = []  # Forward reach per frame (wrist_x - ankle_x)
        self.arm_lengths = []  # Arm length measurements
        self.hip_flexion_angles = []  # Hip angles during reach
        self.back_alignment_angles = []  # Back/spine alignment
        self.knee_extension_angles = []  # Knee angles for validity
        self.reach_symmetry_errors = []  # Left vs right wrist difference
        self.hip_y_positions = deque(maxlen=50)  # For stability variance
        self.max_reach_distance = 0  # Maximum reach achieved
        self.max_reach_frame_data = None  # Keypoints at max reach
        self.valid_frames = 0  # Frames with valid posture
        self.total_frames = 0  # Total processed frames
        self.sitnreach_start_time = None  # type: float | None
        self.sitnreach_test_duration = 0  # Total test time
        
        # Skipping specific tracking
        self.ankle_y_positions = deque(maxlen=200)  # Ankle Y positions for jump detection
        self.ground_y = None  # Ground reference level
        self.jump_state = 'ground'  # 'ground' or 'air'
        self.jump_start_time = None  # type: float | None
        self.jump_durations = []  # Duration of each jump
        self.jump_heights = []  # Height of each jump
        self.jump_count = 0  # Total valid jumps
        self.double_bounce_count = 0  # Double bounce errors
        self.back_angles_skip = deque(maxlen=100)  # Back posture during skipping
        self.knee_bend_angles = deque(maxlen=100)  # Knee angles during jumps
        self.wrist_positions = deque(maxlen=10)  # For arm movement tracking
        self.frames_processed_skip = 0  # Total frames
        self.skip_start_time = None  # type: float | None
        self.last_land_time = None  # type: float | None
        self.min_ankle_y_current_jump = None  # type: float | None
        self.correct_jumps = 0  # Jumps meeting quality criteria
        
        # Jumping jacks specific tracking
        self.jj_state = 'closed'  # 'closed' or 'open'
        self.jj_rep_count = 0  # Total reps
        self.jj_correct_reps = 0  # Reps meeting quality criteria
        self.jj_arm_spreads = deque(maxlen=200)  # Arm spread distances
        self.jj_leg_spreads = deque(maxlen=200)  # Leg spread distances
        self.jj_arm_angles = deque(maxlen=200)  # Arm elevation angles
        self.jj_back_angles = deque(maxlen=100)  # Back posture angles
        self.jj_rep_durations = []  # Duration per rep
        self.jj_rep_start_time = None  # type: float | None
        self.jj_arm_open_time = None  # type: float | None
        self.jj_leg_open_time = None  # type: float | None
        self.jj_coordination_errors = []  # Arm-leg sync errors
        self.jj_max_arm_spread = 0  # Maximum arm spread in current rep
        self.jj_max_leg_spread = 0  # Maximum leg spread in current rep
        self.jj_max_arm_angle = 0  # Maximum arm angle in current rep
        self.jj_start_time = None  # type: float | None
        
        # Vertical jump specific tracking
        self.vjump_ankle_y_positions = deque(maxlen=200)  # Ankle Y positions for jump detection
        self.vjump_ground_y = None  # Ground reference level
        self.vjump_state = 'standing'  # 'standing', 'preparing', 'airborne', 'landing'
        self.vjump_jump_count = 0  # Total jumps
        self.vjump_jump_heights = []  # Height of each jump
        self.vjump_countermovement_depths = []  # Min knee angle before takeoff
        self.vjump_arm_swing_angles = []  # Arm swing angles before takeoff
        self.vjump_takeoff_symmetry_errors = []  # Left vs right ankle takeoff timing
        self.vjump_landing_knee_angles = []  # Knee angles at landing
        self.vjump_valid_jumps = 0  # Jumps meeting quality criteria
        self.vjump_prep_start_time = None  # type: float | None
        self.vjump_takeoff_time = None  # type: float | None
        self.vjump_min_ankle_y = None  # type: float | None
        self.vjump_min_knee_angle = 180  # Minimum knee angle during countermovement
        self.vjump_max_arm_angle = 0  # Maximum arm swing angle
        self.vjump_left_ankle_takeoff_time = None  # type: float | None
        self.vjump_right_ankle_takeoff_time = None  # type: float | None
        self.vjump_start_time = None  # type: float | None
        
        # Broad jump specific tracking
        self.bjump_ankle_x_positions = deque(maxlen=200)  # Ankle X positions for horizontal tracking
        self.bjump_ankle_y_positions = deque(maxlen=200)  # Ankle Y positions for vertical tracking
        self.bjump_start_x = None  # Starting position X coordinate
        self.bjump_ground_y = None  # Ground reference Y coordinate
        self.bjump_state = 'standing'  # 'standing', 'airborne', 'landing'
        self.bjump_jump_count = 0  # Total jumps
        self.bjump_jump_distances = []  # Horizontal distance of each jump
        self.bjump_countermovement_depths = []  # Min knee angle before takeoff
        self.bjump_arm_swing_angles = []  # Arm swing angles before takeoff
        self.bjump_takeoff_symmetry_errors = []  # Left vs right ankle takeoff timing
        self.bjump_landing_stability_errors = []  # Left vs right ankle landing position difference
        self.bjump_valid_jumps = 0  # Jumps meeting quality criteria
        self.bjump_takeoff_time = None  # type: float | None
        self.bjump_takeoff_x = None  # type: float | None
        self.bjump_max_x = None  # type: float | None
        self.bjump_min_knee_angle = 180  # Minimum knee angle during countermovement
        self.bjump_max_arm_angle = 0  # Maximum arm swing angle
        self.bjump_left_ankle_takeoff_time = None  # type: float | None
        self.bjump_right_ankle_takeoff_time = None  # type: float | None
        self.bjump_start_time = None  # type: float | None
        self.bjump_max_distance = 0  # Maximum jump distance achieved
        
        # Thresholds
        self.thresholds = {
            'min_hip_angle': 150,
            'ideal_back_angle': 180,
            'max_back_deviation': 45,
            'ideal_rom': 90,
            'max_arm_asymmetry': 30,
            'squat_parallel': 90,
            'ideal_torso_angle': 35,
            'max_knee_deviation': 50,  # pixels
            # Situp thresholds
            'situp_up_angle': 70,  # Minimum torso inclination for "up"
            'situp_down_angle': 20,  # Maximum angle for "down"/reset
            'situp_good_hip_flexion': 50,  # Good crunch angle
            'situp_foot_lift_threshold': 30,  # pixels
            'situp_momentum_threshold': 50,  # Jerk score
            # Sit-and-Reach thresholds
            'sitnreach_excellent_hip': 60,  # Hip angle for excellent flexibility
            'sitnreach_average_hip': 80,  # Hip angle for average flexibility
            'sitnreach_knee_valid': 165,  # Minimum knee angle for valid test
            'sitnreach_max_symmetry_error': 50,  # pixels
            'sitnreach_max_hip_variance': 100,  # pixels squared
            # Skipping thresholds
            'skip_jump_threshold': 30,  # Pixels above ground to detect jump
            'skip_min_jump_height': 20,  # Minimum jump height for valid skip
            'skip_max_knee_bend': 120,  # Maximum knee bend for efficiency
            'skip_ideal_back_angle': 180,  # Ideal upright posture
            'skip_max_back_deviation': 30,  # Max back angle deviation
            # Jumping jacks thresholds
            'jj_arm_open_threshold': 180,  # Min arm spread for open position (pixels)
            'jj_arm_close_threshold': 120,  # Max arm spread for closed position
            'jj_leg_open_threshold': 120,  # Min leg spread for open position
            'jj_leg_close_threshold': 100,  # Max leg spread for closed position
            'jj_arm_angle_open': 135,  # Min shoulder angle for open (arms raised, ~135-180°)
            'jj_arm_angle_close': 45,  # Max shoulder angle for closed (arms down, ~0-45°)
            'jj_ideal_back_angle': 180,  # Ideal upright posture
            'jj_max_coordination_error': 0.3,  # Max time difference for sync (seconds)
            # Vertical jump thresholds
            'vjump_min_height': 30,  # Minimum jump height in pixels
            'vjump_good_countermovement': 110,  # Good knee bend angle
            'vjump_good_arm_swing': 140,  # Good arm swing angle (upward)
            'vjump_max_symmetry_error': 0.1,  # Maximum takeoff timing difference (seconds)
            'vjump_good_landing_knee': 130,  # Good landing knee flexion angle
            'bjump_min_distance': 30,  # Minimum horizontal jump distance in pixels
            'bjump_good_countermovement': 110,  # Good knee bend angle
            'bjump_good_arm_swing': 140,  # Good arm swing angle
            'bjump_max_symmetry_error': 0.1,  # Maximum takeoff timing difference (seconds)
            'bjump_max_landing_stability': 60,  # Maximum ankle position difference at landing (pixels)
        }
    
    def update_angle_data(self, left_elbow, right_elbow, left_hip, right_hip, 
                      left_shoulder, right_shoulder, left_knee, right_knee,
                      left_ankle, right_ankle, left_wrist, right_wrist):
        """Update tracking data with current frame angles."""
        if left_elbow is not None and right_elbow is not None:
            self.elbow_angles.append((left_elbow, right_elbow))
        
        # Calculate back straightness (spine angle)
        if (left_hip is not None) or (right_hip is not None):
            back_angle = self._calculate_back_angle(left_shoulder, left_hip, left_knee, 
                                                  right_shoulder, right_hip, right_knee)
            if back_angle is not None:
                self.back_angles.append(back_angle)
    
    def _calculate_back_angle(self, left_shoulder, left_hip, left_knee, 
                            right_shoulder, right_hip, right_knee):
        """Calculate average back/spine angle from both sides"""
        angles = []
        if left_hip is not None:
            angles.append(left_hip)
        if right_hip is not None:
            angles.append(right_hip)
        
        if angles:
            return int(np.mean(angles))
        return None
    
    def update_squat_data(self, keypoints, angles, torso_angle, shin_angle, current_time, fps=30):
        """Update squat-specific tracking data per frame."""
        if keypoints is None:
            return
        
        # Track torso and shin angles
        if torso_angle is not None:
            self.torso_angles.append(torso_angle)
        if shin_angle is not None:
            self.shin_angles.append(shin_angle)
        
        # Track knee stability (valgus/varus detection)
        left_knee_conf = keypoints[13][2]
        right_knee_conf = keypoints[14][2]
        left_ankle_conf = keypoints[15][2]
        right_ankle_conf = keypoints[16][2]
        
        if left_knee_conf > 0.5 and left_ankle_conf > 0.5 and right_knee_conf > 0.5 and right_ankle_conf > 0.5:
            left_knee_x = keypoints[13][0]
            right_knee_x = keypoints[14][0]
            left_ankle_x = keypoints[15][0]
            right_ankle_x = keypoints[16][0]
            
            left_deviation = abs(left_knee_x - left_ankle_x)
            right_deviation = abs(right_knee_x - right_ankle_x)
            self.knee_positions.append({
                'left_dev': left_deviation,
                'right_dev': right_deviation,
                'left_x': left_knee_x,
                'right_x': right_knee_x
            })
        
        # Calculate hip velocity for concentric phase
        hip_conf = max(keypoints[11][2], keypoints[12][2])
        if hip_conf > 0.5:
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            
            if self.last_hip_y is not None and self.last_frame_time is not None:
                time_delta = current_time - self.last_frame_time
                if time_delta > 0:
                    velocity = (self.last_hip_y - hip_y) / time_delta  # Positive = moving up
                    self.hip_velocities.append(velocity)
                    
                    # Track sticking point (minimum velocity during ascent)
                    if self.current_phase == 'ascending' and velocity < self.min_velocity:
                        self.min_velocity = velocity
                        knee_angle = angles.get('left_knee') or angles.get('right_knee')
                        if knee_angle is not None:
                            self.min_velocity_angle = knee_angle
            
            self.last_hip_y = hip_y
            self.last_frame_time = current_time
    
    def _calculate_torso_inclination_score(self):
        """Calculate score based on torso angle (0-35 degrees is ideal)."""
        if not self.torso_angles:
            return 0
        avg_angle = np.mean(list(self.torso_angles))
        if avg_angle <= self.thresholds['ideal_torso_angle']:
            return 100
        else:
            deviation = avg_angle - self.thresholds['ideal_torso_angle']
            score = max(0, 100 - (deviation * 2))  # Lose 2 points per degree over ideal
            return int(score)
    
    def _calculate_knee_stability_score(self):
        """Calculate score based on knee stability (valgus/varus)."""
        if not self.knee_positions:
            return 100
        
        violations = 0
        for pos in self.knee_positions:
            if pos['left_dev'] > self.thresholds['max_knee_deviation']:
                violations += 1
            if pos['right_dev'] > self.thresholds['max_knee_deviation']:
                violations += 1
        
        violation_rate = violations / (len(self.knee_positions) * 2)
        score = int((1 - violation_rate) * 100)
        return max(0, score)
    
    def _calculate_depth_consistency(self):
        """Calculate ROM consistency based on depth variation."""
        if len(self.squat_depths) < 2:
            return 100
        std_dev = np.std(self.squat_depths)
        mean_depth = np.mean(self.squat_depths)
        if mean_depth == 0:
            return 0
        cv = std_dev / mean_depth  # Coefficient of variation
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def _calculate_tempo_score(self):
        """Calculate tempo consistency for squats."""
        if len(self.eccentric_times) < 2:
            return 100
        combined_times = self.eccentric_times + self.concentric_times
        if not combined_times:
            return 100
        mean_time = np.mean(combined_times)
        if mean_time == 0:
            return 100
        std_time = np.std(combined_times)
        cv = std_time / mean_time
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def _get_avg_concentric_velocity(self):
        """Get average velocity during concentric phase."""
        if not self.hip_velocities:
            return 0
        positive_velocities = [v for v in self.hip_velocities if v > 0]
        if not positive_velocities:
            return 0
        return np.mean(positive_velocities)
    
    # ---------------------------------------------------------
    # SITUP HELPER METHODS
    # ---------------------------------------------------------
    
    def update_situp_data(self, keypoints, angles, torso_inclination, hip_flexion, current_time):
        """Update situp-specific tracking data per frame."""
        if keypoints is None:
            return
        
        # Track peak torso inclination during rep
        if torso_inclination is not None and torso_inclination > self.max_torso_inclination:
            self.max_torso_inclination = torso_inclination
        
        # Track minimum hip flexion during rep
        if hip_flexion is not None and hip_flexion < self.min_hip_flexion:
            self.min_hip_flexion = hip_flexion
        
        # Establish baseline for foot lift detection (first 30 frames)
        knee_conf = max(keypoints[13][2], keypoints[14][2])
        ankle_conf = max(keypoints[15][2], keypoints[16][2])
        
        if knee_conf > 0.5 and ankle_conf > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
            
            if self.ankle_baseline_y is None:
                self.ankle_baseline_y = ankle_y
                self.knee_baseline_y = knee_y
        
        # Track shoulder positions for momentum detection
        shoulder_conf = max(keypoints[5][2], keypoints[6][2])
        if shoulder_conf > 0.5:
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            self.shoulder_positions.append((current_time, shoulder_y))
    
    def _detect_foot_lift(self, keypoints):
        """Detect if feet/knees lifted during rep."""
        if self.ankle_baseline_y is None or self.knee_baseline_y is None:
            return False
        
        knee_conf = max(keypoints[13][2], keypoints[14][2])
        ankle_conf = max(keypoints[15][2], keypoints[16][2])
        
        if knee_conf > 0.5 and ankle_conf > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
            
            knee_lift = abs(knee_y - self.knee_baseline_y)
            ankle_lift = abs(ankle_y - self.ankle_baseline_y)
            
            if knee_lift > self.thresholds['situp_foot_lift_threshold'] or \
               ankle_lift > self.thresholds['situp_foot_lift_threshold']:
                return True
        
        return False
    
    def _detect_neck_strain(self, keypoints):
        """Detect neck strain by measuring ear-shoulder distance change."""
        ear_conf = max(keypoints[3][2], keypoints[4][2])
        shoulder_conf = max(keypoints[5][2], keypoints[6][2])
        
        if ear_conf > 0.5 and shoulder_conf > 0.5:
            # Use the most confident side
            if keypoints[3][2] > keypoints[4][2]:
                ear_pos = np.array(keypoints[3][:2])
                shoulder_pos = np.array(keypoints[5][:2])
            else:
                ear_pos = np.array(keypoints[4][:2])
                shoulder_pos = np.array(keypoints[6][:2])
            
            distance = np.linalg.norm(ear_pos - shoulder_pos)
            return distance
        
        return None
    
    def _calculate_momentum_score(self):
        """Calculate momentum/jerk score based on shoulder acceleration."""
        if len(self.shoulder_positions) < 3:
            return 0
        
        # Calculate acceleration (2nd derivative)
        positions = list(self.shoulder_positions)
        accelerations = []
        
        for i in range(2, len(positions)):
            t0, y0 = positions[i-2]
            t1, y1 = positions[i-1]
            t2, y2 = positions[i]
            
            dt1 = t1 - t0
            dt2 = t2 - t1
            
            if dt1 > 0 and dt2 > 0:
                v1 = (y1 - y0) / dt1
                v2 = (y2 - y1) / dt2
                accel = (v2 - v1) / ((dt1 + dt2) / 2)
                accelerations.append(abs(accel))
        
        if not accelerations:
            return 0
        
        # Check if initial acceleration is much higher (momentum usage)
        if len(accelerations) >= 3:
            initial_accel = np.mean(accelerations[:3])
            avg_accel = np.mean(accelerations)
            
            if avg_accel > 0:
                jerk_ratio = initial_accel / avg_accel
                # Higher ratio = more momentum
                score = int(min(100, jerk_ratio * 30))
                return score
        
        return 0
    
    def _get_situp_form_score(self):
        """Calculate situp form score (0-100)."""
        if self.situp_valid_reps == 0:
            return 0
        
        total_deductions = 0
        
        # 5 points per foot lift
        total_deductions += sum(self.situp_foot_lifts) * 5
        
        # 2 points per short ROM
        total_deductions += self.situp_short_rom_count * 2
        
        # Calculate score
        score = max(0, 100 - total_deductions)
        return int(score)
    
    def _get_situp_tempo_consistency(self):
        """Calculate tempo consistency for situps."""
        if len(self.situp_concentric_times) < 2:
            return 100
        
        combined = self.situp_concentric_times + self.situp_eccentric_times
        if not combined:
            return 100
        
        mean_time = np.mean(combined)
        if mean_time == 0:
            return 100
        
        std_time = np.std(combined)
        cv = std_time / mean_time
        score = int(max(0, (1 - cv) * 100))
        return score
    
    # ---------------------------------------------------------
    # SIT-AND-REACH HELPER METHODS
    # ---------------------------------------------------------
    
    def update_sitnreach_data(self, keypoints, angles, reach_distance, arm_length, 
                             hip_angle, back_angle, knee_angle, symmetry_error, current_time):
        """Update sit-and-reach tracking data per frame."""
        if keypoints is None:
            return
        
        self.total_frames += 1
        
        # Track reach distance
        if reach_distance is not None:
            self.reach_distances.append(reach_distance)
            if reach_distance > self.max_reach_distance:
                self.max_reach_distance = reach_distance
                self.max_reach_frame_data = keypoints.copy()
        
        # Track arm length
        if arm_length is not None:
            self.arm_lengths.append(arm_length)
        
        # Track hip flexion angle
        if hip_angle is not None:
            self.hip_flexion_angles.append(hip_angle)
        
        # Track back alignment
        if back_angle is not None:
            self.back_alignment_angles.append(back_angle)
        
        # Track knee extension
        if knee_angle is not None:
            self.knee_extension_angles.append(knee_angle)
        
        # Track symmetry
        if symmetry_error is not None:
            self.reach_symmetry_errors.append(symmetry_error)
        
        # Track hip stability
        hip_conf = max(keypoints[11][2], keypoints[12][2])
        if hip_conf > 0.5:
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            self.hip_y_positions.append(hip_y)
        
        # Check if frame is valid (legs straight)
        if knee_angle and knee_angle >= self.thresholds['sitnreach_knee_valid']:
            self.valid_frames += 1
    
    def _calculate_normalized_reach_score(self):
        """Calculate normalized reach score (0-100)."""
        if not self.arm_lengths or self.max_reach_distance == 0:
            return 0
        
        avg_arm_length = np.mean(self.arm_lengths)
        if avg_arm_length == 0:
            return 0
        
        # Normalize reach by arm length
        normalized_reach = self.max_reach_distance / avg_arm_length
        
        # Convert to 0-100 scale (assuming 1.0 = 100%)
        score = int(min(100, normalized_reach * 100))
        return score
    
    def _calculate_trunk_flexibility_score(self):
        """Calculate trunk flexibility based on hip angle."""
        if not self.hip_flexion_angles:
            return 0
        
        min_hip_angle = np.min(self.hip_flexion_angles)
        
        if min_hip_angle < self.thresholds['sitnreach_excellent_hip']:
            score = 100
        elif min_hip_angle < self.thresholds['sitnreach_average_hip']:
            # Linear scale between excellent and average
            range_size = self.thresholds['sitnreach_average_hip'] - self.thresholds['sitnreach_excellent_hip']
            deviation = min_hip_angle - self.thresholds['sitnreach_excellent_hip']
            score = int(100 - (deviation / range_size) * 30)  # 100 to 70
        else:
            # Poor flexibility
            deviation = min_hip_angle - self.thresholds['sitnreach_average_hip']
            score = int(max(0, 70 - (deviation / 20) * 70))  # 70 to 0
        
        return score
    
    def _calculate_back_alignment_score(self):
        """Calculate back alignment score."""
        if not self.back_alignment_angles:
            return 0
        
        # Calculate deviation from ideal straight back (180 degrees)
        deviations = [abs(angle - 180) for angle in self.back_alignment_angles]
        avg_deviation = np.mean(deviations)
        
        # Lower deviation = better score
        score = int(max(0, 100 - (avg_deviation / 45) * 100))
        return score
    
    def _calculate_knee_validity_score(self):
        """Calculate knee validity score (legs straight)."""
        if not self.knee_extension_angles:
            return 0
        
        valid_knee_count = sum(1 for angle in self.knee_extension_angles 
                               if angle >= self.thresholds['sitnreach_knee_valid'])
        
        validity_rate = valid_knee_count / len(self.knee_extension_angles)
        score = int(validity_rate * 100)
        return score
    
    def _calculate_symmetry_score(self):
        """Calculate reach symmetry score (left vs right balance)."""
        if not self.reach_symmetry_errors:
            return 100
        
        avg_error = np.mean(self.reach_symmetry_errors)
        
        # Lower error = better score
        score = int(max(0, 100 - (avg_error / self.thresholds['sitnreach_max_symmetry_error']) * 100))
        return score
    
    def _calculate_hip_stability_score(self):
        """Calculate hip stability score (no bouncing)."""
        if len(self.hip_y_positions) < 2:
            return 100
        
        variance = np.var(list(self.hip_y_positions))
        
        # Lower variance = better score
        score = int(max(0, 100 - (variance / self.thresholds['sitnreach_max_hip_variance']) * 100))
        return score
    
    def _calculate_sitnreach_accuracy(self):
        """Calculate test execution accuracy."""
        if self.total_frames == 0:
            return 0
        
        accuracy = (self.valid_frames / self.total_frames) * 100
        return int(accuracy)
    
    # ---------------------------------------------------------
    # SKIPPING HELPER METHODS
    # ---------------------------------------------------------
    
    def update_skipping_data(self, keypoints, angles, current_time):
        """Update skipping-specific tracking data per frame."""
        if keypoints is None:
            return
        
        # Initialize start time on first call
        if self.skip_start_time is None:
            self.skip_start_time = current_time
        
        self.frames_processed_skip += 1
        
        # Calculate average ankle Y position
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        ankle_conf = max(left_ankle[2], right_ankle[2])
        
        if ankle_conf > 0.5:
            avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            self.ankle_y_positions.append(avg_ankle_y)
            
            # Establish ground reference quickly (median of first 10 frames)
            if self.ground_y is None and len(self.ankle_y_positions) >= 10:
                self.ground_y = np.median(list(self.ankle_y_positions))
            
            # Continuously update ground reference when on ground
            if self.jump_state == 'ground' and len(self.ankle_y_positions) > 20:
                recent_positions = list(self.ankle_y_positions)[-20:]
                self.ground_y = np.median(recent_positions)
            
            # Jump detection logic
            if self.ground_y is not None:
                # Use lower threshold for more reliable detection (20px instead of 30px)
                jump_threshold = 20
                
                # Check if in air (ankle Y < ground Y - threshold)
                is_in_air = avg_ankle_y < (self.ground_y - jump_threshold)
                
                if is_in_air and self.jump_state == 'ground':
                    # Jump started
                    self.jump_state = 'air'
                    self.jump_start_time = current_time
                    self.min_ankle_y_current_jump = avg_ankle_y
                    
                    # Check for double bounce (too soon after last landing)
                    if self.last_land_time is not None:
                        time_since_land = current_time - self.last_land_time
                        if time_since_land < 0.15:  # Less than 150ms
                            self.double_bounce_count += 1
                
                elif is_in_air and self.jump_state == 'air':
                    # Still in air, track minimum Y (maximum height)
                    if self.min_ankle_y_current_jump is None or avg_ankle_y < self.min_ankle_y_current_jump:
                        self.min_ankle_y_current_jump = avg_ankle_y
                
                elif not is_in_air and self.jump_state == 'air':
                    # Landed - complete the jump
                    self.jump_state = 'ground'
                    self.last_land_time = current_time
                    
                    # Calculate jump metrics
                    if self.jump_start_time is not None:
                        jump_duration = current_time - self.jump_start_time
                        self.jump_durations.append(jump_duration)
                    
                    if self.min_ankle_y_current_jump is not None:
                        jump_height = self.ground_y - self.min_ankle_y_current_jump
                        
                        # Only count jumps with meaningful height (lower threshold)
                        if jump_height >= 15:  # Lowered from 20px to 15px
                            self.jump_heights.append(jump_height)
                            self.jump_count += 1
                            
                            # Check if it's a quality jump
                            if jump_height >= self.thresholds['skip_min_jump_height']:
                                self.correct_jumps += 1
                    self.min_ankle_y_current_jump = None
        
        # Track back angle (posture)
        back_angle = angles.get('skip_back_angle')
        if back_angle is not None:
            self.back_angles_skip.append(back_angle)
        
        # Track knee bend
        knee_angle = angles.get('skip_knee_angle')
        if knee_angle is not None:
            self.knee_bend_angles.append(knee_angle)
        
        # Track wrist positions for arm movement
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        wrist_conf = max(left_wrist[2], right_wrist[2])
        
        if wrist_conf > 0.5:
            avg_wrist_x = (left_wrist[0] + right_wrist[0]) / 2
            avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
            self.wrist_positions.append((avg_wrist_x, avg_wrist_y))
    
    def _calculate_tempo_consistency_skip(self):
        """Calculate tempo consistency for skipping."""
        if len(self.jump_durations) < 2:
            return 100
        
        mean_duration = np.mean(self.jump_durations)
        if mean_duration == 0:
            return 0
        
        std_duration = np.std(self.jump_durations)
        cv = std_duration / mean_duration
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def _calculate_jump_height_score(self):
        """Calculate jump height score."""
        if not self.jump_heights:
            return 0
        
        avg_height = np.mean(self.jump_heights)
        # Normalize to 0-100 (assuming 100 pixels = 100 score)
        score = int(min(100, avg_height))
        return score
    
    def _calculate_posture_score_skip(self):
        """Calculate posture score for skipping."""
        if not self.back_angles_skip:
            return 100
        
        deviations = [abs(angle - self.thresholds['skip_ideal_back_angle']) 
                     for angle in self.back_angles_skip]
        avg_deviation = np.mean(deviations)
        
        score = int(max(0, 100 - (avg_deviation / self.thresholds['skip_max_back_deviation']) * 100))
        return score
    
    def _calculate_knee_control_score(self):
        """Calculate knee control score for skipping."""
        if not self.knee_bend_angles:
            return 100
        
        # Count excessive knee bends
        excessive_bends = sum(1 for angle in self.knee_bend_angles 
                             if angle < self.thresholds['skip_max_knee_bend'])
        
        if len(self.knee_bend_angles) == 0:
            return 100
        
        control_rate = 1 - (excessive_bends / len(self.knee_bend_angles))
        score = int(control_rate * 100)
        return score
    
    def _calculate_skip_accuracy(self):
        """Calculate skipping accuracy."""
        if self.jump_count == 0:
            return 0
        
        accuracy = (self.correct_jumps / self.jump_count) * 100
        return int(accuracy)
    
    def _get_skipping_frequency(self):
        """Calculate skips per second."""
        if self.skip_start_time is None:
            return 0
        
        import time
        total_time = time.time() - self.skip_start_time
        if total_time == 0:
            return 0
        
        return self.jump_count / total_time
    
    # ---------------------------------------------------------
    # JUMPING JACKS HELPER METHODS
    # ---------------------------------------------------------
    
    def update_jumpingjacks_data(self, keypoints, angles, current_time):
        """Update jumping jacks tracking data per frame."""
        if keypoints is None:
            return
        
        # Initialize start time on first call
        if self.jj_start_time is None:
            self.jj_start_time = current_time
        
        # Calculate arm spread (wrist distance)
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        wrist_conf = min(left_wrist[2], right_wrist[2])
        
        if wrist_conf > 0.5:
            arm_spread = abs(left_wrist[0] - right_wrist[0])
            self.jj_arm_spreads.append(arm_spread)
        else:
            arm_spread = None
        
        # Calculate leg spread (ankle distance)
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        ankle_conf = min(left_ankle[2], right_ankle[2])
        
        if ankle_conf > 0.5:
            leg_spread = abs(left_ankle[0] - right_ankle[0])
            self.jj_leg_spreads.append(leg_spread)
        else:
            leg_spread = None
        
        # Get arm elevation angle
        arm_angle = angles.get('jj_arm_angle')
        if arm_angle is not None:
            self.jj_arm_angles.append(arm_angle)
        
        # Get back angle
        back_angle = angles.get('jj_back_angle')
        if back_angle is not None:
            self.jj_back_angles.append(back_angle)
        
        # State machine for rep counting - more flexible detection
        if arm_spread is not None and leg_spread is not None and arm_angle is not None:
            # Check for OPEN position - require at least 2 out of 3 conditions
            arm_open = arm_spread >= self.thresholds['jj_arm_open_threshold']
            leg_open = leg_spread >= self.thresholds['jj_leg_open_threshold']
            angle_open = arm_angle >= self.thresholds['jj_arm_angle_open']
            
            # OPEN: At least 2 conditions met (more lenient)
            is_open = sum([arm_open, leg_open, angle_open]) >= 2
            
            # Check for CLOSED position
            arm_closed = arm_spread <= self.thresholds['jj_arm_close_threshold']
            leg_closed = leg_spread <= self.thresholds['jj_leg_close_threshold']
            angle_closed = arm_angle <= self.thresholds['jj_arm_angle_close']
            
            # CLOSED: At least 2 conditions met (more lenient)
            is_closed = sum([arm_closed, leg_closed, angle_closed]) >= 2
            
            if self.jj_state == 'closed' and is_open:
                # Transition to OPEN
                self.jj_state = 'open'
                self.jj_rep_start_time = current_time
                self.jj_arm_open_time = current_time
                self.jj_leg_open_time = current_time
                self.jj_max_arm_spread = arm_spread
                self.jj_max_leg_spread = leg_spread
                self.jj_max_arm_angle = arm_angle
            
            elif self.jj_state == 'open' and is_closed:
                # Transition to CLOSED - complete rep
                self.jj_state = 'closed'
                self.jj_rep_count += 1
                
                # Record rep duration
                if self.jj_rep_start_time is not None:
                    rep_duration = current_time - self.jj_rep_start_time
                    self.jj_rep_durations.append(rep_duration)
                
                # Calculate coordination error (simplified - using same time for both)
                if self.jj_arm_open_time is not None and self.jj_leg_open_time is not None:
                    coord_error = abs(self.jj_arm_open_time - self.jj_leg_open_time)
                    self.jj_coordination_errors.append(coord_error)
                
                # Check if rep was correct (all 3 conditions must have been met)
                is_correct = True
                if self.jj_max_arm_spread < self.thresholds['jj_arm_open_threshold']:
                    is_correct = False
                if self.jj_max_leg_spread < self.thresholds['jj_leg_open_threshold']:
                    is_correct = False
                if self.jj_max_arm_angle < self.thresholds['jj_arm_angle_open']:
                    is_correct = False
                
                if is_correct:
                    self.jj_correct_reps += 1
                
                # Reset for next rep
                self.jj_rep_start_time = None
                self.jj_max_arm_spread = 0
                self.jj_max_leg_spread = 0
                self.jj_max_arm_angle = 0
            
            elif self.jj_state == 'open':
                # Track max values during open phase
                if arm_spread > self.jj_max_arm_spread:
                    self.jj_max_arm_spread = arm_spread
                if leg_spread > self.jj_max_leg_spread:
                    self.jj_max_leg_spread = leg_spread
                if arm_angle > self.jj_max_arm_angle:
                    self.jj_max_arm_angle = arm_angle
    
    def _calculate_jj_coordination_score(self):
        """Calculate arm-leg coordination score."""
        if not self.jj_coordination_errors:
            return 100
        
        avg_error = np.mean(self.jj_coordination_errors)
        # Lower error = better score
        score = int(max(0, 100 - (avg_error / self.thresholds['jj_max_coordination_error']) * 100))
        return score
    
    def _calculate_jj_arm_rom_score(self):
        """Calculate arm range of motion score."""
        if not self.jj_arm_angles:
            return 0
        
        max_angle = np.max(list(self.jj_arm_angles))
        min_angle = np.min(list(self.jj_arm_angles))
        rom = max_angle - min_angle
        
        # Ideal ROM is ~150 degrees (from 0 to 150+)
        ideal_rom = 150
        score = int(min(100, (rom / ideal_rom) * 100))
        return score
    
    def _calculate_jj_leg_rom_score(self):
        """Calculate leg range of motion score."""
        if not self.jj_leg_spreads:
            return 0
        
        max_spread = np.max(list(self.jj_leg_spreads))
        min_spread = np.min(list(self.jj_leg_spreads))
        rom = max_spread - min_spread
        
        # Score based on spread range
        ideal_rom = 200  # pixels
        score = int(min(100, (rom / ideal_rom) * 100))
        return score
    
    def _calculate_jj_tempo_consistency(self):
        """Calculate tempo consistency for jumping jacks."""
        if len(self.jj_rep_durations) < 2:
            return 100
        
        mean_duration = np.mean(self.jj_rep_durations)
        if mean_duration == 0:
            return 0
        
        std_duration = np.std(self.jj_rep_durations)
        cv = std_duration / mean_duration
        score = int(max(0, (1 - cv) * 100))
        return score
    
    def _calculate_jj_posture_score(self):
        """Calculate posture score for jumping jacks."""
        if not self.jj_back_angles:
            return 100
        
        deviations = [abs(angle - self.thresholds['jj_ideal_back_angle']) 
                     for angle in self.jj_back_angles]
        avg_deviation = np.mean(deviations)
        
        score = int(max(0, 100 - (avg_deviation / 45) * 100))
        return score
    
    def _calculate_jj_accuracy(self):
        """Calculate jumping jacks accuracy."""
        if self.jj_rep_count == 0:
            return 0
        
        accuracy = (self.jj_correct_reps / self.jj_rep_count) * 100
        return int(accuracy)
    
    def _get_jj_frequency(self):
        """Calculate reps per second."""
        if self.jj_start_time is None:
            return 0
        
        import time
        total_time = time.time() - self.jj_start_time
        if total_time == 0:
            return 0
        
        return self.jj_rep_count / total_time
    
    # ---------------------------------------------------------
    # VERTICAL JUMP HELPER METHODS
    # ---------------------------------------------------------
    
    def update_vjump_data(self, keypoints, angles, current_time):
        """Update vertical jump tracking data per frame."""
        if keypoints is None:
            return
        
        # Initialize start time on first call
        if self.vjump_start_time is None:
            self.vjump_start_time = current_time
        
        # Get ankle positions
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
            avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            self.vjump_ankle_y_positions.append(avg_ankle_y)
            
            # Establish ground reference more quickly (use first 10 frames or standing frames)
            if self.vjump_ground_y is None:
                if len(self.vjump_ankle_y_positions) >= 10:
                    self.vjump_ground_y = np.median(list(self.vjump_ankle_y_positions))
            
            # Continuously update ground reference during standing
            if self.vjump_state == 'standing' and len(self.vjump_ankle_y_positions) > 15:
                recent_positions = list(self.vjump_ankle_y_positions)[-15:]
                # Only update if variance is low (person is standing still)
                if np.var(recent_positions) < 50:
                    self.vjump_ground_y = np.median(recent_positions)
            
            if self.vjump_ground_y is None:
                return
            
            # State machine for vertical jump detection
            jump_threshold = self.thresholds['vjump_min_height']
            
            if self.vjump_state == 'standing':
                # Detect takeoff directly (ankle Y decreases = person going up)
                # Use stricter threshold to avoid false detections
                if avg_ankle_y < self.vjump_ground_y - 25:  # Increased from 10 to 25px
                    self.vjump_state = 'airborne'
                    self.vjump_takeoff_time = current_time
                    self.vjump_min_ankle_y = avg_ankle_y
                    
                    # Track countermovement angle at takeoff
                    knee_angle = angles.get('vjump_countermovement_angle')
                    if knee_angle:
                        self.vjump_min_knee_angle = knee_angle
                    
                    # Track arm swing at takeoff
                    arm_angle = angles.get('vjump_arm_swing_angle')
                    if arm_angle:
                        self.vjump_max_arm_angle = arm_angle
                    
                    # Record takeoff times for symmetry
                    if left_ankle[1] < self.vjump_ground_y - 8:
                        self.vjump_left_ankle_takeoff_time = current_time
                    if right_ankle[1] < self.vjump_ground_y - 8:
                        self.vjump_right_ankle_takeoff_time = current_time
            
            elif self.vjump_state == 'airborne':
                # Track maximum jump height (lowest Y value)
                if self.vjump_min_ankle_y is None or avg_ankle_y < self.vjump_min_ankle_y:
                    self.vjump_min_ankle_y = avg_ankle_y
                
                # Continue tracking angles during jump
                knee_angle = angles.get('vjump_countermovement_angle')
                if knee_angle and knee_angle < self.vjump_min_knee_angle:
                    self.vjump_min_knee_angle = knee_angle
                
                arm_angle = angles.get('vjump_arm_swing_angle')
                if arm_angle and arm_angle > self.vjump_max_arm_angle:
                    self.vjump_max_arm_angle = arm_angle
                
                # Detect landing (ankle returns near ground level)
                # Stricter threshold to avoid false landings
                if avg_ankle_y > self.vjump_ground_y - 15:  # Changed from 10 to 15px
                    self.vjump_state = 'landing'
                    
                    # Calculate jump metrics
                    if self.vjump_min_ankle_y is not None:
                        jump_height = self.vjump_ground_y - self.vjump_min_ankle_y
                        
                        # Only count jumps with meaningful height (stricter threshold)
                        if jump_height >= 30:  # Increased from 10px to 30px minimum
                            self.vjump_jump_heights.append(jump_height)
                            self.vjump_countermovement_depths.append(self.vjump_min_knee_angle)
                            self.vjump_arm_swing_angles.append(self.vjump_max_arm_angle)
                            
                            # Calculate takeoff symmetry error
                            if self.vjump_left_ankle_takeoff_time and self.vjump_right_ankle_takeoff_time:
                                symmetry_error = abs(self.vjump_left_ankle_takeoff_time - self.vjump_right_ankle_takeoff_time)
                                self.vjump_takeoff_symmetry_errors.append(symmetry_error)
                            
                            # Check if jump meets quality criteria (more lenient)
                            is_valid = (
                                jump_height >= 10 and
                                self.vjump_min_knee_angle < 170  # Had some knee bend
                            )
                            
                            if is_valid:
                                self.vjump_valid_jumps += 1
                            
                            self.vjump_jump_count += 1
                    
                    # Reset for next jump
                    self.vjump_min_ankle_y = None
                    self.vjump_min_knee_angle = 180
                    self.vjump_max_arm_angle = 0
                    self.vjump_left_ankle_takeoff_time = None
                    self.vjump_right_ankle_takeoff_time = None
            
            elif self.vjump_state == 'landing':
                # Record landing knee angle
                landing_knee = angles.get('vjump_landing_knee_angle')
                if landing_knee and len(self.vjump_landing_knee_angles) < self.vjump_jump_count:
                    self.vjump_landing_knee_angles.append(landing_knee)
                
                # Return to standing when stable (within 15px of ground) - increased tolerance
                if abs(avg_ankle_y - self.vjump_ground_y) < 15:
                    self.vjump_state = 'standing'
    
    def _calculate_jump_height_score_vjump(self):
        """Calculate score based on jump height (0-1)."""
        if not self.vjump_jump_heights:
            return 0
        
        max_height = max(self.vjump_jump_heights)
        # Normalize: 30px = 0, 150px = 1.0
        score = min(max_height / 150.0, 1.0)
        return score
    
    def _calculate_countermovement_score(self):
        """Calculate score based on countermovement depth (0-1)."""
        if not self.vjump_countermovement_depths:
            return 0
        
        avg_depth = np.mean(self.vjump_countermovement_depths)
        # Good depth: 90-120 degrees
        if 90 <= avg_depth <= 120:
            return 1.0
        elif avg_depth < 90:
            # Too deep
            return max(0, 1.0 - (90 - avg_depth) / 30.0)
        else:
            # Not deep enough
            return max(0, 1.0 - (avg_depth - 120) / 60.0)
    
    def _calculate_arm_swing_score(self):
        """Calculate score based on arm swing utilization (0-1)."""
        if not self.vjump_arm_swing_angles:
            return 0
        
        avg_arm_swing = np.mean(self.vjump_arm_swing_angles)
        # Good arm swing: 140+ degrees
        if avg_arm_swing >= 140:
            return 1.0
        else:
            return max(0, avg_arm_swing / 140.0)
    
    def _calculate_takeoff_symmetry_score(self):
        """Calculate score based on takeoff symmetry (0-1)."""
        if not self.vjump_takeoff_symmetry_errors:
            return 1.0  # No errors detected, assume perfect
        
        avg_error = np.mean(self.vjump_takeoff_symmetry_errors)
        # Good: < 0.1 seconds difference
        if avg_error < 0.05:
            return 1.0
        elif avg_error < 0.1:
            return 0.8
        else:
            return max(0, 1.0 - avg_error * 5)
    
    def _calculate_landing_control_score(self):
        """Calculate score based on landing control (0-1)."""
        if not self.vjump_landing_knee_angles:
            return 0.5  # Neutral if no data
        
        avg_landing_knee = np.mean(self.vjump_landing_knee_angles)
        # Good landing: 120-140 degrees (controlled flexion)
        if 120 <= avg_landing_knee <= 140:
            return 1.0
        elif avg_landing_knee < 100:
            # Too much flexion
            return max(0, 0.5 - (100 - avg_landing_knee) / 100.0)
        elif avg_landing_knee > 160:
            # Too stiff
            return max(0, 0.8 - (avg_landing_knee - 160) / 20.0)
        else:
            return 0.8
    
    def _calculate_vjump_accuracy(self):
        """Calculate jump accuracy (valid jumps / total jumps)."""
        if self.vjump_jump_count == 0:
            return 0
        return self.vjump_valid_jumps / self.vjump_jump_count
    
    # ---------------------------------------------------------
    # BROAD JUMP HELPER METHODS
    # ---------------------------------------------------------
    
    def update_bjump_data(self, keypoints, angles, current_time):
        """
        Robust Broad Jump Analyzer using State Machine.
        View-agnostic but optimized for Side View.
        Tracks ankle center for stable distance measurement.
        """
        if keypoints is None:
            return
        
        # Initialize start time on first call
        if self.bjump_start_time is None:
            self.bjump_start_time = current_time
        
        # STEP 1: Extract ankle center (average of left and right for stability)
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
            ankle_x = (left_ankle[0] + right_ankle[0]) / 2
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        elif left_ankle[2] > 0.5:
            ankle_x = left_ankle[0]
            ankle_y = left_ankle[1]
        elif right_ankle[2] > 0.5:
            ankle_x = right_ankle[0]
            ankle_y = right_ankle[1]
        else:
            return
        
        # STEP 2: Buffer positions for smoothing (reduces jitter)
        self.bjump_ankle_x_positions.append(ankle_x)
        self.bjump_ankle_y_positions.append(ankle_y)
        
        # STEP 3: Calibrate floor level (dynamic, averaged over first 30 frames)
        if self.bjump_ground_y is None:
            if len(self.bjump_ankle_y_positions) >= 30:
                self.bjump_ground_y = np.mean(list(self.bjump_ankle_y_positions)[-30:])
            return
        
        # STEP 4: Calculate vertical displacement (positive = feet off ground)
        vertical_displacement = self.bjump_ground_y - ankle_y
        
        # Smooth positions to reduce noise
        recent_x = list(self.bjump_ankle_x_positions)[-5:] if len(self.bjump_ankle_x_positions) >= 5 else list(self.bjump_ankle_x_positions)
        recent_y = list(self.bjump_ankle_y_positions)[-5:] if len(self.bjump_ankle_y_positions) >= 5 else list(self.bjump_ankle_y_positions)
        smooth_x = np.mean(recent_x)
        smooth_y = np.mean(recent_y)
        smooth_vertical_displacement = self.bjump_ground_y - smooth_y
        
        # STEP 5: STATE MACHINE with robust thresholds
        AIRBORNE_THRESHOLD = 25       # Feet must rise 25px above ground (increased for reliability)
        LANDED_TOLERANCE = 20         # Within 20px of ground = landed
        MIN_DISTANCE_PIXELS = 30      # Minimum distance to count as valid jump (noise filtering)
        STABILITY_FRAMES = 8          # Frames to wait for stabilization
        
        # ========== STATE: IDLE / READY ==========
        if self.bjump_state == 'standing':
            # Calibrate starting position (use median for stability)
            if self.bjump_start_x is None and len(self.bjump_ankle_x_positions) >= 10:
                self.bjump_start_x = np.median(list(self.bjump_ankle_x_positions)[-10:])
                return
            
            if self.bjump_start_x is None:
                return
            
            # Detect TAKEOFF: feet rise above threshold
            if smooth_vertical_displacement > AIRBORNE_THRESHOLD:
                self.bjump_takeoff_x = smooth_x
                self.bjump_takeoff_time = current_time
                self.bjump_state = 'airborne'
                self.bjump_max_x = smooth_x  # Initialize max tracking
                
                # Record pre-jump biomechanics
                countermovement_angle = angles.get('bjump_countermovement_angle')
                if countermovement_angle:
                    self.bjump_min_knee_angle = min(self.bjump_min_knee_angle, countermovement_angle)
                
                arm_swing_angle = angles.get('bjump_arm_swing_angle')
                if arm_swing_angle:
                    self.bjump_max_arm_angle = max(self.bjump_max_arm_angle, arm_swing_angle)
                
                # Record takeoff times for symmetry analysis
                self.bjump_left_ankle_takeoff_time = current_time
                self.bjump_right_ankle_takeoff_time = current_time
                
                print(f"[BJUMP] Takeoff detected at X={int(smooth_x)}px")
        
        # ========== STATE: AIRBORNE ==========
        elif self.bjump_state == 'airborne':
            # Track maximum X position during flight (farthest point)
            if smooth_x > self.bjump_max_x:
                self.bjump_max_x = smooth_x
            
            # Continue tracking angles during flight
            countermovement_angle = angles.get('bjump_countermovement_angle')
            if countermovement_angle and countermovement_angle < self.bjump_min_knee_angle:
                self.bjump_min_knee_angle = countermovement_angle
            
            arm_swing_angle = angles.get('bjump_arm_swing_angle')
            if arm_swing_angle and arm_swing_angle > self.bjump_max_arm_angle:
                self.bjump_max_arm_angle = arm_swing_angle
            
            # Detect LANDING: feet return to ground level
            if smooth_vertical_displacement < LANDED_TOLERANCE:
                self.bjump_state = 'landing'
                
                # Calculate horizontal distance (Start → Max Position)
                if self.bjump_start_x is not None and self.bjump_max_x is not None:
                    distance_pixels = abs(self.bjump_max_x - self.bjump_start_x)
                    
                    # NOISE FILTER: only count significant jumps
                    if distance_pixels >= MIN_DISTANCE_PIXELS:
                        self.bjump_jump_count += 1
                        self.bjump_jump_distances.append(distance_pixels)
                        
                        # Update personal best
                        if distance_pixels > self.bjump_max_distance:
                            self.bjump_max_distance = distance_pixels
                        
                        # Store biomechanical metrics
                        self.bjump_countermovement_depths.append(self.bjump_min_knee_angle)
                        self.bjump_arm_swing_angles.append(self.bjump_max_arm_angle)
                        
                        # Takeoff symmetry error
                        if self.bjump_left_ankle_takeoff_time and self.bjump_right_ankle_takeoff_time:
                            symmetry_error = abs(self.bjump_left_ankle_takeoff_time - self.bjump_right_ankle_takeoff_time)
                            self.bjump_takeoff_symmetry_errors.append(symmetry_error)
                        
                        # Landing stability (left-right ankle difference)
                        if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
                            landing_stability_error = abs(left_ankle[0] - right_ankle[0])
                            self.bjump_landing_stability_errors.append(landing_stability_error)
                        
                        # Quality validation (meets performance criteria)
                        is_valid = (
                            distance_pixels >= MIN_DISTANCE_PIXELS and
                            self.bjump_min_knee_angle <= self.thresholds['bjump_good_countermovement'] and
                            self.bjump_max_arm_angle >= self.thresholds['bjump_good_arm_swing']
                        )
                        
                        if is_valid:
                            self.bjump_valid_jumps += 1
                        
                        print(f"[BJUMP] Jump #{self.bjump_jump_count}: Distance={int(distance_pixels)}px, Valid={is_valid}")
                    else:
                        print(f"[BJUMP] Jump filtered (too short: {int(distance_pixels)}px < {MIN_DISTANCE_PIXELS}px)")
                
                # Reset metrics for next jump
                self.bjump_min_knee_angle = 180
                self.bjump_max_arm_angle = 0
                self.bjump_left_ankle_takeoff_time = None
                self.bjump_right_ankle_takeoff_time = None
        
        # ========== STATE: LANDING / STABILIZATION ==========
        elif self.bjump_state == 'landing':
            # Wait for stable position before resetting to IDLE
            if len(self.bjump_ankle_x_positions) >= STABILITY_FRAMES:
                recent_x = list(self.bjump_ankle_x_positions)[-STABILITY_FRAMES:]
                recent_y = list(self.bjump_ankle_y_positions)[-STABILITY_FRAMES:]
                
                # Check for stability (low variance = not moving)
                x_variance = np.var(recent_x)
                x_stable = x_variance < 20
                y_stable = abs(smooth_y - self.bjump_ground_y) < 25
                
                if x_stable and y_stable:
                    self.bjump_state = 'standing'
                    # Recalibrate for next jump
                    self.bjump_start_x = np.median(recent_x)
                    self.bjump_ground_y = np.median(recent_y)
                    print(f"[BJUMP] Ready for next jump (Start X={int(self.bjump_start_x)}px)")
    
    def _calculate_jump_distance_score_bjump(self):
        """Calculate score based on jump distance (0-1)."""
        if not self.bjump_jump_distances:
            return 0
        
        max_distance = max(self.bjump_jump_distances)
        # Normalize: 20px = 0, 250px = 1.0
        score = min((max_distance - 20) / 230.0, 1.0)
        return max(0, score)
    
    def _calculate_countermovement_score_bjump(self):
        """Calculate score based on countermovement depth (0-1)."""
        if not self.bjump_countermovement_depths:
            return 0
        
        avg_depth = np.mean(self.bjump_countermovement_depths)
        # Good depth: 90-120 degrees
        if 90 <= avg_depth <= 120:
            return 1.0
        elif avg_depth < 90:
            return max(0, 1.0 - (90 - avg_depth) / 30.0)
        else:
            return max(0, 1.0 - (avg_depth - 120) / 60.0)
    
    def _calculate_arm_swing_score_bjump(self):
        """Calculate score based on arm swing utilization (0-1)."""
        if not self.bjump_arm_swing_angles:
            return 0
        
        avg_arm_swing = np.mean(self.bjump_arm_swing_angles)
        # Good arm swing: 140+ degrees
        if avg_arm_swing >= 140:
            return 1.0
        else:
            return max(0, avg_arm_swing / 140.0)
    
    def _calculate_takeoff_symmetry_score_bjump(self):
        """Calculate score based on takeoff symmetry (0-1)."""
        if not self.bjump_takeoff_symmetry_errors:
            return 1.0
        
        avg_error = np.mean(self.bjump_takeoff_symmetry_errors)
        if avg_error < 0.05:
            return 1.0
        elif avg_error < 0.1:
            return 0.8
        else:
            return max(0, 1.0 - avg_error * 5)
    
    def _calculate_landing_stability_score_bjump(self):
        """Calculate score based on landing stability (0-1)."""
        if not self.bjump_landing_stability_errors:
            return 0.5
        
        avg_stability_error = np.mean(self.bjump_landing_stability_errors)
        # Good landing: < 30px difference between ankles
        if avg_stability_error < 20:
            return 1.0
        elif avg_stability_error < 40:
            return 0.7
        else:
            return max(0, 1.0 - avg_stability_error / 100.0)
    
    def _calculate_bjump_accuracy(self):
        """Calculate jump accuracy (valid jumps / total jumps)."""
        if self.bjump_jump_count == 0:
            return 0
        return self.bjump_valid_jumps / self.bjump_jump_count
    
    def record_rep(self, rep_max, rep_min, duration_seconds, is_good_form):
        """Record a completed repetition."""
        self.rep_angles.append((rep_max, rep_min))
        self.rep_durations.append(duration_seconds)
        
        if is_good_form:
            self.good_reps += 1
        else:
            self.bad_reps += 1
    
    @property
    def total_reps(self):
        return self.good_reps + self.bad_reps

    # ---------------------------------------------------------
    # HELPER CALCULATIONS (Used by pushup_metrics)
    # ---------------------------------------------------------

    def _get_form_score(self):
        if self.total_reps == 0: return 0
        return int((self.good_reps / self.total_reps) * 100)

    def _get_range_of_motion_score(self):
        if not self.rep_angles: return 0
        rom_values = [max_a - min_a for max_a, min_a in self.rep_angles]
        avg_rom = np.mean(rom_values)
        return int(min(100, (avg_rom / self.thresholds['ideal_rom']) * 100))

    def _get_alignment_score(self):
        if not self.back_angles: return 0
        back_angles = list(self.back_angles)
        avg_error = np.mean([abs(a - self.thresholds['ideal_back_angle']) for a in back_angles])
        score = max(0, 1 - (avg_error / self.thresholds['max_back_deviation']))
        return int(score * 100)

    def _get_arm_symmetry_score(self):
        if not self.elbow_angles: return 0
        symmetry_errors = [abs(l - r) for l, r in self.elbow_angles if l and r]
        if not symmetry_errors: return 0
        avg_error = np.mean(symmetry_errors)
        score = max(0, 1 - (avg_error / self.thresholds['max_arm_asymmetry']))
        return int(score * 100)

    def _get_tempo_consistency_score(self):
        if len(self.rep_durations) < 2: return 100
        mean_dur = np.mean(self.rep_durations)
        if mean_dur == 0: return 100
        variation = np.std(self.rep_durations) / mean_dur
        return int(max(0, 1 - variation) * 100)

    def _get_rating(self, score):
        if score >= 90: return "⭐⭐⭐⭐⭐ EXCELLENT"
        elif score >= 75: return "⭐⭐⭐⭐ VERY GOOD"
        elif score >= 60: return "⭐⭐⭐ GOOD"
        elif score >= 45: return "⭐⭐ FAIR"
        elif score >= 30: return "⭐ NEEDS IMPROVEMENT"
        return "KEEP PRACTICING"

    # ---------------------------------------------------------
    # MAIN METRIC WRAPPERS
    # ---------------------------------------------------------

    def pushup_metrics(self):
        """
        Wraps all logic for calculating and displaying Push-up specific metrics.
        """
        if not self.exercise: self.exercise = 'pushup'
        
        # 1. Calculate individual components
        form = self._get_form_score()
        rom = self._get_range_of_motion_score()
        alignment = self._get_alignment_score()
        symmetry = self._get_arm_symmetry_score()
        tempo = self._get_tempo_consistency_score()
        
        # 2. Calculate Weighted Overall Score
        # Weights: Form(25%), ROM(20%), Alignment(20%), Symmetry(15%), Tempo(10%)
        # Note: Sum is 90% in original code, assumed intended or loose math. 
        overall = (form * 0.25) + (rom * 0.20) + (alignment * 0.20) + \
                  (symmetry * 0.15) + (tempo * 0.10)
        
        # Normalize to 100 scale roughly if weights don't add to 1.0, 
        # or just take raw calculation as per original logic.
        overall = int(overall) 
        
        # 3. Generate Feedback Messages
        messages = []
        if form < 50: messages.append("❌ FORM: Focus on maintaining proper form")
        elif form < 80: messages.append("⚠ FORM: Work on consistency")
        else: messages.append("✓ FORM: Excellent form maintained!")
        
        if rom < 60: messages.append("❌ ROM: Push deeper")
        elif rom < 80: messages.append("⚠ ROM: Get lower")
        else: messages.append("✓ ROM: Great depth!")
        
        if alignment < 50: messages.append("❌ ALIGNMENT: Keep back straight")
        elif alignment < 80: messages.append("⚠ ALIGNMENT: Watch your back sag")
        else: messages.append("✓ ALIGNMENT: Excellent spine stability!")

        if symmetry < 60: messages.append("❌ SYMMETRY: Significant arm imbalance")
        elif symmetry < 80: messages.append("⚠ SYMMETRY: Focus on pushing evenly")
        else: messages.append("✓ SYMMETRY: Good left/right balance")

        # 4. Display Report
        print("\n" + "="*70)
        print(" "*15 + "PUSH-UP PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 REPETITIONS:")
        print(f"    ✓ Good Reps: {self.good_reps}")
        print(f"    ✗ Bad Reps:  {self.bad_reps}")
        print(f"    Total:       {self.total_reps}")
        
        print(f"\n📈 DETAILED SCORES:")
        print(f"    Form Quality:    {form}/100")
        print(f"    Range of Motion: {rom}/100")
        print(f"    Body Alignment:  {alignment}/100")
        print(f"    Arm Symmetry:    {symmetry}/100")
        print(f"    Tempo:           {tempo}/100")
        
        print(f"\n" + "="*70)
        print(f"🏆 OVERALL SCORE: {overall}/100")
        print(f"⭐ RATING: {self._get_rating(overall)}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
            
        return {
            'exercise': 'pushup',
            'overall_score': overall,
            'rating': self._get_rating(overall),
            'repetitions': {
                'total': self.total_reps,
                'good': self.good_reps,
                'bad': self.bad_reps
            },
            'scores': {
                'form': form,
                'rom': rom,
                'alignment': alignment,
                'symmetry': symmetry,
                'tempo': tempo
            },
            'feedback': messages
        }

    def squat_metrics(self):
        """
        Comprehensive squat analysis with all biomechanical metrics.
        Based on squat.md specifications.
        """
        if not self.exercise:
            self.exercise = 'squat'
        
        # 1. Squat Depth Analysis
        avg_depth = np.mean(self.squat_depths) if self.squat_depths else 0
        parallel_reps = sum(1 for d in self.squat_depths if d <= self.thresholds['squat_parallel'])
        depth_score = int((parallel_reps / len(self.squat_depths)) * 100) if self.squat_depths else 0
        
        # 2. Back Inclination (Torso Angle)
        torso_score = self._calculate_torso_inclination_score()
        avg_torso_angle = np.mean(list(self.torso_angles)) if self.torso_angles else 0
        
        # 3. Shin Angle
        avg_shin_angle = np.mean(list(self.shin_angles)) if self.shin_angles else 0
        
        # 4. Knee Stability
        knee_stability_score = self._calculate_knee_stability_score()
        knee_cave_detected = knee_stability_score < 70
        
        # 5. Concentric Velocity
        avg_velocity = self._get_avg_concentric_velocity()
        
        # 6. Eccentric Tempo
        avg_eccentric_time = np.mean(self.eccentric_times) if self.eccentric_times else 0
        avg_concentric_time = np.mean(self.concentric_times) if self.concentric_times else 0
        
        # 7. Sticking Point
        sticking_point_angle = self.min_velocity_angle if self.min_velocity_angle else "N/A"
        
        # 8. ROM Consistency
        consistency_score = self._calculate_depth_consistency()
        
        # 9. Tempo Consistency
        tempo_score = self._calculate_tempo_score()
        
        # Overall Score Calculation (weighted average)
        overall = (
            depth_score * 0.25 +
            torso_score * 0.20 +
            knee_stability_score * 0.20 +
            consistency_score * 0.15 +
            tempo_score * 0.10 +
            (100 if not knee_cave_detected else 70) * 0.10
        )
        overall = int(overall)
        
        # Generate Feedback
        messages = []
        if depth_score < 50:
            messages.append("❌ DEPTH: Squat deeper - aim for parallel or below")
        elif depth_score < 80:
            messages.append("⚠ DEPTH: Almost there - get your hips to knee level")
        else:
            messages.append("✓ DEPTH: Excellent depth consistency!")
        
        if torso_score < 50:
            messages.append("❌ TORSO: Excessive forward lean - strengthen core")
        elif torso_score < 80:
            messages.append("⚠ TORSO: Slight forward lean detected")
        else:
            messages.append("✓ TORSO: Great upright posture!")
        
        if knee_cave_detected:
            messages.append("❌ KNEES: Knee valgus detected - push knees outward")
        else:
            messages.append("✓ KNEES: Good knee tracking!")
        
        if consistency_score < 70:
            messages.append("⚠ CONSISTENCY: Depth varies between reps")
        else:
            messages.append("✓ CONSISTENCY: Very consistent depth!")
        
        # Display Report
        print("\n" + "="*70)
        print(" "*15 + "SQUAT PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 REPETITIONS:")
        print(f"    Total Reps:        {self.total_reps}")
        print(f"    Parallel+ Reps:    {parallel_reps}/{len(self.squat_depths)}")
        print(f"    Good Form Reps:    {self.good_reps}")
        print(f"    Poor Form Reps:    {self.bad_reps}")
        
        print(f"\n📐 BIOMECHANICS (Category A - Form):")
        print(f"    Avg Depth:         {avg_depth:.1f}° (Target: ≤{self.thresholds['squat_parallel']}°)")
        print(f"    Depth Score:       {depth_score}/100")
        print(f"    Avg Torso Angle:   {avg_torso_angle:.1f}° (Ideal: 0-{self.thresholds['ideal_torso_angle']}°)")
        print(f"    Torso Score:       {torso_score}/100")
        print(f"    Avg Shin Angle:    {avg_shin_angle:.1f}°")
        print(f"    Knee Stability:    {knee_stability_score}/100")
        if knee_cave_detected:
            print(f"    ⚠️  Knee Cave:       DETECTED")
        
        print(f"\n⚡ PHYSICS (Category B - Power):")
        print(f"    Avg Velocity:      {avg_velocity:.2f} px/s")
        print(f"    Avg Descent Time:  {avg_eccentric_time:.2f}s")
        print(f"    Avg Ascent Time:   {avg_concentric_time:.2f}s")
        if sticking_point_angle != "N/A":
            print(f"    Sticking Point:    {sticking_point_angle}°")
        
        print(f"\n📈 CONSISTENCY (Category C):")
        print(f"    ROM Consistency:   {consistency_score}/100")
        print(f"    Tempo Consistency: {tempo_score}/100")
        
        print(f"\n" + "="*70)
        print(f"🏆 OVERALL SCORE: {overall}/100")
        print(f"⭐ RATING: {self._get_rating(overall)}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
        
        print("\n" + "="*70)
        
        return {
            'exercise': 'squat',
            'overall_score': overall,
            'rating': self._get_rating(overall),
            'repetitions': {
                'total': self.total_reps,
                'parallel_plus': parallel_reps,
                'good_form': self.good_reps,
                'poor_form': self.bad_reps
            },
            'biomechanics': {
                'avg_depth': round(avg_depth, 1),
                'depth_score': depth_score,
                'avg_torso_angle': round(avg_torso_angle, 1),
                'torso_score': torso_score,
                'avg_shin_angle': round(avg_shin_angle, 1),
                'knee_stability': knee_stability_score,
                'knee_cave_detected': knee_cave_detected
            },
            'physics': {
                'avg_velocity': round(avg_velocity, 2),
                'avg_descent_time': round(avg_eccentric_time, 2),
                'avg_ascent_time': round(avg_concentric_time, 2),
                'sticking_point': sticking_point_angle
            },
            'consistency': {
                'rom': consistency_score,
                'tempo': tempo_score
            },
            'feedback': messages
        }
    
    def situp_metrics(self):
        """
        Comprehensive situp analysis with all biomechanical metrics.
        Based on situp.md specifications.
        """
        if not self.exercise:
            self.exercise = 'situp'
        
        # 1. Torso Inclination Analysis
        avg_torso_inclination = np.mean(self.situp_torso_inclinations) if self.situp_torso_inclinations else 0
        good_inclination_reps = sum(1 for t in self.situp_torso_inclinations if t >= self.thresholds['situp_up_angle'])
        inclination_score = int((good_inclination_reps / len(self.situp_torso_inclinations)) * 100) if self.situp_torso_inclinations else 0
        
        # 2. Hip Flexion (Crunch Quality)
        avg_hip_flexion = np.mean(self.situp_hip_flexions) if self.situp_hip_flexions else 0
        good_flexion_reps = sum(1 for h in self.situp_hip_flexions if h <= self.thresholds['situp_good_hip_flexion'])
        flexion_score = int((good_flexion_reps / len(self.situp_hip_flexions)) * 100) if self.situp_hip_flexions else 0
        
        # 3. Foot Lift Detection
        total_foot_lifts = sum(self.situp_foot_lifts)
        foot_lift_rate = (total_foot_lifts / len(self.situp_foot_lifts)) * 100 if self.situp_foot_lifts else 0
        
        # 4. Neck Strain
        avg_neck_distance = np.mean(self.situp_neck_strains) if self.situp_neck_strains else 0
        
        # 5. Tempo Analysis
        avg_up_time = np.mean(self.situp_concentric_times) if self.situp_concentric_times else 0
        avg_down_time = np.mean(self.situp_eccentric_times) if self.situp_eccentric_times else 0
        tempo_consistency = self._get_situp_tempo_consistency()
        
        # 6. Momentum Detection
        avg_momentum_score = np.mean(self.situp_momentum_scores) if self.situp_momentum_scores else 0
        high_momentum_reps = sum(1 for m in self.situp_momentum_scores if m > self.thresholds['situp_momentum_threshold'])
        
        # 7. Valid Rep Count
        valid_rep_rate = (self.situp_valid_reps / self.total_reps) * 100 if self.total_reps > 0 else 0
        
        # 8. Form Score
        form_score = self._get_situp_form_score()
        
        # Overall Score
        overall = (
            inclination_score * 0.20 +
            flexion_score * 0.20 +
            (100 - foot_lift_rate) * 0.15 +
            form_score * 0.20 +
            tempo_consistency * 0.10 +
            (100 - min(100, avg_momentum_score)) * 0.15
        )
        overall = int(overall)
        
        # Generate Feedback
        messages = []
        if inclination_score < 50:
            messages.append("❌ ROM: Sit up higher - reach toward your knees")
        elif inclination_score < 80:
            messages.append("⚠ ROM: Almost there - lift your chest higher")
        else:
            messages.append("✓ ROM: Excellent full range of motion!")
        
        if flexion_score < 50:
            messages.append("❌ CRUNCH: Focus on crunching/folding tighter")
        elif flexion_score < 80:
            messages.append("⚠ CRUNCH: Good, but try to close the gap more")
        else:
            messages.append("✓ CRUNCH: Perfect core engagement!")
        
        if total_foot_lifts > 0:
            messages.append(f"❌ ANCHORING: Feet lifted {total_foot_lifts} times - keep them down")
        else:
            messages.append("✓ ANCHORING: Perfect foot placement!")
        
        if high_momentum_reps > 0:
            messages.append(f"⚠ MOMENTUM: {high_momentum_reps} reps used excessive swing")
        else:
            messages.append("✓ CONTROL: Great controlled movement!")
        
        # Display Report
        print("\n" + "="*70)
        print(" "*15 + "SIT-UP PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 REPETITIONS:")
        print(f"    Total Reps:        {self.total_reps}")
        print(f"    Valid Reps:        {self.situp_valid_reps}")
        print(f"    Good Form Reps:    {self.good_reps}")
        print(f"    Poor Form Reps:    {self.bad_reps}")
        
        print(f"\n📐 BIOMECHANICS (Category A - Form):")
        print(f"    Avg Torso Angle:   {avg_torso_inclination:.1f}° (Target: ≥{self.thresholds['situp_up_angle']}°)")
        print(f"    Inclination Score: {inclination_score}/100")
        print(f"    Avg Hip Flexion:   {avg_hip_flexion:.1f}° (Target: ≤{self.thresholds['situp_good_hip_flexion']}°)")
        print(f"    Flexion Score:     {flexion_score}/100")
        if total_foot_lifts > 0:
            print(f"    ⚠️  Foot Lifts:      {total_foot_lifts} violations")
        else:
            print(f"    ✓  Foot Stability:  Perfect")
        
        print(f"\n⚡ PHYSICS (Category B - Power):")
        print(f"    Avg Up Time:       {avg_up_time:.2f}s")
        print(f"    Avg Down Time:     {avg_down_time:.2f}s")
        print(f"    Tempo Consistency: {tempo_consistency}/100")
        print(f"    Avg Momentum:      {avg_momentum_score:.1f}/100")
        if high_momentum_reps > 0:
            print(f"    ⚠️  High Momentum:   {high_momentum_reps} reps")
        
        print(f"\n📈 SCORING (Category C):")
        print(f"    Form Score:        {form_score}/100")
        print(f"    Valid Rep Rate:    {valid_rep_rate:.1f}%")
        print(f"    Short ROM Count:   {self.situp_short_rom_count}")
        
        print(f"\n" + "="*70)
        print(f"🏆 OVERALL SCORE: {overall}/100")
        print(f"⭐ RATING: {self._get_rating(overall)}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
        
        print("\n" + "="*70)
        
        return {
            'exercise': 'situp',
            'overall_score': overall,
            'rating': self._get_rating(overall),
            'repetitions': {
                'total': self.total_reps,
                'valid': self.situp_valid_reps,
                'good_form': self.good_reps,
                'poor_form': self.bad_reps,
                'short_rom': self.situp_short_rom_count
            },
            'biomechanics': {
                'avg_torso_angle': round(avg_torso_inclination, 1),
                'inclination_score': inclination_score,
                'avg_hip_flexion': round(avg_hip_flexion, 1),
                'flexion_score': flexion_score,
                'foot_lifts': total_foot_lifts,
                'foot_lift_rate': round(foot_lift_rate, 1)
            },
            'physics': {
                'avg_up_time': round(avg_up_time, 2),
                'avg_down_time': round(avg_down_time, 2),
                'tempo_consistency': tempo_consistency,
                'avg_momentum': round(avg_momentum_score, 1),
                'high_momentum_reps': high_momentum_reps
            },
            'scores': {
                'form': form_score,
                'valid_rep_rate': round(valid_rep_rate, 1)
            },
            'feedback': messages
        }
    
    def sitnreach_metrics(self):
        """
        Comprehensive sit-and-reach analysis with all biomechanical metrics.
        Based on sitnreach.md specifications.
        """
        if not self.exercise:
            self.exercise = 'sitnreach'
        
        # Calculate all component scores
        reach_score = self._calculate_normalized_reach_score()
        trunk_flexibility_score = self._calculate_trunk_flexibility_score()
        back_alignment_score = self._calculate_back_alignment_score()
        symmetry_score = self._calculate_symmetry_score()
        knee_validity_score = self._calculate_knee_validity_score()
        hip_stability_score = self._calculate_hip_stability_score()
        accuracy_score = self._calculate_sitnreach_accuracy()
        
        # Calculate averages
        avg_arm_length = np.mean(self.arm_lengths) if self.arm_lengths else 0
        min_hip_angle = np.min(self.hip_flexion_angles) if self.hip_flexion_angles else 0
        avg_back_angle = np.mean(self.back_alignment_angles) if self.back_alignment_angles else 0
        avg_knee_angle = np.mean(self.knee_extension_angles) if self.knee_extension_angles else 0
        avg_symmetry_error = np.mean(self.reach_symmetry_errors) if self.reach_symmetry_errors else 0
        hip_variance = np.var(list(self.hip_y_positions)) if len(self.hip_y_positions) > 1 else 0
        
        # Final Performance Score (weighted formula from sitnreach.md)
        final_score = (
            reach_score * 0.40 +
            trunk_flexibility_score * 0.25 +
            back_alignment_score * 0.15 +
            symmetry_score * 0.10 +
            knee_validity_score * 0.10
        )
        final_score = int(final_score)
        
        # Generate Feedback
        messages = []
        
        if reach_score >= 80:
            messages.append("✓ REACH: Excellent forward reach!")
        elif reach_score >= 60:
            messages.append("⚠ REACH: Good, but can improve")
        else:
            messages.append("❌ REACH: Work on flexibility - reach further")
        
        if trunk_flexibility_score >= 80:
            messages.append("✓ HIP FLEXIBILITY: Excellent trunk flexion!")
        elif trunk_flexibility_score >= 60:
            messages.append("⚠ HIP FLEXIBILITY: Average flexibility")
        else:
            messages.append("❌ HIP FLEXIBILITY: Limited hip flexion - stretch more")
        
        if knee_validity_score < 70:
            messages.append("❌ KNEE POSITION: Keep legs straight during test!")
        else:
            messages.append("✓ KNEE POSITION: Good leg extension")
        
        if symmetry_score < 70:
            messages.append("⚠ SYMMETRY: Uneven reach - balance both sides")
        else:
            messages.append("✓ SYMMETRY: Balanced bilateral reach")
        
        if hip_stability_score < 70:
            messages.append("⚠ STABILITY: Reduce bouncing - smooth reach motion")
        else:
            messages.append("✓ STABILITY: Controlled, steady movement")
        
        # Flexibility category
        if final_score >= 80:
            flexibility_rating = "EXCELLENT FLEXIBILITY"
        elif final_score >= 60:
            flexibility_rating = "GOOD FLEXIBILITY"
        elif final_score >= 40:
            flexibility_rating = "AVERAGE FLEXIBILITY"
        else:
            flexibility_rating = "POOR FLEXIBILITY - NEEDS IMPROVEMENT"
        
        # Display Report
        print("\n" + "="*70)
        print(" "*15 + "SIT-AND-REACH PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📏 PRIMARY METRICS:")
        print(f"    Max Reach Distance:    {self.max_reach_distance:.1f} px")
        print(f"    Avg Arm Length:        {avg_arm_length:.1f} px")
        print(f"    Normalized Reach:      {(self.max_reach_distance/avg_arm_length if avg_arm_length > 0 else 0):.2f}")
        print(f"    Reach Score:           {reach_score}/100")
        
        print(f"\n📐 BIOMECHANICS:")
        print(f"    Min Hip Angle:         {min_hip_angle:.1f}° (Target: <{self.thresholds['sitnreach_excellent_hip']}°)")
        print(f"    Trunk Flexibility:     {trunk_flexibility_score}/100")
        print(f"    Avg Back Angle:        {avg_back_angle:.1f}° (Ideal: 180°)")
        print(f"    Back Alignment:        {back_alignment_score}/100")
        print(f"    Avg Knee Angle:        {avg_knee_angle:.1f}° (Target: ≥{self.thresholds['sitnreach_knee_valid']}°)")
        print(f"    Knee Validity:         {knee_validity_score}/100")
        
        print(f"\n⚖️ BALANCE & CONTROL:")
        print(f"    Avg Symmetry Error:    {avg_symmetry_error:.1f} px")
        print(f"    Symmetry Score:        {symmetry_score}/100")
        print(f"    Hip Variance:          {hip_variance:.1f} px²")
        print(f"    Stability Score:       {hip_stability_score}/100")
        
        print(f"\n📊 TEST VALIDITY:")
        print(f"    Valid Frames:          {self.valid_frames}/{self.total_frames}")
        print(f"    Accuracy:              {accuracy_score}%")
        print(f"    Test Duration:         {self.sitnreach_test_duration:.1f}s")
        
        print(f"\n" + "="*70)
        print(f"🏆 FINAL PERFORMANCE SCORE: {final_score}/100")
        print(f"⭐ FLEXIBILITY RATING: {flexibility_rating}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
        
        print("\n" + "="*70)
        
        return {
            'exercise': 'sitnreach',
            'final_score': final_score,
            'flexibility_rating': flexibility_rating,
            'primary_metrics': {
                'max_reach_distance': round(self.max_reach_distance, 1),
                'avg_arm_length': round(avg_arm_length, 1),
                'normalized_reach': round(self.max_reach_distance/avg_arm_length if avg_arm_length > 0 else 0, 2),
                'reach_score': reach_score
            },
            'biomechanics': {
                'min_hip_angle': round(min_hip_angle, 1),
                'trunk_flexibility': trunk_flexibility_score,
                'avg_back_angle': round(avg_back_angle, 1),
                'back_alignment': back_alignment_score,
                'avg_knee_angle': round(avg_knee_angle, 1),
                'knee_validity': knee_validity_score
            },
            'balance_control': {
                'avg_symmetry_error': round(avg_symmetry_error, 1),
                'symmetry_score': symmetry_score,
                'hip_variance': round(hip_variance, 1),
                'stability_score': hip_stability_score
            },
            'test_validity': {
                'valid_frames': self.valid_frames,
                'total_frames': self.total_frames,
                'accuracy': accuracy_score,
                'test_duration': round(self.sitnreach_test_duration, 1)
            },
            'feedback': messages
        }
    
    def skipping_metrics(self):
        """
        Comprehensive skipping (jump rope) analysis with all biomechanical metrics.
        Based on skipping.md specifications.
        """
        if not self.exercise:
            self.exercise = 'skipping'
        
        # ---------------------------------------------------------
        # CALCULATE ALL METRICS
        # ---------------------------------------------------------
        
        # Jump metrics
        total_jumps = self.jump_count
        correct_jumps = self.correct_jumps
        accuracy = self._calculate_skip_accuracy()
        
        # Jump characteristics
        avg_jump_height = np.mean(self.jump_heights) if self.jump_heights else 0
        max_jump_height = np.max(self.jump_heights) if self.jump_heights else 0
        min_jump_height = np.min(self.jump_heights) if self.jump_heights else 0
        
        # Tempo metrics
        avg_jump_duration = np.mean(self.jump_durations) if self.jump_durations else 0
        tempo_consistency = self._calculate_tempo_consistency_skip()
        skips_per_second = self._get_skipping_frequency()
        
        # Component scores
        jump_height_score = self._calculate_jump_height_score()
        posture_score = self._calculate_posture_score_skip()
        knee_control_score = self._calculate_knee_control_score()
        
        # Posture details
        avg_back_angle = np.mean(list(self.back_angles_skip)) if self.back_angles_skip else 180
        avg_knee_angle = np.mean(list(self.knee_bend_angles)) if self.knee_bend_angles else 180
        
        # ---------------------------------------------------------
        # FINAL SCORE CALCULATION
        # ---------------------------------------------------------
        # Formula from skipping.md:
        # 0.35 × Accuracy + 0.20 × Tempo + 0.20 × Height + 0.15 × Posture + 0.10 × Knee
        
        final_score = int(
            0.35 * accuracy +
            0.20 * tempo_consistency +
            0.20 * jump_height_score +
            0.15 * posture_score +
            0.10 * knee_control_score
        )
        
        # Performance rating
        if final_score >= 80:
            rating = "EXCELLENT - Professional skipping form"
        elif final_score >= 60:
            rating = "GOOD - Solid technique"
        elif final_score >= 40:
            rating = "AVERAGE - Keep practicing"
        else:
            rating = "NEEDS IMPROVEMENT - Focus on fundamentals"
        
        # ---------------------------------------------------------
        # FEEDBACK GENERATION
        # ---------------------------------------------------------
        messages = []
        
        if accuracy < 70:
            messages.append("⚠️ Work on consistent jump height and landing control")
        elif accuracy >= 90:
            messages.append("✓ Excellent jump accuracy!")
        
        if tempo_consistency < 70:
            messages.append("⚠️ Improve rhythm - try counting or music")
        elif tempo_consistency >= 85:
            messages.append("✓ Great rhythm and consistency!")
        
        if avg_jump_height < self.thresholds['skip_min_jump_height']:
            messages.append("⚠️ Jump higher for better clearance")
        elif avg_jump_height > 50:
            messages.append("✓ Excellent jump height!")
        
        if posture_score < 70:
            messages.append("⚠️ Keep your back straight and upright")
        elif posture_score >= 85:
            messages.append("✓ Perfect upright posture!")
        
        if knee_control_score < 70:
            messages.append("⚠️ Keep knees less bent for efficiency")
        
        if self.double_bounce_count > 0:
            messages.append(f"⚠️ {self.double_bounce_count} double bounces detected - maintain rhythm")
        
        if skips_per_second > 3:
            messages.append("✓ Excellent speed! Advanced level")
        elif skips_per_second > 2:
            messages.append("✓ Good speed - intermediate level")
        
        if not messages:
            messages.append("✓ Keep up the great work!")
        
        # ---------------------------------------------------------
        # DISPLAY REPORT
        # ---------------------------------------------------------
        print("\n" + "="*70)
        print(" "*15 + "SKIPPING (JUMP ROPE) PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n🏃 JUMP STATISTICS:")
        print(f"    Total Jumps:           {total_jumps}")
        print(f"    Correct Jumps:         {correct_jumps}")
        print(f"    Accuracy:              {accuracy}%")
        print(f"    Double Bounces:        {self.double_bounce_count}")
        print(f"    Skips per Second:      {skips_per_second:.2f}")
        
        print(f"\n📏 JUMP CHARACTERISTICS:")
        print(f"    Avg Jump Height:       {avg_jump_height:.1f} px")
        print(f"    Max Jump Height:       {max_jump_height:.1f} px")
        print(f"    Min Jump Height:       {min_jump_height:.1f} px")
        print(f"    Jump Height Score:     {jump_height_score}/100")
        
        print(f"\n⏱️ TEMPO & RHYTHM:")
        print(f"    Avg Jump Duration:     {avg_jump_duration:.3f}s")
        print(f"    Tempo Consistency:     {tempo_consistency}/100")
        if len(self.jump_durations) > 1:
            print(f"    Duration StdDev:       {np.std(self.jump_durations):.3f}s")
        
        print(f"\n📐 POSTURE & FORM:")
        print(f"    Avg Back Angle:        {avg_back_angle:.1f}° (Ideal: {self.thresholds['skip_ideal_back_angle']}°)")
        print(f"    Posture Score:         {posture_score}/100")
        print(f"    Avg Knee Angle:        {avg_knee_angle:.1f}° (Target: >{self.thresholds['skip_max_knee_bend']}°)")
        print(f"    Knee Control Score:    {knee_control_score}/100")
        
        print(f"\n" + "="*70)
        print(f"🏆 FINAL PERFORMANCE SCORE: {final_score}/100")
        print(f"⭐ RATING: {rating}")
        print("="*70)
        
        print(f"\n💡 FEEDBACK:")
        for msg in messages:
            print(f"    {msg}")
        
        print("\n" + "="*70)
        
        return {
            'exercise': 'skipping',
            'final_score': final_score,
            'rating': rating,
            'jump_statistics': {
                'total_jumps': total_jumps,
                'correct_jumps': correct_jumps,
                'accuracy': accuracy,
                'double_bounces': self.double_bounce_count,
                'skips_per_second': round(skips_per_second, 2)
            },
            'jump_characteristics': {
                'avg_height': round(avg_jump_height, 1),
                'max_height': round(max_jump_height, 1),
                'min_height': round(min_jump_height, 1),
                'height_score': jump_height_score
            },
            'tempo_rhythm': {
                'avg_duration': round(avg_jump_duration, 3),
                'tempo_consistency': tempo_consistency,
                'duration_std': round(np.std(self.jump_durations), 3) if len(self.jump_durations) > 1 else 0
            },
            'posture_form': {
                'avg_back_angle': round(avg_back_angle, 1),
                'posture_score': posture_score,
                'avg_knee_angle': round(avg_knee_angle, 1),
                'knee_control_score': knee_control_score
            },
            'feedback': messages
        }
    
    def jumpingjacks_metrics(self):
        """Return detailed jumping jacks metrics."""
        import time
        duration = time.time() - self.jj_start_time if self.jj_start_time else 0
        
        # Calculate individual metric scores
        coordination_score = self._calculate_jj_coordination_score()
        arm_rom_score = self._calculate_jj_arm_rom_score()
        leg_rom_score = self._calculate_jj_leg_rom_score()
        tempo_score = self._calculate_jj_tempo_consistency()
        posture_score = self._calculate_jj_posture_score()
        accuracy = self._calculate_jj_accuracy()
        
        # Calculate overall performance score
        final_score = (
            0.25 * coordination_score +
            0.20 * arm_rom_score +
            0.20 * leg_rom_score +
            0.15 * tempo_score +
            0.10 * posture_score +
            0.10 * accuracy
        )
        
        # Calculate frequency
        frequency = self._get_jj_frequency()
        
        # Compile detailed report
        report = {
            'exercise': 'Jumping Jacks',
            'duration_seconds': round(duration, 1),
            'total_reps': self.jj_rep_count,
            'correct_reps': self.jj_correct_reps,
            'accuracy': round(accuracy * 100, 1),
            'frequency_per_sec': round(frequency, 2),
            
            # Individual metrics
            'coordination_score': round(coordination_score, 2),
            'arm_rom_score': round(arm_rom_score, 2),
            'leg_rom_score': round(leg_rom_score, 2),
            'tempo_consistency': round(tempo_score, 2),
            'posture_score': round(posture_score, 2),
            
            # Detailed measurements
            'avg_arm_spread': round(np.mean(list(self.jj_arm_spreads)), 1) if self.jj_arm_spreads else 0,
            'avg_leg_spread': round(np.mean(list(self.jj_leg_spreads)), 1) if self.jj_leg_spreads else 0,
            'avg_arm_angle': round(np.mean(list(self.jj_arm_angles)), 1) if self.jj_arm_angles else 0,
            'avg_back_angle': round(np.mean(list(self.jj_back_angles)), 1) if self.jj_back_angles else 0,
            'avg_rep_duration': round(np.mean(self.jj_rep_durations), 2) if self.jj_rep_durations else 0,
            
            # Overall performance
            'final_score': round(final_score, 2),
            'rating': self._get_rating(final_score)
        }
        
        # Print formatted report
        print("\n" + "="*60)
        print(f"JUMPING JACKS PERFORMANCE REPORT")
        print("="*60)
        print(f"Duration: {report['duration_seconds']}s")
        print(f"Total Reps: {report['total_reps']} | Correct Reps: {report['correct_reps']}")
        print(f"Accuracy: {report['accuracy']}%")
        print(f"Frequency: {report['frequency_per_sec']} reps/sec")
        print()
        print("METRIC BREAKDOWN:")
        print("-" * 60)
        print(f"Coordination Score: {report['coordination_score']} | {self._get_rating(coordination_score)}")
        print(f"Arm ROM Score: {report['arm_rom_score']} | {self._get_rating(arm_rom_score)}")
        print(f"Leg ROM Score: {report['leg_rom_score']} | {self._get_rating(leg_rom_score)}")
        print(f"Tempo Consistency: {report['tempo_consistency']} | {self._get_rating(tempo_score)}")
        print(f"Posture Score: {report['posture_score']} | {self._get_rating(posture_score)}")
        print()
        print("DETAILED MEASUREMENTS:")
        print("-" * 60)
        print(f"Average Arm Spread: {report['avg_arm_spread']} px")
        print(f"Average Leg Spread: {report['avg_leg_spread']} px")
        print(f"Average Arm Angle: {report['avg_arm_angle']}°")
        print(f"Average Back Angle: {report['avg_back_angle']}° (180° = ideal)")
        print(f"Average Rep Duration: {report['avg_rep_duration']}s")
        print()
        print("=" * 60)
        print(f"FINAL PERFORMANCE SCORE: {report['final_score']} | {report['rating']}")
        print("=" * 60)
        
        return report
    
    def bjump_metrics(self):
        """Return detailed broad jump metrics."""
        import time
        duration = time.time() - self.bjump_start_time if self.bjump_start_time else 0
        
        # Calculate individual metric scores
        distance_score = self._calculate_jump_distance_score_bjump()
        countermovement_score = self._calculate_countermovement_score_bjump()
        arm_swing_score = self._calculate_arm_swing_score_bjump()
        symmetry_score = self._calculate_takeoff_symmetry_score_bjump()
        landing_stability_score = self._calculate_landing_stability_score_bjump()
        accuracy = self._calculate_bjump_accuracy()
        
        # Calculate overall performance score (formula from bjump.md)
        final_score = (
            0.45 * distance_score +
            0.20 * countermovement_score +
            0.15 * arm_swing_score +
            0.10 * symmetry_score +
            0.10 * landing_stability_score
        )
        
        # Compile detailed report
        report = {
            'exercise': 'Standing Broad Jump',
            'duration_seconds': round(duration, 1),
            'total_jumps': self.bjump_jump_count,
            'valid_jumps': self.bjump_valid_jumps,
            'accuracy': round(accuracy * 100, 1),
            
            # Individual metrics
            'distance_score': round(distance_score, 2),
            'countermovement_score': round(countermovement_score, 2),
            'arm_swing_score': round(arm_swing_score, 2),
            'symmetry_score': round(symmetry_score, 2),
            'landing_stability_score': round(landing_stability_score, 2),
            
            # Detailed measurements
            'max_jump_distance': round(self.bjump_max_distance, 1) if self.bjump_max_distance > 0 else 0,
            'avg_jump_distance': round(np.mean(self.bjump_jump_distances), 1) if self.bjump_jump_distances else 0,
            'avg_countermovement': round(np.mean(self.bjump_countermovement_depths), 1) if self.bjump_countermovement_depths else 0,
            'avg_arm_swing': round(np.mean(self.bjump_arm_swing_angles), 1) if self.bjump_arm_swing_angles else 0,
            'avg_symmetry_error': round(np.mean(self.bjump_takeoff_symmetry_errors), 3) if self.bjump_takeoff_symmetry_errors else 0,
            'avg_landing_stability': round(np.mean(self.bjump_landing_stability_errors), 1) if self.bjump_landing_stability_errors else 0,
            
            # Overall performance
            'final_score': round(final_score, 2),
            'rating': self._get_rating(final_score)
        }
        
        # Print formatted report
        print("\n" + "="*60)
        print(f"STANDING BROAD JUMP PERFORMANCE REPORT")
        print("="*60)
        print(f"Duration: {report['duration_seconds']}s")
        print(f"Total Jumps: {report['total_jumps']} | Valid Jumps: {report['valid_jumps']}")
        print(f"Accuracy: {report['accuracy']}%")
        print()
        print("METRIC BREAKDOWN:")
        print("-" * 60)
        print(f"Jump Distance Score: {report['distance_score']} | {self._get_rating(distance_score)}")
        print(f"Countermovement Score: {report['countermovement_score']} | {self._get_rating(countermovement_score)}")
        print(f"Arm Swing Score: {report['arm_swing_score']} | {self._get_rating(arm_swing_score)}")
        print(f"Takeoff Symmetry: {report['symmetry_score']} | {self._get_rating(symmetry_score)}")
        print(f"Landing Stability: {report['landing_stability_score']} | {self._get_rating(landing_stability_score)}")
        print()
        print("DETAILED MEASUREMENTS:")
        print("-" * 60)
        print(f"Max Jump Distance: {report['max_jump_distance']} px")
        print(f"Average Jump Distance: {report['avg_jump_distance']} px")
        print(f"Average Countermovement: {report['avg_countermovement']}° (Good: 90-120°)")
        print(f"Average Arm Swing: {report['avg_arm_swing']}° (Good: 140+°)")
        print(f"Average Symmetry Error: {report['avg_symmetry_error']}s (Good: <0.1s)")
        print(f"Average Landing Stability: {report['avg_landing_stability']}px (Good: <30px)")
        print()
        print("=" * 60)
        print(f"FINAL PERFORMANCE SCORE: {report['final_score']} | {report['rating']}")
        print("=" * 60)
        
        return report
    
    def vjump_metrics(self):
        """Return detailed vertical jump metrics."""
        import time
        duration = time.time() - self.vjump_start_time if self.vjump_start_time else 0
        
        # Calculate individual metric scores
        jump_height_score = self._calculate_jump_height_score_vjump()
        countermovement_score = self._calculate_countermovement_score()
        arm_swing_score = self._calculate_arm_swing_score()
        symmetry_score = self._calculate_takeoff_symmetry_score()
        landing_control_score = self._calculate_landing_control_score()
        accuracy = self._calculate_vjump_accuracy()
        
        # Calculate overall performance score (formula from vjump.md)
        final_score = (
            0.40 * jump_height_score +
            0.20 * countermovement_score +
            0.15 * arm_swing_score +
            0.15 * symmetry_score +
            0.10 * landing_control_score
        )
        
        # Compile detailed report
        report = {
            'exercise': 'Vertical Jump',
            'duration_seconds': round(duration, 1),
            'total_jumps': self.vjump_jump_count,
            'valid_jumps': self.vjump_valid_jumps,
            'accuracy': round(accuracy * 100, 1),
            
            # Individual metrics
            'jump_height_score': round(jump_height_score, 2),
            'countermovement_score': round(countermovement_score, 2),
            'arm_swing_score': round(arm_swing_score, 2),
            'symmetry_score': round(symmetry_score, 2),
            'landing_control_score': round(landing_control_score, 2),
            
            # Detailed measurements
            'max_jump_height': round(max(self.vjump_jump_heights), 1) if self.vjump_jump_heights else 0,
            'avg_jump_height': round(np.mean(self.vjump_jump_heights), 1) if self.vjump_jump_heights else 0,
            'avg_countermovement': round(np.mean(self.vjump_countermovement_depths), 1) if self.vjump_countermovement_depths else 0,
            'avg_arm_swing': round(np.mean(self.vjump_arm_swing_angles), 1) if self.vjump_arm_swing_angles else 0,
            'avg_symmetry_error': round(np.mean(self.vjump_takeoff_symmetry_errors), 3) if self.vjump_takeoff_symmetry_errors else 0,
            'avg_landing_knee': round(np.mean(self.vjump_landing_knee_angles), 1) if self.vjump_landing_knee_angles else 0,
            
            # Overall performance
            'final_score': round(final_score, 2),
            'rating': self._get_rating(final_score)
        }
        
        # Print formatted report
        print("\n" + "="*60)
        print(f"VERTICAL JUMP PERFORMANCE REPORT")
        print("="*60)
        print(f"Duration: {report['duration_seconds']}s")
        print(f"Total Jumps: {report['total_jumps']} | Valid Jumps: {report['valid_jumps']}")
        print(f"Accuracy: {report['accuracy']}%")
        print()
        print("METRIC BREAKDOWN:")
        print("-" * 60)
        print(f"Jump Height Score: {report['jump_height_score']} | {self._get_rating(jump_height_score)}")
        print(f"Countermovement Score: {report['countermovement_score']} | {self._get_rating(countermovement_score)}")
        print(f"Arm Swing Score: {report['arm_swing_score']} | {self._get_rating(arm_swing_score)}")
        print(f"Takeoff Symmetry: {report['symmetry_score']} | {self._get_rating(symmetry_score)}")
        print(f"Landing Control: {report['landing_control_score']} | {self._get_rating(landing_control_score)}")
        print()
        print("DETAILED MEASUREMENTS:")
        print("-" * 60)
        print(f"Max Jump Height: {report['max_jump_height']} px")
        print(f"Average Jump Height: {report['avg_jump_height']} px")
        print(f"Average Countermovement: {report['avg_countermovement']}° (Good: 90-120°)")
        print(f"Average Arm Swing: {report['avg_arm_swing']}° (Good: 140+°)")
        print(f"Average Symmetry Error: {report['avg_symmetry_error']}s (Good: <0.1s)")
        print(f"Average Landing Knee: {report['avg_landing_knee']}° (Good: 120-140°)")
        print()
        print("=" * 60)
        print(f"FINAL PERFORMANCE SCORE: {report['final_score']} | {report['rating']}")
        print("=" * 60)
        
        return report
