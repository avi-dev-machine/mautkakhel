
import cv2
import numpy as np
import argparse
import time
# Assuming PoseCalibrator exists in utils.py as per original context
from utils import PoseCalibrator 
from metrics import PerformanceMetrics
from ai import analyze_exercise_metrics

class ExerciseEvaluator:
    def __init__(self):
        self.calibrator = PoseCalibrator(model_path='yolo11n-pose.pt')
        self.metrics = PerformanceMetrics()
        
        # Exercise State Variables
        self.current_exercise = None
        self.counter = 0
        self.stage = "up"  # Start in UP state for squats
        self.feedback = "Setup"
        self.start_time = None
        
        # Simple moving average for smoothing (last 3 frames)
        self.last_knee_angles = []
        
        # Thresholds - SIMPLE and WORKING
        self.thresholds = {
            'pushup': {'down': 90, 'up': 140, 'form_hip_min': 130},  # More lenient for camera angles
            'squat': {'down': 135, 'up': 155, 'deep': 90},  # Simple fixed thresholds that work
            'situp': {'up': 70, 'down': 20, 'good_crunch': 50},
            'sitnreach': {'excellent_hip': 60, 'average_hip': 80, 'knee_valid': 165},
            'skipping': {'jump_threshold': 30, 'min_height': 20},
            'jumpingjacks': {'arm_open': 150, 'leg_open': 150},
            'vjump': {'min_height': 30, 'good_countermovement': 110},
            'bjump': {'min_distance': 50, 'good_countermovement': 110}
        }
    
    def _smooth_knee_angle(self, knee_angle):
        """Simple 3-frame moving average"""
        if knee_angle is None:
            return None
        
        self.last_knee_angles.append(knee_angle)
        if len(self.last_knee_angles) > 3:
            self.last_knee_angles.pop(0)
        
        return int(np.mean(self.last_knee_angles))
    
    def _smooth_elbow_angle(self, elbow_angle):
        """Simple 3-frame moving average for elbow"""
        if elbow_angle is None:
            return None
        
        if not hasattr(self, 'last_elbow_angles'):
            self.last_elbow_angles = []
        
        self.last_elbow_angles.append(elbow_angle)
        if len(self.last_elbow_angles) > 3:
            self.last_elbow_angles.pop(0)
        
        return int(np.mean(self.last_elbow_angles))

    def _draw_dashboard(self, frame, exercise_name):
        """Draws the exercise statistics overlay."""
        if exercise_name == 'sitnreach':
            # Special dashboard for sit-and-reach
            # Box 1: REACH DISTANCE (max reach in pixels)
            cv2.rectangle(frame, (0, 0), (240, 90), (16, 117, 245), -1)
            cv2.putText(frame, 'MAX REACH (px)', (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            max_distance = int(self.metrics.max_reach_distance) if self.metrics.max_reach_distance > 0 else 0
            cv2.putText(frame, str(max_distance), (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Box 2: CURRENT DISTANCE - positioned to the right without overlap
            cv2.rectangle(frame, (250, 0), (490, 90), (245, 117, 16), -1)
            cv2.putText(frame, 'CURRENT (px)', (260, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            current_distance = int(self.metrics.reach_distances[-1]) if self.metrics.reach_distances else 0
            cv2.putText(frame, str(current_distance), (270, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Feedback bar below
            color = (0, 255, 0) if self.stage == "VALID" else (0, 165, 255)
            cv2.rectangle(frame, (0, 95), (490, 130), (255, 255, 255), -1)
            cv2.putText(frame, self.feedback, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        elif exercise_name == 'skipping':
            # Special dashboard for skipping
            # Box 1: JUMP COUNT
            cv2.rectangle(frame, (0, 0), (180, 90), (117, 245, 16), -1)
            cv2.putText(frame, 'JUMPS', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.metrics.jump_count), (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Box 2: FREQUENCY (skips per second)
            cv2.rectangle(frame, (190, 0), (380, 90), (245, 200, 16), -1)
            cv2.putText(frame, 'SKIPS/SEC', (200, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            frequency = self.metrics._get_skipping_frequency() if hasattr(self.metrics, '_get_skipping_frequency') else 0
            cv2.putText(frame, f"{frequency:.1f}", (210, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Feedback bar below
            color = (0, 255, 0) if self.stage == "AIR" else (200, 200, 200)
            cv2.rectangle(frame, (0, 95), (380, 130), (255, 255, 255), -1)
            cv2.putText(frame, self.feedback, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        elif exercise_name == 'jumpingjacks':
            # Special dashboard for jumping jacks
            # Box 1: REP COUNT
            cv2.rectangle(frame, (0, 0), (180, 90), (200, 117, 245), -1)
            cv2.putText(frame, 'REPS', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.metrics.jj_rep_count), (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Box 2: STATE (OPEN/CLOSED)
            cv2.rectangle(frame, (190, 0), (380, 90), (117, 200, 245), -1)
            cv2.putText(frame, 'STATE', (200, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            state_text = self.metrics.jj_state.upper()
            cv2.putText(frame, state_text, (200, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Feedback bar below
            color = (0, 255, 0) if self.metrics.jj_state == 'open' else (200, 200, 200)
            cv2.rectangle(frame, (0, 95), (380, 130), (255, 255, 255), -1)
            cv2.putText(frame, self.feedback, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        elif exercise_name == 'vjump':
            # Special dashboard for vertical jump
            # Box 1: JUMP COUNT
            cv2.rectangle(frame, (0, 0), (180, 90), (16, 245, 117), -1)
            cv2.putText(frame, 'JUMPS', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.metrics.vjump_jump_count), (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Box 2: MAX JUMP HEIGHT
            cv2.rectangle(frame, (190, 0), (400, 90), (245, 117, 245), -1)
            cv2.putText(frame, 'MAX (px)', (200, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            max_height = int(max(self.metrics.vjump_jump_heights)) if self.metrics.vjump_jump_heights else 0
            cv2.putText(frame, str(max_height), (210, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Feedback bar below
            state_colors = {'standing': (200, 200, 200), 'airborne': (0, 255, 0), 'landing': (255, 165, 0)}
            color = state_colors.get(self.metrics.vjump_state, (200, 200, 200))
            cv2.rectangle(frame, (0, 95), (400, 130), (255, 255, 255), -1)
            cv2.putText(frame, self.feedback, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        elif exercise_name == 'bjump':
            # Special dashboard for broad jump
            # Box 1: JUMP COUNT
            cv2.rectangle(frame, (0, 0), (180, 90), (245, 117, 16), -1)
            cv2.putText(frame, 'JUMPS', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.metrics.bjump_jump_count), (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Box 2: MAX DISTANCE
            cv2.rectangle(frame, (190, 0), (400, 90), (16, 200, 245), -1)
            cv2.putText(frame, 'MAX DIST (px)', (200, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            max_dist = int(self.metrics.bjump_max_distance)
            cv2.putText(frame, str(max_dist), (210, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Feedback bar below
            state_colors_bjump = {'standing': (200, 200, 200), 'airborne': (0, 255, 0), 'landing': (255, 165, 0)}
            color = state_colors_bjump.get(self.metrics.bjump_state, (200, 200, 200))
            cv2.rectangle(frame, (0, 95), (400, 130), (255, 255, 255), -1)
            cv2.putText(frame, self.feedback, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        else:
            # Standard dashboard for other exercises
            cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)
            
            cv2.putText(frame, 'REPS', (15, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.counter), (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(frame, 'STAGE', (65, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(self.stage if self.stage else '-'), (60, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            color = (0, 255, 0) if self.feedback == "Good Form" else (0, 0, 255)
            cv2.rectangle(frame, (0, 73), (225, 103), (255, 255, 255), -1)
            cv2.putText(frame, self.feedback, (15, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        
        return frame

    def process_pushup(self, angles, keypoints):
        """Logic for Pushups - works from multiple camera angles"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        
        # Try to use the side with better visibility
        left_elbow = angles.get('left_elbow')
        right_elbow = angles.get('right_elbow')
        left_hip = angles.get('left_hip')
        right_hip = angles.get('right_hip')
        left_shoulder = angles.get('left_shoulder')
        right_shoulder = angles.get('right_shoulder')
        left_knee = angles.get('left_knee')
        right_knee = angles.get('right_knee')
        left_ankle = angles.get('left_ankle')
        right_ankle = angles.get('right_ankle')
        left_wrist = angles.get('left_wrist')
        right_wrist = angles.get('right_wrist')
        
        # Use whichever side has valid angles
        elbow = None
        hip = None
        
        if left_elbow is not None and left_hip is not None:
            elbow = left_elbow
            hip = left_hip
        elif right_elbow is not None and right_hip is not None:
            elbow = right_elbow
            hip = right_hip
        elif left_elbow is not None and right_hip is not None:
            elbow = left_elbow
            hip = right_hip
        elif right_elbow is not None and left_hip is not None:
            elbow = right_elbow
            hip = left_hip

        if elbow is None:
            self.feedback = "⚠ Position yourself so arms are visible"
            return
        
        # Apply smoothing to elbow angle
        elbow_raw = elbow
        elbow = self._smooth_elbow_angle(elbow)
        if elbow is None:
            return
        
        # Debug: Print angle and stage (only first few to avoid spam)
        if self.counter <= 3:
            print(f"Elbow: {elbow}°, Stage: {self.stage}, Thresholds: UP>{self.thresholds['pushup']['up']}, DOWN<{self.thresholds['pushup']['down']}")
        
        if hip is None:
            # If hip not visible, just track elbow movement
            hip = 160  # Assume good form if not visible

        # Update metrics with angle data - pass all joint angles for comprehensive tracking
        self.metrics.update_angle_data(
            left_elbow, right_elbow,
            left_hip, right_hip,
            left_shoulder, right_shoulder, left_knee, right_knee,
            left_ankle, right_ankle, left_wrist, right_wrist
        )

        # Form check
        if hip < self.thresholds['pushup']['form_hip_min'] - 20:
            self.feedback = "Fix Back!"
        else:
            self.feedback = "Good Form"

        # Initialize stage if needed
        if self.stage is None or self.stage == "":
            # Start in UP position for pushups
            self.stage = "UP" if elbow > 140 else "DOWN"

        # State machine: UP -> DOWN -> UP = 1 rep
        if elbow > self.thresholds['pushup']['up']:  # 140° - arms extended
            if self.stage == "DOWN":
                # Completed a rep
                self.counter += 1
                is_good = (self.feedback == "Good Form")
                self.metrics.record_rep(
                    rep_max=self.thresholds['pushup']['up'],
                    rep_min=elbow,
                    duration_seconds=1.0,
                    is_good_form=is_good
                )
                print(f"✓ Pushup Rep {self.counter} (Elbow: {elbow}°)")
            self.stage = "UP"
        
        elif elbow < self.thresholds['pushup']['down']:  # 90° - arms bent
            if self.stage != "DOWN":
                print(f"  → Going down (Elbow: {elbow}°)")
            self.stage = "DOWN"

    def process_squat(self, angles, keypoints):
        """Logic for Squats - works from multiple camera angles"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        
        # Try both knees
        left_knee = angles.get('left_knee')
        right_knee = angles.get('right_knee')
        
        knee = None
        if left_knee is not None:
            knee = left_knee
        elif right_knee is not None:
            knee = right_knee

        if knee is None:
            self.feedback = "⚠ Move back - show full body"
            return
        
        # Simple smoothing (3-frame average)
        knee = self._smooth_knee_angle(knee)
        if knee is None:
            return
        
        # Get current time
        current_time = time.time()
        
        # Extract additional angles (calculated by utils.py)
        torso_angle = angles.get('torso_angle')
        shin_angle_left = angles.get('shin_angle_left')
        shin_angle_right = angles.get('shin_angle_right')
        
        # Use the most confident shin angle
        left_knee_conf = keypoints[13][2]
        right_knee_conf = keypoints[14][2]
        if left_knee_conf > right_knee_conf:
            shin_angle = shin_angle_left
        else:
            shin_angle = shin_angle_right
        
        # Update squat-specific metrics
        self.metrics.update_squat_data(keypoints, angles, torso_angle, shin_angle, current_time)
        
        # Simple state machine: UP -> DOWN -> UP = 1 rep
        if knee > self.thresholds['squat']['up']:  # 155° - standing
            if self.stage == "DOWN":
                # Completing a rep - record concentric time
                if self.metrics.rep_bottom_time is not None:
                    concentric_time = current_time - self.metrics.rep_bottom_time
                    self.metrics.concentric_times.append(concentric_time)
                    
                    # Record sticking point if tracked
                    if self.metrics.min_velocity_angle is not None:
                        self.metrics.sticking_points.append(self.metrics.min_velocity_angle)
                    
                    # Reset for next rep
                    self.metrics.min_velocity = float('inf')
                    self.metrics.min_velocity_angle = None
                    self.metrics.rep_bottom_time = None
            
            self.stage = "UP"
            self.metrics.current_phase = 'standing'
            
            # Start new rep timer
            if self.metrics.rep_start_time is None:
                self.metrics.rep_start_time = current_time
        
        elif knee < self.thresholds['squat']['down']:  # 135° - squatting
            self.metrics.current_phase = 'descending'
            
            if self.stage == "UP" or self.stage == "up":
                # Reached bottom of squat
                self.stage = "DOWN"
                self.counter += 1
                self.metrics.current_phase = 'bottom'
                
                # Record eccentric time (descent)
                if self.metrics.rep_start_time is not None:
                    eccentric_time = current_time - self.metrics.rep_start_time
                    self.metrics.eccentric_times.append(eccentric_time)
                
                # Mark bottom time for concentric phase
                self.metrics.rep_bottom_time = current_time
                
                # Reset rep start for next rep
                self.metrics.rep_start_time = None
                
                # Record depth
                self.metrics.squat_depths.append(knee)
                
                # Evaluate form
                if knee < self.thresholds['squat']['deep']:
                    self.feedback = "Great Depth!"
                    is_good_form = True
                else:
                    self.feedback = "Go Lower"
                    is_good_form = False
                
                # Check torso angle
                if torso_angle and torso_angle > 45:
                    self.feedback = "Too Much Lean!"
                    is_good_form = False
                
                # Record rep data
                self.metrics.record_rep(
                    rep_max=self.thresholds['squat']['up'],
                    rep_min=knee,
                    duration_seconds=1.0,
                    is_good_form=is_good_form
                )
                
                print(f"Squat Count: {self.counter}")
        
        else:
            # Transitioning between states
            if self.stage == 'UP':
                self.metrics.current_phase = 'descending'
            elif self.stage == 'DOWN':
                self.metrics.current_phase = 'ascending'
            
            if self.stage != "DOWN":
                self.feedback = "Squat"

    def process_situp(self, angles, keypoints):
        """Logic for Sit-ups with comprehensive biomechanical tracking"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        
        current_time = time.time()
        
        # Extract situp-specific angles
        torso_inclination = angles.get('torso_inclination_horizontal')
        hip_flexion = angles.get('hip_flexion_angle')
        
        if torso_inclination is None:
            self.feedback = "Body not visible"
            return
        
        # Update situp-specific metrics
        self.metrics.update_situp_data(keypoints, angles, torso_inclination, hip_flexion, current_time)
        
        # Detect foot lift
        foot_lifted = self.metrics._detect_foot_lift(keypoints)
        
        # Detect neck strain
        neck_distance = self.metrics._detect_neck_strain(keypoints)
        if neck_distance:
            self.metrics.situp_neck_strains.append(neck_distance)
        
        # State machine for sit-up counting
        # State 0: Rest (torso angle < 20°)
        # State 1: Ascending (angle increasing)
        # State 2: Peak (angle > 70° OR hip flexion < 50°)
        # State 3: Descending (angle decreasing back to rest)
        
        if torso_inclination <= self.thresholds['situp']['down']:
            # In rest/down position
            if self.metrics.situp_state == 'descending':
                # Completing a rep
                if self.metrics.situp_peak_time is not None:
                    eccentric_time = current_time - self.metrics.situp_peak_time
                    self.metrics.situp_eccentric_times.append(eccentric_time)
                    self.metrics.situp_peak_time = None
                
                # Calculate momentum score for this rep
                momentum_score = self.metrics._calculate_momentum_score()
                self.metrics.situp_momentum_scores.append(momentum_score)
                
                # Reset shoulder positions for next rep
                self.metrics.shoulder_positions.clear()
                
                # Reset tracking variables for next rep
                self.metrics.max_torso_inclination = 0
                self.metrics.min_hip_flexion = 180
                
            self.metrics.situp_state = 'rest'
            self.stage = "DOWN"
            
            # Start new rep timer
            if self.metrics.situp_rep_start_time is None:
                self.metrics.situp_rep_start_time = current_time
        
        elif torso_inclination >= self.thresholds['situp']['up'] or \
             (hip_flexion is not None and hip_flexion <= self.thresholds['situp']['good_crunch']):
            # At peak position
            if self.metrics.situp_state in ['rest', 'ascending']:
                # Just reached peak
                self.counter += 1
                self.metrics.situp_state = 'peak'
                self.stage = "UP"
                
                # Record concentric time
                if self.metrics.situp_rep_start_time is not None:
                    concentric_time = current_time - self.metrics.situp_rep_start_time
                    self.metrics.situp_concentric_times.append(concentric_time)
                    self.metrics.situp_rep_start_time = None
                
                # Mark peak time for eccentric phase
                self.metrics.situp_peak_time = current_time
                
                # Record peak torso and hip flexion
                self.metrics.situp_torso_inclinations.append(self.metrics.max_torso_inclination)
                self.metrics.situp_hip_flexions.append(self.metrics.min_hip_flexion if hip_flexion else 180)
                
                # Record foot lift for this rep
                self.metrics.situp_foot_lifts.append(1 if foot_lifted else 0)
                
                # Evaluate form
                good_rom = torso_inclination >= self.thresholds['situp']['up']
                good_crunch = hip_flexion is not None and hip_flexion <= self.thresholds['situp']['good_crunch']
                
                if good_rom and good_crunch:
                    self.feedback = "Perfect Rep!"
                    self.metrics.good_reps += 1
                    self.metrics.situp_valid_reps += 1
                    is_good_form = True
                elif good_rom:
                    self.feedback = "Good - Crunch Tighter"
                    self.metrics.good_reps += 1
                    self.metrics.situp_valid_reps += 1
                    is_good_form = True
                else:
                    self.feedback = "Go Higher!"
                    self.metrics.bad_reps += 1
                    self.metrics.situp_short_rom_count += 1
                    is_good_form = False
                
                if foot_lifted:
                    self.feedback += " - Feet Lifted!"
                    is_good_form = False
                
                # Record rep
                self.metrics.record_rep(
                    rep_max=torso_inclination,
                    rep_min=0,
                    duration_seconds=1.0,
                    is_good_form=is_good_form
                )
                
                print(f"Sit-up Count: {self.counter}")
            
            # Stay in peak until descending
            if torso_inclination < self.thresholds['situp']['up'] - 10:
                self.metrics.situp_state = 'descending'
        
        else:
            # Transitioning
            if self.metrics.situp_state == 'rest':
                self.metrics.situp_state = 'ascending'
                self.feedback = "Keep Going Up"
            elif self.metrics.situp_state == 'peak':
                self.metrics.situp_state = 'descending'
                self.feedback = "Controlled Down"

    def process_sitnreach(self, angles, keypoints):
        """Logic for Sit-and-Reach with comprehensive flexibility tracking"""
        import time
        current_time = time.time()
        
        # Initialize start time
        if self.metrics.sitnreach_start_time is None:
            self.metrics.sitnreach_start_time = current_time
        
        # Calculate duration
        self.metrics.sitnreach_test_duration = current_time - self.metrics.sitnreach_start_time
        
        # Extract sit-and-reach specific measurements
        reach_distance = angles.get('reach_distance')
        arm_length = angles.get('arm_length')
        hip_angle = angles.get('sitnreach_hip_angle')
        back_angle = angles.get('sitnreach_back_angle')
        knee_angle = angles.get('sitnreach_knee_angle')
        symmetry_error = angles.get('reach_symmetry')
        
        if reach_distance is None:
            self.feedback = "Position not visible"
            return
        
        # Update sit-and-reach metrics
        self.metrics.update_sitnreach_data(
            keypoints, angles, reach_distance, arm_length,
            hip_angle, back_angle, knee_angle, symmetry_error, current_time
        )
        
        # Real-time feedback based on current measurements
        feedback_parts = []
        
        # Check reach progress
        if reach_distance > self.metrics.max_reach_distance * 0.95:
            feedback_parts.append("MAX REACH!")
        elif reach_distance > self.metrics.max_reach_distance * 0.85:
            feedback_parts.append("Keep Reaching")
        else:
            feedback_parts.append("Stretch Forward")
        
        # Check knee validity
        if knee_angle and knee_angle < self.thresholds['sitnreach']['knee_valid']:
            feedback_parts.append("Straighten Legs!")
        
        # Check symmetry
        if symmetry_error and symmetry_error > 50:
            feedback_parts.append("Balance Both Sides")
        
        # Check hip flexibility
        if hip_angle:
            if hip_angle < self.thresholds['sitnreach']['excellent_hip']:
                feedback_parts.append("Excellent Flex!")
            elif hip_angle < self.thresholds['sitnreach']['average_hip']:
                feedback_parts.append("Good Flex")
            else:
                feedback_parts.append("Bend More")
        
        self.feedback = " | ".join(feedback_parts)
        
        # Update counter with max reach distance
        self.counter = int(self.metrics.max_reach_distance)
        
    def process_skipping(self, angles, keypoints):
        """Logic for Skipping (Jump Rope) with comprehensive biomechanical tracking"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        current_time = time.time()
        
        # Initialize start time
        if self.metrics.skip_start_time is None:
            self.metrics.skip_start_time = current_time
        
        # Update skipping metrics
        self.metrics.update_skipping_data(keypoints, angles, current_time)
        
        # Real-time feedback based on jump state
        feedback_parts = []
        
        if self.metrics.jump_state == 'air':
            self.stage = "AIR"
            feedback_parts.append("IN AIR")
        else:
            self.stage = "GROUND"
            feedback_parts.append("Ready")
        
        # Check posture
        back_angle = angles.get('skip_back_angle')
        if back_angle and abs(back_angle - 180) > 30:
            feedback_parts.append("Stand Upright!")
        
        # Check knee control
        knee_angle = angles.get('skip_knee_angle')
        if knee_angle and knee_angle < 120:
            feedback_parts.append("Less Knee Bend")
        
        # Check jump frequency
        if self.metrics.jump_count > 10:
            frequency = self.metrics._get_skipping_frequency()
            if frequency > 3:
                feedback_parts.append("Excellent Speed!")
            elif frequency < 1.5:
                feedback_parts.append("Speed Up")
        
        self.feedback = " | ".join(feedback_parts)
        
        # Update counter with jump count
        self.counter = self.metrics.jump_count
        self.stage = self.metrics.jump_state.upper()

    def process_jumpingjacks(self, angles, keypoints):
        """Logic for Jumping Jacks with comprehensive biomechanical tracking"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        current_time = time.time()
        
        # Initialize start time
        if self.metrics.jj_start_time is None:
            self.metrics.jj_start_time = current_time
        
        # Update jumping jacks metrics
        self.metrics.update_jumpingjacks_data(keypoints, angles, current_time)
        
        # Real-time feedback based on position
        feedback_parts = []
        
        # Get current measurements
        arm_spread = list(self.metrics.jj_arm_spreads)[-1] if self.metrics.jj_arm_spreads else 0
        leg_spread = list(self.metrics.jj_leg_spreads)[-1] if self.metrics.jj_leg_spreads else 0
        arm_angle = list(self.metrics.jj_arm_angles)[-1] if self.metrics.jj_arm_angles else 0
        
        # Show current state
        state_text = f"STATE: {self.metrics.jj_state.upper()}"
        feedback_parts.append(state_text)
        
        # Show detection status for each component
        arm_status = "✓" if arm_angle >= 135 else ("✗" if arm_angle <= 45 else "~")
        leg_status = "✓" if leg_spread >= 120 else ("✗" if leg_spread <= 100 else "~")
        spread_status = "✓" if arm_spread >= 180 else ("✗" if arm_spread <= 120 else "~")
        
        # Show measurements with status indicators
        feedback_parts.append(f"Angle:{int(arm_angle)}°{arm_status}")
        feedback_parts.append(f"ArmSpread:{int(arm_spread)}px{spread_status}")
        feedback_parts.append(f"LegSpread:{int(leg_spread)}px{leg_status}")
        
        # Position-specific guidance
        if self.metrics.jj_state == 'open':
            if arm_angle < 120:
                feedback_parts.append("↑ RAISE ARMS!")
        else:  # closed
            if arm_angle > 60:
                feedback_parts.append("↓ LOWER ARMS!")
            if leg_spread > 120:
                feedback_parts.append("→← FEET TOGETHER!")
        
        self.feedback = " | ".join(feedback_parts)
        
        # Update counter and stage
        self.counter = self.metrics.jj_rep_count
        self.stage = self.metrics.jj_state.upper()

    def process_vjump(self, angles, keypoints):
        """Logic for Vertical Jump with comprehensive biomechanical tracking"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        current_time = time.time()
        
        # Initialize start time
        if self.metrics.vjump_start_time is None:
            self.metrics.vjump_start_time = current_time
        
        # Update vertical jump metrics
        self.metrics.update_vjump_data(keypoints, angles, current_time)
        
        # Real-time feedback based on state
        feedback_parts = []
        
        # State-specific feedback
        state_text = f"STATE: {self.metrics.vjump_state.upper()}"
        feedback_parts.append(state_text)
        
        if self.metrics.vjump_state == 'standing':
            feedback_parts.append("Ready to jump")
        elif self.metrics.vjump_state == 'preparing':
            knee_angle = angles.get('vjump_countermovement_angle')
            if knee_angle:
                feedback_parts.append(f"Knee bend: {knee_angle}°")
                if knee_angle < 100:
                    feedback_parts.append("Good depth!")
            feedback_parts.append("Swing arms up!")
        elif self.metrics.vjump_state == 'airborne':
            feedback_parts.append("In the air!")
            if self.metrics.vjump_min_ankle_y:
                current_height = self.metrics.vjump_ground_y - self.metrics.vjump_min_ankle_y
                feedback_parts.append(f"Height: {int(current_height)}px")
        elif self.metrics.vjump_state == 'landing':
            feedback_parts.append("Prepare for landing")
            landing_knee = angles.get('vjump_landing_knee_angle')
            if landing_knee:
                if landing_knee < 100:
                    feedback_parts.append("Too much bend!")
                elif landing_knee > 160:
                    feedback_parts.append("Land softer!")
                else:
                    feedback_parts.append("Good landing!")
        
        # Show jump performance
        if self.metrics.vjump_jump_count > 0:
            max_jump = int(max(self.metrics.vjump_jump_heights))
            feedback_parts.append(f"Best: {max_jump}px")
        
        self.feedback = " | ".join(feedback_parts)
        
        # Update counter and stage with jump count
        self.counter = self.metrics.vjump_jump_count
        self.stage = self.metrics.vjump_state.upper()

    def process_bjump(self, angles, keypoints):
        """Logic for Broad Jump with robust state machine"""
        # Validate keypoints array
        if keypoints is None or len(keypoints) < 17:
            self.feedback = "Body not detected"
            return
        current_time = time.time()
        
        # Initialize start time
        if self.metrics.bjump_start_time is None:
            self.metrics.bjump_start_time = current_time
        
        # Update broad jump metrics using robust state machine
        self.metrics.update_bjump_data(keypoints, angles, current_time)
        
        # Real-time feedback with visual clarity
        feedback_parts = []
        
        # State display with emoji indicators
        state_icons = {
            'standing': '🟢 READY',
            'airborne': '🚀 FLYING',
            'landing': '🎯 LANDING'
        }
        state_display = state_icons.get(self.metrics.bjump_state, self.metrics.bjump_state.upper())
        feedback_parts.append(state_display)
        
        # Real-time distance during jump
        if self.metrics.bjump_state == 'airborne':
            if self.metrics.bjump_max_x and self.metrics.bjump_start_x:
                current_dist = abs(self.metrics.bjump_max_x - self.metrics.bjump_start_x)
                feedback_parts.append(f"Distance: {int(current_dist)}px")
        
        # Performance summary
        if self.metrics.bjump_jump_count > 0:
            feedback_parts.append(f"Jumps: {self.metrics.bjump_jump_count}")
            
            # Show best distance
            if self.metrics.bjump_max_distance > 0:
                feedback_parts.append(f"Best: {int(self.metrics.bjump_max_distance)}px")
            
            # Show last jump
            if self.metrics.bjump_jump_distances:
                last_dist = self.metrics.bjump_jump_distances[-1]
                feedback_parts.append(f"Last: {int(last_dist)}px")
        
        self.feedback = " | ".join(feedback_parts)
        
        # Update counter and stage
        self.counter = self.metrics.bjump_jump_count
        self.stage = self.metrics.bjump_state.upper()

    def _resize_for_display(self, frame, max_width=1280, max_height=720):
        h, w = frame.shape[:2]
        if w == 0 or h == 0: return frame
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def run(self, exercise_type='pushup', source='0', save_output=True):
        if source == '0' or source == 0:
            cap = cv2.VideoCapture(0)
            input_is_camera = True
            print("Using webcam...")
        else:
            cap = cv2.VideoCapture(source)
            input_is_camera = False
            print(f"Loading video from: {source}")
            
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return

        writer = None
        if save_output and not input_is_camera:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            import os
            filename, ext = os.path.splitext(source)
            output_path = f"{filename}_processed.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")

        self.current_exercise = exercise_type
        self.metrics.exercise = exercise_type
        self.start_time = time.time()
        print(f"\nStarting {exercise_type.upper()} analysis on {source}...")
        print("Press 'Q' to quit\n")
        
        cv2.namedWindow('AI Trainer', cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame, keypoints, angles = self.calibrator.process_frame(frame, show_angles_panel=False)
            
            if keypoints is not None:
                if exercise_type == 'pushup':
                    self.process_pushup(angles, keypoints)
                elif exercise_type == 'squat':
                    self.process_squat(angles, keypoints)
                elif exercise_type == 'situp':
                    self.process_situp(angles, keypoints)
                elif exercise_type == 'sitnreach':
                    self.process_sitnreach(angles, keypoints)
                elif exercise_type == 'skipping':
                    self.process_skipping(angles, keypoints)
                elif exercise_type == 'jumpingjacks':
                    self.process_jumpingjacks(angles, keypoints)
                elif exercise_type == 'vjump':
                    self.process_vjump(angles, keypoints)
                elif exercise_type == 'bjump':
                    self.process_bjump(angles, keypoints)
            
            frame = self._draw_dashboard(frame, exercise_type)
            display_frame = self._resize_for_display(frame)
            cv2.resizeWindow('AI Trainer', display_frame.shape[1], display_frame.shape[0])
            cv2.imshow('AI Trainer', display_frame)
            
            if writer: writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print("Analysis Complete!")
        print("="*50)
        
        # ---------------------------------------------------------
        # CALLING SPECIFIC METRICS FUNCTIONS AND SAVING RESULTS
        # ---------------------------------------------------------
        import json
        from datetime import datetime
        
        result = None
        if exercise_type == 'pushup':
            result = self.metrics.pushup_metrics()
        elif exercise_type == 'squat':
            result = self.metrics.squat_metrics()
        elif exercise_type == 'situp':
            result = self.metrics.situp_metrics()
        elif exercise_type == 'sitnreach':
            result = self.metrics.sitnreach_metrics()
        elif exercise_type == 'skipping':
            result = self.metrics.skipping_metrics()
        elif exercise_type == 'jumpingjacks':
            result = self.metrics.jumpingjacks_metrics()
        elif exercise_type == 'vjump':
            result = self.metrics.vjump_metrics()
        elif exercise_type == 'bjump':
            result = self.metrics.bjump_metrics()
        
        if result:
            try:
                # Add timestamp to the result
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open('exercise_metrics.txt', 'a') as f:
                    f.write(json.dumps(result, indent=2, default=lambda o: float(o)))
                print("\n✓ Metrics data saved to exercise_metrics.txt")
            except Exception as e:
                print(f"\n⚠ Could not save metrics data: {e}")

# # CLI Helpers
# def display_menu():
#     print("\n" + "="*50)
#     print("     AI EXERCISE TRAINER")
#     print("="*50)
#     print("\nSelect Exercise:")
#     print("1. Push-ups")
#     print("2. Squats")
#     print("3. Sit-ups")
#     print("4. Sit-and-Reach (Flexibility)")
#     print("5. Skipping (Jump Rope)")
#     print("6. Jumping Jacks")
#     print("7. Vertical Jump")
#     print("8. Broad Jump (Horizontal)")
#     print("\nSelect Input Source:")
#     print("1. Webcam (Live)")
#     print("2. Video File")
#     print("\nSave output video? (y/n)")

# def get_exercise_type():
#     while True:
#         display_menu()
#         choice = input("\nEnter Exercise (1-8): ").strip()
#         if choice == '1': return 'pushup'
#         elif choice == '2': return 'squat'
#         elif choice == '3': return 'situp'
#         elif choice == '4': return 'sitnreach'
#         elif choice == '5': return 'skipping'
#         elif choice == '6': return 'jumpingjacks'
#         elif choice == '7': return 'vjump'
#         elif choice == '8': return 'bjump'
#         else: print("Invalid choice. Try again.")

# def get_source():
#     while True:
#         choice = input("Enter Source (1 for Webcam, 2 for Video): ").strip()
#         if choice == '1': return '0'
#         elif choice == '2':
#             video_path = input("Enter video file path/name: ").strip()
#             if video_path: return video_path
#             else: print("Invalid path!")
#         else: print("Invalid choice! Please enter 1 or 2")

# def get_save_option():
#     while True:
#         choice = input("Save output video? (y/n): ").strip().lower()
#         if choice == 'y': return True
#         elif choice == 'n': return False
#         else: print("Invalid choice!")

# if __name__ == "__main__":
#     exercise_type = get_exercise_type()
#     source = get_source()
#     save_output = get_save_option()
    
#     trainer = ExerciseEvaluator()
#     trainer.run(
#         exercise_type=exercise_type,
#         source=source,
#         save_output=save_output
#     )
def display_menu(completed_exercises):
    print("\n" + "=" * 50)
    print("     AI EXERCISE TRAINER")
    print("=" * 50)
    print("\nSelect Exercise:")
    
    exercises = {
        "1": ("Push-ups", "pushup"),
        "2": ("Squats", "squat"),
        "3": ("Sit-ups", "situp"),
        "4": ("Sit-and-Reach (Flexibility)", "sitnreach"),
        "5": ("Skipping (Jump Rope)", "skipping"),
        "6": ("Jumping Jacks", "jumpingjacks"),
        "7": ("Vertical Jump", "vjump"),
        "8": ("Broad Jump (Horizontal)", "bjump"),
    }
    
    for key, (name, ex_type) in exercises.items():
        status = " ✓ COMPLETED" if ex_type in completed_exercises else ""
        print(f"{key}. {name}{status}")
    
    print("\n9. View AI Analysis")
    print("0. Exit")
    
    return exercises


def get_exercise_type(completed_exercises):
    while True:
        exercises = display_menu(completed_exercises)
        choice = input("\nEnter Exercise (1-8), 9 for AI Analysis, or 0 to Exit: ").strip()
        
        if choice == "0":
            return None, "exit"
        elif choice == "9":
            return None, "analyze"
        elif choice in exercises:
            exercise_name, exercise_type = exercises[choice]
            if exercise_type in completed_exercises:
                redo = input(f"\n{exercise_name} already completed. Do it again? (y/n): ").strip().lower()
                if redo == "y":
                    return exercise_type, "normal"
                else:
                    continue
            return exercise_type, "normal"
        else:
            print("Invalid choice. Try again.")


def get_source():
    while True:
        choice = input("Enter Source (1 for Webcam, 2 for Video): ").strip()
        if choice == "1":
            return "0"
        elif choice == "2":
            video_path = input("Enter video file path/name: ").strip()
            if video_path:
                return video_path
            else:
                print("Invalid path!")
        else:
            print("Invalid choice! Please enter 1 or 2")


def get_save_option():
    while True:
        choice = input("Save output video? (y/n): ").strip().lower()
        if choice == "y":
            return True
        elif choice == "n":
            return False
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    from ai import analyze_exercise_metrics
    
    completed_exercises = set()
    
    while True:
        exercise_type, action = get_exercise_type(completed_exercises)
        
        if action == "exit":
            print("\nExiting... Thank you for using AI Exercise Trainer!")
            break
        elif action == "analyze":
            print("\n" + "=" * 50)
            print("     AI ANALYSIS")
            print("=" * 50)
            try:
                analysis = analyze_exercise_metrics()
                print("\n" + analysis)
            except Exception as e:
                print(f"\nError generating analysis: {e}")
            input("\nPress Enter to continue...")
            continue
        
        source = get_source()
        save_output = get_save_option()
        
        trainer = ExerciseEvaluator()
        trainer.run(
            exercise_type=exercise_type,
            source=source,
            save_output=save_output
        )
        # print(
        #     f"\nSelected Exercise: {exercise_type}, Source: {source}, Save Output: {save_output}"
        # )
        
        # Mark as completed
        completed_exercises.add(exercise_type)
        print(f"\n✓ {exercise_type.upper()} marked as completed!")
        input("\nPress Enter to continue...")
