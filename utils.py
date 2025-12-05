
import cv2
import numpy as np
import math
from ultralytics import YOLO
import time


class PoseCalibrator:
    """
    A comprehensive pose calibration system for exercise tracking.
    Supports YOLO11 pose estimation with 17 keypoints.
    """
    
    def __init__(self, model_path='yolo11n-pose.pt', calibration_time=0):
        self.model = YOLO(model_path)
        self.calibrated = False
        self.calibration_start_time = None
        self.calibration_time = calibration_time
        self.calibration_elapsed = 0
        
        self.keypoint_names = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
        }
        
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        self.joint_angles = {
            'left_elbow': (5, 7, 9),
            'right_elbow': (6, 8, 10),
            'left_shoulder': (7, 5, 11),
            'right_shoulder': (8, 6, 12),
            'left_hip': (5, 11, 13),
            'right_hip': (6, 12, 14),
            'left_knee': (11, 13, 15),
            'right_knee': (12, 14, 16),
        }
        
        self.colors = {
            'keypoint': (0, 255, 0),
            'keypoint_border': (255, 0, 255),
            'skeleton': (255, 255, 255),
            'angle_text': (0, 255, 255),
            'calibration_text': (0, 255, 0),
            'warning_text': (0, 0, 255)
        }
    
    
    def detect_pose(self, frame):
        """Detect pose keypoints using YOLOv8"""
        results = self.model(frame, verbose=False)
        
        if len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            return keypoints
        
        return None
    
    
    def calculate_angle(self, pt1, pt2, pt3):
        """Calculate angle between three points (in degrees)"""
        pt1, pt2, pt3 = np.array(pt1), np.array(pt2), np.array(pt3)
        
        v1 = pt1 - pt2
        v2 = pt3 - pt2
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    
    def get_all_joint_angles(self, keypoints):
        """Calculate angles for all major joints"""
        angles = {}
        
        # Safety check
        if keypoints is None or len(keypoints) < 17:
            return {key: None for key in self.joint_angles.keys()}
        
        for joint_name, (idx1, idx2, idx3) in self.joint_angles.items():
            try:
                if (keypoints[idx1][2] > 0.5 and 
                    keypoints[idx2][2] > 0.5 and 
                    keypoints[idx3][2] > 0.5):
                    
                    pt1 = tuple(keypoints[idx1][:2])
                    pt2 = tuple(keypoints[idx2][:2])
                    pt3 = tuple(keypoints[idx3][:2])
                    
                    angle = self.calculate_angle(pt1, pt2, pt3)
                    angles[joint_name] = angle
                else:
                    angles[joint_name] = None
            except (IndexError, TypeError):
                angles[joint_name] = None
        
        # Calculate torso angle (relative to vertical)
        angles['torso_angle'] = self.calculate_torso_angle(keypoints)
        
        # Calculate shin angle (relative to vertical)
        angles['shin_angle_left'] = self.calculate_shin_angle(keypoints, side='left')
        angles['shin_angle_right'] = self.calculate_shin_angle(keypoints, side='right')
        
        # Calculate situp-specific angles (relative to horizontal)
        angles['torso_inclination_horizontal'] = self.calculate_torso_inclination_horizontal(keypoints)
        angles['hip_flexion_angle'] = self.calculate_hip_flexion_angle(keypoints)
        
        # Calculate sit-and-reach specific measurements
        angles['reach_distance'] = self.calculate_reach_distance(keypoints)
        angles['arm_length'] = self.calculate_arm_length(keypoints)
        angles['sitnreach_hip_angle'] = self.calculate_sitnreach_hip_angle(keypoints)
        angles['sitnreach_back_angle'] = self.calculate_sitnreach_back_angle(keypoints)
        angles['sitnreach_knee_angle'] = self.calculate_sitnreach_knee_angle(keypoints)
        angles['reach_symmetry'] = self.calculate_reach_symmetry(keypoints)
        
        # Calculate skipping-specific measurements
        angles['skip_back_angle'] = self.calculate_skip_back_angle(keypoints)
        angles['skip_knee_angle'] = self.calculate_skip_knee_angle(keypoints)
        
        # Calculate jumping jacks-specific measurements
        angles['jj_arm_angle'] = self.calculate_jj_arm_angle(keypoints)
        angles['jj_back_angle'] = self.calculate_jj_back_angle(keypoints)
        
        # Calculate vertical jump-specific measurements
        angles['vjump_countermovement_angle'] = self.calculate_vjump_countermovement_angle(keypoints)
        angles['vjump_arm_swing_angle'] = self.calculate_vjump_arm_swing_angle(keypoints)
        angles['vjump_landing_knee_angle'] = self.calculate_vjump_landing_knee_angle(keypoints)
        
        # Calculate broad jump-specific measurements (same as vertical jump)
        angles['bjump_countermovement_angle'] = self.calculate_vjump_countermovement_angle(keypoints)
        angles['bjump_arm_swing_angle'] = self.calculate_vjump_arm_swing_angle(keypoints)
        
        return angles
    
    def calculate_torso_angle(self, keypoints):
        """Calculate torso inclination angle relative to vertical axis"""
        # Use shoulder and hip to define torso line
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Check confidence
        if (left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5 and
            left_hip[2] > 0.5 and right_hip[2] > 0.5):
            
            # Calculate midpoints
            shoulder_mid = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                    (left_shoulder[1] + right_shoulder[1]) / 2])
            hip_mid = np.array([(left_hip[0] + right_hip[0]) / 2,
                               (left_hip[1] + right_hip[1]) / 2])
            
            # Vector from hip to shoulder
            torso_vector = shoulder_mid - hip_mid
            
            # Vertical vector (pointing up in image coordinates: negative y)
            vertical_vector = np.array([0, -1])
            
            # Calculate angle
            torso_norm = np.linalg.norm(torso_vector)
            if torso_norm == 0:
                return None
            
            cos_angle = np.dot(torso_vector, vertical_vector) / torso_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return int(round(angle))
        
        return None
    
    def calculate_shin_angle(self, keypoints, side='left'):
        """Calculate shin angle relative to vertical axis"""
        if side == 'left':
            knee = keypoints[13]
            ankle = keypoints[15]
        else:
            knee = keypoints[14]
            ankle = keypoints[16]
        
        if knee[2] > 0.5 and ankle[2] > 0.5:
            # Vector from ankle to knee
            shin_vector = np.array([knee[0] - ankle[0], knee[1] - ankle[1]])
            
            # Vertical vector (pointing up)
            vertical_vector = np.array([0, -1])
            
            # Calculate angle
            shin_norm = np.linalg.norm(shin_vector)
            if shin_norm == 0:
                return None
            
            cos_angle = np.dot(shin_vector, vertical_vector) / shin_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return int(round(angle))
        
        return None
    
    def calculate_torso_inclination_horizontal(self, keypoints):
        """Calculate torso inclination angle relative to horizontal (for situps)."""
        # Use shoulder and hip to define torso line
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Check confidence
        if (left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5 and
            left_hip[2] > 0.5 and right_hip[2] > 0.5):
            
            # Calculate midpoints
            shoulder_mid = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                    (left_shoulder[1] + right_shoulder[1]) / 2])
            hip_mid = np.array([(left_hip[0] + right_hip[0]) / 2,
                               (left_hip[1] + right_hip[1]) / 2])
            
            # Calculate angle using atan2 (relative to horizontal)
            dy = shoulder_mid[1] - hip_mid[1]  # Negative because y increases downward
            dx = shoulder_mid[0] - hip_mid[0]
            
            # Calculate angle from horizontal
            angle = np.degrees(np.arctan2(-dy, dx))
            
            # Return absolute angle (0-90 range typically)
            return int(round(abs(angle)))
        
        return None
    
    def calculate_hip_flexion_angle(self, keypoints):
        """Calculate hip flexion angle (shoulder-hip-knee) for situp crunch measurement."""
        # Use the more confident side
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        
        # Choose the side with better confidence
        left_conf = min(left_shoulder[2], left_hip[2], left_knee[2])
        right_conf = min(right_shoulder[2], right_hip[2], right_knee[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array(left_shoulder[:2])
            hip = np.array(left_hip[:2])
            knee = np.array(left_knee[:2])
        elif right_conf > 0.5:
            shoulder = np.array(right_shoulder[:2])
            hip = np.array(right_hip[:2])
            knee = np.array(right_knee[:2])
        else:
            return None
        
        # Calculate angle at hip vertex
        v1 = shoulder - hip
        v2 = knee - hip
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_reach_distance(self, keypoints):
        """Calculate forward reach distance (wrist_x - ankle_x)."""
        # Use both wrists and take average
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        wrist_conf = max(left_wrist[2], right_wrist[2])
        ankle_conf = max(left_ankle[2], right_ankle[2])
        
        if wrist_conf > 0.5 and ankle_conf > 0.5:
            # Average wrist and ankle positions
            avg_wrist_x = (left_wrist[0] + right_wrist[0]) / 2
            avg_ankle_x = (left_ankle[0] + right_ankle[0]) / 2
            
            reach_distance = avg_wrist_x - avg_ankle_x
            return reach_distance
        
        return None
    
    def calculate_arm_length(self, keypoints):
        """Calculate arm length (shoulder to wrist distance)."""
        # Use the more confident side
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        left_conf = min(left_shoulder[2], left_wrist[2])
        right_conf = min(right_shoulder[2], right_wrist[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array(left_shoulder[:2])
            wrist = np.array(left_wrist[:2])
        elif right_conf > 0.5:
            shoulder = np.array(right_shoulder[:2])
            wrist = np.array(right_wrist[:2])
        else:
            return None
        
        arm_length = np.linalg.norm(wrist - shoulder)
        return arm_length
    
    def calculate_sitnreach_hip_angle(self, keypoints):
        """Calculate hip angle for sit-and-reach (shoulder-hip-knee)."""
        # Same as hip_flexion_angle but used for sit-and-reach context
        return self.calculate_hip_flexion_angle(keypoints)
    
    def calculate_sitnreach_back_angle(self, keypoints):
        """Calculate back alignment angle (shoulder-hip-knee for spine straightness)."""
        # Use the more confident side
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        
        left_conf = min(left_shoulder[2], left_hip[2], left_knee[2])
        right_conf = min(right_shoulder[2], right_hip[2], right_knee[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array(left_shoulder[:2])
            hip = np.array(left_hip[:2])
            knee = np.array(left_knee[:2])
        elif right_conf > 0.5:
            shoulder = np.array(right_shoulder[:2])
            hip = np.array(right_hip[:2])
            knee = np.array(right_knee[:2])
        else:
            return None
        
        # Calculate angle at hip
        v1 = shoulder - hip
        v2 = knee - hip
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_sitnreach_knee_angle(self, keypoints):
        """Calculate knee angle for validity check (hip-knee-ankle)."""
        # Use the more confident side
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        left_conf = min(left_hip[2], left_knee[2], left_ankle[2])
        right_conf = min(right_hip[2], right_knee[2], right_ankle[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            hip = np.array(left_hip[:2])
            knee = np.array(left_knee[:2])
            ankle = np.array(left_ankle[:2])
        elif right_conf > 0.5:
            hip = np.array(right_hip[:2])
            knee = np.array(right_knee[:2])
            ankle = np.array(right_ankle[:2])
        else:
            return None
        
        # Calculate angle at knee
        v1 = hip - knee
        v2 = ankle - knee
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_reach_symmetry(self, keypoints):
        """Calculate reach symmetry error (left vs right wrist difference)."""
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        if left_wrist[2] > 0.5 and right_wrist[2] > 0.5:
            symmetry_error = abs(left_wrist[0] - right_wrist[0])
            return symmetry_error
        
        return None
    
    
    def draw_keypoints(self, frame, keypoints, min_confidence=0.5):
        """Draw keypoints on frame"""
        # Safety check
        if keypoints is None or len(keypoints) < 17:
            return frame
            
        for i, (x, y, conf) in enumerate(keypoints):
            try:
                if conf > min_confidence:
                    x, y = int(x), int(y)
                    cv2.circle(frame, (x, y), 6, self.colors['keypoint'], -1)
                    cv2.circle(frame, (x, y), 9, self.colors['keypoint_border'], 2)
            except (IndexError, TypeError, ValueError):
                continue
        
        return frame
    
    
    def draw_skeleton(self, frame, keypoints, min_confidence=0.5):
        """Draw skeleton connections"""
        # Safety check
        if keypoints is None or len(keypoints) < 17:
            return frame
            
        for idx1, idx2 in self.skeleton_connections:
            try:
                if (keypoints[idx1][2] > min_confidence and 
                    keypoints[idx2][2] > min_confidence):
                    
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    
                    cv2.line(frame, pt1, pt2, self.colors['skeleton'], 3)
            except (IndexError, TypeError):
                continue
        
        return frame
    
    
    def draw_joint_angles(self, frame, keypoints, angles):
        """Draw joint angles on frame"""
        # Safety check
        if keypoints is None or len(keypoints) < 17 or not angles:
            return frame
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Skip angles that are not traditional joint angles (torso, shin, situp angles, sitnreach metrics, skipping metrics, jumping jacks metrics, vjump metrics, bjump metrics)
        skip_angles = ['torso_angle', 'shin_angle_left', 'shin_angle_right', 
                      'torso_inclination_horizontal', 'hip_flexion_angle',
                      'reach_distance', 'arm_length', 'sitnreach_hip_angle',
                      'sitnreach_back_angle', 'sitnreach_knee_angle', 'reach_symmetry',
                      'skip_back_angle', 'skip_knee_angle',
                      'jj_arm_angle', 'jj_back_angle',
                      'vjump_countermovement_angle', 'vjump_arm_swing_angle', 'vjump_landing_knee_angle',
                      'bjump_countermovement_angle', 'bjump_arm_swing_angle']
        
        for joint_name, angle in angles.items():
            try:
                if angle is not None and joint_name not in skip_angles and joint_name in self.joint_angles:
                    vertex_idx = self.joint_angles[joint_name][1]
                    
                    if keypoints[vertex_idx][2] > 0.5:
                        x, y = int(keypoints[vertex_idx][0]), int(keypoints[vertex_idx][1])
                        
                        text = f"{angle}"
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        
                        offset_x, offset_y = 15, -10
                        text_x, text_y = x + offset_x, y + offset_y
                        
                        cv2.rectangle(frame, 
                                    (text_x - 2, text_y - text_size[1] - 2),
                                    (text_x + text_size[0] + 2, text_y + 2),
                                    (0, 0, 0), -1)
                        
                        cv2.putText(frame, text, (text_x, text_y),
                                  font, font_scale, self.colors['angle_text'], thickness)
            except (IndexError, KeyError, TypeError):
                continue
        
        return frame
    
    
    def draw_calibration_status(self, frame, keypoints):
        """Draw calibration status"""
        h, w = frame.shape[:2]
        
        if self.calibrated:
            text = f"CALIBRATION SUCCESSFUL!"
            color = self.colors['calibration_text']
            font_scale = 0.8
        elif keypoints is None:
            text = "Please stand in front of the camera"
            color = self.colors['warning_text']
            font_scale = 0.8
            self.calibration_start_time = None
            self.calibration_elapsed = 0
        else:
            if self.calibration_start_time is None:
                self.calibration_start_time = time.time()
            
            self.calibration_elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.calibration_time - self.calibration_elapsed)
            
            # Avoid division by zero
            if self.calibration_time > 0:
                progress = min(100, int((self.calibration_elapsed / self.calibration_time) * 100))
            else:
                progress = 100
            
            if self.calibration_elapsed >= self.calibration_time:
                text = f"CALIBRATION SUCCESSFUL!"
                color = self.colors['calibration_text']
                font_scale = 0.8
                self.calibrated = True
            else:
                text = f"Calibrating... {progress}% ({int(remaining)}s remaining)"
                color = (0, 255, 255)
                font_scale = 0.7
        
        banner_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 35
        
        cv2.putText(frame, text, (text_x, text_y),
                   font, font_scale, color, thickness)
        
        return frame
    
    
    def process_frame(self, frame, show_angles_panel=True):
        """Process single frame: detect pose, draw skeleton"""
        keypoints = self.detect_pose(frame)
        angles = {}
        
        if keypoints is not None:
            angles = self.get_all_joint_angles(keypoints)
            
            frame = self.draw_skeleton(frame, keypoints)
            frame = self.draw_keypoints(frame, keypoints)
            frame = self.draw_joint_angles(frame, keypoints, angles)
        
        frame = self.draw_calibration_status(frame, keypoints)
        
        return frame, keypoints, angles
    
    def calculate_skip_back_angle(self, keypoints):
        """Calculate back angle for skipping posture (shoulder-hip-knee)."""
        # Use the more confident side
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        
        left_conf = min(left_shoulder[2], left_hip[2], left_knee[2])
        right_conf = min(right_shoulder[2], right_hip[2], right_knee[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array(left_shoulder[:2])
            hip = np.array(left_hip[:2])
            knee = np.array(left_knee[:2])
        elif right_conf > 0.5:
            shoulder = np.array(right_shoulder[:2])
            hip = np.array(right_hip[:2])
            knee = np.array(right_knee[:2])
        else:
            return None
        
        # Calculate angle at hip
        v1 = shoulder - hip
        v2 = knee - hip
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_skip_knee_angle(self, keypoints):
        """Calculate knee angle for skipping (hip-knee-ankle)."""
        # Use the more confident side
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        left_conf = min(left_hip[2], left_knee[2], left_ankle[2])
        right_conf = min(right_hip[2], right_knee[2], right_ankle[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            hip = np.array(left_hip[:2])
            knee = np.array(left_knee[:2])
            ankle = np.array(left_ankle[:2])
        elif right_conf > 0.5:
            hip = np.array(right_hip[:2])
            knee = np.array(right_knee[:2])
            ankle = np.array(right_ankle[:2])
        else:
            return None
        
        # Calculate angle at knee
        v1 = hip - knee
        v2 = ankle - knee
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_jj_arm_angle(self, keypoints):
        """Calculate arm elevation angle for jumping jacks using shoulder joint angle (torso-shoulder-wrist)."""
        # Use the more confident side
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        left_conf = min(left_shoulder[2], left_wrist[2], left_hip[2])
        right_conf = min(right_shoulder[2], right_wrist[2], right_hip[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array(left_shoulder[:2])
            wrist = np.array(left_wrist[:2])
            hip = np.array(left_hip[:2])
        elif right_conf > 0.5:
            shoulder = np.array(right_shoulder[:2])
            wrist = np.array(right_wrist[:2])
            hip = np.array(right_hip[:2])
        else:
            return None
        
        # Calculate shoulder joint angle: hip-shoulder-wrist
        # This is the angle formed at the shoulder between torso and arm
        v1 = hip - shoulder  # Vector from shoulder to hip (torso direction)
        v2 = wrist - shoulder  # Vector from shoulder to wrist (arm direction)
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Result:
        # ~0° when arm is down along body (closed position)
        # ~90° when arm is horizontal (transitioning)
        # ~180° when arm is raised overhead (open position)
        return int(round(angle))
    
    def calculate_jj_back_angle(self, keypoints):
        """Calculate back angle for jumping jacks posture (shoulder-hip-knee)."""
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        
        left_conf = min(left_shoulder[2], left_hip[2], left_knee[2])
        right_conf = min(right_shoulder[2], right_hip[2], right_knee[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array([left_shoulder[0], left_shoulder[1]])
            hip = np.array([left_hip[0], left_hip[1]])
            knee = np.array([left_knee[0], left_knee[1]])
        elif right_conf > 0.5:
            shoulder = np.array([right_shoulder[0], right_shoulder[1]])
            hip = np.array([right_hip[0], right_hip[1]])
            knee = np.array([right_knee[0], right_knee[1]])
        else:
            return None
        
        v1 = shoulder - hip
        v2 = knee - hip
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_vjump_countermovement_angle(self, keypoints):
        """Calculate knee angle during countermovement (hip-knee-ankle)."""
        # Use the more confident side
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        left_conf = min(left_hip[2], left_knee[2], left_ankle[2])
        right_conf = min(right_hip[2], right_knee[2], right_ankle[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            hip = np.array([left_hip[0], left_hip[1]])
            knee = np.array([left_knee[0], left_knee[1]])
            ankle = np.array([left_ankle[0], left_ankle[1]])
        elif right_conf > 0.5:
            hip = np.array([right_hip[0], right_hip[1]])
            knee = np.array([right_knee[0], right_knee[1]])
            ankle = np.array([right_ankle[0], right_ankle[1]])
        else:
            return None
        
        v1 = hip - knee
        v2 = ankle - knee
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_vjump_arm_swing_angle(self, keypoints):
        """Calculate arm swing angle (shoulder-elbow-wrist) for vertical jump."""
        # Use the more confident side
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        left_conf = min(left_shoulder[2], left_elbow[2], left_wrist[2])
        right_conf = min(right_shoulder[2], right_elbow[2], right_wrist[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            shoulder = np.array([left_shoulder[0], left_shoulder[1]])
            elbow = np.array([left_elbow[0], left_elbow[1]])
            wrist = np.array([left_wrist[0], left_wrist[1]])
        elif right_conf > 0.5:
            shoulder = np.array([right_shoulder[0], right_shoulder[1]])
            elbow = np.array([right_elbow[0], right_elbow[1]])
            wrist = np.array([right_wrist[0], right_wrist[1]])
        else:
            return None
        
        v1 = shoulder - elbow
        v2 = wrist - elbow
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
    
    def calculate_vjump_landing_knee_angle(self, keypoints):
        """Calculate knee angle during landing (hip-knee-ankle)."""
        # Use the more confident side
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        left_conf = min(left_hip[2], left_knee[2], left_ankle[2])
        right_conf = min(right_hip[2], right_knee[2], right_ankle[2])
        
        if left_conf > right_conf and left_conf > 0.5:
            hip = np.array([left_hip[0], left_hip[1]])
            knee = np.array([left_knee[0], left_knee[1]])
            ankle = np.array([left_ankle[0], left_ankle[1]])
        elif right_conf > 0.5:
            hip = np.array([right_hip[0], right_hip[1]])
            knee = np.array([right_knee[0], right_knee[1]])
            ankle = np.array([right_ankle[0], right_ankle[1]])
        else:
            return None
        
        # Calculate angle at knee
        v1 = hip - knee
        v2 = ankle - knee
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return None
        
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return int(round(angle))
