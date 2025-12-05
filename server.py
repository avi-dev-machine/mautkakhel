import os
import json
import uuid
import time
import shutil
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from utils import PoseCalibrator
from metrics import PerformanceMetrics
from cheat import DeepfakeVideoDetector

# ===========================
# CONFIGURATION
# ===========================

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = FastAPI(title="AI Exercise Trainer API", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Global session storage
sessions: Dict[str, dict] = {}

# Initialize cheat detector once at startup
print("\n[STARTUP] Initializing cheat detection system...")
try:
    cheat_detector = DeepfakeVideoDetector()
    print("[STARTUP] ✓ Cheat detection system ready!\n")
except Exception as e:
    print(f"[STARTUP] ⚠ Warning: Cheat detector initialization failed: {e}\n")
    cheat_detector = None

# ===========================
# MODELS
# ===========================

class SessionCreate(BaseModel):
    """Create a new exercise session"""
    pass

class ExerciseAnalysisRequest(BaseModel):
    """Start exercise analysis"""
    session_id: str
    exercise_type: str  # pushup, squat, situp, sitnreach, skipping, jumpingjacks, vjump, bjump

class SessionResponse(BaseModel):
    """Session creation response"""
    session_id: str
    message: str
    created_at: str

class AnalysisStatus(BaseModel):
    """Analysis status response"""
    session_id: str
    status: str  # processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str

class PerformanceReport(BaseModel):
    """Complete performance report"""
    session_id: str
    exercise: str
    report: dict
    timestamp: str

# ===========================
# HELPER CLASSES
# ===========================

class VideoProcessor:
    """Video processing with metrics tracking"""
    
    def __init__(self, session_id: str, exercise_type: str, video_path: str):
        self.session_id = session_id
        self.exercise_type = exercise_type
        self.video_path = video_path
        self.calibrator = PoseCalibrator(model_path='yolo11n-pose.pt')
        self.metrics = PerformanceMetrics()
        
        # Exercise thresholds
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
        
        self.counter = 0
        self.stage = 'up'  # Start in UP state for squats like test.py
        self.start_time = None
        
        # Angle smoothing for stability
        self.last_elbow_angles = []
        self.last_knee_angles = []
    
    def _smooth_elbow_angle(self, elbow_angle):
        """Simple 3-frame moving average for elbow"""
        if elbow_angle is None:
            return None
        self.last_elbow_angles.append(elbow_angle)
        if len(self.last_elbow_angles) > 3:
            self.last_elbow_angles.pop(0)
        return int(sum(self.last_elbow_angles) / len(self.last_elbow_angles))
    
    def _smooth_knee_angle(self, knee_angle):
        """Simple 3-frame moving average for knee"""
        if knee_angle is None:
            return None
        self.last_knee_angles.append(knee_angle)
        if len(self.last_knee_angles) > 3:
            self.last_knee_angles.pop(0)
        return int(sum(self.last_knee_angles) / len(self.last_knee_angles))
        
        # Angle smoothing for stability
        self.last_elbow_angles = []
        self.last_knee_angles = []
        
    def process_frame(self, frame, frame_time):
        """Process single frame with exercise-specific logic"""
        _, keypoints, angles = self.calibrator.process_frame(frame, show_angles_panel=False)
        
        # Skip if no keypoints detected
        if keypoints is None or len(keypoints) < 17:
            return
        
        if self.exercise_type == 'pushup':
            self._process_pushup(angles, keypoints, frame_time)
        elif self.exercise_type == 'squat':
            self._process_squat(angles, keypoints, frame_time)
        elif self.exercise_type == 'situp':
            self._process_situp(angles, keypoints, frame_time)
        elif self.exercise_type == 'sitnreach':
            self._process_sitnreach(angles, keypoints, frame_time)
        elif self.exercise_type == 'skipping':
            self._process_skipping(angles, keypoints, frame_time)
        elif self.exercise_type == 'jumpingjacks':
            self._process_jumpingjacks(angles, keypoints, frame_time)
        elif self.exercise_type == 'vjump':
            self._process_vjump(angles, keypoints, frame_time)
        elif self.exercise_type == 'bjump':
            self._process_bjump(angles, keypoints, frame_time)
    
    def _process_pushup(self, angles, keypoints, frame_time):
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
        elbow = left_elbow if left_elbow else right_elbow
        hip = left_hip if left_hip else right_hip
        
        if not elbow:
            return
        
        # Apply smoothing
        elbow = self._smooth_elbow_angle(elbow)
        if not elbow:
            return
        
        # Update metrics with angle data - pass all joint angles for comprehensive tracking
        self.metrics.update_angle_data(
            left_elbow, right_elbow, left_hip, right_hip,
            left_shoulder, right_shoulder, left_knee, right_knee,
            left_ankle, right_ankle, left_wrist, right_wrist
        )
        
        # Initialize stage if needed
        if self.stage is None or self.stage == "" or self.stage == 'up':
            # Start in UP position for pushups
            self.stage = "UP" if elbow > 140 else "DOWN"
        
        # State machine: UP -> DOWN -> UP = 1 rep
        if elbow > self.thresholds['pushup']['up']:  # 140° - arms extended
            if self.stage in ('DOWN', 'down'):
                self.counter += 1
                is_good = hip and hip >= self.thresholds['pushup']['form_hip_min'] - 20
                self.metrics.record_rep(
                    rep_max=self.thresholds['pushup']['up'],
                    rep_min=elbow,
                    duration_seconds=1.0,
                    is_good_form=is_good
                )
            self.stage = 'UP'
        elif elbow < self.thresholds['pushup']['down']:  # 90° - arms bent
            self.stage = 'DOWN'
    
    def _process_squat(self, angles, keypoints, frame_time):
        left_knee = angles.get('left_knee')
        right_knee = angles.get('right_knee')
        knee = left_knee if left_knee else right_knee
        
        if not knee:
            return
        
        # Apply smoothing
        knee = self._smooth_knee_angle(knee)
        if not knee:
            return
        
        # Get additional angles
        torso_angle = angles.get('torso_angle')
        shin_angle_left = angles.get('shin_angle_left')
        shin_angle_right = angles.get('shin_angle_right')
        
        # Use most confident shin angle
        left_knee_conf = keypoints[13][2] if len(keypoints) > 13 else 0
        right_knee_conf = keypoints[14][2] if len(keypoints) > 14 else 0
        shin_angle = shin_angle_left if left_knee_conf > right_knee_conf else shin_angle_right
        
        # Update squat-specific metrics
        self.metrics.update_squat_data(keypoints, angles, torso_angle, shin_angle, frame_time)
        
        # Initialize stage if needed
        if self.stage is None:
            self.stage = 'up'
        
        # State machine: up -> down -> up = 1 rep
        if knee > self.thresholds['squat']['up']:  # 155° - standing
            if self.stage in ('down', 'DOWN'):  # Handle both cases
                if self.metrics.rep_bottom_time is not None:
                    concentric_time = frame_time - self.metrics.rep_bottom_time
                    self.metrics.concentric_times.append(concentric_time)
                    
                    # Record sticking point if tracked
                    if hasattr(self.metrics, 'min_velocity_angle') and self.metrics.min_velocity_angle is not None:
                        if not hasattr(self.metrics, 'sticking_points'):
                            self.metrics.sticking_points = []
                        self.metrics.sticking_points.append(self.metrics.min_velocity_angle)
                    
                    # Reset for next rep
                    if hasattr(self.metrics, 'min_velocity'):
                        self.metrics.min_velocity = float('inf')
                        self.metrics.min_velocity_angle = None
                    self.metrics.rep_bottom_time = None
            
            self.stage = 'up'
            self.metrics.current_phase = 'standing'
            if self.metrics.rep_start_time is None:
                self.metrics.rep_start_time = frame_time
        
        elif knee < self.thresholds['squat']['down']:  # 135° - squatting
            self.metrics.current_phase = 'descending'
            if self.stage in ('up', 'UP'):  # Handle both cases like test.py
                self.stage = 'down'
                self.counter += 1
                self.metrics.current_phase = 'bottom'
                
                if self.metrics.rep_start_time is not None:
                    eccentric_time = frame_time - self.metrics.rep_start_time
                    self.metrics.eccentric_times.append(eccentric_time)
                    self.metrics.rep_bottom_time = frame_time
                    self.metrics.rep_start_time = None
                
                # Record depth
                self.metrics.squat_depths.append(knee)
                
                # Record rep
                is_good = knee < self.thresholds['squat']['deep']
                self.metrics.record_rep(
                    rep_max=self.thresholds['squat']['up'],
                    rep_min=knee,
                    duration_seconds=1.0,
                    is_good_form=is_good
                )
        else:
            if self.stage == 'up':
                self.metrics.current_phase = 'descending'
            elif self.stage == 'down':
                self.metrics.current_phase = 'ascending'
    
    def _process_situp(self, angles, keypoints, frame_time):
        torso_inclination = angles.get('torso_inclination_horizontal')
        hip_flexion = angles.get('hip_flexion_angle')
        
        if torso_inclination is None:
            return
        
        # Update situp-specific metrics
        self.metrics.update_situp_data(keypoints, angles, torso_inclination, hip_flexion, frame_time)
        
        # State machine for sit-up counting
        # rest -> ascending -> peak -> descending -> rest (rep complete)
        
        if torso_inclination <= self.thresholds['situp']['down']:
            # In rest/down position
            if self.metrics.situp_state == 'descending':
                # Completing a rep - only count here
                if self.metrics.situp_peak_time is not None:
                    eccentric_time = frame_time - self.metrics.situp_peak_time
                    self.metrics.situp_eccentric_times.append(eccentric_time)
                    self.metrics.situp_peak_time = None
                
            self.metrics.situp_state = 'rest'
            self.stage = 'down'
            
            # Start new rep timer
            if self.metrics.situp_rep_start_time is None:
                self.metrics.situp_rep_start_time = frame_time
        
        elif torso_inclination >= self.thresholds['situp']['up'] or \
             (hip_flexion is not None and hip_flexion <= self.thresholds['situp']['good_crunch']):
            # At peak position
            if self.metrics.situp_state in ['rest', 'ascending']:
                # Just reached peak - count the rep here
                self.counter += 1
                self.metrics.situp_state = 'peak'
                self.stage = 'up'
                
                # Record concentric time
                if self.metrics.situp_rep_start_time is not None:
                    concentric_time = frame_time - self.metrics.situp_rep_start_time
                    self.metrics.situp_concentric_times.append(concentric_time)
                    self.metrics.situp_rep_start_time = None
                
                # Mark peak time for eccentric phase
                self.metrics.situp_peak_time = frame_time
                
                # Evaluate form like test.py
                good_rom = torso_inclination >= self.thresholds['situp']['up']
                good_crunch = hip_flexion is not None and hip_flexion <= self.thresholds['situp']['good_crunch']
                
                if good_rom and good_crunch:
                    self.metrics.good_reps += 1
                    if hasattr(self.metrics, 'situp_valid_reps'):
                        self.metrics.situp_valid_reps += 1
                    is_good_form = True
                elif good_rom:
                    self.metrics.good_reps += 1
                    if hasattr(self.metrics, 'situp_valid_reps'):
                        self.metrics.situp_valid_reps += 1
                    is_good_form = True
                else:
                    self.metrics.bad_reps += 1
                    if hasattr(self.metrics, 'situp_short_rom_count'):
                        self.metrics.situp_short_rom_count += 1
                    is_good_form = False
                
                # Record rep
                self.metrics.record_rep(
                    rep_max=torso_inclination,
                    rep_min=0,
                    duration_seconds=1.0,
                    is_good_form=is_good_form
                )
        
        else:
            # Transitioning between states
            if self.metrics.situp_state == 'rest':
                self.metrics.situp_state = 'ascending'
            elif self.metrics.situp_state == 'peak':
                self.metrics.situp_state = 'descending'
    
    def _process_sitnreach(self, angles, keypoints, frame_time):
        # Extract sit-and-reach specific measurements
        reach_distance = angles.get('reach_distance')
        arm_length = angles.get('arm_length')
        hip_angle = angles.get('sitnreach_hip_angle')
        back_angle = angles.get('sitnreach_back_angle')
        knee_angle = angles.get('sitnreach_knee_angle')
        symmetry_error = angles.get('reach_symmetry')
        
        if reach_distance is None:
            return
        
        # Update sit-and-reach metrics
        self.metrics.update_sitnreach_data(
            keypoints, angles, reach_distance, arm_length,
            hip_angle, back_angle, knee_angle, symmetry_error, frame_time
        )
        
        self.counter = int(self.metrics.max_reach_distance)
    
    def _process_skipping(self, angles, keypoints, frame_time):
        self.metrics.update_skipping_data(keypoints, angles, frame_time)
        self.counter = self.metrics.jump_count
    
    def _process_jumpingjacks(self, angles, keypoints, frame_time):
        self.metrics.update_jumpingjacks_data(keypoints, angles, frame_time)
        self.counter = self.metrics.jj_rep_count
    
    def _process_vjump(self, angles, keypoints, frame_time):
        self.metrics.update_vjump_data(keypoints, angles, frame_time)
        self.counter = self.metrics.vjump_jump_count
    
    def _process_bjump(self, angles, keypoints, frame_time):
        self.metrics.update_bjump_data(keypoints, angles, frame_time)
        self.counter = self.metrics.bjump_jump_count
    
    def analyze_video(self):
        """Main video analysis loop"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_time = time.time()
        frame_count = 0
        
        # Update session status
        sessions[self.session_id]['status'] = 'processing'
        sessions[self.session_id]['total_frames'] = total_frames
        
        print(f"\n[{self.session_id}] Starting {self.exercise_type.upper()} analysis...")
        print(f"[{self.session_id}] Total frames: {total_frames}, FPS: {fps}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time = time.time()
            self.process_frame(frame, frame_time)
            
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0
            sessions[self.session_id]['progress'] = progress
            sessions[self.session_id]['frames_processed'] = frame_count
            
            # Progress logging every 10%
            if frame_count % max(1, total_frames // 10) == 0:
                print(f"[{self.session_id}] Progress: {progress*100:.0f}%")
        
        cap.release()
        
        # Generate final report
        print(f"\n[{self.session_id}] Analysis complete! Generating report...")
        duration = time.time() - self.start_time
        
        report = self._generate_report()
        report['processing_time'] = round(duration, 2)
        report['frames_processed'] = frame_count
        report['video_fps'] = fps
        
        sessions[self.session_id]['status'] = 'completed'
        sessions[self.session_id]['progress'] = 1.0
        sessions[self.session_id]['report'] = report
        sessions[self.session_id]['completed_at'] = datetime.now().isoformat()
        
        print(f"[{self.session_id}] Report generated successfully!")
        return report
    
    def _generate_report(self):
        """Generate exercise-specific performance report"""
        report = None
        if self.exercise_type == 'pushup':
            report = self.metrics.pushup_metrics()
        elif self.exercise_type == 'squat':
            report = self.metrics.squat_metrics()
        elif self.exercise_type == 'situp':
            report = self.metrics.situp_metrics()
        elif self.exercise_type == 'sitnreach':
            report = self.metrics.sitnreach_metrics()
        elif self.exercise_type == 'skipping':
            report = self.metrics.skipping_metrics()
        elif self.exercise_type == 'jumpingjacks':
            report = self.metrics.jumpingjacks_metrics()
        elif self.exercise_type == 'vjump':
            report = self.metrics.vjump_metrics()
        elif self.exercise_type == 'bjump':
            report = self.metrics.bjump_metrics()
        
        # Convert numpy types to native Python types
        if report:
            report = self._convert_numpy_types(report)
        return report if report else {}
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

# ===========================
# HELPER FUNCTIONS
# ===========================

def convert_numpy_to_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64, np.int8, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    return obj

# ===========================
# API ENDPOINTS
# ===========================

@app.get("/", response_class=HTMLResponse)
async def root():
    """API health check with web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Exercise Trainer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.95);
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                color: #333;
            }
            h1 { color: #667eea; margin-top: 0; }
            .status { 
                background: #10b981; 
                color: white; 
                padding: 10px 20px; 
                border-radius: 5px; 
                display: inline-block;
                margin: 10px 0;
            }
            .exercises {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 20px 0;
            }
            .exercise-tag {
                background: #667eea;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .upload-section {
                margin: 30px 0;
                padding: 20px;
                border: 2px dashed #667eea;
                border-radius: 10px;
                text-align: center;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            select, button {
                padding: 10px 20px;
                margin: 10px 5px;
                border-radius: 5px;
                border: none;
                font-size: 16px;
            }
            button {
                background: #667eea;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background: #5568d3;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #f3f4f6;
                border-radius: 5px;
                display: none;
            }
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .info-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }
            .info-label { font-weight: bold; color: #667eea; }
            .info-value { font-size: 24px; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💪 AI Exercise Trainer</h1>
            <div class="status">🟢 API Online</div>
            
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-label">Version</div>
                    <div class="info-value">2.0</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Model</div>
                    <div class="info-value">YOLO11n-pose</div>
                </div>
                <div class="info-card">
                    <div class="info-label">Active Sessions</div>
                    <div class="info-value" id="sessions">0</div>
                </div>
            </div>

            <h2>Supported Exercises</h2>
            <div class="exercises">
                <div class="exercise-tag">💪 Pushup</div>
                <div class="exercise-tag">🏋️ Squat</div>
                <div class="exercise-tag">🤸 Situp</div>
                <div class="exercise-tag">🧘 Sit & Reach</div>
                <div class="exercise-tag">🪢 Skipping</div>
                <div class="exercise-tag">🤾 Jumping Jacks</div>
                <div class="exercise-tag">⬆️ Vertical Jump</div>
                <div class="exercise-tag">➡️ Broad Jump</div>
            </div>

            <div class="upload-section">
                <h2>Test Exercise Analysis</h2>
                <input type="file" id="videoFile" accept="video/*">
                <br>
                <select id="exerciseType">
                    <option value="pushup">Pushup</option>
                    <option value="squat">Squat</option>
                    <option value="situp">Situp</option>
                    <option value="sitnreach">Sit & Reach</option>
                    <option value="skipping">Skipping</option>
                    <option value="jumpingjacks">Jumping Jacks</option>
                    <option value="vjump">Vertical Jump</option>
                    <option value="bjump">Broad Jump</option>
                </select>
                <br>
                <button onclick="uploadVideo()">Analyze Video</button>
                <div id="loading" style="display:none; margin-top:10px;">
                    <p>⏳ Analyzing video...</p>
                </div>
            </div>

            <div class="result" id="result">
                <h3>Analysis Result</h3>
                <pre id="resultData"></pre>
            </div>

            <h2>API Endpoints</h2>
            <div class="info-card">
                <p><strong>POST /upload-video/</strong> - Upload and analyze exercise video</p>
                <p><strong>GET /</strong> - API status and info</p>
                <p><strong>GET /docs</strong> - API documentation (if enabled)</p>
            </div>
        </div>

        <script>
            // Update active sessions
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if(data.active_sessions !== undefined) {
                        document.getElementById('sessions').textContent = data.active_sessions;
                    }
                })
                .catch(() => {});

            async function uploadVideo() {
                const fileInput = document.getElementById('videoFile');
                const exerciseType = document.getElementById('exerciseType').value;
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                const resultData = document.getElementById('resultData');

                if (!fileInput.files[0]) {
                    alert('Please select a video file');
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('exercise_type', exerciseType);

                loading.style.display = 'block';
                result.style.display = 'none';

                try {
                    const response = await fetch('/upload-video/', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    resultData.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    loading.style.display = 'none';
                    alert('Error: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/status")
async def api_status():
    """API status endpoint for web interface"""
    return {
        "status": "online",
        "service": "AI Exercise Trainer API",
        "version": "2.0",
        "exercises": [
            "pushup", "squat", "situp", "sitnreach",
            "skipping", "jumpingjacks", "vjump", "bjump"
        ],
        "active_sessions": len(sessions)
    }

@app.post("/cheat-detection/")
async def cheat_detection_and_analysis(file: UploadFile = File(...), exercise_type: str = "pushup"):
    """
    Cheat Detection & Exercise Analysis Endpoint
    Analyzes video for cheating/manipulation before performing exercise analysis
    Includes MANDATORY cheat detection - blocks analysis if MEDIUM or HIGH risk detected
    """
    # Create session
    session_id = str(uuid.uuid4())
    session_data = {
        "session_id": session_id,
        "exercise_type": exercise_type,
        "created_at": datetime.now().isoformat(),
        "status": "processing"
    }
    sessions[session_id] = session_data
    
    # Save uploaded file
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    video_path = session_dir / file.filename
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    session_data["video_path"] = str(video_path)
    
    # ============================================================
    # STEP 1: MANDATORY CHEAT DETECTION
    # ============================================================
    cheat_result = None
    if cheat_detector is not None:
        try:
            print(f"\n[{session_id}] Running cheat detection...")
            cheat_result = cheat_detector.analyze_video(str(video_path), num_frames=16)
            
            if cheat_result:
                overall_score = cheat_result.get('overall_score', 0)
                risk_level = cheat_result.get('risk_level', 'UNKNOWN')
                verdict = cheat_result.get('verdict', '')
                violations = cheat_result.get('violations', {})
                
                print(f"\n[{session_id}] CHEAT DETECTION RESULTS:")
                print(f"  Risk Level: {risk_level}")
                print(f"  Score: {overall_score:.1f}/100")
                
                # ============================================================
                # BLOCKING LOGIC: MEDIUM (score >= 40) or HIGH (score >= 60)
                # ============================================================
                if overall_score >= 40 or risk_level in ['MEDIUM', 'HIGH', 'CRITICAL']:
                    print(f"[{session_id}] ❌ ANALYSIS BLOCKED - {risk_level} RISK DETECTED\n")
                    
                    session_data["status"] = "blocked_cheating"
                    session_data["cheat_detection"] = cheat_result
                    
                    # Convert numpy types to JSON-serializable format
                    detail_data = {
                        "error": "Analysis blocked due to cheating detection",
                        "risk_level": str(risk_level),
                        "overall_score": float(overall_score),
                        "verdict": str(verdict),
                        "violations": convert_numpy_to_json_serializable(violations),
                        "message": "Video contains suspicious content. Analysis cannot proceed.",
                        "cheat_detection_full_report": convert_numpy_to_json_serializable(cheat_result)
                    }
                    
                    raise HTTPException(
                        status_code=403,
                        detail=detail_data
                    )
                else:
                    print(f"[{session_id}] ✓ CHEAT CHECK PASSED - Proceeding...\n")
                    session_data["cheat_detection"] = cheat_result
                    
        except HTTPException:
            raise  # Re-raise blocking exceptions
        except Exception as e:
            print(f"[{session_id}] ⚠ Cheat detection error: {e}\n")
            session_data["cheat_detection_error"] = str(e)
    
    # ============================================================
    # STEP 2: PROCEED WITH EXERCISE ANALYSIS (if cheat check passed)
    # ============================================================
    try:
        processor = VideoProcessor(session_id, exercise_type, str(video_path))
        results = processor.analyze_video()
        
        session_data["status"] = "completed"
        session_data["results"] = results
        
        return {
            "session_id": session_id,
            "status": "completed",
            "exercise_type": exercise_type,
            "cheat_detection": {
                "passed": True,
                "risk_level": cheat_result.get('risk_level', 'UNKNOWN') if cheat_result else 'NOT_CHECKED',
                "score": cheat_result.get('overall_score', 0) if cheat_result else 0
            },
            "results": results
        }
    except Exception as e:
        session_data["status"] = "error"
        session_data["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """
    Create a new exercise session with auto-generated ID
    
    Returns:
        - session_id: Unique identifier for this session
        - message: Confirmation message
        - created_at: Timestamp
    """
    session_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    sessions[session_id] = {
        'session_id': session_id,
        'created_at': created_at,
        'status': 'created',
        'progress': 0.0,
        'exercise_type': None,
        'video_path': None,
        'report': None,
        'frames_processed': 0,
        'total_frames': 0
    }
    
    print(f"\n{'='*60}")
    print(f"  NEW SESSION CREATED")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Created At: {created_at}")
    print(f"{'='*60}\n")
    
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
        created_at=created_at
    )

@app.post("/session/{session_id}/upload")
async def upload_video(
    session_id: str,
    exercise_type: str,
    video: UploadFile = File(...)
):
    """
    Upload video for analysis
    
    Args:
        - session_id: Session identifier
        - exercise_type: Exercise to analyze (pushup, squat, situp, etc.)
        - video: Video file
    
    Returns:
        - message: Upload confirmation
        - file_info: Video details
    """
    # Validate session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate exercise type
    valid_exercises = ['pushup', 'squat', 'situp', 'sitnreach', 
                      'skipping', 'jumpingjacks', 'vjump', 'bjump']
    if exercise_type not in valid_exercises:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid exercise type. Valid options: {valid_exercises}"
        )
    
    # Save video file
    video_filename = f"{session_id}_{exercise_type}_{video.filename}"
    video_path = UPLOAD_DIR / video_filename
    
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)
    
    # Update session
    sessions[session_id]['exercise_type'] = exercise_type
    sessions[session_id]['video_path'] = str(video_path)
    sessions[session_id]['status'] = 'uploaded'
    sessions[session_id]['video_filename'] = video.filename
    sessions[session_id]['video_size'] = len(content)
    
    print(f"\n{'='*60}")
    print(f"  VIDEO UPLOADED")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Exercise: {exercise_type.upper()}")
    print(f"Filename: {video.filename}")
    print(f"Size: {len(content) / (1024*1024):.2f} MB")
    print(f"{'='*60}\n")
    
    return {
        "message": "Video uploaded successfully",
        "session_id": session_id,
        "exercise_type": exercise_type,
        "file_info": {
            "filename": video.filename,
            "size_bytes": len(content),
            "size_mb": round(len(content) / (1024*1024), 2)
        }
    }

@app.post("/session/{session_id}/analyze")
async def analyze_exercise(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """
    Start video analysis (runs in background)
    Includes MANDATORY cheat detection - blocks analysis if MEDIUM or HIGH risk detected
    
    Args:
        - session_id: Session identifier
    
    Returns:
        - message: Analysis started confirmation OR cheat detection blocked message
        - session_id: Session ID for tracking
        - cheat_detection: Cheat detection results (if applicable)
    """
    # Validate session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session['status'] != 'uploaded':
        raise HTTPException(
            status_code=400, 
            detail=f"Session not ready for analysis. Current status: {session['status']}"
        )
    
    # Start background analysis
    exercise_type = session['exercise_type']
    video_path = session['video_path']
    
    print(f"\n{'='*60}")
    print(f"  PRE-ANALYSIS CHEAT DETECTION")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Exercise: {exercise_type.upper()}")
    print(f"Video: {video_path}")
    print(f"{'='*60}\n")
    
    # ============================================================
    # STEP 1: MANDATORY CHEAT DETECTION
    # ============================================================
    cheat_result = None
    if cheat_detector is not None:
        try:
            print(f"[{session_id}] Running cheat detection...")
            cheat_result = cheat_detector.analyze_video(video_path, num_frames=16)
            
            if cheat_result:
                overall_score = cheat_result.get('overall_score', 0)
                risk_level = cheat_result.get('risk_level', 'UNKNOWN')
                verdict = cheat_result.get('verdict', '')
                violations = cheat_result.get('violations', {})
                
                print(f"\n[{session_id}] CHEAT DETECTION RESULTS:")
                print(f"  Risk Level: {risk_level}")
                print(f"  Score: {overall_score:.1f}/100")
                print(f"  Verdict: {verdict}")
                print(f"  Has Text: {violations.get('has_text', False)}")
                print(f"  Is Deepfake: {violations.get('is_deepfake', False)}")
                print(f"  Has Effects: {violations.get('has_effects', False)}")
                
                # ============================================================
                # BLOCKING LOGIC: MEDIUM (score >= 40) or HIGH (score >= 60)
                # ============================================================
                if overall_score >= 40 or risk_level in ['MEDIUM', 'HIGH', 'CRITICAL']:
                    print(f"\n[{session_id}] ❌ ANALYSIS BLOCKED - {risk_level} RISK DETECTED")
                    print(f"[{session_id}] Score: {overall_score:.1f}/100 (Threshold: 40)")
                    print(f"{'='*60}\n")
                    
                    # Update session status
                    sessions[session_id]['status'] = 'blocked_cheating'
                    sessions[session_id]['cheat_detection'] = cheat_result
                    sessions[session_id]['blocked_reason'] = f"{risk_level} risk detected - Score: {overall_score:.1f}/100"
                    
                    # Convert numpy types to JSON-serializable format
                    detail_data = {
                        "error": "Analysis blocked due to cheating detection",
                        "risk_level": str(risk_level),
                        "overall_score": float(overall_score),
                        "verdict": str(verdict),
                        "violations": convert_numpy_to_json_serializable(violations),
                        "message": "Video contains suspicious content. Analysis cannot proceed.",
                        "cheat_detection_full_report": convert_numpy_to_json_serializable(cheat_result)
                    }
                    
                    # Return blocking response
                    raise HTTPException(
                        status_code=403,
                        detail=detail_data
                    )
                
                else:
                    print(f"\n[{session_id}] ✓ CHEAT CHECK PASSED - {risk_level} RISK")
                    print(f"[{session_id}] Score: {overall_score:.1f}/100 (Below threshold: 40)")
                    print(f"[{session_id}] Proceeding with exercise analysis...\n")
                    
                    # Store cheat detection results for reference
                    sessions[session_id]['cheat_detection'] = cheat_result
                    sessions[session_id]['cheat_check_passed'] = True
                    
        except HTTPException:
            raise  # Re-raise HTTP exceptions (blocking)
        except Exception as e:
            print(f"[{session_id}] ⚠ Cheat detection error: {e}")
            print(f"[{session_id}] Proceeding with analysis (cheat detection failed)...\n")
            sessions[session_id]['cheat_detection_error'] = str(e)
    else:
        print(f"[{session_id}] ⚠ Cheat detector not available, skipping check...\n")
    
    # ============================================================
    # STEP 2: PROCEED WITH EXERCISE ANALYSIS (if cheat check passed)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  EXERCISE ANALYSIS STARTED")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Exercise: {exercise_type.upper()}")
    print(f"Video: {video_path}")
    print(f"{'='*60}\n")
    
    def run_analysis():
        try:
            processor = VideoProcessor(session_id, exercise_type, video_path)
            processor.analyze_video()
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"[{session_id}] ERROR: {error_msg}")
            print(f"[{session_id}] TRACEBACK:\n{traceback_str}")
            sessions[session_id]['status'] = 'failed'
            sessions[session_id]['error'] = error_msg
            sessions[session_id]['traceback'] = traceback_str
    
    background_tasks.add_task(run_analysis)
    
    return {
        "message": "Analysis started - Cheat check passed",
        "session_id": session_id,
        "exercise_type": exercise_type,
        "status": "processing",
        "cheat_detection": {
            "passed": True,
            "risk_level": cheat_result.get('risk_level', 'UNKNOWN') if cheat_result else 'NOT_CHECKED',
            "score": cheat_result.get('overall_score', 0) if cheat_result else 0
        }
    }

@app.get("/session/{session_id}/status")
async def get_analysis_status(session_id: str):
    """
    Check analysis progress
    
    Args:
        - session_id: Session identifier
    
    Returns:
        - status: current status (created, uploaded, processing, completed, failed, blocked_cheating)
        - progress: 0.0 to 1.0
        - message: Status description
        - cheat_detection: Cheat detection results (if available)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    status = session['status']
    progress = session.get('progress', 0.0)
    
    messages = {
        'created': 'Session created, waiting for video upload',
        'uploaded': 'Video uploaded, ready for analysis',
        'processing': f'Analyzing video... {progress*100:.0f}% complete',
        'completed': 'Analysis complete! Report ready',
        'failed': f'Analysis failed: {session.get("error", "Unknown error")}',
        'blocked_cheating': 'Analysis blocked due to cheating detection'
    }
    
    response = {
        "session_id": session_id,
        "status": status,
        "progress": progress,
        "message": messages.get(status, 'Unknown status')
    }
    
    # Include cheat detection info if available
    if 'cheat_detection' in session:
        cheat_result = session['cheat_detection']
        response['cheat_detection'] = {
            'risk_level': cheat_result.get('risk_level', 'UNKNOWN'),
            'overall_score': cheat_result.get('overall_score', 0),
            'verdict': cheat_result.get('verdict', ''),
            'violations': cheat_result.get('violations', {})
        }
    
    if status == 'blocked_cheating':
        response['blocked_reason'] = session.get('blocked_reason', 'Cheating detected')
    
    return response

@app.get("/session/{session_id}/report")
async def get_performance_report(session_id: str):
    """
    Retrieve complete performance report
    
    Args:
        - session_id: Session identifier
    
    Returns:
        - Complete performance report with metrics and cheat detection results
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session['status'] == 'blocked_cheating':
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Analysis was blocked due to cheating detection",
                "cheat_detection": session.get('cheat_detection'),
                "blocked_reason": session.get('blocked_reason')
            }
        )
    
    if session['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Current status: {session['status']}"
        )
    
    report = session.get('report')
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Print formatted report to console (replicating test.py output)
    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE!")
    print(f"{'='*60}\n")
    
    # Print exercise-specific report
    _print_terminal_report(session['exercise_type'], report)
    
    response = {
        "session_id": session_id,
        "exercise": session['exercise_type'],
        "report": report,
        "timestamp": session['completed_at']
    }
    
    # Include cheat detection results if available
    if 'cheat_detection' in session:
        response['cheat_detection'] = {
            'passed': session.get('cheat_check_passed', False),
            'risk_level': session['cheat_detection'].get('risk_level', 'UNKNOWN'),
            'overall_score': session['cheat_detection'].get('overall_score', 0),
            'verdict': session['cheat_detection'].get('verdict', '')
        }
    
    return response

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete session and cleanup files
    
    Args:
        - session_id: Session identifier
    
    Returns:
        - message: Deletion confirmation
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Delete video file
    if session.get('video_path'):
        video_path = Path(session['video_path'])
        if video_path.exists():
            video_path.unlink()
    
    # Remove session
    del sessions[session_id]
    
    print(f"\n[{session_id}] Session deleted and cleaned up\n")
    
    return {
        "message": "Session deleted successfully",
        "session_id": session_id
    }

@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    
    Returns:
        - List of session summaries
    """
    session_list = []
    for sid, session in sessions.items():
        session_list.append({
            "session_id": sid,
            "exercise_type": session.get('exercise_type'),
            "status": session['status'],
            "created_at": session['created_at'],
            "progress": session.get('progress', 0.0)
        })
    
    return {
        "total_sessions": len(session_list),
        "sessions": session_list
    }

# ===========================
# HELPER FUNCTIONS
# ===========================

def _print_terminal_report(exercise_type: str, report: dict):
    """Print formatted report to terminal (matching test.py output)"""
    
    if exercise_type == 'bjump':
        print("="*60)
        print(f"STANDING BROAD JUMP PERFORMANCE REPORT")
        print("="*60)
        print(f"Duration: {report.get('duration_seconds', 0)}s")
        print(f"Total Jumps: {report.get('total_jumps', 0)} | Valid Jumps: {report.get('valid_jumps', 0)}")
        print(f"Accuracy: {report.get('accuracy', 0)}%")
        print()
        print("METRIC BREAKDOWN:")
        print("-" * 60)
        print(f"Jump Distance Score: {report.get('distance_score', 0)} | {report.get('rating', 'N/A')}")
        print(f"Countermovement Score: {report.get('countermovement_score', 0)} | {report.get('rating', 'N/A')}")
        print(f"Arm Swing Score: {report.get('arm_swing_score', 0)} | {report.get('rating', 'N/A')}")
        print(f"Takeoff Symmetry: {report.get('symmetry_score', 0)} | {report.get('rating', 'N/A')}")
        print(f"Landing Stability: {report.get('landing_stability_score', 0)} | {report.get('rating', 'N/A')}")
        print()
        print("DETAILED MEASUREMENTS:")
        print("-" * 60)
        print(f"Max Jump Distance: {report.get('max_jump_distance', 0)} px")
        print(f"Average Jump Distance: {report.get('avg_jump_distance', 0)} px")
        print(f"Average Countermovement: {report.get('avg_countermovement', 0)}° (Good: 90-120°)")
        print(f"Average Arm Swing: {report.get('avg_arm_swing', 0)}° (Good: 140+°)")
        print(f"Average Symmetry Error: {report.get('avg_symmetry_error', 0)}s (Good: <0.1s)")
        print(f"Average Landing Stability: {report.get('avg_landing_stability', 0)}px (Good: <30px)")
        print()
        print("="*60)
        print(f"FINAL PERFORMANCE SCORE: {report.get('final_score', 0)} | {report.get('rating', 'N/A')}")
        print("="*60)
    
    elif exercise_type == 'vjump':
        print("="*60)
        print(f"VERTICAL JUMP PERFORMANCE REPORT")
        print("="*60)
        print(f"Duration: {report.get('duration_seconds', 0)}s")
        print(f"Total Jumps: {report.get('total_jumps', 0)} | Valid Jumps: {report.get('valid_jumps', 0)}")
        print(f"Accuracy: {report.get('accuracy', 0)}%")
        print()
        print("METRIC BREAKDOWN:")
        print("-" * 60)
        print(f"Jump Height Score: {report.get('jump_height_score', 0)}")
        print(f"Countermovement Score: {report.get('countermovement_score', 0)}")
        print(f"Arm Swing Score: {report.get('arm_swing_score', 0)}")
        print(f"Takeoff Symmetry: {report.get('symmetry_score', 0)}")
        print(f"Landing Control: {report.get('landing_control_score', 0)}")
        print()
        print("="*60)
        print(f"FINAL PERFORMANCE SCORE: {report.get('final_score', 0)} | {report.get('rating', 'N/A')}")
        print("="*60)
    
    else:
        # Generic report for other exercises
        print("="*60)
        print(f"{exercise_type.upper()} PERFORMANCE REPORT")
        print("="*60)
        print(json.dumps(report, indent=2))
        print("="*60)

# ===========================
# STARTUP MESSAGE
# ===========================

@app.on_event("startup")
async def startup_event():
    """Print startup banner"""
    print("\n" + "="*60)
    print("  AI EXERCISE TRAINER API SERVER")
    print("="*60)
    print(f"Version: 2.0")
    print(f"Model: yolo11n-pose.pt")
    print(f"Exercises: 8 types supported")
    print(f"Upload Dir: {UPLOAD_DIR.absolute()}")
    print(f"Results Dir: {RESULTS_DIR.absolute()}")
    print("="*60)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
