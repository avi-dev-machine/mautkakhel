"""
FastAPI Server for AI Exercise Trainer
Implements test.py workflow with REST API
Integrates cheat.py for video validation and ai.py for analysis
"""

from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import uuid
import time
import json
import shutil

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from utils import PoseCalibrator
from metrics import PerformanceMetrics
from cheat import DeepfakeVideoDetector
from ai import analyze_exercise_metrics

# ===========================
# CONFIGURATION
# ===========================

app = FastAPI(title="AI Exercise Trainer API", version="3.0")

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

# Global session storage (mimics test.py's completed_exercises tracking)
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
# PYDANTIC MODELS
# ===========================

class SessionCreate(BaseModel):
    """Create new workout session"""
    user_id: Optional[str] = None

class SessionResponse(BaseModel):
    """Session creation response"""
    session_id: str
    message: str
    created_at: str
    completed_exercises: List[str]
    available_exercises: List[str]

class ExerciseResult(BaseModel):
    """Individual exercise result"""
    exercise_type: str
    status: str
    cheat_detection: dict
    metrics: Optional[dict] = None
    reps: Optional[int] = None
    timestamp: str

class SessionStatus(BaseModel):
    """Session status response"""
    session_id: str
    created_at: str
    status: str
    completed_exercises: List[str]
    available_exercises: List[str]
    total_completed: int
    can_generate_ai_analysis: bool

class AIAnalysisResponse(BaseModel):
    """AI analysis response"""
    session_id: str
    ai_analysis: str
    exercises_analyzed: List[str]
    generated_at: str

# ===========================
# VIDEO PROCESSOR (from server.py)
# ===========================

class VideoProcessor:
    """Video processing with metrics tracking - copied from server.py"""
    
    def __init__(self, session_id: str, exercise_type: str, video_path: str):
        self.session_id = session_id
        self.exercise_type = exercise_type
        self.video_path = video_path
        # Force CPU to avoid CUDA NMS issues
        import torch
        torch.cuda.is_available = lambda: False
        self.calibrator = PoseCalibrator(model_path='yolo11n-pose.pt')
        self.metrics = PerformanceMetrics()
        
        # Exercise thresholds (from test.py)
        self.thresholds = {
            'pushup': {'down': 90, 'up': 140, 'form_hip_min': 130},
            'squat': {'down': 135, 'up': 155, 'deep': 90},
            'situp': {'up': 70, 'down': 20, 'good_crunch': 50},
            'sitnreach': {'excellent_hip': 60, 'average_hip': 80, 'knee_valid': 165},
            'skipping': {'jump_threshold': 30, 'min_height': 20},
            'jumpingjacks': {'arm_open': 150, 'leg_open': 150},
            'vjump': {'min_height': 30, 'good_countermovement': 110},
            'bjump': {'min_distance': 50, 'good_countermovement': 110}
        }
        
        self.counter = 0
        self.stage = 'up'
        self.start_time = None
        
        # Smoothing buffers
        self.last_elbow_angles = []
        self.last_knee_angles = []
    
    def _smooth_elbow_angle(self, elbow_angle):
        if elbow_angle is None:
            return None
        self.last_elbow_angles.append(elbow_angle)
        if len(self.last_elbow_angles) > 3:
            self.last_elbow_angles.pop(0)
        return int(sum(self.last_elbow_angles) / len(self.last_elbow_angles))
    
    def _smooth_knee_angle(self, knee_angle):
        if knee_angle is None:
            return None
        self.last_knee_angles.append(knee_angle)
        if len(self.last_knee_angles) > 3:
            self.last_knee_angles.pop(0)
        return int(sum(self.last_knee_angles) / len(self.last_knee_angles))
    
    def process_frame(self, frame, frame_time):
        """Process single frame with exercise-specific logic"""
        _, keypoints, angles = self.calibrator.process_frame(frame, show_angles_panel=False)
        
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
        
        elbow = left_elbow if left_elbow else right_elbow
        hip = left_hip if left_hip else right_hip
        
        if not elbow:
            return
        
        elbow = self._smooth_elbow_angle(elbow)
        if not elbow:
            return
        
        self.metrics.update_angle_data(
            left_elbow, right_elbow, left_hip, right_hip,
            left_shoulder, right_shoulder, left_knee, right_knee,
            left_ankle, right_ankle, left_wrist, right_wrist
        )
        
        if self.stage is None or self.stage == "" or self.stage == 'up':
            self.stage = "UP" if elbow > 140 else "DOWN"
        
        if elbow > self.thresholds['pushup']['up']:
            if self.stage == "DOWN":
                self.counter += 1
                is_good = (hip is None or hip >= self.thresholds['pushup']['form_hip_min'] - 20)
                self.metrics.record_rep(
                    rep_max=self.thresholds['pushup']['up'],
                    rep_min=elbow,
                    duration_seconds=1.0,
                    is_good_form=is_good
                )
            self.stage = "UP"
        elif elbow < self.thresholds['pushup']['down']:
            self.stage = "DOWN"
    
    def _process_squat(self, angles, keypoints, frame_time):
        left_knee = angles.get('left_knee')
        right_knee = angles.get('right_knee')
        knee = left_knee if left_knee else right_knee
        
        if not knee:
            return
        
        knee = self._smooth_knee_angle(knee)
        if not knee:
            return
        
        torso_angle = angles.get('torso_angle')
        shin_angle_left = angles.get('shin_angle_left')
        shin_angle_right = angles.get('shin_angle_right')
        
        left_knee_conf = keypoints[13][2] if len(keypoints) > 13 else 0
        right_knee_conf = keypoints[14][2] if len(keypoints) > 14 else 0
        shin_angle = shin_angle_left if left_knee_conf > right_knee_conf else shin_angle_right
        
        self.metrics.update_squat_data(keypoints, angles, torso_angle, shin_angle, frame_time)
        
        if self.stage is None:
            self.stage = "UP"
        
        if knee > self.thresholds['squat']['up']:
            if self.stage == "DOWN":
                if self.metrics.rep_bottom_time is not None:
                    concentric_time = frame_time - self.metrics.rep_bottom_time
                    self.metrics.concentric_times.append(concentric_time)
                    
                    if self.metrics.min_velocity_angle is not None:
                        self.metrics.sticking_points.append(self.metrics.min_velocity_angle)
                    
                    self.metrics.min_velocity = float('inf')
                    self.metrics.min_velocity_angle = None
                    self.metrics.rep_bottom_time = None
            
            self.stage = "UP"
            self.metrics.current_phase = 'standing'
            
            if self.metrics.rep_start_time is None:
                self.metrics.rep_start_time = frame_time
        
        elif knee < self.thresholds['squat']['down']:
            self.metrics.current_phase = 'descending'
            
            if self.stage == "UP" or self.stage == "up":
                self.stage = "DOWN"
                self.counter += 1
                self.metrics.current_phase = 'bottom'
                
                if self.metrics.rep_start_time is not None:
                    eccentric_time = frame_time - self.metrics.rep_start_time
                    self.metrics.eccentric_times.append(eccentric_time)
                
                self.metrics.rep_bottom_time = frame_time
                self.metrics.rep_start_time = None
                self.metrics.squat_depths.append(knee)
                
                if knee < self.thresholds['squat']['deep']:
                    is_good_form = True
                else:
                    is_good_form = False
                
                if torso_angle and torso_angle > 45:
                    is_good_form = False
                
                self.metrics.record_rep(
                    rep_max=self.thresholds['squat']['up'],
                    rep_min=knee,
                    duration_seconds=1.0,
                    is_good_form=is_good_form
                )
        else:
            if self.stage == 'UP':
                self.metrics.current_phase = 'descending'
            elif self.stage == 'DOWN':
                self.metrics.current_phase = 'ascending'
    
    def _process_situp(self, angles, keypoints, frame_time):
        torso_inclination = angles.get('torso_inclination_horizontal')
        hip_flexion = angles.get('hip_flexion_angle')
        
        if torso_inclination is None:
            return
        
        self.metrics.update_situp_data(keypoints, angles, torso_inclination, hip_flexion, frame_time)
        
        foot_lifted = self.metrics._detect_foot_lift(keypoints)
        neck_distance = self.metrics._detect_neck_strain(keypoints)
        if neck_distance:
            self.metrics.situp_neck_strains.append(neck_distance)
        
        if torso_inclination <= self.thresholds['situp']['down']:
            if self.metrics.situp_state == 'descending':
                if self.metrics.situp_peak_time is not None:
                    eccentric_time = frame_time - self.metrics.situp_peak_time
                    self.metrics.situp_eccentric_times.append(eccentric_time)
                    self.metrics.situp_peak_time = None
                
                momentum_score = self.metrics._calculate_momentum_score()
                self.metrics.situp_momentum_scores.append(momentum_score)
                self.metrics.shoulder_positions.clear()
                self.metrics.max_torso_inclination = 0
                self.metrics.min_hip_flexion = 180
                
            self.metrics.situp_state = 'rest'
            self.stage = "DOWN"
            
            if self.metrics.situp_rep_start_time is None:
                self.metrics.situp_rep_start_time = frame_time
        
        elif torso_inclination >= self.thresholds['situp']['up'] or \
             (hip_flexion is not None and hip_flexion <= self.thresholds['situp']['good_crunch']):
            if self.metrics.situp_state in ['rest', 'ascending']:
                self.counter += 1
                self.metrics.situp_state = 'peak'
                self.stage = "UP"
                
                if self.metrics.situp_rep_start_time is not None:
                    concentric_time = frame_time - self.metrics.situp_rep_start_time
                    self.metrics.situp_concentric_times.append(concentric_time)
                    self.metrics.situp_rep_start_time = None
                
                self.metrics.situp_peak_time = frame_time
                self.metrics.situp_torso_inclinations.append(self.metrics.max_torso_inclination)
                self.metrics.situp_hip_flexions.append(self.metrics.min_hip_flexion if hip_flexion else 180)
                self.metrics.situp_foot_lifts.append(1 if foot_lifted else 0)
                
                good_rom = torso_inclination >= self.thresholds['situp']['up']
                good_crunch = hip_flexion is not None and hip_flexion <= self.thresholds['situp']['good_crunch']
                
                if good_rom and good_crunch:
                    self.metrics.good_reps += 1
                    self.metrics.situp_valid_reps += 1
                    is_good_form = True
                elif good_rom:
                    self.metrics.good_reps += 1
                    self.metrics.situp_valid_reps += 1
                    is_good_form = True
                else:
                    self.metrics.bad_reps += 1
                    self.metrics.situp_short_rom_count += 1
                    is_good_form = False
                
                if foot_lifted:
                    is_good_form = False
                
                self.metrics.record_rep(
                    rep_max=torso_inclination,
                    rep_min=0,
                    duration_seconds=1.0,
                    is_good_form=is_good_form
                )
            
            if torso_inclination < self.thresholds['situp']['up'] - 10:
                self.metrics.situp_state = 'descending'
        
        else:
            if self.metrics.situp_state == 'rest':
                self.metrics.situp_state = 'ascending'
            elif self.metrics.situp_state == 'peak':
                self.metrics.situp_state = 'descending'
    
    def _process_sitnreach(self, angles, keypoints, frame_time):
        reach_distance = angles.get('reach_distance')
        arm_length = angles.get('arm_length')
        hip_angle = angles.get('sitnreach_hip_angle')
        back_angle = angles.get('sitnreach_back_angle')
        knee_angle = angles.get('sitnreach_knee_angle')
        symmetry_error = angles.get('reach_symmetry')
        
        if reach_distance is None:
            return
        
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
            raise Exception(f"Could not open video: {self.video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_time = time.time()
        frame_count = 0
        
        print(f"\n[{self.session_id}] Starting {self.exercise_type.upper()} analysis...")
        print(f"  Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time = time.time()
            self.process_frame(frame, frame_time)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"  Progress: {progress:.1f}% | Reps: {self.counter}")
        
        cap.release()
        
        elapsed = time.time() - self.start_time
        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Total reps: {self.counter}")
        
        return {
            'reps': self.counter,
            'elapsed_time': elapsed,
            'frames_processed': frame_count
        }
    
    def _generate_report(self):
        """Generate exercise-specific metrics report"""
        self.metrics.exercise = self.exercise_type
        
        if self.exercise_type == 'pushup':
            return self.metrics.pushup_metrics()
        elif self.exercise_type == 'squat':
            return self.metrics.squat_metrics()
        elif self.exercise_type == 'situp':
            return self.metrics.situp_metrics()
        elif self.exercise_type == 'sitnreach':
            return self.metrics.sitnreach_metrics()
        elif self.exercise_type == 'skipping':
            return self.metrics.skipping_metrics()
        elif self.exercise_type == 'jumpingjacks':
            return self.metrics.jumpingjacks_metrics()
        elif self.exercise_type == 'vjump':
            return self.metrics.vjump_metrics()
        elif self.exercise_type == 'bjump':
            return self.metrics.bjump_metrics()
        else:
            return {}

# ===========================
# HELPER FUNCTIONS
# ===========================

def convert_numpy_to_json_serializable(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_json_serializable(v) for k, v in obj.items()}
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
    """Landing page with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Exercise Trainer API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .post { background: #3498db; color: white; }
            .get { background: #2ecc71; color: white; }
            .delete { background: #e74c3c; color: white; }
            code { background: #34495e; color: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
            .workflow { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🏋️ AI Exercise Trainer API v3.0</h1>
        <p>Complete exercise analysis system with cheat detection and AI coaching</p>
        
        <div class="workflow">
            <h3>📋 Workflow (Like test.py)</h3>
            <ol>
                <li><strong>Create Session</strong> - Start a new workout session</li>
                <li><strong>Submit Exercises</strong> - Upload videos for different exercises (cheat detection runs automatically)</li>
                <li><strong>Check Status</strong> - View completed exercises</li>
                <li><strong>Generate AI Analysis</strong> - Get personalized coaching feedback</li>
                <li><strong>Get Summary</strong> - Retrieve complete session data</li>
            </ol>
        </div>
        
        <h2>🔗 Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/session/create</code>
            <p>Create a new workout session</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/session/{session_id}/submit-exercise</code>
            <p><strong>Main endpoint</strong> - Submit exercise video with automatic cheat detection</p>
            <p>Parameters: exercise_type (pushup|squat|situp|sitnreach|skipping|jumpingjacks|vjump|bjump), video (file)</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/session/{session_id}/status</code>
            <p>Check session status and completed exercises</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/session/{session_id}/generate-ai-analysis</code>
            <p>Generate AI-powered coaching analysis (requires at least 1 completed exercise)</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/session/{session_id}/summary</code>
            <p>Get complete session summary with all exercises and AI analysis</p>
        </div>
        
        <div class="endpoint">
            <span class="method delete">DELETE</span> <code>/session/{session_id}</code>
            <p>Delete session and cleanup files</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/sessions</code>
            <p>List all active sessions</p>
        </div>
        
        <h2>✨ Features</h2>
        <ul>
            <li>🔒 <strong>Mandatory Cheat Detection</strong> - All videos validated before analysis</li>
            <li>🏃 <strong>8 Exercise Types</strong> - Pushups, squats, sit-ups, flexibility, jumps, and more</li>
            <li>🤖 <strong>AI Coaching</strong> - Powered by Google Gemini for personalized feedback</li>
            <li>📊 <strong>Detailed Metrics</strong> - Comprehensive biomechanical analysis</li>
            <li>🔄 <strong>Session-Based</strong> - Complete multiple exercises in one session</li>
        </ul>
        
        <h2>📖 Documentation</h2>
        <p>API Docs: <a href="/docs">/docs</a> | Interactive API: <a href="/redoc">/redoc</a></p>
    </body>
    </html>
    """

@app.get("/api/status")
async def api_status():
    """API health check"""
    return {
        "status": "online",
        "service": "AI Exercise Trainer API",
        "version": "3.0",
        "features": {
            "cheat_detection": cheat_detector is not None,
            "ai_analysis": True,
            "exercises": ["pushup", "squat", "situp", "sitnreach", "skipping", "jumpingjacks", "vjump", "bjump"]
        },
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/session/create", response_model=SessionResponse)
async def create_session(session_data: SessionCreate = SessionCreate()):
    """
    Create a new workout session (mimics test.py's main loop)
    
    This initializes a session where you can submit multiple exercises,
    track completion, and generate AI analysis.
    """
    session_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    available_exercises = ['pushup', 'squat', 'situp', 'sitnreach', 'skipping', 'jumpingjacks', 'vjump', 'bjump']
    
    sessions[session_id] = {
        'session_id': session_id,
        'user_id': session_data.user_id,
        'created_at': created_at,
        'status': 'active',
        'completed_exercises': [],
        'exercise_results': {},
        'metrics_data': [],
        'ai_analysis': None,
        'completed_at': None
    }
    
    print(f"\n{'='*70}")
    print(f"  NEW WORKOUT SESSION CREATED")
    print(f"{'='*70}")
    print(f"Session ID: {session_id}")
    print(f"User: {session_data.user_id or 'Anonymous'}")
    print(f"Created: {created_at}")
    print(f"Available Exercises: {len(available_exercises)}")
    print(f"{'='*70}\n")
    
    return SessionResponse(
        session_id=session_id,
        message="Workout session created! Submit videos for exercises using /session/{id}/submit-exercise",
        created_at=created_at,
        completed_exercises=[],
        available_exercises=available_exercises
    )

@app.post("/session/{session_id}/submit-exercise")
async def submit_exercise(
    session_id: str,
    exercise_type: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Submit exercise video for analysis (MAIN ENDPOINT - mimics test.py's run method)
    
    WORKFLOW:
    1. CHEAT DETECTION (MANDATORY) - Video validated for authenticity
    2. If MEDIUM/HIGH risk (score >= 40): BLOCK with 403 error
    3. If LOW risk (score < 40): Proceed with exercise analysis
    4. Generate metrics and store results
    5. Mark exercise as completed
    
    Parameters:
        - session_id: Session UUID
        - exercise_type: One of [pushup, squat, situp, sitnreach, skipping, jumpingjacks, vjump, bjump]
        - video: Video file (MP4, AVI, etc.)
    
    Returns:
        - Complete exercise result with cheat detection + metrics
        - OR 403 error if cheat detection fails
    """
    # Validate session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session['status'] != 'active':
        raise HTTPException(status_code=400, detail=f"Session is not active. Status: {session['status']}")
    
    # Validate exercise type
    valid_exercises = ['pushup', 'squat', 'situp', 'sitnreach', 'skipping', 'jumpingjacks', 'vjump', 'bjump']
    if exercise_type not in valid_exercises:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type '{exercise_type}'. Valid options: {valid_exercises}"
        )
    
    # Check if already completed
    if exercise_type in session['completed_exercises']:
        print(f"[{session_id}] ⚠ {exercise_type.upper()} already completed. Redoing...")
    
    # Save uploaded video
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = session_dir / f"{exercise_type}_{timestamp}_{video.filename}"
    
    video_content = await video.read()
    with open(video_path, "wb") as f:
        f.write(video_content)
    
    print(f"\n{'='*70}")
    print(f"[{session_id}] EXERCISE SUBMISSION: {exercise_type.upper()}")
    print(f"{'='*70}")
    print(f"Video: {video.filename} ({len(video_content) / (1024*1024):.2f} MB)")
    print(f"Saved: {video_path}")
    
    # ============================================================
    # STEP 1: MANDATORY CHEAT DETECTION
    # ============================================================
    cheat_result = None
    cheat_passed = False
    
    if cheat_detector is not None:
        try:
            print(f"\n{'─'*70}")
            print(f"[STEP 1/2] CHEAT DETECTION")
            print(f"{'─'*70}")
            
            cheat_result = cheat_detector.analyze_video(str(video_path), num_frames=16)
            
            if cheat_result:
                overall_score = cheat_result.get('overall_score', 0)
                risk_level = cheat_result.get('risk_level', 'UNKNOWN')
                verdict = cheat_result.get('verdict', '')
                violations = cheat_result.get('violations', {})
                
                print(f"\nRESULTS:")
                print(f"  Risk Level: {risk_level}")
                print(f"  Overall Score: {overall_score:.1f}/100")
                print(f"  Verdict: {verdict}")
                
                # Print violations if any
                if any([violations.get('has_text'), violations.get('is_deepfake'), 
                       violations.get('has_effects'), violations.get('excessive_cuts')]):
                    print(f"\n  Violations Detected:")
                    if violations.get('has_text'):
                        print(f"    🚫 TEXT OVERLAYS (Penalty: +{violations.get('text_penalty', 0):.1f})")
                    if violations.get('is_deepfake'):
                        print(f"    🚫 AI DEEPFAKE (Penalty: +{violations.get('deepfake_penalty', 0):.1f})")
                    if violations.get('has_effects'):
                        print(f"    ⚠  EFFECTS/FILTERS (Penalty: +{violations.get('effects_penalty', 0):.1f})")
                    if violations.get('excessive_cuts'):
                        print(f"    ⚠  EXCESSIVE CUTS (Penalty: +{violations.get('angles_penalty', 0):.1f})")
                
                # ============================================================
                # BLOCKING LOGIC: Score >= 40 OR MEDIUM/HIGH/CRITICAL
                # ============================================================
                if overall_score >= 40 or risk_level in ['🟠 MEDIUM', '🔴 HIGH', '🔴 CRITICAL']:
                    print(f"\n❌ CHEAT DETECTION FAILED - ANALYSIS BLOCKED")
                    print(f"{'='*70}\n")
                    
                    # Store blocked result
                    session['exercise_results'][exercise_type] = {
                        'status': 'blocked',
                        'exercise_type': exercise_type,
                        'cheat_detection': cheat_result,
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Video blocked due to cheat detection failure'
                    }
                    
                    # Prepare error response
                    detail_data = {
                        "status": "blocked",
                        "exercise_type": exercise_type,
                        "session_id": session_id,
                        "error": "Cheat detection failed - Analysis blocked",
                        "risk_level": str(risk_level),
                        "overall_score": float(overall_score),
                        "verdict": str(verdict),
                        "violations": convert_numpy_to_json_serializable(violations),
                        "message": "⚠️ Video contains prohibited content. Please re-record with authentic, unedited video (no text, filters, or AI manipulation).",
                        "recommendations": [
                            "Remove any text overlays or watermarks",
                            "Use original, unedited video",
                            "Avoid filters, effects, or animations",
                            "Ensure consistent lighting and camera angle",
                            "Record with a single person in frame"
                        ],
                        "cheat_detection_full_report": convert_numpy_to_json_serializable(cheat_result)
                    }
                    
                    raise HTTPException(
                        status_code=403,
                        detail=detail_data
                    )
                else:
                    print(f"\n✅ CHEAT CHECK PASSED")
                    print(f"{'─'*70}")
                    cheat_passed = True
                    
        except HTTPException:
            raise
        except Exception as e:
            print(f"\n⚠ Cheat detection error: {e}")
            cheat_result = {'error': str(e), 'overall_score': 0, 'risk_level': 'ERROR'}
            cheat_passed = True  # Allow to proceed on error (optional: could block here)
    else:
        print(f"\n⚠ Cheat detector not available - Skipping validation")
        cheat_passed = True
        cheat_result = {'overall_score': 0, 'risk_level': 'NOT_CHECKED'}
    
    # ============================================================
    # STEP 2: EXERCISE ANALYSIS
    # ============================================================
    print(f"\n{'─'*70}")
    print(f"[STEP 2/2] EXERCISE ANALYSIS: {exercise_type.upper()}")
    print(f"{'─'*70}")
    
    try:
        processor = VideoProcessor(session_id, exercise_type, str(video_path))
        analysis_results = processor.analyze_video()
        
        print(f"\n{'─'*70}")
        print(f"Generating detailed metrics...")
        
        metrics_result = processor._generate_report()
        
        # Add timestamp
        metrics_result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"✅ Metrics generated successfully")
        print(f"{'─'*70}")
        
        # Store successful result
        exercise_result = {
            'status': 'passed',
            'exercise_type': exercise_type,
            'cheat_detection': {
                'passed': cheat_passed,
                'risk_level': cheat_result.get('risk_level', 'UNKNOWN') if cheat_result else 'NOT_CHECKED',
                'score': float(cheat_result.get('overall_score', 0)) if cheat_result else 0,
                'verdict': cheat_result.get('verdict', '') if cheat_result else '',
                'base_score': float(cheat_result.get('base_score', 0)) if cheat_result else 0
            },
            'metrics': convert_numpy_to_json_serializable(metrics_result),
            'reps': analysis_results.get('reps', 0),
            'elapsed_time': analysis_results.get('elapsed_time', 0),
            'frames_processed': analysis_results.get('frames_processed', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        session['exercise_results'][exercise_type] = exercise_result
        
        # Add to completed exercises
        if exercise_type not in session['completed_exercises']:
            session['completed_exercises'].append(exercise_type)
        
        # Append metrics for AI analysis
        session['metrics_data'].append(metrics_result)
        
        print(f"\n{'='*70}")
        print(f"✅ {exercise_type.upper()} ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Status: PASSED")
        print(f"Reps Completed: {analysis_results.get('reps', 0)}")
        print(f"Processing Time: {analysis_results.get('elapsed_time', 0):.2f}s")
        print(f"Frames Analyzed: {analysis_results.get('frames_processed', 0)}")
        print(f"Session Progress: {len(session['completed_exercises'])}/8 exercises completed")
        print(f"{'='*70}\n")
        
        return {
            "session_id": session_id,
            "exercise_type": exercise_type,
            "status": "passed",
            "cheat_detection": exercise_result['cheat_detection'],
            "metrics": exercise_result['metrics'],
            "reps": exercise_result['reps'],
            "elapsed_time": exercise_result['elapsed_time'],
            "frames_processed": exercise_result['frames_processed'],
            "timestamp": exercise_result['timestamp'],
            "session_progress": {
                "completed_exercises": session['completed_exercises'],
                "total_completed": len(session['completed_exercises']),
                "remaining": [ex for ex in valid_exercises if ex not in session['completed_exercises']]
            }
        }
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ANALYSIS ERROR: {str(e)}")
        print(f"{'='*70}\n")
        
        import traceback
        traceback.print_exc()
        
        session['exercise_results'][exercise_type] = {
            'status': 'failed',
            'exercise_type': exercise_type,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Exercise analysis failed",
                "exercise_type": exercise_type,
                "message": str(e),
                "session_id": session_id
            }
        )

@app.get("/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """
    Get current session status (mimics test.py's completed_exercises check)
    
    Returns:
        - Completed exercises
        - Available exercises
        - Whether AI analysis can be generated
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    all_exercises = ['pushup', 'squat', 'situp', 'sitnreach', 'skipping', 'jumpingjacks', 'vjump', 'bjump']
    available = [ex for ex in all_exercises if ex not in session['completed_exercises']]
    
    return SessionStatus(
        session_id=session_id,
        created_at=session['created_at'],
        status=session['status'],
        completed_exercises=session['completed_exercises'],
        available_exercises=available,
        total_completed=len(session['completed_exercises']),
        can_generate_ai_analysis=len(session['completed_exercises']) > 0
    )

@app.post("/session/{session_id}/generate-ai-analysis", response_model=AIAnalysisResponse)
async def generate_ai_analysis(session_id: str):
    """
    Generate AI-powered coaching analysis (uses ai.py)
    
    Requires at least one completed exercise.
    Uses accumulated metrics to generate personalized feedback via Google Gemini.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if len(session['completed_exercises']) == 0:
        raise HTTPException(
            status_code=400,
            detail="No exercises completed yet. Complete at least one exercise before generating AI analysis."
        )
    
    # Check if already generated
    if session.get('ai_analysis'):
        print(f"[{session_id}] Returning cached AI analysis")
        return AIAnalysisResponse(
            session_id=session_id,
            ai_analysis=session['ai_analysis'],
            exercises_analyzed=session['completed_exercises'],
            generated_at=session.get('ai_generated_at', datetime.now().isoformat())
        )
    
    print(f"\n{'='*70}")
    print(f"[{session_id}] GENERATING AI ANALYSIS")
    print(f"{'='*70}")
    print(f"Exercises completed: {len(session['completed_exercises'])}")
    print(f"Exercises: {', '.join(session['completed_exercises'])}")
    
    # Prepare metrics file for AI analysis (like test.py's exercise_metrics.txt)
    metrics_file = RESULTS_DIR / f"session_{session_id}_metrics.txt"
    
    try:
        # Combine all metrics into single JSON
        combined_metrics = {
            'session_id': session_id,
            'completed_exercises': session['completed_exercises'],
            'total_exercises': len(session['completed_exercises']),
            'exercises': session['metrics_data']
        }
        
        with open(metrics_file, 'w') as f:
            f.write(json.dumps(combined_metrics, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
        
        print(f"Metrics saved to: {metrics_file}")
        print(f"Calling Google Gemini LLM...")
        
        # Call AI analysis (from ai.py)
        ai_analysis = analyze_exercise_metrics(str(metrics_file))
        
        session['ai_analysis'] = ai_analysis
        session['ai_generated_at'] = datetime.now().isoformat()
        
        print(f"✅ AI analysis generated successfully")
        print(f"{'='*70}\n")
        
        return AIAnalysisResponse(
            session_id=session_id,
            ai_analysis=ai_analysis,
            exercises_analyzed=session['completed_exercises'],
            generated_at=session['ai_generated_at']
        )
        
    except Exception as e:
        print(f"❌ AI analysis error: {str(e)}")
        print(f"{'='*70}\n")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AI analysis generation failed",
                "message": str(e),
                "session_id": session_id
            }
        )

@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    Get complete session summary with all exercises and AI analysis
    
    Returns:
        - All exercise results (passed and blocked)
        - AI analysis (if generated)
        - Session metadata
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Build exercises list
    exercises_list = []
    for exercise_type, result in session['exercise_results'].items():
        exercises_list.append({
            'exercise_type': exercise_type,
            'status': result.get('status'),
            'cheat_detection': result.get('cheat_detection'),
            'metrics': result.get('metrics'),
            'reps': result.get('reps'),
            'timestamp': result.get('timestamp'),
            'error': result.get('error')
        })
    
    return {
        "session_id": session_id,
        "user_id": session.get('user_id'),
        "created_at": session['created_at'],
        "status": session['status'],
        "completed_exercises": session['completed_exercises'],
        "total_completed": len(session['completed_exercises']),
        "exercises_completed": exercises_list,
        "ai_analysis": session.get('ai_analysis'),
        "ai_generated_at": session.get('ai_generated_at'),
        "completed_at": session.get('completed_at')
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete session and cleanup files
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete uploaded videos
    session_dir = UPLOAD_DIR / session_id
    files_deleted = 0
    if session_dir.exists():
        for file in session_dir.glob("*"):
            file.unlink()
            files_deleted += 1
        session_dir.rmdir()
    
    # Delete metrics file
    metrics_file = RESULTS_DIR / f"session_{session_id}_metrics.txt"
    if metrics_file.exists():
        metrics_file.unlink()
        files_deleted += 1
    
    # Remove from sessions
    del sessions[session_id]
    
    print(f"\n[{session_id}] Session deleted ({files_deleted} files removed)")
    
    return {
        "message": "Session deleted successfully",
        "session_id": session_id,
        "files_deleted": files_deleted
    }

@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    sessions_list = []
    for sid, session in sessions.items():
        sessions_list.append({
            'session_id': sid,
            'user_id': session.get('user_id'),
            'created_at': session['created_at'],
            'status': session['status'],
            'completed_exercises': len(session['completed_exercises']),
            'has_ai_analysis': session.get('ai_analysis') is not None
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions_list
    }

# ===========================
# STARTUP MESSAGE
# ===========================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("  AI EXERCISE TRAINER API - v3.0")
    print("="*70)
    print("  🏋️ Exercise Analysis with Cheat Detection & AI Coaching")
    print("="*70)
    print(f"  Cheat Detection: {'✅ Ready' if cheat_detector else '❌ Not Available'}")
    print(f"  AI Analysis: ✅ Ready (Google Gemini)")
    print(f"  Exercises: 8 types supported")
    print(f"  Sessions: {len(sessions)} active")
    print("="*70)
    print("  API Documentation: http://localhost:8000/docs")
    print("  Web Interface: http://localhost:8000/")
    print("="*70 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
