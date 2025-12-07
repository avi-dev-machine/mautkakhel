# Server.py API - Comprehensive Documentation

## Overview
`server.py` is a FastAPI-based REST API that implements the exercise analysis workflow from `test.py` with integrated cheat detection from `cheat.py`. The API follows a session-based approach where users can submit multiple exercises, get them validated through cheat detection, and generate AI-powered analysis.

---

## Architecture

### Workflow (Mimics test.py)

```
1. CREATE SESSION
   └─ POST /session/create
      └─ Returns: session_id
              ↓
2. SUBMIT EXERCISES (Repeat for each exercise)
   └─ POST /session/{id}/submit-exercise
      ├─ STEP 1: Cheat Detection (MANDATORY)
      │  ├─ Score < 40: PASS → Continue
      │  └─ Score >= 40: BLOCK → Return 403
      ├─ STEP 2: Exercise Analysis (if passed)
      │  ├─ YOLO11 Pose Detection
      │  ├─ Biomechanical Analysis
      │  └─ Metrics Generation
      └─ Store Result + Mark as Completed
              ↓
3. VIEW SESSION STATUS
   └─ GET /session/{id}/status
      └─ Returns: completed_exercises, available_exercises
              ↓
4. GENERATE AI ANALYSIS
   └─ POST /session/{id}/generate-ai-analysis
      └─ Uses accumulated metrics → Gemini LLM → Personalized feedback
              ↓
5. GET FULL SUMMARY
   └─ GET /session/{id}/summary
      └─ Returns: All exercises + AI analysis + Complete session data
```

---

## API Endpoints

### 1. POST /session/create
**Purpose:** Create a new workout session (like test.py's main loop initialization)

**Request Body:**
```json
{
  "user_id": "optional_user_identifier"
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "message": "Workout session created! Submit videos for exercises.",
  "created_at": "2025-12-07T10:30:00",
  "completed_exercises": []
}
```

---

### 2. POST /session/{session_id}/submit-exercise
**Purpose:** Submit an exercise video with MANDATORY cheat detection

**Parameters:**
- `session_id` (path): Session UUID
- `exercise_type` (form): One of [pushup, squat, situp, sitnreach, skipping, jumpingjacks, vjump, bjump]
- `video` (file): MP4/AVI video file

**Workflow:**
1. Save uploaded video
2. Run cheat detection (4 checkpoints)
3. If score >= 40: BLOCK (return 403)
4. If score < 40: Proceed with analysis
5. Store metrics for AI analysis
6. Mark exercise as completed

**Success Response (200):**
```json
{
  "session_id": "uuid",
  "exercise_type": "pushup",
  "status": "passed",
  "cheat_detection": {
    "passed": true,
    "risk_level": "🟢 LOW",
    "score": 15.2,
    "verdict": "LOW RISK - Appears Authentic"
  },
  "metrics": {
    "exercise": "pushup",
    "reps": {
      "total": 20,
      "good_form": 18,
      "bad_form": 2
    },
    "overall_score": 88.5,
    ...
  },
  "reps": 20,
  "timestamp": "2025-12-07T10:35:00",
  "completed_exercises": ["pushup"],
  "total_completed": 1
}
```

**Blocked Response (403):**
```json
{
  "detail": {
    "status": "blocked",
    "exercise_type": "pushup",
    "error": "Cheat detection failed - Analysis blocked",
    "risk_level": "🔴 HIGH",
    "overall_score": 67.5,
    "verdict": "HIGH RISK - Heavily Manipulated",
    "violations": {
      "has_text": true,
      "is_deepfake": false,
      "has_effects": true,
      "excessive_cuts": false,
      "text_penalty": 52.5,
      ...
    },
    "message": "Video contains prohibited content (text overlays, deepfakes, or heavy manipulation). Please re-record with authentic video.",
    "cheat_detection_full_report": {...}
  }
}
```

---

### 3. GET /session/{session_id}/status
**Purpose:** Check session status and completed exercises

**Response:**
```json
{
  "session_id": "uuid",
  "created_at": "2025-12-07T10:30:00",
  "completed_exercises": ["pushup", "squat", "situp"],
  "available_exercises": ["sitnreach", "skipping", "jumpingjacks", "vjump", "bjump"],
  "total_exercises_completed": 3,
  "can_generate_ai_analysis": true
}
```

---

### 4. POST /session/{session_id}/generate-ai-analysis
**Purpose:** Generate AI-powered analysis using Google Gemini (like ai.py)

**Requirements:**
- At least one exercise completed in session

**Response:**
```json
{
  "session_id": "uuid",
  "ai_analysis": "Exercise Performance Analysis\n============================\n\nOverall Performance: 87.5/100 - EXCELLENT\n\nStrengths:\n✓ Excellent depth consistency\n✓ Good tempo control\n...",
  "exercises_analyzed": ["pushup", "squat", "situp"],
  "generated_at": "2025-12-07T10:40:00"
}
```

---

### 5. GET /session/{session_id}/summary
**Purpose:** Get complete session summary with all exercises and AI analysis

**Response:**
```json
{
  "session_id": "uuid",
  "exercises_completed": [
    {
      "exercise_type": "pushup",
      "status": "passed",
      "cheat_detection": {...},
      "metrics": {...},
      "timestamp": "2025-12-07T10:35:00"
    },
    {
      "exercise_type": "squat",
      "status": "blocked",
      "cheat_detection": {...},
      "timestamp": "2025-12-07T10:38:00"
    }
  ],
  "ai_analysis": "Complete analysis text...",
  "created_at": "2025-12-07T10:30:00",
  "completed_at": "2025-12-07T10:45:00"
}
```

---

### 6. DELETE /session/{session_id}
**Purpose:** Delete session and cleanup files

**Response:**
```json
{
  "message": "Session deleted successfully",
  "session_id": "uuid",
  "files_deleted": 3
}
```

---

### 7. GET /sessions
**Purpose:** List all active sessions

**Response:**
```json
{
  "total_sessions": 5,
  "sessions": [
    {
      "session_id": "uuid1",
      "created_at": "2025-12-07T10:30:00",
      "status": "active",
      "completed_exercises": 3
    },
    ...
  ]
}
```

---

## Key Features

### 1. Cheat Detection Integration
- **Mandatory for all exercises**
- Runs automatically before analysis
- Blocks videos with score >= 40
- Zero tolerance for text overlays and deepfakes

### 2. Session Management
- Persistent session storage
- Tracks completed exercises (like test.py)
- Prevents duplicate exercise submissions (with redo option)
- Accumulates metrics for AI analysis

### 3. Exercise Tracking
- 8 supported exercises
- Individual cheat detection per exercise
- Metrics stored for AI analysis
- Completed exercises list maintained

### 4. AI Analysis Generation
- Uses accumulated metrics from all exercises
- Google Gemini 2.5 Flash LLM
- Personalized feedback and sport recommendations
- Generated on-demand after exercises

---

## Implementation Details

### Session Structure
```python
sessions[session_id] = {
    'session_id': 'uuid',
    'user_id': 'optional',
    'created_at': 'timestamp',
    'status': 'active',
    'completed_exercises': ['pushup', 'squat'],  # Like test.py
    'exercise_results': {
        'pushup': {
            'status': 'passed',
            'cheat_detection': {...},
            'metrics': {...},
            'timestamp': '...'
        }
    },
    'metrics_data': [...],  # For AI analysis
    'ai_analysis': None,
    'completed_at': None
}
```

### Cheat Detection Workflow
```python
# 1. Upload video
video_path = save_video(file)

# 2. Run cheat detection
cheat_result = cheat_detector.analyze_video(video_path)

# 3. Check score
if cheat_result['overall_score'] >= 40:
    # BLOCK
    raise HTTPException(403, detail={...})
else:
    # PROCEED
    run_exercise_analysis()
```

### Exercise Analysis Workflow
```python
# 1. Initialize processor
processor = VideoProcessor(session_id, exercise_type, video_path)

# 2. Process video frames
results = processor.analyze_video()

# 3. Generate metrics
metrics = processor._generate_report()

# 4. Store in session
session['exercise_results'][exercise_type] = {
    'status': 'passed',
    'metrics': metrics,
    ...
}

# 5. Mark as completed
session['completed_exercises'].append(exercise_type)
```

---

## Usage Example

### Complete Workflow
```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Create session
response = requests.post(f"{BASE_URL}/session/create", json={"user_id": "user123"})
session_id = response.json()["session_id"]

# 2. Submit pushups
with open("pushup_video.mp4", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/session/{session_id}/submit-exercise",
        params={"exercise_type": "pushup"},
        files={"video": f}
    )
    
if response.status_code == 200:
    print("Pushup analysis passed!")
elif response.status_code == 403:
    print("Pushup blocked due to cheat detection!")

# 3. Submit squats
with open("squat_video.mp4", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/session/{session_id}/submit-exercise",
        params={"exercise_type": "squat"},
        files={"video": f}
    )

# 4. Check status
response = requests.get(f"{BASE_URL}/session/{session_id}/status")
status = response.json()
print(f"Completed: {status['completed_exercises']}")

# 5. Generate AI analysis
response = requests.post(f"{BASE_URL}/session/{session_id}/generate-ai-analysis")
ai_analysis = response.json()["ai_analysis"]
print(ai_analysis)

# 6. Get summary
response = requests.get(f"{BASE_URL}/session/{session_id}/summary")
summary = response.json()
```

---

## Comparison: test.py vs server.py

| Feature | test.py | server.py API |
|---------|---------|---------------|
| Session Management | Local `completed_exercises` set | REST API session storage |
| Exercise Loop | CLI menu loop | POST requests per exercise |
| Cheat Detection | Manual call | Automatic before analysis |
| Metrics Storage | `exercise_metrics.txt` | Session `metrics_data` array |
| AI Analysis | `ai.py` function call | POST /generate-ai-analysis |
| Redo Logic | Interactive prompt | Automatic (overwrites) |
| Video Source | Webcam or file path | File upload only |
| Real-time Display | OpenCV window | No (server-side) |

---

## Error Handling

### Common Errors

**404 - Session Not Found**
```json
{
  "detail": "Session not found"
}
```

**400 - Invalid Exercise Type**
```json
{
  "detail": "Invalid exercise type. Must be one of: [pushup, squat, ...]"
}
```

**403 - Cheat Detection Block**
```json
{
  "detail": {
    "status": "blocked",
    "risk_level": "🔴 HIGH",
    "overall_score": 67.5,
    ...
  }
}
```

**500 - Analysis Error**
```json
{
  "detail": "Analysis error: Could not process video"
}
```

---

## Summary

`server.py` is a production-ready FastAPI application that:

✅ **Mimics test.py workflow** with session-based exercise tracking
✅ **Mandatory cheat detection** for all video submissions
✅ **Zero-tolerance blocking** for MEDIUM/HIGH risk videos
✅ **Exercise completion tracking** like test.py's completed_exercises
✅ **Accumulated metrics** for comprehensive AI analysis
✅ **Google Gemini integration** for personalized coaching
✅ **RESTful API** for web/mobile client integration
✅ **Robust error handling** with detailed feedback
✅ **File management** with automatic cleanup
✅ **Session persistence** across multiple exercise submissions

**Key Innovation**: Combines the interactive workflow of test.py with the authentication power of cheat.py in a scalable REST API, enabling multi-exercise workout sessions with AI-powered analysis.
