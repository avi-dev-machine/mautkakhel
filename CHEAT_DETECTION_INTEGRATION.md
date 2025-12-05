# Cheat Detection Integration - Documentation

## Overview
Successfully integrated cheat detection from `cheat.py` into `server.py` with **MANDATORY blocking** for MEDIUM and HIGH risk videos.

## What Was Changed

### 1. Import Integration
- Added `DeepfakeVideoDetector` import from `cheat.py`
- Initialized global cheat detector at startup (loaded once, reused for all sessions)

### 2. Cheat Detection Logic

#### Blocking Thresholds:
- **LOW RISK** (score < 40): ✅ Analysis proceeds normally
- **MEDIUM RISK** (score >= 40): ❌ Analysis BLOCKED
- **HIGH RISK** (score >= 60): ❌ Analysis BLOCKED
- **CRITICAL** (text/deepfake detected): ❌ Analysis BLOCKED

#### Detection Checkpoints (from cheat.py):
1. **Camera Angle Changes** - Detects excessive cuts/edits
2. **Effects & Text** - Detects filters, overlays, text, animations
3. **Multiple People** - Detects if more than one person appears
4. **AI Deepfake** - Detects AI-generated or manipulated content

## API Endpoints Modified

### 1. `/upload-video/` (Simple Upload & Analyze)
**Added:** Pre-analysis cheat detection
- Video uploaded → Cheat check runs → Blocks if MEDIUM/HIGH
- Returns `403 Forbidden` with full cheat detection report if blocked
- Proceeds to exercise analysis only if risk is LOW

**Response when BLOCKED:**
```json
{
  "error": "Analysis blocked due to cheating detection",
  "risk_level": "MEDIUM" | "HIGH" | "CRITICAL",
  "overall_score": 45.2,
  "verdict": "Edited video with suspicious content",
  "violations": {
    "has_text": true,
    "is_deepfake": false,
    "has_effects": true,
    "excessive_cuts": false
  },
  "message": "Video contains suspicious content. Analysis cannot proceed.",
  "cheat_detection_full_report": { ... }
}
```

**Response when PASSED:**
```json
{
  "session_id": "uuid",
  "status": "completed",
  "exercise_type": "pushup",
  "cheat_detection": {
    "passed": true,
    "risk_level": "LOW",
    "score": 15.3
  },
  "results": { ... }
}
```

### 2. `/session/{session_id}/analyze` (Background Analysis)
**Added:** Pre-analysis cheat detection with same blocking logic
- Runs cheat check before starting exercise analysis
- Raises `403 HTTPException` if MEDIUM/HIGH risk detected
- Updates session status to `blocked_cheating`
- Stores full cheat detection results in session

**Response when BLOCKED:**
```json
{
  "detail": {
    "error": "Analysis blocked due to cheating detection",
    "risk_level": "HIGH",
    "overall_score": 72.5,
    "verdict": "CRITICAL: AI-generated deepfake detected",
    "violations": { ... },
    "message": "Video contains suspicious content. Analysis cannot proceed.",
    "cheat_detection_full_report": { ... }
  }
}
```

**Response when PASSED:**
```json
{
  "message": "Analysis started - Cheat check passed",
  "session_id": "uuid",
  "exercise_type": "squat",
  "status": "processing",
  "cheat_detection": {
    "passed": true,
    "risk_level": "LOW",
    "score": 12.8
  }
}
```

### 3. `/session/{session_id}/status` (Check Status)
**Enhanced:** Now includes cheat detection information

**Response:**
```json
{
  "session_id": "uuid",
  "status": "blocked_cheating" | "processing" | "completed",
  "progress": 0.75,
  "message": "Analysis blocked due to cheating detection",
  "cheat_detection": {
    "risk_level": "MEDIUM",
    "overall_score": 48.3,
    "verdict": "Edited with effects and filters",
    "violations": {
      "has_text": false,
      "is_deepfake": false,
      "has_effects": true,
      "excessive_cuts": true
    }
  },
  "blocked_reason": "MEDIUM risk detected - Score: 48.3/100"
}
```

### 4. `/session/{session_id}/report` (Get Report)
**Enhanced:** Returns `403` if session was blocked due to cheating

**Response when blocked:**
```json
{
  "detail": {
    "error": "Analysis was blocked due to cheating detection",
    "cheat_detection": { ... },
    "blocked_reason": "HIGH risk detected - Score: 65.2/100"
  }
}
```

**Response when completed:**
```json
{
  "session_id": "uuid",
  "exercise": "pushup",
  "report": { ... },
  "timestamp": "2025-12-06T...",
  "cheat_detection": {
    "passed": true,
    "risk_level": "LOW",
    "overall_score": 18.4,
    "verdict": "Video appears authentic"
  }
}
```

## Session States

New session state added:
- `created` - Session initialized
- `uploaded` - Video uploaded, ready for analysis
- **`blocked_cheating`** - ⚠️ **NEW**: Cheat detection blocked analysis
- `processing` - Exercise analysis in progress
- `completed` - Analysis finished successfully
- `failed` - Analysis error

## Console Output

When cheat detection runs:
```
============================================================
  PRE-ANALYSIS CHEAT DETECTION
============================================================
Session ID: abc-123
Exercise: PUSHUP
Video: uploads/video.mp4
============================================================

[abc-123] Running cheat detection...
Using device: cpu

Loading models for checkpoints...
  [1/4] Loading deepfake detection model...
  ✓ Deepfake detector loaded
  ...

[abc-123] CHEAT DETECTION RESULTS:
  Risk Level: MEDIUM
  Score: 45.2/100
  Verdict: Edited video with effects
  Has Text: false
  Is Deepfake: false
  Has Effects: true

[abc-123] ❌ ANALYSIS BLOCKED - MEDIUM RISK DETECTED
[abc-123] Score: 45.2/100 (Threshold: 40)
============================================================
```

Or when passed:
```
[abc-123] ✓ CHEAT CHECK PASSED - LOW RISK
[abc-123] Score: 15.3/100 (Below threshold: 40)
[abc-123] Proceeding with exercise analysis...

============================================================
  EXERCISE ANALYSIS STARTED
============================================================
```

## Error Handling

1. **Cheat detector initialization fails**: 
   - System continues without cheat detection
   - Warning logged to console
   - All videos are allowed (no blocking)

2. **Cheat detection runtime error**:
   - Error logged
   - Session proceeds with exercise analysis
   - `cheat_detection_error` stored in session

3. **Cheat detector unavailable**:
   - Analysis proceeds normally
   - No blocking occurs

## Testing

### Test with legitimate video (should pass):
```bash
curl -X POST "http://localhost:8000/upload-video/" \
  -F "file=@legitimate_exercise.mp4" \
  -F "exercise_type=pushup"
```

### Test with edited video (should block if score >= 40):
```bash
curl -X POST "http://localhost:8000/upload-video/" \
  -F "file=@edited_with_text.mp4" \
  -F "exercise_type=squat"
```

Expected response: `403 Forbidden` with cheat detection report

### Using Session API:
```bash
# 1. Create session
curl -X POST "http://localhost:8000/session/create"

# 2. Upload video
curl -X POST "http://localhost:8000/session/{session_id}/upload?exercise_type=pushup" \
  -F "video=@video.mp4"

# 3. Start analysis (will block if cheating detected)
curl -X POST "http://localhost:8000/session/{session_id}/analyze"

# 4. Check status
curl "http://localhost:8000/session/{session_id}/status"

# 5. Get report (if not blocked)
curl "http://localhost:8000/session/{session_id}/report"
```

## Key Features

✅ **Zero-tolerance for cheating**: MEDIUM and HIGH risk videos are blocked  
✅ **Comprehensive detection**: 4 checkpoints analyze video authenticity  
✅ **Detailed reports**: Full breakdown of violations and risk factors  
✅ **Session persistence**: Cheat detection results stored with session  
✅ **Graceful degradation**: System continues if cheat detector unavailable  
✅ **Clear HTTP status codes**: `403 Forbidden` for blocked videos  
✅ **Transparent logging**: Detailed console output for debugging  

## Summary

The system now **MANDATORILY** checks all uploaded videos for cheating before allowing exercise analysis. Videos with MEDIUM (score >= 40) or HIGH (score >= 60) risk levels are **BLOCKED** with a `403 Forbidden` response containing the full cheat detection report. Only LOW-risk videos (score < 40) proceed to exercise analysis.
