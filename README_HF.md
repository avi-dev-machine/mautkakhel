---
title: AI Exercise Trainer
emoji: 🏋️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🏋️ AI Exercise Trainer API

**Complete exercise analysis system with cheat detection and AI coaching**

## 🚀 Features

- **🔒 Mandatory Cheat Detection** - All videos validated for authenticity before analysis
- **🏃 8 Exercise Types** - Pushups, squats, sit-ups, flexibility tests, jumping exercises
- **🤖 AI Coaching** - Powered by Google Gemini for personalized feedback
- **📊 Detailed Metrics** - Comprehensive biomechanical analysis with YOLO11 pose detection
- **🔄 Session-Based** - Complete multiple exercises in one workout session

## 📋 API Workflow

1. **Create Session** - Start a new workout session (`POST /session/create`)
2. **Submit Exercises** - Upload videos with automatic cheat detection (`POST /session/{id}/submit-exercise`)
3. **Check Status** - View completed exercises (`GET /session/{id}/status`)
4. **Generate AI Analysis** - Get personalized coaching feedback (`POST /session/{id}/generate-ai-analysis`)
5. **Get Summary** - Retrieve complete session data (`GET /session/{id}/summary`)

## 🔗 API Endpoints

### Core Endpoints
- `POST /session/create` - Create new workout session
- `POST /session/{session_id}/submit-exercise` - Submit exercise video (with cheat detection)
- `GET /session/{session_id}/status` - Check session status
- `POST /session/{session_id}/generate-ai-analysis` - Generate AI coaching
- `GET /session/{session_id}/summary` - Get complete session summary
- `DELETE /session/{session_id}` - Delete session
- `GET /sessions` - List all active sessions

### Info Endpoints
- `GET /` - Landing page with documentation
- `GET /api/status` - API health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## 🎯 Supported Exercises

1. **Pushups** - Upper body strength with form analysis
2. **Squats** - Lower body strength with depth tracking
3. **Sit-ups** - Core strength with ROM validation
4. **Sit & Reach** - Flexibility assessment
5. **Skipping** - Cardio with jump height tracking
6. **Jumping Jacks** - Full-body coordination
7. **Vertical Jump** - Explosive power measurement
8. **Broad Jump** - Horizontal power assessment

## 🔐 Cheat Detection System

4-checkpoint verification system:
- ✅ Camera angle consistency
- ✅ Text/watermark detection
- ✅ Multiple people detection
- ✅ AI deepfake detection

**Blocking Logic**: Videos with score ≥ 40 or MEDIUM/HIGH risk are automatically blocked.

## 💡 Usage Example

```bash
# 1. Create session
curl -X POST "https://your-space.hf.space/session/create"

# 2. Submit exercise
curl -X POST "https://your-space.hf.space/session/{session_id}/submit-exercise" \
  -F "exercise_type=pushup" \
  -F "video=@pushup_video.mp4"

# 3. Generate AI analysis
curl -X POST "https://your-space.hf.space/session/{session_id}/generate-ai-analysis"

# 4. Get summary
curl "https://your-space.hf.space/session/{session_id}/summary"
```

## 🛠️ Technical Stack

- **Backend**: FastAPI + Uvicorn
- **Pose Detection**: YOLO11n-pose (Ultralytics)
- **Cheat Detection**: Transformers (VideoMAE, CLIP, EasyOCR)
- **AI Analysis**: Google Gemini 2.5 Flash via LangChain
- **Video Processing**: OpenCV + FFmpeg

## ⚙️ Environment Variables

Required for AI analysis (optional):
```bash
GOOGLE_API_KEY=your_gemini_api_key
```

## 📊 Output Format

Each exercise returns comprehensive metrics including:
- Rep counts and form quality
- Biomechanical angles and measurements
- Tempo and timing analysis
- Form violations and recommendations
- Cheat detection results

## 🚦 Status Codes

- `200` - Success
- `403` - Cheat detection failed (video blocked)
- `404` - Session not found
- `500` - Analysis error

## 📝 Notes

- Maximum video size: Recommended < 50MB
- Processing time: ~5-30 seconds per video
- Session data is stored in memory (not persistent)
- AI analysis requires at least 1 completed exercise

## 🔗 Links

- [API Documentation](/docs)
- [Alternative Docs](/redoc)
- [Health Check](/api/status)

## 📄 License

MIT License - See LICENSE file for details
