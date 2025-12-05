---
title: AI Exercise Trainer
emoji: 💪
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "4.36.0"
app_file: server.py
pinned: false
---

# AI Exercise Trainer

Real-time exercise form evaluation using YOLO11n-pose for accurate rep counting and form feedback.

## Features
- 💪 Pushup detection with elbow angle tracking
- 🏋️ Squat detection with knee angle tracking
- 📊 Real-time rep counting
- ✅ Form quality assessment
- 🎯 Angle smoothing for stable detection

## API Endpoints
- `POST /upload-video/` - Upload video for exercise analysis
- Returns: Rep counts, form feedback, and performance metrics
