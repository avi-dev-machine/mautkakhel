# 🚀 Hugging Face Spaces Deployment Guide

## 📋 Prerequisites

1. Hugging Face account ([sign up here](https://huggingface.co/join))
2. Git installed locally
3. Google API key for Gemini ([get it here](https://makersuite.google.com/app/apikey))

## 🛠️ Deployment Steps

### 1. Create New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `ai-exercise-trainer` (or your choice)
   - **License**: MIT
   - **Select SDK**: Docker
   - **Space hardware**: CPU basic (free tier works!)
   - **Visibility**: Public or Private

### 2. Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-exercise-trainer
cd ai-exercise-trainer
```

### 3. Copy Application Files

Copy these files from your project to the space directory:

**Required Files:**
```bash
# Core application
server.py
utils.py
metrics.py
cheat.py
ai.py

# Models
yolo11n-pose.pt
yolo11n-pose.onnx  # Optional, for faster inference

# Configuration
Dockerfile
requirements.txt
requirements_api.txt
.dockerignore
README.md  # Rename README_HF.md to README.md
```

**File Checklist:**
- ✅ `server.py` - Main FastAPI application
- ✅ `utils.py` - Pose calibration utilities
- ✅ `metrics.py` - Performance metrics calculation
- ✅ `cheat.py` - Cheat detection system
- ✅ `ai.py` - Google Gemini AI analysis
- ✅ `yolo11n-pose.pt` - YOLO11 pose model (~6MB)
- ✅ `Dockerfile` - Container configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `requirements_api.txt` - API dependencies
- ✅ `.dockerignore` - Files to exclude from build
- ✅ `README.md` - Space documentation (use README_HF.md)

### 4. Create README with Metadata

Your `README.md` must include the metadata header:

```yaml
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
```

### 5. Configure Secrets (Required for AI Analysis)

1. Go to your Space settings
2. Navigate to **"Repository secrets"**
3. Add secret:
   - **Name**: `GOOGLE_API_KEY`
   - **Value**: Your Google Gemini API key

### 6. Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment of AI Exercise Trainer"
git push
```

### 7. Monitor Build

1. Go to your Space page on Hugging Face
2. Click **"Building"** tab to see build logs
3. Wait for build to complete (~5-10 minutes)
4. Status should change to **"Running"**

## 🧪 Testing Your Deployment

### 1. Access the API

Your space will be available at:
```
https://YOUR_USERNAME-ai-exercise-trainer.hf.space
```

### 2. Test Health Check

```bash
curl https://YOUR_USERNAME-ai-exercise-trainer.hf.space/api/status
```

Expected response:
```json
{
  "status": "online",
  "service": "AI Exercise Trainer API",
  "version": "3.0",
  "features": {
    "cheat_detection": true,
    "ai_analysis": true,
    "exercises": ["pushup", "squat", "situp", "sitnreach", "skipping", "jumpingjacks", "vjump", "bjump"]
  }
}
```

### 3. Test Exercise Analysis

```bash
# Create session
SESSION_ID=$(curl -X POST "https://YOUR_USERNAME-ai-exercise-trainer.hf.space/session/create" | jq -r '.session_id')

# Submit exercise video
curl -X POST "https://YOUR_USERNAME-ai-exercise-trainer.hf.space/session/$SESSION_ID/submit-exercise" \
  -F "exercise_type=pushup" \
  -F "video=@test_pushup.mp4"

# Generate AI analysis
curl -X POST "https://YOUR_USERNAME-ai-exercise-trainer.hf.space/session/$SESSION_ID/generate-ai-analysis"

# Get summary
curl "https://YOUR_USERNAME-ai-exercise-trainer.hf.space/session/$SESSION_ID/summary"
```

## 📊 Performance Considerations

### CPU vs GPU

- **CPU (Free Tier)**: Works well, ~10-30s per video
- **GPU (Paid)**: Faster inference, ~5-15s per video
- Cheat detection models run efficiently on CPU

### Memory Usage

- Base: ~2GB RAM
- With model loaded: ~3-4GB RAM
- Per session: ~100-500MB (depends on video size)

### Optimization Tips

1. **Use ONNX model** for faster inference (if available)
2. **Limit video resolution** to 720p or lower
3. **Process videos in chunks** for large files
4. **Enable caching** for repeated requests

## 🔧 Troubleshooting

### Build Fails

**Issue**: Docker build timeout
**Solution**: 
- Reduce model sizes
- Use pre-built wheels for torch
- Remove unnecessary dependencies

**Issue**: Out of memory during build
**Solution**:
- Upgrade to paid tier
- Optimize Dockerfile layers
- Use multi-stage builds

### Runtime Issues

**Issue**: Port 7860 not accessible
**Solution**: 
- Check `app_port: 7860` in README.md
- Ensure CMD uses correct port in Dockerfile

**Issue**: Cheat detection fails
**Solution**:
- Check transformers cache is writable
- Verify model downloads complete
- Check logs for specific errors

**Issue**: AI analysis not working
**Solution**:
- Verify `GOOGLE_API_KEY` is set in secrets
- Check quota limits on Gemini API
- Review ai.py error logs

### Performance Issues

**Issue**: Slow video processing
**Solution**:
- Use ONNX runtime
- Reduce num_frames in cheat detection
- Process at lower FPS

**Issue**: High memory usage
**Solution**:
- Clear session data regularly
- Implement session timeouts
- Use delete endpoint after analysis

## 🔄 Updating Your Space

```bash
# Pull latest changes
git pull

# Make your updates
# Edit files...

# Commit and push
git add .
git commit -m "Update description"
git push
```

Hugging Face will automatically rebuild and redeploy.

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker SDK Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)

## 💡 Best Practices

1. **Enable authentication** for production use
2. **Set rate limits** to prevent abuse
3. **Monitor usage** via HF Space analytics
4. **Regular updates** for security patches
5. **Backup session data** if needed
6. **Document API changes** in README

## 🆘 Support

- **Hugging Face Forums**: [discuss.huggingface.co](https://discuss.huggingface.co/)
- **GitHub Issues**: Create issues for bugs
- **API Documentation**: Check `/docs` endpoint

## 📄 License

MIT License - See LICENSE file for details

---

**Happy Deploying! 🚀**
