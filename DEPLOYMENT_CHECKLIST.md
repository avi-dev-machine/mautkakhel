# 📋 Hugging Face Deployment Checklist

## ✅ Pre-Deployment

### Files Required
- [ ] `server.py` - Main FastAPI application
- [ ] `utils.py` - Pose calibration utilities  
- [ ] `metrics.py` - Performance metrics
- [ ] `cheat.py` - Cheat detection system
- [ ] `ai.py` - Google Gemini AI integration
- [ ] `yolo11n-pose.pt` - YOLO pose model (~6MB)
- [ ] `Dockerfile` - Optimized for HF Spaces
- [ ] `requirements.txt` - Core dependencies
- [ ] `requirements_api.txt` - API dependencies
- [ ] `.dockerignore` - Build optimization
- [ ] `README.md` - With HF metadata header

### Optional Files
- [ ] `yolo11n-pose.onnx` - Faster inference model
- [ ] `.env.example` - Configuration template
- [ ] `app.py` - Alternative entry point

### Configuration
- [ ] Hugging Face account created
- [ ] Google API key obtained (for AI analysis)
- [ ] Git configured locally
- [ ] Space created on HF platform

## 🚀 Deployment Steps

### 1. Create Space
- [ ] Navigate to https://huggingface.co/spaces
- [ ] Click "Create new Space"
- [ ] Set name: `ai-exercise-trainer`
- [ ] Select SDK: **Docker**
- [ ] Set license: MIT
- [ ] Choose visibility (Public/Private)
- [ ] Click "Create Space"

### 2. Clone Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-exercise-trainer
cd ai-exercise-trainer
```
- [ ] Space cloned successfully

### 3. Copy Files
**Option A: Manual**
```bash
cp server.py utils.py metrics.py cheat.py ai.py ai-exercise-trainer/
cp yolo11n-pose.pt ai-exercise-trainer/
cp Dockerfile requirements.txt requirements_api.txt .dockerignore ai-exercise-trainer/
cp README_HF.md ai-exercise-trainer/README.md
```

**Option B: Use deployment script**
```bash
# Linux/Mac
bash deploy_hf.sh

# Windows
.\deploy_hf.ps1
```
- [ ] All files copied

### 4. Configure README Metadata
Ensure README.md starts with:
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
- [ ] README metadata configured

### 5. Set Secrets
1. Go to Space → Settings → Repository secrets
2. Add secret:
   - Name: `GOOGLE_API_KEY`
   - Value: `your_actual_api_key`
- [ ] GOOGLE_API_KEY configured

### 6. Push to HF
```bash
cd ai-exercise-trainer
git add .
git commit -m "Initial deployment"
git push
```
- [ ] Files pushed to HF

### 7. Monitor Build
- [ ] Build started (check "Building" tab)
- [ ] Build completed successfully (~5-10 min)
- [ ] Status shows "Running"

## 🧪 Post-Deployment Testing

### Basic Tests
- [ ] Landing page loads: `https://YOUR_USER-ai-exercise-trainer.hf.space`
- [ ] API status works: `https://YOUR_USER-ai-exercise-trainer.hf.space/api/status`
- [ ] Swagger docs load: `https://YOUR_USER-ai-exercise-trainer.hf.space/docs`

### Functional Tests
```bash
# Set your space URL
SPACE_URL="https://YOUR_USER-ai-exercise-trainer.hf.space"

# 1. Health check
curl "$SPACE_URL/api/status"
```
- [ ] Health check returns 200

```bash
# 2. Create session
SESSION_ID=$(curl -X POST "$SPACE_URL/session/create" | jq -r '.session_id')
echo "Session ID: $SESSION_ID"
```
- [ ] Session created successfully

```bash
# 3. Submit exercise (requires test video)
curl -X POST "$SPACE_URL/session/$SESSION_ID/submit-exercise" \
  -F "exercise_type=pushup" \
  -F "video=@test_video.mp4"
```
- [ ] Video upload works
- [ ] Cheat detection runs
- [ ] Exercise analysis completes

```bash
# 4. Generate AI analysis
curl -X POST "$SPACE_URL/session/$SESSION_ID/generate-ai-analysis"
```
- [ ] AI analysis generates (if API key set)

```bash
# 5. Get summary
curl "$SPACE_URL/session/$SESSION_ID/summary"
```
- [ ] Summary returns complete data

### Performance Tests
- [ ] Video processing time < 30s (CPU)
- [ ] API response time < 2s
- [ ] Memory usage stable
- [ ] No crashes under load

## 📊 Verification

### Space Configuration
- [ ] Title displays correctly
- [ ] Emoji shows: 🏋️
- [ ] SDK set to Docker
- [ ] Port 7860 accessible
- [ ] Logs show no errors

### Features
- [ ] 8 exercise types supported
- [ ] Cheat detection operational
- [ ] Metrics generation working
- [ ] AI analysis functional (with API key)
- [ ] Session management working

### Documentation
- [ ] README renders properly
- [ ] API docs accessible at /docs
- [ ] ReDoc accessible at /redoc
- [ ] Examples work correctly

## 🔧 Troubleshooting

### Build Issues
- [ ] Check Dockerfile syntax
- [ ] Verify all dependencies listed
- [ ] Check file paths in COPY commands
- [ ] Review build logs for errors

### Runtime Issues
- [ ] Check container logs
- [ ] Verify port 7860 exposed
- [ ] Test API endpoints individually
- [ ] Check environment variables

### Performance Issues
- [ ] Monitor memory usage
- [ ] Check CPU utilization
- [ ] Review processing times
- [ ] Consider GPU upgrade if needed

## 📝 Post-Deployment

### Documentation
- [ ] Update README with space URL
- [ ] Add usage examples
- [ ] Document API endpoints
- [ ] Create tutorial videos

### Monitoring
- [ ] Set up error tracking
- [ ] Monitor usage metrics
- [ ] Check user feedback
- [ ] Review performance logs

### Maintenance
- [ ] Schedule regular updates
- [ ] Monitor dependency vulnerabilities
- [ ] Backup important data
- [ ] Plan feature enhancements

## 🎉 Deployment Complete!

Your AI Exercise Trainer is now live on Hugging Face Spaces!

**Space URL**: `https://YOUR_USERNAME-ai-exercise-trainer.hf.space`

**Next Steps**:
1. Share with users
2. Gather feedback
3. Monitor performance
4. Plan updates

---

**Need Help?**
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [GitHub Issues](your-repo-url)
