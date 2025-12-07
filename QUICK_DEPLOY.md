# 🚀 Hugging Face Deployment - Quick Reference

## 📦 Files Needed (Copy These)
```
✅ server.py           - Main API
✅ utils.py            - Pose detection
✅ metrics.py          - Metrics calculation
✅ cheat.py            - Cheat detection
✅ ai.py               - AI analysis
✅ yolo11n-pose.pt     - Model file (~6MB)
✅ Dockerfile          - Container config
✅ requirements.txt    - Dependencies
✅ requirements_api.txt - API dependencies
✅ .dockerignore       - Build optimization
✅ README.md           - Copy from README_HF.md
```

## ⚡ 30-Second Deploy

```bash
# 1. Create Space at https://huggingface.co/spaces (SDK: Docker)

# 2. Clone & Copy
git clone https://huggingface.co/spaces/YOUR_USER/your-space-name
cd your-space-name
# Copy all files above into this directory

# 3. Push
git add .
git commit -m "Initial deployment"
git push

# 4. Set Secret in HF Space Settings
# GOOGLE_API_KEY = your_gemini_api_key

# Done! Monitor at https://huggingface.co/spaces/YOUR_USER/your-space-name
```

## 🎯 Automated Deploy (Even Faster)

**Windows**:
```powershell
.\deploy_hf.ps1
```

**Linux/Mac**:
```bash
bash deploy_hf.sh
```

Follow prompts → Done!

## 📝 Critical: README.md Header

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

## 🔑 Required Secret

**Name**: `GOOGLE_API_KEY`  
**Value**: Your Gemini API key  
**Set in**: Space Settings → Repository secrets

## ✅ Test After Deploy

```bash
SPACE="https://YOUR_USER-your-space.hf.space"

# Health check
curl $SPACE/api/status

# Create session
curl -X POST $SPACE/session/create

# View docs
open $SPACE/docs  # or visit in browser
```

## ⏱️ Expected Timeline

- Build: 5-10 minutes
- First request: ~30 seconds (model loading)
- Video analysis: 10-30 seconds (CPU)

## 🆘 Common Issues

| Issue | Fix |
|-------|-----|
| Build timeout | Upgrade to paid tier or optimize Dockerfile |
| Port not accessible | Ensure `app_port: 7860` in README |
| AI fails | Check GOOGLE_API_KEY in secrets |
| Slow processing | Normal on CPU, upgrade to GPU for speed |

## 📚 Full Docs

- Setup: `DEPLOYMENT_GUIDE.md`
- Checklist: `DEPLOYMENT_CHECKLIST.md`
- Summary: `DEPLOYMENT_SUMMARY.md`

## 🎉 Success = Running + Green Status

Visit: `https://YOUR_USER-your-space.hf.space`

---

**Need help?** Check `DEPLOYMENT_GUIDE.md` for detailed instructions.
