# ============================================================================
# AI Exercise Trainer - Hugging Face Spaces Deployment Script (PowerShell)
# ============================================================================

Write-Host "========================================================================"
Write-Host "  AI EXERCISE TRAINER - HF SPACES DEPLOYMENT"
Write-Host "========================================================================"
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "server.py")) {
    Write-Host "❌ Error: server.py not found. Run this script from the project root." -ForegroundColor Red
    exit 1
}

# Get space name from user
$HF_USERNAME = Read-Host "Enter your Hugging Face username"
$SPACE_NAME_INPUT = Read-Host "Enter your space name (default: ai-exercise-trainer)"
$SPACE_NAME = if ($SPACE_NAME_INPUT) { $SPACE_NAME_INPUT } else { "ai-exercise-trainer" }

Write-Host ""
Write-Host "📦 Preparing files for deployment..."
Write-Host ""

# Create temporary deployment directory
$DEPLOY_DIR = "hf_deployment_temp"
New-Item -ItemType Directory -Force -Path $DEPLOY_DIR | Out-Null

# Copy required files
Write-Host "✓ Copying application files..."
Copy-Item "server.py" -Destination "$DEPLOY_DIR/"
Copy-Item "utils.py" -Destination "$DEPLOY_DIR/"
Copy-Item "metrics.py" -Destination "$DEPLOY_DIR/"
Copy-Item "cheat.py" -Destination "$DEPLOY_DIR/"
Copy-Item "ai.py" -Destination "$DEPLOY_DIR/"

Write-Host "✓ Copying model files..."
Copy-Item "yolo11n-pose.pt" -Destination "$DEPLOY_DIR/"
if (Test-Path "yolo11n-pose.onnx") {
    Copy-Item "yolo11n-pose.onnx" -Destination "$DEPLOY_DIR/"
}

Write-Host "✓ Copying configuration files..."
Copy-Item "Dockerfile" -Destination "$DEPLOY_DIR/"
Copy-Item "requirements.txt" -Destination "$DEPLOY_DIR/"
Copy-Item "requirements_api.txt" -Destination "$DEPLOY_DIR/"
Copy-Item ".dockerignore" -Destination "$DEPLOY_DIR/"

Write-Host "✓ Creating README.md..."
Copy-Item "README_HF.md" -Destination "$DEPLOY_DIR/README.md"

Write-Host ""
Write-Host "========================================================================"
Write-Host "  FILES READY FOR DEPLOYMENT"
Write-Host "========================================================================"
Write-Host ""
Write-Host "Files copied to: $DEPLOY_DIR/"
Write-Host ""
Write-Host "Next steps:"
Write-Host ""
Write-Host "1. Clone your Hugging Face Space:"
Write-Host "   git clone https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
Write-Host ""
Write-Host "2. Copy files to the space directory:"
Write-Host "   Copy-Item -Recurse $DEPLOY_DIR\* -Destination $SPACE_NAME\"
Write-Host ""
Write-Host "3. Configure secrets in HF Space settings:"
Write-Host "   - GOOGLE_API_KEY: Your Google Gemini API key"
Write-Host ""
Write-Host "4. Push to Hugging Face:"
Write-Host "   cd $SPACE_NAME"
Write-Host "   git add ."
Write-Host "   git commit -m ""Deploy AI Exercise Trainer"""
Write-Host "   git push"
Write-Host ""
Write-Host "5. Monitor build at:"
Write-Host "   https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
Write-Host ""
Write-Host "========================================================================"
Write-Host ""

# Ask if user wants to automatically continue
$AUTO_DEPLOY = Read-Host "Do you want to clone and deploy automatically? (y/n)"

if ($AUTO_DEPLOY -eq "y" -or $AUTO_DEPLOY -eq "Y") {
    Write-Host ""
    Write-Host "🚀 Starting automatic deployment..." -ForegroundColor Green
    Write-Host ""
    
    # Clone space
    Write-Host "Cloning space repository..."
    git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone space. Please check your username and space name." -ForegroundColor Red
        exit 1
    }
    
    # Copy files
    Write-Host "Copying files to space..."
    Copy-Item -Recurse "$DEPLOY_DIR\*" -Destination "$SPACE_NAME\" -Force
    
    # Navigate to space
    Set-Location $SPACE_NAME
    
    # Git operations
    Write-Host "Committing changes..."
    git add .
    git commit -m "Deploy AI Exercise Trainer v3.0"
    
    Write-Host "Pushing to Hugging Face..."
    git push
    
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "  ✅ DEPLOYMENT COMPLETE!" -ForegroundColor Green
    Write-Host "========================================================================"
    Write-Host ""
    Write-Host "Your space will be available at:"
    Write-Host "https://$HF_USERNAME-$SPACE_NAME.hf.space" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Monitor build progress at:"
    Write-Host "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Don't forget to set GOOGLE_API_KEY in Space secrets!" -ForegroundColor Yellow
    Write-Host ""
    
    # Return to original directory
    Set-Location ..
} else {
    Write-Host ""
    Write-Host "✅ Files prepared in: $DEPLOY_DIR/" -ForegroundColor Green
    Write-Host "Follow the manual steps above to complete deployment."
    Write-Host ""
}

Write-Host "========================================================================"
