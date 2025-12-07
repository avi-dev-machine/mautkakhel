#!/bin/bash

# ============================================================================
# AI Exercise Trainer - Hugging Face Spaces Deployment Script
# ============================================================================

echo "========================================================================"
echo "  AI EXERCISE TRAINER - HF SPACES DEPLOYMENT"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: server.py not found. Run this script from the project root."
    exit 1
fi

# Get space name from user
read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter your space name (default: ai-exercise-trainer): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-ai-exercise-trainer}

echo ""
echo "📦 Preparing files for deployment..."
echo ""

# Create temporary deployment directory
DEPLOY_DIR="hf_deployment_temp"
mkdir -p "$DEPLOY_DIR"

# Copy required files
echo "✓ Copying application files..."
cp server.py "$DEPLOY_DIR/"
cp utils.py "$DEPLOY_DIR/"
cp metrics.py "$DEPLOY_DIR/"
cp cheat.py "$DEPLOY_DIR/"
cp ai.py "$DEPLOY_DIR/"

echo "✓ Copying model files..."
cp yolo11n-pose.pt "$DEPLOY_DIR/"
[ -f yolo11n-pose.onnx ] && cp yolo11n-pose.onnx "$DEPLOY_DIR/"

echo "✓ Copying configuration files..."
cp Dockerfile "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"
cp requirements_api.txt "$DEPLOY_DIR/"
cp .dockerignore "$DEPLOY_DIR/"

echo "✓ Creating README.md..."
cp README_HF.md "$DEPLOY_DIR/README.md"

echo ""
echo "========================================================================"
echo "  FILES READY FOR DEPLOYMENT"
echo "========================================================================"
echo ""
echo "Files copied to: $DEPLOY_DIR/"
echo ""
echo "Next steps:"
echo ""
echo "1. Clone your Hugging Face Space:"
echo "   git clone https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""
echo "2. Copy files to the space directory:"
echo "   cp -r $DEPLOY_DIR/* $SPACE_NAME/"
echo ""
echo "3. Configure secrets in HF Space settings:"
echo "   - GOOGLE_API_KEY: Your Google Gemini API key"
echo ""
echo "4. Push to Hugging Face:"
echo "   cd $SPACE_NAME"
echo "   git add ."
echo "   git commit -m \"Deploy AI Exercise Trainer\""
echo "   git push"
echo ""
echo "5. Monitor build at:"
echo "   https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""
echo "========================================================================"
echo ""

# Ask if user wants to automatically continue
read -p "Do you want to clone and deploy automatically? (y/n): " AUTO_DEPLOY

if [ "$AUTO_DEPLOY" = "y" ] || [ "$AUTO_DEPLOY" = "Y" ]; then
    echo ""
    echo "🚀 Starting automatic deployment..."
    echo ""
    
    # Clone space
    echo "Cloning space repository..."
    git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" || {
        echo "❌ Failed to clone space. Please check your username and space name."
        exit 1
    }
    
    # Copy files
    echo "Copying files to space..."
    cp -r "$DEPLOY_DIR"/* "$SPACE_NAME/"
    
    # Navigate to space
    cd "$SPACE_NAME" || exit 1
    
    # Git operations
    echo "Committing changes..."
    git add .
    git commit -m "Deploy AI Exercise Trainer v3.0"
    
    echo "Pushing to Hugging Face..."
    git push
    
    echo ""
    echo "========================================================================"
    echo "  ✅ DEPLOYMENT COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Your space will be available at:"
    echo "https://$HF_USERNAME-$SPACE_NAME.hf.space"
    echo ""
    echo "Monitor build progress at:"
    echo "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    echo ""
    echo "Don't forget to set GOOGLE_API_KEY in Space secrets!"
    echo ""
else
    echo ""
    echo "✅ Files prepared in: $DEPLOY_DIR/"
    echo "Follow the manual steps above to complete deployment."
    echo ""
fi

echo "========================================================================"
