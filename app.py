"""
Hugging Face Spaces Entry Point
Wraps FastAPI server for better HF integration
"""

import os
import subprocess
import sys

def main():
    """Start the FastAPI server"""
    print("="*70)
    print("  AI EXERCISE TRAINER - Hugging Face Spaces")
    print("="*70)
    print("  Starting FastAPI server on port 7860...")
    print("="*70)
    
    # Check for environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not set - AI analysis will be unavailable")
    else:
        print("✅ Google API Key detected - AI analysis enabled")
    
    print("="*70)
    print()
    
    # Start uvicorn
    try:
        subprocess.run([
            "uvicorn",
            "server:app",
            "--host", "0.0.0.0",
            "--port", "7860",
            "--workers", "1",
            "--timeout-keep-alive", "120",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
