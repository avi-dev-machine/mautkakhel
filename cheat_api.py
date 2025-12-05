"""
REST API for Video Deepfake Detection System
Deploy with FastAPI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepfake_detector import DeepfakeVideoDetector
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import uvicorn

app = FastAPI(
    title="Video Deepfake Detection API",
    description="Analyze videos for deepfakes, tampering, and suspicious edits",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector once at startup
print("Initializing detector...")
detector = DeepfakeVideoDetector()
print("Detector ready!")

# Create directories
UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("results")
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)


@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'Video Deepfake Detector',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }


@app.post('/analyze')
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    num_frames: Optional[int] = Form(16, description="Number of frames to analyze")
):
    """
    Analyze uploaded video file
    
    POST /analyze
    Form data:
        - video: video file (required)
        - num_frames: number of frames to analyze (optional, default=16)
    
    Returns:
        JSON with analysis results
    """
    # Check if video file is present
    if not video.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    
    # Validate file extension
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    file_ext = Path(video.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
        )
    
    # Save uploaded file temporarily
    temp_path = UPLOAD_FOLDER / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video.filename}"
    
    try:
        # Save uploaded file
        with open(temp_path, 'wb') as f:
            content = await video.read()
            f.write(content)
        
        # Analyze video
        results = detector.analyze_video(str(temp_path), num_frames=num_frames)
        
        if results is None:
            raise HTTPException(status_code=500, detail='Could not analyze video file')
        
        # Save results to file
        result_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = RESULTS_FOLDER / f"result_{result_id}.json"
        
        # Prepare response with safe access to nested keys
        try:
            response_data = {
                'result_id': result_id,
                'timestamp': datetime.now().isoformat(),
                'filename': video.filename,
                'analysis': {
                    'overall_score': float(results.get('overall_score', 0)),
                    'base_score': float(results.get('base_score', 0)),
                    'risk_level': str(results.get('risk_level', 'UNKNOWN')),
                    'verdict': str(results.get('verdict', 'Analysis incomplete')),
                    'frames_analyzed': int(results.get('frames_analyzed', 0)),
                    'violations': {
                        'has_text': bool(results.get('violations', {}).get('has_text', False)),
                        'is_deepfake': bool(results.get('violations', {}).get('is_deepfake', False)),
                        'has_effects': bool(results.get('violations', {}).get('has_effects', False)),
                        'excessive_cuts': bool(results.get('violations', {}).get('excessive_cuts', False)),
                        'text_penalty': float(results.get('violations', {}).get('text_penalty', 0)),
                        'deepfake_penalty': float(results.get('violations', {}).get('deepfake_penalty', 0)),
                        'effects_penalty': float(results.get('violations', {}).get('effects_penalty', 0)),
                        'angles_penalty': float(results.get('violations', {}).get('angles_penalty', 0))
                    },
                    'checkpoints': {
                        'camera_angles': {
                            'score': float(results.get('checkpoint_1_camera_angles', {}).get('score', 0)),
                            'total_changes': int(results.get('checkpoint_1_camera_angles', {}).get('total_changes', 0)),
                            'is_suspicious': bool(results.get('checkpoint_1_camera_angles', {}).get('is_suspicious', False))
                        },
                        'effects_text': {
                            'score': float(results.get('checkpoint_2_effects_text', {}).get('score', 0)),
                            'has_text': bool(results.get('checkpoint_2_effects_text', {}).get('has_text', False)),
                            'has_effects': bool(results.get('checkpoint_2_effects_text', {}).get('has_effects', False)),
                            'text_frames': int(results.get('checkpoint_2_effects_text', {}).get('text_frames', 0)),
                            'effects_detected': int(results.get('checkpoint_2_effects_text', {}).get('effects_detected', 0))
                        },
                        'people_detection': {
                            'score': float(results.get('checkpoint_3_people', {}).get('score', 0)),
                            'max_people': int(results.get('checkpoint_3_people', {}).get('max_people', 0)),
                            'has_multiple_people': bool(results.get('checkpoint_3_people', {}).get('has_multiple_people', False))
                        },
                        'deepfake': {
                            'score': float(results.get('checkpoint_4_deepfake', {}).get('score', 0)),
                            'is_deepfake': bool(results.get('checkpoint_4_deepfake', {}).get('is_deepfake', False)),
                            'confidence': float(results.get('checkpoint_4_deepfake', {}).get('confidence', 0))
                        }
                    }
                },
                'system_decision': get_system_decision(results)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error building response: {str(e)}. Raw results: {str(results)[:200]}')
        
        # Save to file
        with open(result_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Analysis error: {str(e)}')
        
    finally:
        # Clean up temporary file
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass


@app.get('/results/{result_id}')
async def get_result(result_id: str):
    """
    Retrieve previous analysis result
    
    GET /results/{result_id}
    """
    result_file = RESULTS_FOLDER / f"result_{result_id}.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail='Result not found')
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    return data


@app.get('/results')
async def list_results():
    """
    List all available results
    
    GET /results
    """
    results = []
    for result_file in RESULTS_FOLDER.glob("result_*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            results.append({
                'result_id': data['result_id'],
                'timestamp': data['timestamp'],
                'filename': data['filename'],
                'verdict': data['analysis']['verdict'],
                'score': data['analysis']['overall_score']
            })
    
    # Sort by timestamp descending
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        'count': len(results),
        'results': results
    }


def get_system_decision(results):
    """Generate system decision text"""
    violations = results.get('violations', {})
    score = results.get('overall_score', 0)
    
    if violations.get('has_text') or violations.get('is_deepfake'):
        return {
            'status': 'REJECTED',
            'reason': 'CRITICAL VIOLATIONS: Text overlays and/or AI-generated content detected',
            'action': 'Video must be re-recorded without any overlays or manipulation'
        }
    elif score >= 40:
        return {
            'status': 'FLAGGED',
            'reason': 'Significant editing detected',
            'action': 'Submit for manual review before acceptance'
        }
    elif score >= 20:
        return {
            'status': 'WARNING',
            'reason': 'Minor edits detected',
            'action': 'Acceptable but flagged for monitoring'
        }
    else:
        return {
            'status': 'APPROVED',
            'reason': 'Video appears authentic',
            'action': 'Approved for use in the system'
        }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("VIDEO DEEPFAKE DETECTION API - FastAPI")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /analyze             - Analyze video")
    print("  GET  /results/{id}        - Get result by ID")
    print("  GET  /results             - List all results")
    print("  GET  /docs                - Interactive API documentation")
    print("\nExample:")
    print("  curl -X POST -F 'video=@video.mp4' http://localhost:8000/analyze")
    print("\n" + "="*70 + "\n")
    
    # Run server with uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
