"""
Test script to demonstrate cheat detection integration with server.py
"""

import requests
import json
import time
from pathlib import Path

# Server configuration
BASE_URL = "http://localhost:8000"

def print_separator():
    print("\n" + "="*70)

def test_simple_upload(video_path: str, exercise_type: str = "pushup"):
    """Test the simple upload-and-analyze endpoint with cheat detection"""
    
    print_separator()
    print(f"TEST: Simple Upload - {Path(video_path).name}")
    print_separator()
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📤 Uploading video: {video_path}")
    print(f"🏋️ Exercise type: {exercise_type}")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {'exercise_type': exercise_type}
            
            response = requests.post(
                f"{BASE_URL}/upload-video/",
                files=files,
                data=data,
                timeout=300
            )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            # Analysis completed successfully
            result = response.json()
            print("\n✅ ANALYSIS COMPLETED!")
            print(f"Session ID: {result.get('session_id')}")
            
            cheat_detection = result.get('cheat_detection', {})
            print(f"\n🔍 CHEAT DETECTION:")
            print(f"  ✓ Passed: {cheat_detection.get('passed')}")
            print(f"  Risk Level: {cheat_detection.get('risk_level')}")
            print(f"  Score: {cheat_detection.get('score'):.1f}/100")
            
            print(f"\n📈 EXERCISE RESULTS:")
            results = result.get('results', {})
            print(json.dumps(results, indent=2)[:500] + "...")
            
        elif response.status_code == 403:
            # Blocked by cheat detection
            error_detail = response.json().get('detail', {})
            print("\n❌ ANALYSIS BLOCKED - CHEATING DETECTED!")
            print(f"  Risk Level: {error_detail.get('risk_level')}")
            print(f"  Score: {error_detail.get('overall_score'):.1f}/100")
            print(f"  Verdict: {error_detail.get('verdict')}")
            
            violations = error_detail.get('violations', {})
            print(f"\n🚨 VIOLATIONS:")
            print(f"  Has Text: {violations.get('has_text')}")
            print(f"  Is Deepfake: {violations.get('is_deepfake')}")
            print(f"  Has Effects: {violations.get('has_effects')}")
            print(f"  Excessive Cuts: {violations.get('excessive_cuts')}")
            
            print(f"\n💬 Message: {error_detail.get('message')}")
            
        else:
            print(f"\n⚠️ Unexpected status code: {response.status_code}")
            print(response.text[:500])
    
    except requests.exceptions.Timeout:
        print("\n⏱️ Request timeout - video analysis taking too long")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_session_workflow(video_path: str, exercise_type: str = "squat"):
    """Test the full session workflow with cheat detection"""
    
    print_separator()
    print(f"TEST: Session Workflow - {Path(video_path).name}")
    print_separator()
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    try:
        # Step 1: Create session
        print("\n📝 Step 1: Creating session...")
        response = requests.post(f"{BASE_URL}/session/create")
        session = response.json()
        session_id = session['session_id']
        print(f"✓ Session created: {session_id}")
        
        # Step 2: Upload video
        print(f"\n📤 Step 2: Uploading video...")
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(
                f"{BASE_URL}/session/{session_id}/upload?exercise_type={exercise_type}",
                files=files
            )
        print(f"✓ Video uploaded: {response.json().get('message')}")
        
        # Step 3: Start analysis (cheat detection happens here)
        print(f"\n🔍 Step 3: Starting analysis (with cheat detection)...")
        response = requests.post(f"{BASE_URL}/session/{session_id}/analyze")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Analysis started!")
            
            cheat_detection = result.get('cheat_detection', {})
            print(f"\n🔍 CHEAT DETECTION:")
            print(f"  ✓ Passed: {cheat_detection.get('passed')}")
            print(f"  Risk Level: {cheat_detection.get('risk_level')}")
            print(f"  Score: {cheat_detection.get('score'):.1f}/100")
            
            # Step 4: Poll status
            print(f"\n⏳ Step 4: Waiting for analysis to complete...")
            max_attempts = 60
            for i in range(max_attempts):
                time.sleep(2)
                response = requests.get(f"{BASE_URL}/session/{session_id}/status")
                status_data = response.json()
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                print(f"  Status: {status} ({progress*100:.0f}%)", end='\r')
                
                if status == 'completed':
                    print(f"\n✓ Analysis completed!")
                    break
                elif status == 'failed':
                    print(f"\n❌ Analysis failed: {status_data.get('message')}")
                    return
            
            # Step 5: Get report
            print(f"\n📊 Step 5: Retrieving report...")
            response = requests.get(f"{BASE_URL}/session/{session_id}/report")
            report_data = response.json()
            
            print(f"✓ Report retrieved!")
            print(f"\n📈 EXERCISE: {report_data.get('exercise').upper()}")
            
            cheat_info = report_data.get('cheat_detection', {})
            if cheat_info:
                print(f"\n🔍 CHEAT DETECTION SUMMARY:")
                print(f"  Risk Level: {cheat_info.get('risk_level')}")
                print(f"  Score: {cheat_info.get('score'):.1f}/100")
                print(f"  Verdict: {cheat_info.get('verdict')}")
            
            report = report_data.get('report', {})
            print(f"\n📊 REPORT SUMMARY:")
            print(json.dumps(report, indent=2)[:500] + "...")
        
        elif response.status_code == 403:
            # Blocked by cheat detection
            error_detail = response.json().get('detail', {})
            print(f"\n❌ ANALYSIS BLOCKED - CHEATING DETECTED!")
            print(f"  Risk Level: {error_detail.get('risk_level')}")
            print(f"  Score: {error_detail.get('overall_score'):.1f}/100")
            print(f"  Verdict: {error_detail.get('verdict')}")
            
            violations = error_detail.get('violations', {})
            print(f"\n🚨 VIOLATIONS:")
            print(f"  Has Text: {violations.get('has_text')}")
            print(f"  Is Deepfake: {violations.get('is_deepfake')}")
            print(f"  Has Effects: {violations.get('has_effects')}")
            print(f"  Excessive Cuts: {violations.get('excessive_cuts')}")
            
            # Check status of blocked session
            print(f"\n📊 Checking session status...")
            response = requests.get(f"{BASE_URL}/session/{session_id}/status")
            status_data = response.json()
            print(f"  Status: {status_data.get('status')}")
            print(f"  Blocked Reason: {status_data.get('blocked_reason')}")
        
        else:
            print(f"\n⚠️ Unexpected status code: {response.status_code}")
            print(response.text[:500])
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main test runner"""
    
    print("\n" + "="*70)
    print("  CHEAT DETECTION INTEGRATION - TEST SUITE")
    print("="*70)
    
    print("\n📋 This test suite demonstrates:")
    print("  1. Cheat detection runs BEFORE exercise analysis")
    print("  2. Videos with MEDIUM (>=40) or HIGH (>=60) risk are BLOCKED")
    print("  3. Only LOW risk videos proceed to exercise analysis")
    print("  4. Full cheat detection reports are returned when blocked")
    
    print("\n⚙️ Server must be running at:", BASE_URL)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        if response.status_code == 200:
            print("✓ Server is online!")
        else:
            print("❌ Server responded but status check failed")
            return
    except:
        print("❌ Server is not running!")
        print("\nPlease start the server first:")
        print("  python server.py")
        return
    
    print("\n" + "="*70)
    print("  IMPORTANT: Replace 'video.mp4' with your actual video files")
    print("="*70)
    
    # Example test cases - REPLACE WITH YOUR ACTUAL VIDEO FILES
    
    # Test 1: Simple upload with legitimate video
    print("\n\n🧪 TEST 1: Simple Upload (Legitimate Video)")
    print("Expected: Should pass cheat detection and complete analysis")
    # test_simple_upload("legitimate_video.mp4", "pushup")
    
    # Test 2: Simple upload with edited video
    print("\n\n🧪 TEST 2: Simple Upload (Edited Video with Text)")
    print("Expected: Should be blocked by cheat detection")
    # test_simple_upload("edited_video_with_text.mp4", "squat")
    
    # Test 3: Session workflow with legitimate video
    print("\n\n🧪 TEST 3: Session Workflow (Legitimate Video)")
    print("Expected: Should pass cheat detection and complete analysis")
    # test_session_workflow("legitimate_video.mp4", "situp")
    
    # Test 4: Session workflow with deepfake video
    print("\n\n🧪 TEST 4: Session Workflow (Deepfake Video)")
    print("Expected: Should be blocked by cheat detection")
    # test_session_workflow("deepfake_video.mp4", "pushup")
    
    print("\n" + "="*70)
    print("  Uncomment test cases above and provide video paths to test!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
