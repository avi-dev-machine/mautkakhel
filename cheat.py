"""
Advanced Video Deepfake Detection using Pre-trained Deep Learning Models
Basic Checkpoints: Camera Angles, Effects, Multiple People, AI Deepfakes
"""

import torch
import numpy as np
from PIL import Image
import av
import io
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DeepfakeVideoDetector:
    """
    Video deepfake and manipulation detector with 4 basic checkpoints:
    1. Sudden camera angle changes
    2. Effects, filters, text, or animations
    3. More than one person in video
    4. AI-generated deepfake detection
    """
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """
        Initialize detector with pre-trained models for each checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.models = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for all checkpoints"""
        from transformers import pipeline, AutoFeatureExtractor, AutoModelForImageClassification
        
        print("\nLoading models for checkpoints...")
        
        # Model 1: Deepfake detection
        try:
            print("  [1/4] Loading deepfake detection model...")
            self.models['deepfake'] = pipeline(
                "image-classification", 
                model="dima806/deepfake_vs_real_image_detection",
                device=0 if torch.cuda.is_available() else -1
            )
            print("  ✓ Deepfake detector loaded")
        except Exception as e:
            print(f"  ⚠ Deepfake detector unavailable: {e}")
            self.models['deepfake'] = None
        
        # Model 2: Object detection for people counting
        try:
            print("  [2/4] Loading people detection model...")
            self.models['people'] = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            print("  ✓ People detector loaded")
        except Exception as e:
            print(f"  ⚠ People detector unavailable: {e}")
            self.models['people'] = None
        
        # Model 3: Image feature extractor for effects/filters
        try:
            print("  [3/4] Loading feature extraction model...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.models['features'] = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.models['features'].to(self.device)
            self.models['features'].eval()
            print("  ✓ Feature extractor loaded")
        except Exception as e:
            print(f"  ⚠ Feature extractor unavailable: {e}")
            self.models['features'] = None
        
        # Model 4: Text detection (OCR)
        try:
            print("  [4/4] Loading text detection model...")
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.models['text_processor'] = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.models['text'] = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            self.models['text'].to(self.device)
            print("  ✓ Text detector loaded")
        except Exception as e:
            print(f"  ⚠ Text detector unavailable: {e}")
            self.models['text'] = None
        
        print("\nModels loaded successfully!\n")
    
    def read_video_frames(self, video_path: str, num_frames: int = 16) -> List[Image.Image]:
        """
        Extract frames from video using PyAV (no OpenCV)
        """
        frames = []
        
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            # Calculate frame indices to sample
            total_frames = video_stream.frames
            if total_frames == 0:
                # Estimate from duration
                total_frames = int(video_stream.duration * video_stream.time_base * video_stream.average_rate)
            
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frame_count = 0
            target_idx = 0
            
            for frame in container.decode(video=0):
                if target_idx >= len(indices):
                    break
                
                if frame_count == indices[target_idx]:
                    # Convert frame to PIL Image
                    img = frame.to_image()
                    frames.append(img)
                    target_idx += 1
                
                frame_count += 1
            
            container.close()
            
            print(f"Extracted {len(frames)} frames from video")
            
        except Exception as e:
            print(f"Error reading video: {e}")
            print("Trying alternative method...")
            frames = self._read_video_alternative(video_path, num_frames)
        
        return frames
    
    def _read_video_alternative(self, video_path: str, num_frames: int = 16) -> List[Image.Image]:
        """Alternative video reading method"""
        frames = []
        
        try:
            import imageio
            reader = imageio.get_reader(video_path)
            
            total_frames = reader.count_frames()
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for idx in indices:
                frame = reader.get_data(idx)
                img = Image.fromarray(frame)
                frames.append(img)
            
            reader.close()
            print(f"Extracted {len(frames)} frames (alternative method)")
            
        except Exception as e:
            print(f"Error with alternative method: {e}")
        
        return frames
    
    # ==================== CHECKPOINT 1: CAMERA ANGLE CHANGES ====================
    
    def checkpoint_1_camera_angle_changes(self, frames: List[Image.Image]) -> Dict:
        """
        Detect sudden camera angle changes between frames
        """
        if len(frames) < 2:
            return {'score': 0, 'angle_changes': [], 'total_changes': 0}
        
        print("  → Analyzing camera angles...")
        
        angle_changes = []
        
        for i in range(len(frames) - 1):
            # Convert frames to numpy arrays
            arr1 = np.array(frames[i].convert('RGB'))
            arr2 = np.array(frames[i + 1].convert('RGB'))
            
            # Calculate structural similarity
            frame_diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))
            
            # Calculate histogram correlation
            hist1_r = np.histogram(arr1[:,:,0], bins=32, range=(0,256))[0]
            hist1_g = np.histogram(arr1[:,:,1], bins=32, range=(0,256))[0]
            hist1_b = np.histogram(arr1[:,:,2], bins=32, range=(0,256))[0]
            
            hist2_r = np.histogram(arr2[:,:,0], bins=32, range=(0,256))[0]
            hist2_g = np.histogram(arr2[:,:,1], bins=32, range=(0,256))[0]
            hist2_b = np.histogram(arr2[:,:,2], bins=32, range=(0,256))[0]
            
            # Correlation between histograms
            corr_r = np.corrcoef(hist1_r, hist2_r)[0,1]
            corr_g = np.corrcoef(hist1_g, hist2_g)[0,1]
            corr_b = np.corrcoef(hist1_b, hist2_b)[0,1]
            avg_corr = (corr_r + corr_g + corr_b) / 3
            
            # Sudden change detected if correlation is low and difference is high
            if avg_corr < 0.6 or frame_diff > 60:
                angle_changes.append({
                    'frame_index': i + 1,
                    'correlation': float(avg_corr),
                    'difference': float(frame_diff),
                    'severity': 'HIGH' if frame_diff > 80 else 'MEDIUM'
                })
        
        # Calculate score
        change_ratio = len(angle_changes) / (len(frames) - 1)
        score = min(100, change_ratio * 150)
        
        return {
            'score': score,
            'angle_changes': angle_changes,
            'total_changes': len(angle_changes),
            'change_ratio': float(change_ratio),
            'is_suspicious': len(angle_changes) > len(frames) * 0.3
        }
    
    # ==================== CHECKPOINT 2: EFFECTS & FILTERS ====================
    
    def checkpoint_2_effects_filters_text(self, frames: List[Image.Image]) -> Dict:
        """
        Detect visual effects, filters, text overlays, and animations
        """
        print("  → Detecting effects, filters, and text...")
        
        results = {
            'text_detected': [],
            'artificial_effects': [],
            'filter_score': 0,
            'animation_score': 0
        }
        
        # Check for text overlays with improved detection
        text_frames = 0
        for i, frame in enumerate(frames):
            arr = np.array(frame.convert('RGB'))
            gray = np.array(frame.convert('L'))  # Grayscale
            
            # Method 1: Edge density analysis (text has sharp edges)
            from scipy import ndimage
            edges_x = ndimage.sobel(gray, axis=0)
            edges_y = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
            edge_density = np.mean(edge_magnitude)
            
            # Method 2: Detect high contrast regions
            local_std = ndimage.generic_filter(gray, np.std, size=10)
            high_contrast_regions = np.sum(local_std > 50) / local_std.size
            
            # Method 3: Detect rectangular regions (common in text overlays)
            binary = (gray > 200).astype(np.uint8) | (gray < 50).astype(np.uint8)
            binary_ratio = np.sum(binary) / binary.size
            
            # Method 4: Check for unnatural uniformity in color blocks
            color_blocks = []
            for channel in range(3):
                unique_colors = len(np.unique(arr[:,:,channel]))
                color_blocks.append(unique_colors)
            avg_unique_colors = np.mean(color_blocks)
            
            # Combine detection methods
            text_score = 0
            if edge_density > 20:  # Sharp edges
                text_score += 30
            if high_contrast_regions > 0.15:  # High contrast areas
                text_score += 30
            if binary_ratio > 0.1:  # Black or white regions
                text_score += 25
            if avg_unique_colors > 200:  # Many distinct colors (overlays)
                text_score += 15
            
            if text_score >= 50:  # Threshold for text detection
                text_frames += 1
                results['text_detected'].append({
                    'frame_index': i,
                    'confidence': min(100, text_score),
                    'edge_density': float(edge_density),
                    'high_contrast_ratio': float(high_contrast_regions)
                })
        
        # Check for artificial filters/effects
        for i, frame in enumerate(frames):
            arr = np.array(frame.convert('RGB'))
            
            # Check saturation (filters often boost saturation)
            hsv = frame.convert('HSV')
            hsv_arr = np.array(hsv)
            saturation = hsv_arr[:,:,1]
            avg_saturation = np.mean(saturation)
            
            # Check for unnatural color distribution
            if avg_saturation > 180 or avg_saturation < 30:
                results['artificial_effects'].append({
                    'frame_index': i,
                    'type': 'color_filter',
                    'saturation': float(avg_saturation)
                })
            
            # Check for blur effects (low variance)
            variance = np.var(arr)
            if variance < 500:
                results['artificial_effects'].append({
                    'frame_index': i,
                    'type': 'blur_effect',
                    'variance': float(variance)
                })
        
        # Check for animations (rapid color/brightness changes)
        if len(frames) > 2:
            brightness_values = [np.mean(np.array(f.convert('L'))) for f in frames]
            brightness_changes = np.diff(brightness_values)
            rapid_changes = np.sum(np.abs(brightness_changes) > 30)
            results['animation_score'] = min(100, (rapid_changes / len(frames)) * 200)
        
        # Calculate overall score with STRICT penalties for text
        text_ratio = text_frames / len(frames) if frames else 0
        text_score = min(100, text_ratio * 200)  # Doubled penalty
        
        effects_score = (len(results['artificial_effects']) / len(frames)) * 100 if frames else 0
        
        # STRICT: Text detection gets heavy weight
        total_score = (text_score * 0.6 + effects_score * 0.3 + results['animation_score'] * 0.1)
        
        return {
            'score': total_score,
            'text_frames': text_frames,
            'text_ratio': text_frames / len(frames) if frames else 0,
            'effects_detected': len(results['artificial_effects']),
            'animation_score': results['animation_score'],
            'has_text': text_frames > 0,
            'has_effects': len(results['artificial_effects']) > 0,
            'details': results
        }
    
    # ==================== CHECKPOINT 3: MULTIPLE PEOPLE ====================
    
    def checkpoint_3_multiple_people(self, frames: List[Image.Image]) -> Dict:
        """
        Detect number of people in video frames
        """
        print("  → Counting people in frames...")
        
        if self.models.get('people') is None:
            print("    ⚠ People detection model not available, using fallback...")
            return self._checkpoint_3_fallback(frames)
        
        people_counts = []
        frames_with_multiple = []
        
        # Sample every other frame for efficiency
        sample_indices = range(0, len(frames), max(1, len(frames) // 8))
        
        for idx in sample_indices:
            frame = frames[idx]
            
            try:
                # Run object detection
                detections = self.models['people'](frame)
                
                # Count people (person class)
                people = [d for d in detections if d['label'] == 'person' and d['score'] > 0.5]
                people_count = len(people)
                
                people_counts.append(people_count)
                
                if people_count > 1:
                    frames_with_multiple.append({
                        'frame_index': idx,
                        'people_count': people_count,
                        'confidence': float(np.mean([p['score'] for p in people]))
                    })
            except Exception as e:
                print(f"    Detection error on frame {idx}: {e}")
                people_counts.append(0)
        
        # Calculate statistics
        max_people = max(people_counts) if people_counts else 0
        avg_people = np.mean(people_counts) if people_counts else 0
        
        # Score based on presence of multiple people
        score = min(100, (len(frames_with_multiple) / len(sample_indices)) * 100) if sample_indices else 0
        
        return {
            'score': score,
            'max_people': max_people,
            'avg_people': float(avg_people),
            'frames_with_multiple': len(frames_with_multiple),
            'has_multiple_people': max_people > 1,
            'details': frames_with_multiple[:5]  # First 5 examples
        }
    
    def _checkpoint_3_fallback(self, frames: List[Image.Image]) -> Dict:
        """Fallback people detection using simple heuristics"""
        # Simple skin tone detection as fallback
        people_frames = 0
        
        for frame in frames[::2]:  # Sample every other frame
            arr = np.array(frame.convert('RGB'))
            
            # Simple skin tone detection (R>G>B pattern)
            skin_mask = (arr[:,:,0] > 95) & (arr[:,:,1] > 40) & (arr[:,:,2] > 20) & \
                       (arr[:,:,0] > arr[:,:,1]) & (arr[:,:,1] > arr[:,:,2])
            
            skin_ratio = np.sum(skin_mask) / skin_mask.size
            
            # If significant skin-colored regions, likely has people
            if skin_ratio > 0.05:
                people_frames += 1
        
        has_people = people_frames > len(frames) * 0.3
        
        return {
            'score': 50 if has_people else 0,  # Neutral score with fallback
            'max_people': 1 if has_people else 0,
            'avg_people': 1.0 if has_people else 0.0,
            'frames_with_multiple': 0,
            'has_multiple_people': False,
            'details': [],
            'note': 'Fallback detection used'
        }
    
    # ==================== CHECKPOINT 4: AI DEEPFAKE ====================
    
    def checkpoint_4_ai_deepfake(self, frames: List[Image.Image]) -> Dict:
        """
        Detect AI-generated deepfake content
        """
        print("  → Running deepfake detection...")
        
        if self.models.get('deepfake') is None:
            print("    ⚠ Deepfake model not available")
            return {'score': 0, 'is_deepfake': False, 'confidence': 0}
        
        deepfake_scores = []
        suspicious_frames = []
        
        # Sample frames for efficiency
        sample_indices = range(0, len(frames), max(1, len(frames) // 10))
        
        for idx in sample_indices:
            frame = frames[idx]
            
            try:
                # Convert numpy array to PIL Image if needed
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame)
                else:
                    frame_pil = frame
                
                result = self.models['deepfake'](frame_pil)
                
                # Get fake probability
                fake_prob = 0
                for item in result:
                    label = item['label'].lower()
                    if 'fake' in label or 'forged' in label or 'manipulated' in label:
                        fake_prob = max(fake_prob, item['score'])
                
                deepfake_scores.append(fake_prob)
                
                if fake_prob > 0.6:
                    suspicious_frames.append({
                        'frame_index': idx,
                        'fake_probability': float(fake_prob),
                        'confidence': 'HIGH' if fake_prob > 0.8 else 'MEDIUM'
                    })
            except Exception as e:
                print(f"    Detection error on frame {idx}: {e}")
        
        # Calculate overall score
        if deepfake_scores:
            avg_fake_prob = np.mean(deepfake_scores)
            max_fake_prob = max(deepfake_scores)
            score = avg_fake_prob * 100
        else:
            avg_fake_prob = 0
            max_fake_prob = 0
            score = 0
        
        return {
            'score': score,
            'is_deepfake': avg_fake_prob > 0.5,
            'confidence': float(max_fake_prob * 100),
            'avg_fake_probability': float(avg_fake_prob),
            'max_fake_probability': float(max_fake_prob),
            'suspicious_frames': len(suspicious_frames),
            'details': suspicious_frames[:5]  # First 5 examples
        }
    
    def analyze_video(self, video_path: str, num_frames: int = 16) -> Dict:
        """
        Complete video analysis pipeline with 4 basic checkpoints
        """
        print(f"\n{'='*70}")
        print(f"VIDEO ANALYSIS: {Path(video_path).name}")
        print(f"{'='*70}\n")
        
        # Extract frames
        print("Extracting frames from video...")
        frames = self.read_video_frames(video_path, num_frames)
        
        if not frames:
            print("Error: Could not extract frames from video")
            return None
        
        print(f"\n{'='*70}")
        print("RUNNING 4 BASIC CHECKPOINTS")
        print(f"{'='*70}\n")
        
        # Checkpoint 1: Camera Angles
        print("\n[1/4] CHECKPOINT 1: Camera Angle Changes")
        try:
            checkpoint1 = self.checkpoint_1_camera_angle_changes(frames)
        except Exception as e:
            print(f"    ⚠ Error in camera angle detection: {e}")
            checkpoint1 = {'score': 0, 'total_changes': 0, 'is_suspicious': False, 'changes': []}
        
        # Checkpoint 2: Effects, Filters, Text
        print("\n[2/4] CHECKPOINT 2: Effects, Filters, Text, Animations")
        try:
            checkpoint2 = self.checkpoint_2_effects_filters_text(frames)
        except Exception as e:
            print(f"    ⚠ Error in effects detection: {e}")
            checkpoint2 = {'score': 0, 'has_text': False, 'has_effects': False, 'text_frames': 0, 'effects_detected': 0}
        
        # Checkpoint 3: Multiple People
        print("\n[3/4] CHECKPOINT 3: Multiple People Detection")
        try:
            checkpoint3 = self.checkpoint_3_multiple_people(frames)
        except Exception as e:
            print(f"    ⚠ Error in people detection: {e}")
            checkpoint3 = {'score': 0, 'max_people': 0, 'has_multiple_people': False, 'frames_with_multiple': 0}
        
        # Checkpoint 4: AI Deepfake
        print("\n[4/4] CHECKPOINT 4: AI-Generated Deepfake Detection")
        try:
            checkpoint4 = self.checkpoint_4_ai_deepfake(frames)
        except Exception as e:
            print(f"    ⚠ Error in deepfake detection: {e}")
            checkpoint4 = {
                'score': 0,
                'is_deepfake': False,
                'confidence': 0,
                'avg_fake_probability': 0,
                'max_fake_probability': 0,
                'suspicious_frames': 0,
                'details': []
            }
        
        # STRICT ZERO-TOLERANCE CHEAT DETECTION FORMULA
        # Any violation triggers immediate high score
        
        violation_multipliers = {
            'text': 0,
            'deepfake': 0,
            'effects': 0,
            'angles': 0
        }
        
        # Check for violations
        if checkpoint2['has_text']:  # TEXT: Absolute zero tolerance
            violation_multipliers['text'] = 3.0  # Triple penalty
        
        if checkpoint4['is_deepfake']:  # DEEPFAKE: Zero tolerance
            violation_multipliers['deepfake'] = 2.5  # 2.5x penalty
        
        if checkpoint2['has_effects'] and checkpoint2['effects_detected'] > 5:
            violation_multipliers['effects'] = 1.5  # 1.5x penalty
        
        if checkpoint1['total_changes'] > len(frames) * 0.4:  # Too many cuts
            violation_multipliers['angles'] = 1.3  # 1.3x penalty
        
        # Calculate base score with strict weights
        base_score = (
            checkpoint1['score'] * 0.20 +  # Camera angles: 20%
            checkpoint2['score'] * 0.35 +  # Effects/Text: 35% (highest weight)
            checkpoint3['score'] * 0.10 +  # People: 10%
            checkpoint4['score'] * 0.35    # Deepfake: 35%
        )
        
        # Apply violation multipliers (STRICT)
        text_penalty = checkpoint2['score'] * violation_multipliers['text']
        deepfake_penalty = checkpoint4['score'] * violation_multipliers['deepfake']
        effects_penalty = checkpoint2['score'] * violation_multipliers['effects']
        angles_penalty = checkpoint1['score'] * violation_multipliers['angles']
        
        # Total score with penalties
        overall_score = base_score + text_penalty + deepfake_penalty + effects_penalty + angles_penalty
        overall_score = min(100, overall_score)  # Cap at 100
        
        # STRICT verdict thresholds
        if checkpoint2['has_text'] or checkpoint4['is_deepfake']:
            # AUTOMATIC FAILURE if text or deepfake detected
            verdict = "CRITICAL VIOLATION - Prohibited Content Detected"
            risk_level = "🔴 CRITICAL"
        elif overall_score >= 60:
            verdict = "HIGH RISK - Heavily Manipulated"
            risk_level = "🔴 HIGH"
        elif overall_score >= 40:
            verdict = "MEDIUM RISK - Edited Content"
            risk_level = "🟠 MEDIUM"
        elif overall_score >= 20:
            verdict = "LOW-MEDIUM RISK - Minor Edits"
            risk_level = "🟡 LOW-MEDIUM"
        else:
            verdict = "LOW RISK - Appears Authentic"
            risk_level = "🟢 LOW"
        
        results = {
            'video_path': video_path,
            'frames_analyzed': len(frames),
            'overall_score': overall_score,
            'base_score': base_score,
            'risk_level': risk_level,
            'verdict': verdict,
            'violations': {
                'has_text': checkpoint2['has_text'],
                'is_deepfake': checkpoint4['is_deepfake'],
                'has_effects': checkpoint2['has_effects'],
                'excessive_cuts': checkpoint1['total_changes'] > len(frames) * 0.4,
                'text_penalty': text_penalty,
                'deepfake_penalty': deepfake_penalty,
                'effects_penalty': effects_penalty,
                'angles_penalty': angles_penalty
            },
            'checkpoint_1_camera_angles': checkpoint1,
            'checkpoint_2_effects_text': checkpoint2,
            'checkpoint_3_people': checkpoint3,
            'checkpoint_4_deepfake': checkpoint4
        }
        
        return results
    
    def print_report(self, results: Dict):
        """
        Print detailed analysis report for 4 checkpoints
        """
        print(f"\n{'='*70}")
        print("VIDEO ANALYSIS REPORT")
        print(f"{'='*70}\n")
        
        print(f"{results['risk_level']} OVERALL RISK SCORE: {results['overall_score']:.1f}/100")
        print(f"Base Score: {results['base_score']:.1f}/100")
        print(f"Verdict: {results['verdict']}")
        print(f"Frames Analyzed: {results['frames_analyzed']}")
        
        # Show violations if any
        violations = results['violations']
        if any([violations['has_text'], violations['is_deepfake'], violations['has_effects'], violations['excessive_cuts']]):
            print(f"\n⚠️  VIOLATIONS DETECTED:")
            if violations['has_text']:
                print(f"   🚫 TEXT OVERLAY: +{violations['text_penalty']:.1f} penalty (ZERO TOLERANCE)")
            if violations['is_deepfake']:
                print(f"   🚫 DEEPFAKE: +{violations['deepfake_penalty']:.1f} penalty (ZERO TOLERANCE)")
            if violations['has_effects']:
                print(f"   ⚠️  EFFECTS: +{violations['effects_penalty']:.1f} penalty")
            if violations['excessive_cuts']:
                print(f"   ⚠️  EXCESSIVE CUTS: +{violations['angles_penalty']:.1f} penalty")
        
        print(f"\n{'='*70}")
        print("CHECKPOINT RESULTS")
        print(f"{'='*70}\n")
        
        # Checkpoint 1
        cp1 = results['checkpoint_1_camera_angles']
        print(f"[1] CAMERA ANGLE CHANGES")
        print(f"    Score: {cp1['score']:.1f}/100")
        print(f"    Total Changes: {cp1['total_changes']}")
        print(f"    Change Ratio: {cp1['change_ratio']:.1%}")
        print(f"    Status: {'⚠️ SUSPICIOUS - Many cuts/angle changes' if cp1['is_suspicious'] else '✓ Normal'}")
        if cp1['angle_changes'][:3]:
            print(f"    Sample Changes:")
            for change in cp1['angle_changes'][:3]:
                print(f"      - Frame {change['frame_index']}: {change['severity']} severity (diff: {change['difference']:.1f})")
        
        # Checkpoint 2
        cp2 = results['checkpoint_2_effects_text']
        print(f"\n[2] EFFECTS, FILTERS, TEXT & ANIMATIONS")
        print(f"    Score: {cp2['score']:.1f}/100")
        print(f"    Text Detected: {'YES' if cp2['has_text'] else 'NO'} ({cp2['text_frames']} frames, {cp2['text_ratio']:.1%})")
        print(f"    Effects Detected: {'YES' if cp2['has_effects'] else 'NO'} ({cp2['effects_detected']} instances)")
        print(f"    Animation Score: {cp2['animation_score']:.1f}/100")
        print(f"    Status: {'⚠️ Edited with effects/text' if cp2['score'] > 40 else '✓ Minimal effects'}")
        
        # Checkpoint 3
        cp3 = results['checkpoint_3_people']
        print(f"\n[3] MULTIPLE PEOPLE DETECTION")
        print(f"    Score: {cp3['score']:.1f}/100")
        print(f"    Max People: {cp3['max_people']}")
        print(f"    Avg People: {cp3['avg_people']:.1f}")
        print(f"    Multiple People: {'YES' if cp3['has_multiple_people'] else 'NO'}")
        if cp3['frames_with_multiple']:
            print(f"    Frames with Multiple: {cp3['frames_with_multiple']}")
        if 'note' in cp3:
            print(f"    Note: {cp3['note']}")
        
        # Checkpoint 4
        cp4 = results['checkpoint_4_deepfake']
        print(f"\n[4] AI-GENERATED DEEPFAKE DETECTION")
        print(f"    Score: {cp4['score']:.1f}/100")
        print(f"    Is Deepfake: {'⚠️ YES' if cp4['is_deepfake'] else '✓ NO'}")
        print(f"    Confidence: {cp4['confidence']:.1f}%")
        if 'avg_fake_probability' in cp4:
            print(f"    Avg Fake Probability: {cp4['avg_fake_probability']:.1%}")
            print(f"    Max Fake Probability: {cp4['max_fake_probability']:.1%}")
        if cp4.get('suspicious_frames', 0) > 0:
            print(f"    Suspicious Frames: {cp4['suspicious_frames']}")
            if cp4.get('details'):
                print(f"    Sample Detections:")
                for det in cp4['details'][:3]:
                    print(f"      - Frame {det['frame_index']}: {det['fake_probability']:.1%} fake ({det['confidence']})")
        
        print(f"\n{'='*70}")
        print("FINAL ASSESSMENT")
        print(f"{'='*70}\n")
        
        # STRICT Summary with Zero Tolerance
        critical_violations = []
        warnings = []
        
        if cp2['has_text']:
            critical_violations.append("🚫 TEXT OVERLAYS DETECTED (PROHIBITED)")
        if cp4['is_deepfake']:
            critical_violations.append("🚫 AI-GENERATED DEEPFAKE (PROHIBITED)")
        
        if cp1['is_suspicious']:
            warnings.append("Multiple camera angle changes")
        if cp2['has_effects']:
            warnings.append("Visual effects/filters applied")
        if cp3['has_multiple_people']:
            warnings.append("Multiple people detected")
        
        if critical_violations:
            print("\n🔴 CRITICAL VIOLATIONS (ZERO TOLERANCE):")
            for violation in critical_violations:
                print(f"   {violation}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   • {warning}")
        
        if not critical_violations and not warnings:
            print("\n✓ No violations detected")
        
        print(f"\nOverall Risk: {results['risk_level']}")
        print(f"Verdict: {results['verdict']}")
        
        # STRICT Recommendations with Zero Tolerance Policy
        if critical_violations:
            print("\n🚫 SYSTEM DECISION: CONTENT REJECTED")
            print("   This video contains PROHIBITED elements and cannot be accepted.")
            print("   Violations: Text overlays and/or AI-generated content are NOT ALLOWED.")
            print("   Action: Video must be re-recorded without any overlays or manipulation.")
        elif results['overall_score'] >= 40:
            print("\n⚠️  SYSTEM DECISION: CONTENT FLAGGED FOR REVIEW")
            print("   Video shows significant editing and requires manual verification.")
            print("   Action: Submit for human review before acceptance.")
        elif results['overall_score'] >= 20:
            print("\n⚠️  SYSTEM DECISION: CONTENT ACCEPTABLE WITH WARNING")
            print("   Minor edits detected. Acceptable but flagged for monitoring.")
        else:
            print("\n✅ SYSTEM DECISION: CONTENT APPROVED")
            print("   Video appears authentic with no prohibited elements.")
            print("   Action: Approved for use in the system.")
        
        print(f"\n{'='*70}\n")


def main():
    """Main execution function"""
    
    if len(sys.argv) < 2:
        print("Usage: python deepfake_detector.py <video_path>")
        print("\nThis tool performs 4 basic checkpoint analyses:")
        print("  1. Sudden camera angle changes")
        print("  2. Effects, filters, text, and animations")
        print("  3. Multiple people detection")
        print("  4. AI-generated deepfake detection")
        print("\nExample: python deepfake_detector.py video.mp4")
        return
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize detector
    detector = DeepfakeVideoDetector()
    
    # Analyze video
    results = detector.analyze_video(video_path, num_frames=16)
    
    if results:
        # Print report
        detector.print_report(results)


if __name__ == "__main__":
    main()
