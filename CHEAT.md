# Cheat.py - Comprehensive Documentation

## Overview
`cheat.py` is an advanced video deepfake and manipulation detection system that uses pre-trained deep learning models to identify fraudulent or edited videos. The module implements a **4-checkpoint validation system** with strict zero-tolerance policies for prohibited content like text overlays and AI-generated deepfakes.

---

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [Detection Checkpoints](#detection-checkpoints)
3. [Formulas & Calculations](#formulas--calculations)
4. [Scoring System](#scoring-system)
5. [AI Models Used](#ai-models-used)
6. [Important Functions](#important-functions)
7. [Detection Algorithms](#detection-algorithms)
8. [Zero-Tolerance Policy](#zero-tolerance-policy)
9. [Usage Examples](#usage-examples)

---

## Core Architecture

### Class: DeepfakeVideoDetector

The main detection engine that orchestrates all 4 checkpoints.

**Key Components:**
```python
class DeepfakeVideoDetector:
    - device: torch.device           # CPU or CUDA GPU
    - model_name: str                # Base feature extraction model
    - models: dict                   # Dictionary of all AI models
    - feature_extractor: AutoFeatureExtractor
```

**Loaded Models:**
1. **Deepfake Detector**: `dima806/deepfake_vs_real_image_detection`
2. **People Detector**: `facebook/detr-resnet-50` (DETR object detection)
3. **Feature Extractor**: `microsoft/resnet-50` (ResNet-50)
4. **Text Detector**: `microsoft/trocr-base-printed` (TrOCR)

---

## Detection Checkpoints

### 4-Checkpoint System

```
┌──────────────────────────────────────────────────────────┐
│ CHECKPOINT 1: CAMERA ANGLE CHANGES                       │
│ Detects sudden cuts, angle shifts, scene transitions     │
│ Weight: 20%                                              │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ CHECKPOINT 2: EFFECTS, FILTERS, TEXT, ANIMATIONS         │
│ Detects overlays, filters, artificial effects, text     │
│ Weight: 35% (HIGHEST - ZERO TOLERANCE FOR TEXT)         │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ CHECKPOINT 3: MULTIPLE PEOPLE DETECTION                  │
│ Counts number of people in video frames                 │
│ Weight: 10%                                              │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ CHECKPOINT 4: AI-GENERATED DEEPFAKE DETECTION            │
│ Detects AI-generated/manipulated faces                  │
│ Weight: 35% (ZERO TOLERANCE FOR DEEPFAKES)              │
└──────────────────────────────────────────────────────────┘
                        ↓
              ┌─────────────────┐
              │  FINAL VERDICT  │
              │  Risk Score     │
              │  0-100 scale    │
              └─────────────────┘
```

---

## Formulas & Calculations

### Overall Score Calculation

#### Step 1: Base Score (Weighted Average)
```python
base_score = (
    checkpoint1_score × 0.20 +    # Camera angles: 20%
    checkpoint2_score × 0.35 +    # Effects/Text: 35%
    checkpoint3_score × 0.10 +    # People: 10%
    checkpoint4_score × 0.35      # Deepfake: 35%
)
```

#### Step 2: Violation Multipliers (STRICT PENALTIES)
```python
violation_multipliers = {
    'text': 3.0,        # Triple penalty for text
    'deepfake': 2.5,    # 2.5x penalty for deepfakes
    'effects': 1.5,     # 1.5x penalty for heavy effects
    'angles': 1.3       # 1.3x penalty for excessive cuts
}

# Calculate penalties
text_penalty = checkpoint2_score × 3.0     (if has_text)
deepfake_penalty = checkpoint4_score × 2.5 (if is_deepfake)
effects_penalty = checkpoint2_score × 1.5  (if effects_count > 5)
angles_penalty = checkpoint1_score × 1.3   (if cuts > 40% of frames)
```

#### Step 3: Total Score with Penalties
```python
overall_score = base_score + text_penalty + deepfake_penalty + 
                effects_penalty + angles_penalty

overall_score = min(100, overall_score)  # Cap at 100
```

#### Step 4: Risk Level Classification
```python
if has_text OR is_deepfake:
    risk_level = "🔴 CRITICAL"
    verdict = "CRITICAL VIOLATION - Prohibited Content Detected"
elif overall_score >= 60:
    risk_level = "🔴 HIGH"
    verdict = "HIGH RISK - Heavily Manipulated"
elif overall_score >= 40:
    risk_level = "🟠 MEDIUM"
    verdict = "MEDIUM RISK - Edited Content"
elif overall_score >= 20:
    risk_level = "🟡 LOW-MEDIUM"
    verdict = "LOW-MEDIUM RISK - Minor Edits"
else:
    risk_level = "🟢 LOW"
    verdict = "LOW RISK - Appears Authentic"
```

---

## CHECKPOINT 1: Camera Angle Changes

### Detection Algorithm

#### Frame Difference Calculation:
```python
# Convert frames to RGB arrays
frame1_rgb = np.array(frame_t.convert('RGB'))
frame2_rgb = np.array(frame_t+1.convert('RGB'))

# Calculate pixel-wise difference
frame_diff = mean(|frame1 - frame2|)

# Large difference indicates scene change
```

#### Histogram Correlation:
```python
# Calculate color histograms (32 bins per channel)
hist_r1 = histogram(frame1[:,:,0], bins=32, range=(0,256))
hist_g1 = histogram(frame1[:,:,1], bins=32, range=(0,256))
hist_b1 = histogram(frame1[:,:,2], bins=32, range=(0,256))

# Same for frame2
hist_r2 = histogram(frame2[:,:,0], bins=32, range=(0,256))
hist_g2 = histogram(frame2[:,:,1], bins=32, range=(0,256))
hist_b2 = histogram(frame2[:,:,2], bins=32, range=(0,256))

# Calculate correlation coefficients
corr_r = correlation(hist_r1, hist_r2)
corr_g = correlation(hist_g1, hist_g2)
corr_b = correlation(hist_b1, hist_b2)

avg_correlation = (corr_r + corr_g + corr_b) / 3
```

#### Change Detection Formula:
```python
# Detect sudden change if:
if avg_correlation < 0.6 OR frame_diff > 60:
    angle_change_detected = True
    
    # Severity classification
    if frame_diff > 80:
        severity = "HIGH"
    else:
        severity = "MEDIUM"
```

#### Scoring Formula:
```python
# Calculate change ratio
change_ratio = num_angle_changes / (total_frames - 1)

# Score calculation (capped at 100)
checkpoint1_score = min(100, change_ratio × 150)

# Example:
# 10 changes in 16 frames → ratio = 10/15 = 0.667
# Score = 0.667 × 150 = 100 (capped)
```

#### Output Structure:
```python
{
    'score': 45.0,
    'angle_changes': [
        {
            'frame_index': 5,
            'correlation': 0.45,
            'difference': 72.3,
            'severity': 'HIGH'
        },
        ...
    ],
    'total_changes': 3,
    'change_ratio': 0.20,
    'is_suspicious': False  # True if changes > 30% of frames
}
```

---

## CHECKPOINT 2: Effects, Filters, Text, Animations

### Text Detection Algorithm (Multi-Method)

#### Method 1: Edge Density Analysis
```python
# Convert to grayscale
gray = convert_to_grayscale(frame)

# Sobel edge detection
edges_x = sobel(gray, axis=0)  # Horizontal edges
edges_y = sobel(gray, axis=1)  # Vertical edges

# Edge magnitude
edge_magnitude = sqrt(edges_x² + edges_y²)

# Edge density (text has sharp edges)
edge_density = mean(edge_magnitude)

# Text indicator
if edge_density > 20:
    text_score += 30
```

#### Method 2: High Contrast Region Detection
```python
# Calculate local standard deviation (10x10 window)
local_std = generic_filter(gray, std_function, size=10)

# High contrast ratio (text regions have high local std)
high_contrast_regions = sum(local_std > 50) / total_pixels

# Text indicator
if high_contrast_regions > 0.15:
    text_score += 30
```

#### Method 3: Binary Region Analysis
```python
# Detect pure black or white regions (common in text)
binary_mask = (gray > 200) | (gray < 50)
binary_ratio = sum(binary_mask) / total_pixels

# Text indicator
if binary_ratio > 0.1:
    text_score += 25
```

#### Method 4: Color Uniformity Analysis
```python
# Count unique colors per channel
for channel in [R, G, B]:
    unique_colors = len(unique(frame[:,:,channel]))

avg_unique_colors = mean([unique_R, unique_G, unique_B])

# Overlays often have many distinct colors
if avg_unique_colors > 200:
    text_score += 15
```

#### Combined Text Detection:
```python
# Total text score (0-100)
text_score = edge_score + contrast_score + binary_score + color_score

# Threshold for text detection
if text_score >= 50:
    text_detected = True
```

### Filter & Effects Detection

#### Saturation Analysis:
```python
# Convert to HSV
hsv_frame = convert_to_HSV(frame)
saturation_channel = hsv_frame[:,:,1]

avg_saturation = mean(saturation_channel)

# Unnatural saturation indicates filters
if avg_saturation > 180 OR avg_saturation < 30:
    filter_detected = True
    effect_type = "color_filter"
```

#### Blur Detection:
```python
# Calculate image variance
variance = var(frame)

# Low variance indicates blur effect
if variance < 500:
    blur_effect_detected = True
```

### Animation Detection

#### Brightness Tracking:
```python
# Track brightness across frames
brightness_values = [mean(grayscale(frame_i)) for frame_i in frames]

# Calculate frame-to-frame changes
brightness_changes = diff(brightness_values)

# Count rapid changes (animations)
rapid_changes = sum(|brightness_changes| > 30)

# Animation score
animation_score = min(100, (rapid_changes / num_frames) × 200)
```

### Checkpoint 2 Scoring Formula:
```python
# Calculate ratios
text_ratio = text_frames / total_frames
text_score = min(100, text_ratio × 200)  # Double penalty

effects_ratio = effects_detected / total_frames
effects_score = effects_ratio × 100

# Weighted total (TEXT GETS 60% WEIGHT!)
checkpoint2_score = (
    text_score × 0.60 +          # 60% weight on text
    effects_score × 0.30 +       # 30% weight on effects
    animation_score × 0.10       # 10% weight on animations
)
```

#### Output Structure:
```python
{
    'score': 85.5,
    'text_frames': 12,
    'text_ratio': 0.75,
    'effects_detected': 8,
    'animation_score': 45.2,
    'has_text': True,
    'has_effects': True,
    'details': {
        'text_detected': [
            {
                'frame_index': 3,
                'confidence': 87.5,
                'edge_density': 25.3,
                'high_contrast_ratio': 0.18
            },
            ...
        ],
        'artificial_effects': [
            {
                'frame_index': 5,
                'type': 'color_filter',
                'saturation': 195.2
            },
            ...
        ]
    }
}
```

---

## CHECKPOINT 3: Multiple People Detection

### AI-Based Detection (Primary Method)

#### DETR Object Detection:
```python
# Run DETR model on frame
detections = detr_model(frame)

# Filter for 'person' class with confidence > 0.5
people = [d for d in detections 
          if d['label'] == 'person' and d['score'] > 0.5]

people_count = len(people)

# Record frames with multiple people
if people_count > 1:
    frames_with_multiple.append({
        'frame_index': i,
        'people_count': people_count,
        'confidence': mean([p['score'] for p in people])
    })
```

#### Scoring Formula:
```python
# Calculate statistics
max_people = max(people_counts)
avg_people = mean(people_counts)

# Frames with multiple people ratio
multiple_ratio = len(frames_with_multiple) / sampled_frames

# Score based on presence of multiple people
checkpoint3_score = min(100, multiple_ratio × 100)
```

### Fallback Method (No AI Model)

#### Skin Tone Detection:
```python
# RGB pattern for skin tones
skin_mask = (R > 95) & (G > 40) & (B > 20) & 
            (R > G) & (G > B)

# Skin pixel ratio
skin_ratio = sum(skin_mask) / total_pixels

# If significant skin regions, likely has people
if skin_ratio > 0.05:
    has_people = True
```

#### Fallback Scoring:
```python
if has_people:
    checkpoint3_score = 50  # Neutral score
    max_people = 1
else:
    checkpoint3_score = 0
    max_people = 0
```

#### Output Structure:
```python
{
    'score': 75.0,
    'max_people': 3,
    'avg_people': 2.1,
    'frames_with_multiple': 6,
    'has_multiple_people': True,
    'details': [
        {
            'frame_index': 2,
            'people_count': 3,
            'confidence': 0.87
        },
        ...
    ]
}
```

---

## CHECKPOINT 4: AI-Generated Deepfake Detection

### Deepfake Detection Model

#### Model: `dima806/deepfake_vs_real_image_detection`
Pre-trained transformer model specialized in detecting AI-generated faces.

#### Detection Process:
```python
# Run deepfake classifier on frame
result = deepfake_model(frame_image)

# Extract fake probability
fake_probability = 0
for item in result:
    label = item['label'].lower()
    if 'fake' in label OR 'forged' in label OR 'manipulated' in label:
        fake_probability = max(fake_probability, item['score'])
```

#### Frame Classification:
```python
# Classify frame based on fake probability
if fake_probability > 0.8:
    confidence = "HIGH"
elif fake_probability > 0.6:
    confidence = "MEDIUM"
else:
    confidence = "LOW"

# Record suspicious frames
if fake_probability > 0.6:
    suspicious_frames.append({
        'frame_index': i,
        'fake_probability': fake_probability,
        'confidence': confidence
    })
```

#### Scoring Formula:
```python
# Calculate statistics across all frames
avg_fake_probability = mean([p for p in fake_probabilities])
max_fake_probability = max(fake_probabilities)

# Score (0-100)
checkpoint4_score = avg_fake_probability × 100

# Binary classification
is_deepfake = (avg_fake_probability > 0.5)
```

#### Output Structure:
```python
{
    'score': 72.5,
    'is_deepfake': True,
    'confidence': 85.3,
    'avg_fake_probability': 0.725,
    'max_fake_probability': 0.853,
    'suspicious_frames': 8,
    'details': [
        {
            'frame_index': 4,
            'fake_probability': 0.85,
            'confidence': 'HIGH'
        },
        ...
    ]
}
```

---

## Zero-Tolerance Policy

### Critical Violations (Automatic Rejection)

#### Text Detection:
```python
if checkpoint2['has_text']:
    violation_multipliers['text'] = 3.0  # Triple penalty
    critical_violation = True
    verdict = "CRITICAL VIOLATION - Text Overlays PROHIBITED"
```

#### Deepfake Detection:
```python
if checkpoint4['is_deepfake']:
    violation_multipliers['deepfake'] = 2.5  # 2.5x penalty
    critical_violation = True
    verdict = "CRITICAL VIOLATION - AI-Generated Content PROHIBITED"
```

#### System Decision Logic:
```python
if has_text OR is_deepfake:
    decision = "CONTENT REJECTED"
    action = "Video must be re-recorded without prohibited elements"
    
elif overall_score >= 40:
    decision = "CONTENT FLAGGED FOR REVIEW"
    action = "Submit for manual verification"
    
elif overall_score >= 20:
    decision = "CONTENT ACCEPTABLE WITH WARNING"
    action = "Approved but flagged for monitoring"
    
else:
    decision = "CONTENT APPROVED"
    action = "Cleared for system use"
```

---

## Important Functions

### 1. `__init__(model_name="microsoft/resnet-50")`
Initializes detector and loads all 4 AI models.

**Process:**
```python
1. Detect GPU/CPU device
2. Load deepfake detection model (dima806/deepfake_vs_real_image_detection)
3. Load DETR object detection (facebook/detr-resnet-50)
4. Load ResNet feature extractor (microsoft/resnet-50)
5. Load TrOCR text detection (microsoft/trocr-base-printed)
```

**Error Handling:**
- If model fails to load, sets model to `None`
- Continues with available models
- Prints warnings for unavailable models

---

### 2. `read_video_frames(video_path, num_frames=16)`
Extracts evenly-spaced frames from video using PyAV.

**Algorithm:**
```python
# Open video container
container = av.open(video_path)
video_stream = container.streams.video[0]

# Calculate frame sampling indices
total_frames = video_stream.frames
indices = linspace(0, total_frames - 1, num_frames, dtype=int)

# Extract frames at specified indices
for frame_idx, frame in enumerate(decode(video)):
    if frame_idx in indices:
        pil_image = frame.to_image()
        frames.append(pil_image)
```

**Fallback Method:**
```python
# If PyAV fails, use imageio
reader = imageio.get_reader(video_path)
for idx in indices:
    frame_array = reader.get_data(idx)
    pil_image = Image.fromarray(frame_array)
    frames.append(pil_image)
```

**Returns:**
- List of PIL Image objects
- Length: `num_frames` (default 16)

---

### 3. `checkpoint_1_camera_angle_changes(frames)`
Detects sudden camera movements and scene transitions.

**Key Calculations:**

#### Pixel Difference:
```python
frame_diff = mean(|frame_t - frame_t+1|)
```

#### Histogram Correlation:
```python
corr_r = correlation(hist_r1, hist_r2)
corr_g = correlation(hist_g1, hist_g2)
corr_b = correlation(hist_b1, hist_b2)
avg_corr = (corr_r + corr_g + corr_b) / 3
```

#### Change Detection:
```python
if avg_corr < 0.6 OR frame_diff > 60:
    angle_change_detected = True
```

**Returns:** Dict with score, changes list, and statistics

---

### 4. `checkpoint_2_effects_filters_text(frames)`
Detects text overlays, filters, effects, and animations.

**Detection Pipeline:**
```
Frame → Grayscale Conversion
        ↓
    Edge Detection (Sobel)
        ↓
    Contrast Analysis (Local Std)
        ↓
    Binary Region Detection
        ↓
    Color Uniformity Check
        ↓
    Combined Text Score
        ↓
    (score ≥ 50) → TEXT DETECTED
```

**Saturation Check:**
```python
hsv = frame.convert('HSV')
saturation = hsv[:,:,1]
avg_sat = mean(saturation)

if avg_sat > 180 OR avg_sat < 30:
    filter_detected = True
```

**Animation Detection:**
```python
brightness_changes = diff([mean(gray(f)) for f in frames])
rapid_changes = sum(|changes| > 30)
animation_score = min(100, rapid_changes/frames × 200)
```

**Returns:** Dict with text detection, effects count, animation score

---

### 5. `checkpoint_3_multiple_people(frames)`
Counts number of people using DETR object detection.

**Detection Process:**
```python
# Run DETR on sampled frames
detections = detr_model(frame)

# Filter for people with confidence > 0.5
people = [d for d in detections 
          if d['label'] == 'person' and d['score'] > 0.5]

people_count = len(people)
```

**Statistics Calculation:**
```python
max_people = max(all_counts)
avg_people = mean(all_counts)
multiple_frames = count(counts > 1)

score = (multiple_frames / sampled_frames) × 100
```

**Returns:** Dict with people count, statistics, and detection details

---

### 6. `_checkpoint_3_fallback(frames)`
Fallback people detection using skin tone heuristics.

**Skin Detection:**
```python
# RGB thresholds for skin tones
skin_mask = (R > 95) & (G > 40) & (B > 20) & 
            (R > G) & (G > B)

skin_ratio = sum(skin_mask) / total_pixels

# If > 5% skin pixels, likely has person
has_people = (skin_ratio > 0.05)
```

**Returns:** Dict with basic people detection (max_people: 0 or 1)

---

### 7. `checkpoint_4_ai_deepfake(frames)`
Detects AI-generated or manipulated faces.

**Model Inference:**
```python
# Run deepfake classifier
result = deepfake_model(frame)

# Extract fake probability
for item in result:
    if 'fake' in item['label'].lower():
        fake_prob = max(fake_prob, item['score'])
```

**Scoring:**
```python
avg_fake_prob = mean(fake_probabilities)
max_fake_prob = max(fake_probabilities)

checkpoint4_score = avg_fake_prob × 100
is_deepfake = (avg_fake_prob > 0.5)
```

**Returns:** Dict with deepfake probability, confidence, suspicious frames

---

### 8. `analyze_video(video_path, num_frames=16)`
Main analysis pipeline orchestrating all 4 checkpoints.

**Execution Flow:**
```
1. Extract frames from video (PyAV)
   ↓
2. Run Checkpoint 1 (Camera Angles)
   ↓
3. Run Checkpoint 2 (Effects/Text)
   ↓
4. Run Checkpoint 3 (People Count)
   ↓
5. Run Checkpoint 4 (Deepfake)
   ↓
6. Calculate base score (weighted average)
   ↓
7. Apply violation penalties
   ↓
8. Determine risk level and verdict
   ↓
9. Return comprehensive results dict
```

**Scoring Pipeline:**
```python
# Step 1: Base score
base = cp1×0.20 + cp2×0.35 + cp3×0.10 + cp4×0.35

# Step 2: Penalties
penalties = text_penalty + deepfake_penalty + 
            effects_penalty + angles_penalty

# Step 3: Total
overall = min(100, base + penalties)

# Step 4: Risk classification
if has_text OR is_deepfake:
    risk = "CRITICAL"
elif overall >= 60:
    risk = "HIGH"
elif overall >= 40:
    risk = "MEDIUM"
elif overall >= 20:
    risk = "LOW-MEDIUM"
else:
    risk = "LOW"
```

**Returns:** Comprehensive results dictionary

---

### 9. `print_report(results)`
Generates detailed human-readable analysis report.

**Report Sections:**
```
┌─────────────────────────────────┐
│ OVERALL RISK SCORE              │
│ Base Score + Penalties          │
│ Risk Level + Verdict            │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ VIOLATIONS DETECTED (if any)    │
│ • Text: +penalty                │
│ • Deepfake: +penalty            │
│ • Effects: +penalty             │
│ • Cuts: +penalty                │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ CHECKPOINT 1: Camera Angles     │
│ Score, changes, ratio           │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ CHECKPOINT 2: Effects/Text      │
│ Text frames, effects, animations│
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ CHECKPOINT 3: People Count      │
│ Max, avg, multiple detection    │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ CHECKPOINT 4: Deepfake          │
│ Fake probability, confidence    │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│ FINAL ASSESSMENT                │
│ Critical violations summary     │
│ Warnings list                   │
│ System decision + action        │
└─────────────────────────────────┘
```

---

## AI Models Used

### 1. Deepfake Detection Model
**Model:** `dima806/deepfake_vs_real_image_detection`
- **Type:** Image Classification (Binary)
- **Purpose:** Detect AI-generated/manipulated faces
- **Output:** Probability scores for 'real' vs 'fake'
- **Threshold:** 0.5 (>0.5 = deepfake)

### 2. Object Detection Model
**Model:** `facebook/detr-resnet-50`
- **Type:** DEtection TRansformer (DETR)
- **Purpose:** Detect people and objects in frames
- **Output:** Bounding boxes + class labels + confidence
- **Classes:** 80 COCO classes (including 'person')

### 3. Feature Extraction Model
**Model:** `microsoft/resnet-50`
- **Type:** Convolutional Neural Network (ResNet)
- **Purpose:** Extract visual features for analysis
- **Layers:** 50 layers (residual connections)
- **Output:** Feature embeddings

### 4. Text Detection Model
**Model:** `microsoft/trocr-base-printed`
- **Type:** Transformer OCR (TrOCR)
- **Purpose:** Detect text overlays in frames
- **Architecture:** Vision Encoder + Decoder
- **Output:** Text strings detected in image

---

## Detection Algorithms

### Edge Detection (Sobel Operator)

#### Horizontal Edges:
```python
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

edges_x = convolve2d(gray_image, sobel_x)
```

#### Vertical Edges:
```python
sobel_y = [[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]]

edges_y = convolve2d(gray_image, sobel_y)
```

#### Edge Magnitude:
```python
edge_magnitude = sqrt(edges_x² + edges_y²)
```

### Histogram Correlation

#### Pearson Correlation Coefficient:
```python
# For two histograms H1 and H2
mean_H1 = mean(H1)
mean_H2 = mean(H2)

numerator = sum((H1 - mean_H1) × (H2 - mean_H2))
denominator = sqrt(sum((H1 - mean_H1)²) × sum((H2 - mean_H2)²))

correlation = numerator / denominator

# Range: -1 to +1
# +1 = perfect positive correlation (similar)
#  0 = no correlation
# -1 = perfect negative correlation (opposite)
```

### Local Standard Deviation

#### Generic Filter:
```python
# For each pixel, calculate std of surrounding 10x10 window
local_std[i,j] = std(image[i-5:i+5, j-5:j+5])

# High local_std indicates high contrast (text regions)
```

---

## Usage Examples

### Example 1: Analyze Single Video
```python
# Initialize detector
detector = DeepfakeVideoDetector()

# Analyze video with 16 frames
results = detector.analyze_video("workout_video.mp4", num_frames=16)

# Print detailed report
detector.print_report(results)
```

### Example 2: Check Specific Violations
```python
results = detector.analyze_video("video.mp4")

# Check for text
if results['violations']['has_text']:
    print("⚠️ Text overlay detected!")
    
# Check for deepfake
if results['violations']['is_deepfake']:
    print("⚠️ AI-generated content detected!")
    
# Check overall risk
if results['overall_score'] >= 40:
    print("🚫 Video REJECTED")
else:
    print("✅ Video APPROVED")
```

### Example 3: Custom Frame Count
```python
# Analyze with more frames for better accuracy
results = detector.analyze_video(
    "video.mp4",
    num_frames=32  # Double the default
)
```

### Example 4: CLI Usage
```bash
# Command line interface
python cheat.py video.mp4

# Output: Full analysis report with all 4 checkpoints
```

### Example 5: Integration with Server
```python
# In server.py
from cheat import DeepfakeVideoDetector

# Initialize once at startup
cheat_detector = DeepfakeVideoDetector()

# Analyze uploaded video
def check_video_authenticity(video_path):
    results = cheat_detector.analyze_video(video_path)
    
    # Block if MEDIUM or HIGH risk
    if results['overall_score'] >= 40:
        return {
            'blocked': True,
            'reason': results['verdict'],
            'score': results['overall_score']
        }
    
    return {'blocked': False, 'score': results['overall_score']}
```

---

## Output Structure

### Complete Results Dictionary:
```python
{
    'video_path': 'path/to/video.mp4',
    'frames_analyzed': 16,
    'overall_score': 67.5,
    'base_score': 45.0,
    'risk_level': '🔴 HIGH',
    'verdict': 'HIGH RISK - Heavily Manipulated',
    
    'violations': {
        'has_text': True,
        'is_deepfake': False,
        'has_effects': True,
        'excessive_cuts': False,
        'text_penalty': 52.5,
        'deepfake_penalty': 0.0,
        'effects_penalty': 22.5,
        'angles_penalty': 0.0
    },
    
    'checkpoint_1_camera_angles': {
        'score': 25.0,
        'angle_changes': [...],
        'total_changes': 2,
        'change_ratio': 0.133,
        'is_suspicious': False
    },
    
    'checkpoint_2_effects_text': {
        'score': 75.0,
        'text_frames': 12,
        'text_ratio': 0.75,
        'effects_detected': 6,
        'animation_score': 35.0,
        'has_text': True,
        'has_effects': True,
        'details': {...}
    },
    
    'checkpoint_3_people': {
        'score': 50.0,
        'max_people': 2,
        'avg_people': 1.5,
        'frames_with_multiple': 4,
        'has_multiple_people': True,
        'details': [...]
    },
    
    'checkpoint_4_deepfake': {
        'score': 30.0,
        'is_deepfake': False,
        'confidence': 45.0,
        'avg_fake_probability': 0.30,
        'max_fake_probability': 0.45,
        'suspicious_frames': 0,
        'details': []
    }
}
```

---

## Error Handling

### Robust Model Loading:
```python
try:
    model = load_model(model_name)
except Exception as e:
    print(f"⚠ Model unavailable: {e}")
    model = None  # Continue with other models
```

### Frame Extraction Fallback:
```python
# Try PyAV first
try:
    frames = read_with_pyav(video_path)
except Exception as e:
    print("⚠ PyAV failed, trying imageio...")
    frames = read_with_imageio(video_path)
```

### Checkpoint Error Handling:
```python
try:
    checkpoint_result = run_checkpoint(frames)
except Exception as e:
    print(f"⚠ Checkpoint error: {e}")
    checkpoint_result = default_result()  # Safe fallback
```

---

## Performance Considerations

### Frame Sampling Strategy:
- **Default:** 16 frames evenly spaced
- **Rationale:** Balance accuracy vs speed
- **Sampling:** `linspace(0, total_frames-1, 16)`

### GPU Acceleration:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Models automatically use GPU if available
```

### Memory Management:
- Frames processed individually
- No full video loaded into memory
- Models loaded once at initialization

### Optimization Tips:
```python
# Reduce frames for faster processing
results = detector.analyze_video(video, num_frames=8)

# Sample frames instead of all
sample_indices = range(0, len(frames), 2)  # Every other frame
```

---

## Technical Specifications

### Input Requirements:
- **Format:** MP4, AVI, MOV (any format supported by PyAV/imageio)
- **Resolution:** Any (resized internally by models)
- **Duration:** Any (frames sampled evenly)
- **FPS:** Any (time-independent)

### Output Format:
- **Type:** Python dictionary (JSON-serializable)
- **Size:** ~5-20 KB depending on detections
- **Precision:** Float values to 1 decimal place

### Processing Time:
- **16 frames:** ~5-15 seconds (GPU) / ~30-60 seconds (CPU)
- **32 frames:** ~10-30 seconds (GPU) / ~60-120 seconds (CPU)

---

## Summary

`cheat.py` is a comprehensive video authentication system that:

✅ **4-Checkpoint Analysis**: Camera angles, effects/text, people, deepfakes
✅ **AI-Powered Detection**: 4 pre-trained deep learning models
✅ **Zero-Tolerance Policy**: Automatic rejection for text/deepfakes
✅ **Multi-Method Detection**: Combines edge detection, histograms, AI classifiers
✅ **Strict Penalty System**: 3x penalty for text, 2.5x for deepfakes
✅ **Robust Error Handling**: Fallback methods for each checkpoint
✅ **Comprehensive Reporting**: Detailed analysis with actionable decisions
✅ **Frame Sampling**: Efficient processing with representative frames
✅ **GPU Acceleration**: Automatic CUDA support for faster inference
✅ **Flexible Integration**: Easy to use standalone or in server pipeline

**Key Innovation**: Combines computer vision (edge detection, histograms), deep learning (DETR, ResNet, TrOCR), and strict policy enforcement to provide enterprise-grade video authentication with zero tolerance for prohibited content.

**Use Case**: Perfect for exercise analysis systems, authentication platforms, or any application requiring verification that videos are genuine, unedited recordings without overlays or AI manipulation.
