# Kinemation 3D Pose Estimation Pipeline - Improved Workflow

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [How 2D Pose Estimation Works](#how-2d-pose-estimation-works)
3. [How VideoPose3D Works](#how-videopose3d-works)
4. [Original Implementation Issues](#original-implementation-issues)
5. [Changes Made](#changes-made)
6. [Line-by-Line Code Explanations](#line-by-line-code-explanations)
7. [Testing and Validation](#testing-and-validation)
8. [Remaining Limitations](#remaining-limitations)
9. [planv4.md Implementations](#planv4md-implementations-visualization-improvements)
10. [planv5.md Implementation](#planv5md-implementation-joint-adaptive-smoothing)
11. [VideoPose3D vs MotionBERT](#videopose3d-vs-motionbert-a-comprehensive-comparison)
12. [final/ vs vidpose-amrita](#final-vs-vidpose-amrita-pipeline-comparison)
13. [planv6.md Implementation: True 3D Visualization](#planv6md-implementation-true-3d-visualization)

---

## Executive Summary

This document describes the improvements made to the Kinemation 3D pose estimation pipeline. **Three critical bugs were fixed:**

| Bug | Impact | Fix |
|-----|--------|-----|
| **Wrong Input Format** | Completely distorted 3D skeletons | Feed COCO format, not H36M |
| **Wrong Axis Flips** | 90° rotated skeleton orientation | Remove X/Y flips |
| **3D Projection Math Wrong** | 4.54x skeleton inflation | Bbox-anchored projection |
| **Aspect Ratio Distortion** | 40% detection loss on portrait videos | Aspect-preserving resize |

**Key Discovery:** VideoPose3D's `pretrained_h36m_detectron_coco.bin` model:
- **Expects INPUT in COCO 17-keypoint format** (not H36M!)
- **Outputs in H36M 17-joint format** (3D coordinates)
- **Uses same coordinate system as screen** (+X right, +Y down)

Additionally, defensive improvements were made:
- IoU threshold added to tracking (prevents identity swaps)
- Clear documentation of coordinate spaces
- Explicit labeling of display heuristics vs true projection

---

## How 2D Pose Estimation Works

2D pose estimation is the foundation of the entire pipeline. It converts raw video frames into structured skeleton data that can be processed by the 3D lifting model.

### Understanding 2D Pose Estimation

**What is a 2D Pose?**
A 2D pose is a set of keypoints (joints) detected on a human body in an image. Each keypoint has an (x, y) coordinate in pixel space. For example:
- The nose might be at pixel (450, 200)
- The left shoulder might be at pixel (420, 280)
- The right ankle might be at pixel (380, 650)

These keypoints form a skeleton when connected, allowing us to understand human body positioning.

**Why 2D Before 3D?**
We extract 2D poses first because:
1. Direct 3D estimation from images is computationally expensive
2. 2D pose estimation is a mature, well-solved problem
3. 2D→3D "lifting" allows specialized models for each task
4. The same 2D poses can feed different 3D models

### Path A: YOLO Detection + MediaPipe Landmarking (`final/`)

This is the **primary production pipeline** used in Kinemation. It uses two models in sequence:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Frame   │ →  │ Preprocess   │ →  │   YOLOv8n    │ →  │  MediaPipe   │
│              │    │ (CLAHE+Blur) │    │ (Detection)  │    │   (Pose)     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                               │                    │
                                               ▼                    ▼
                                        Bounding boxes      33 landmarks
                                        for each person     per person
```

#### Step 1: Preprocessing (Why We Do It)

Before detection, we preprocess each frame to improve model performance:

**a) CLAHE Enhancement (Contrast Limited Adaptive Histogram Equalization)**

CLAHE improves image contrast, especially in poorly lit areas:

```python
def apply_clahe(frame):
    # Convert BGR to LAB color space
    # LAB separates lightness (L) from color (A,B)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE only to lightness channel
    # clipLimit=2.0 prevents over-amplification of noise
    # tileGridSize=(8,8) divides image into 8x8 tiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
```

**Why LAB space?** LAB separates brightness from color, so we can enhance contrast without affecting color accuracy.

**b) Aspect-Preserving Resize**

We resize frames to max_dim=800 to speed up processing while maintaining proportions:

```python
def preprocess_frame(frame, max_dim=800):
    h, w = frame.shape[:2]
    
    # Only resize if larger than max_dim
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    return frame
```

**Why 800?** It's a balance:
- Smaller = faster but less accurate
- Larger = more accurate but slower
- 800 works well for most videos

**c) Gaussian Blur (Optional)**

A slight blur can reduce noise that causes detection jitter:

```python
frame = cv2.GaussianBlur(frame, (3, 3), 0)
```

#### Step 2: YOLO Person Detection

**What is YOLO?**
YOLO (You Only Look Once) is a real-time object detection model. YOLOv8n is the "nano" variant - smallest and fastest.

**How Detection Works:**
1. Frame is divided into a grid
2. Each grid cell predicts bounding boxes
3. Non-maximum suppression removes overlapping boxes
4. Final output: list of [x1, y1, x2, y2, confidence, class]

```python
def detect_persons(yolo_model, frame, confidence=0.5):
    # Run YOLO inference
    results = yolo_model(frame, verbose=False)[0]
    
    # Filter for person class (class 0) with confidence threshold
    boxes = []
    for box in results.boxes:
        if box.cls == 0 and box.conf >= confidence:  # class 0 = person
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
    
    return boxes
```

**Output Format:**
Each box is `[x1, y1, x2, y2]` where:
- (x1, y1) = top-left corner
- (x2, y2) = bottom-right corner

#### Step 3: ROI Cropping (Region of Interest)

For each detected person, we extract a crop to feed to MediaPipe:

```python
def get_person_crop(frame, box, padding_w=0.15, padding_h=0.10):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # Add padding (15% width, 10% height)
    # This ensures the whole body is visible
    pad_w = w * padding_w
    pad_h = h * padding_h
    
    # Expand bounding box
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(frame.shape[1], x2 + pad_w)
    y2 = min(frame.shape[0], y2 + pad_h)
    
    return frame[int(y1):int(y2), int(x1):int(x2)]
```

**Why Padding?** YOLO boxes are tight. MediaPipe needs to see the full body with some margin.

#### Step 4: MediaPipe Pose Landmarking

**What is MediaPipe Pose?**
MediaPipe Pose is Google's real-time pose estimation model. It detects 33 landmarks on the human body.

**33 Landmark Breakdown:**
- **Face (10):** nose, eyes, ears, mouth corners
- **Upper Body (4):** shoulders, elbows
- **Hands (6):** wrists, pinky, index, thumb
- **Lower Body (4):** hips, knees
- **Feet (6):** ankles, heels, toes
- **Torso (3):** additional reference points

```python
def estimate_pose_mediapipe(landmarker, frame, box):
    # Crop the person region
    crop = get_person_crop(frame, box)
    
    # Convert BGR to RGB (MediaPipe expects RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    
    # Run pose detection
    result = landmarker.detect(mp_image)
    
    if not result.pose_landmarks:
        return None
    
    # Extract landmarks (normalized 0-1 coordinates)
    landmarks = np.zeros((33, 2), dtype=np.float32)
    for i, lm in enumerate(result.pose_landmarks[0]):
        landmarks[i] = [lm.x, lm.y]  # x, y in [0, 1] range
    
    return landmarks
```

#### Coordinate Mapping: Crop to Full Frame

MediaPipe returns landmarks in crop-relative coordinates [0, 1]. We must convert to full-frame pixels:

```python
def map_landmarks_to_frame(landmarks, box, frame_shape):
    """
    Convert landmarks from crop coordinates to full frame coordinates.
    
    landmarks: (33, 2) in [0, 1] range relative to crop
    box: [x1, y1, x2, y2] of the crop in frame coordinates
    """
    x1, y1, x2, y2 = box
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    # Scale and offset
    # landmark_x in [0,1] → frame_x in [x1, x2]
    frame_landmarks = np.zeros_like(landmarks)
    frame_landmarks[:, 0] = x1 + landmarks[:, 0] * crop_w
    frame_landmarks[:, 1] = y1 + landmarks[:, 1] * crop_h
    
    return frame_landmarks
```

**Example:**
- Crop starts at (100, 50) and is 200×400 pixels
- MediaPipe detects nose at (0.5, 0.2) in crop
- Frame coordinates: (100 + 0.5×200, 50 + 0.2×400) = (200, 130)

### Path B: YOLO11 Pose (`vidpose-amrita/`)

This is a **simpler alternative pipeline** that uses a single pose-estimation model:

```
┌──────────────┐    ┌───────────────────────┐
│  Raw Frame   │ →  │  YOLO11-pose          │
│              │    │ (Detection + Pose)    │
└──────────────┘    └───────────────────────┘
                              │
                              ▼
                    17 COCO keypoints
                    per detected person
```

**How YOLO11-pose Works:**
- Single model does detection AND pose estimation
- Directly outputs 17 keypoints in COCO format
- No need for cropping or secondary model

```python
def extract_yolo_keypoints(frame):
    # Single model call
    results = yolo_pose_model(frame, verbose=False)[0]
    
    # Each person has 17 keypoints
    if results.keypoints is not None:
        keypoints = results.keypoints.xy.cpu().numpy()  # (num_people, 17, 2)
        return keypoints
    
    return None
```

**COCO 17 Keypoints:**
| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

**Comparison:**
| Aspect | Path A (YOLOv8n + MediaPipe) | Path B (YOLO11-pose) |
|--------|------------------------------|---------------------|
| Models | 2 | 1 |
| Keypoints | 33 | 17 |
| Face detail | ✅ Yes (eyes, ears, mouth) | ⚠️ Basic (nose, eyes, ears) |
| Speed | Slower | Faster |
| Accuracy | Higher | Good |
| Complexity | Higher | Lower |

---

## How VideoPose3D Works

### What is 3D Pose Estimation?

While 2D pose estimation tells us where body joints appear in an image (x, y pixels), 3D pose estimation adds depth information, telling us how far each joint is from the camera. This transforms a flat skeleton into a 3D model.

**The "Lifting" Concept:**
"Lifting" means taking 2D joint positions and predicting their 3D positions. This is challenging because:
- Multiple 3D poses can project to the same 2D pose
- A person with arm extended forward vs. sideways might look similar in 2D
- The model must learn to disambiguate using context

### VideoPose3D Overview

VideoPose3D is a neural network that "lifts" 2D poses to 3D. Its key innovation is using **temporal context** - it looks at many frames to make better predictions.

```
┌─────────────────────────────────────────────────────────────────┐
│                    VideoPose3D Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Frame N-121        Frame N          Frame N+121               │
│      ↓                  ↓                  ↓                     │
│   ┌────────┐        ┌────────┐        ┌────────┐               │
│   │  2D    │        │  2D    │        │  2D    │               │
│   │ Pose   │        │ Pose   │        │ Pose   │               │
│   └────────┘        └────────┘        └────────┘               │
│        ↓                ↓                  ↓                    │
│   ┌────────────────────────────────────────────┐                │
│   │     Temporal Convolutional Network         │                │
│   │     (TCN with 243-frame receptive field)   │                │
│   └────────────────────────────────────────────┘                │
│                          ↓                                      │
│                    ┌────────┐                                   │
│                    │   3D   │                                   │
│                    │  Pose  │                                   │
│                    │ Frame N│                                   │
│                    └────────┘                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Expected Input Format

VideoPose3D's `pretrained_h36m_detectron_coco.bin` model expects **COCO 17-keypoint format**:

| COCO Index | Joint Name | Description |
|------------|------------|-------------|
| 0 | nose | Center of face |
| 1 | left_eye | Left eye position |
| 2 | right_eye | Right eye position |
| 3 | left_ear | Left ear position |
| 4 | right_ear | Right ear position |
| 5 | left_shoulder | Left shoulder joint |
| 6 | right_shoulder | Right shoulder joint |
| 7 | left_elbow | Left elbow joint |
| 8 | right_elbow | Right elbow joint |
| 9 | left_wrist | Left wrist joint |
| 10 | right_wrist | Right wrist joint |
| 11 | left_hip | Left hip joint |
| 12 | right_hip | Right hip joint |
| 13 | left_knee | Left knee joint |
| 14 | right_knee | Right knee joint |
| 15 | left_ankle | Left ankle joint |
| 16 | right_ankle | Right ankle joint |

**Critical Note:** The model name `h36m_detectron_coco` tells us:
- **h36m**: Trained on Human3.6M dataset (3D ground truth)
- **detectron**: 2D detections from Detectron2
- **coco**: Input must be in COCO format

### Output Format: H36M 17-Joint

The model **outputs** in H36M format (different from input!):

| H36M Index | Joint Name | Notes |
|------------|------------|-------|
| 0 | hip_center | Root joint (origin) |
| 1 | right_hip | |
| 2 | right_knee | |
| 3 | right_ankle | |
| 4 | left_hip | |
| 5 | left_knee | |
| 6 | left_ankle | |
| 7 | spine | Midpoint between hip and thorax |
| 8 | thorax | Chest center |
| 9 | neck | Base of neck |
| 10 | head | Top of head (often approximated) |
| 11 | left_shoulder | |
| 12 | left_elbow | |
| 13 | left_wrist | |
| 14 | right_shoulder | |
| 15 | right_elbow | |
| 16 | right_wrist | |

### H36M Conversion

**Converting MediaPipe (33 landmarks) to COCO (17 keypoints):**

MediaPipe has many more landmarks than needed. We select the 17 that correspond to COCO:

```python
# MediaPipe landmark indices for each COCO keypoint
MEDIAPIPE_TO_COCO = [
    0,   # COCO 0 (nose)      ← MediaPipe 0 (nose)
    2,   # COCO 1 (left_eye)  ← MediaPipe 2 (left_eye_outer)
    5,   # COCO 2 (right_eye) ← MediaPipe 5 (right_eye_outer)
    7,   # COCO 3 (left_ear)  ← MediaPipe 7 (left_ear)
    8,   # COCO 4 (right_ear) ← MediaPipe 8 (right_ear)
    11,  # COCO 5 (l_shoulder)← MediaPipe 11 (left_shoulder)
    12,  # COCO 6 (r_shoulder)← MediaPipe 12 (right_shoulder)
    13,  # COCO 7 (l_elbow)   ← MediaPipe 13 (left_elbow)
    14,  # COCO 8 (r_elbow)   ← MediaPipe 14 (right_elbow)
    15,  # COCO 9 (l_wrist)   ← MediaPipe 15 (left_wrist)
    16,  # COCO 10 (r_wrist)  ← MediaPipe 16 (right_wrist)
    23,  # COCO 11 (l_hip)    ← MediaPipe 23 (left_hip)
    24,  # COCO 12 (r_hip)    ← MediaPipe 24 (right_hip)
    25,  # COCO 13 (l_knee)   ← MediaPipe 25 (left_knee)
    26,  # COCO 14 (r_knee)   ← MediaPipe 26 (right_knee)
    27,  # COCO 15 (l_ankle)  ← MediaPipe 27 (left_ankle)
    28,  # COCO 16 (r_ankle)  ← MediaPipe 28 (right_ankle)
]

def mediapipe_to_coco(landmarks):
    """
    Convert 33 MediaPipe landmarks to 17 COCO keypoints.
    
    Input: landmarks (33, 2) - MediaPipe BlazePose landmarks
    Output: coco_kps (17, 2) - COCO format keypoints
    """
    return landmarks[MEDIAPIPE_TO_COCO]
```

### Normalization

Before feeding to VideoPose3D, 2D keypoints must be normalized to a consistent coordinate space:

```python
def normalize_screen_coordinates(X, w, h):
    """
    Normalize 2D keypoints to a centered coordinate system.
    
    Input:
        X: (N, 17, 2) keypoints in pixel coordinates
        w: frame width in pixels
        h: frame height in pixels
    
    Output:
        Normalized coordinates where:
        - X axis: [-1, 1] from left to right edge
        - Y axis: [-h/w, h/w] from top to bottom (preserves aspect ratio)
    
    Formula:
        x_norm = (x_pixel / w) * 2 - 1
        y_norm = (y_pixel / w) * 2 - (h / w)
    """
    return X / w * 2 - np.array([1, h / w])
```

**Why This Normalization?**
1. **Resolution independence**: Different video resolutions produce same normalized values for same body positions
2. **Centered origin**: (0, 0) is at the center of the frame
3. **Aspect ratio preservation**: Y-range varies with aspect ratio, preventing distortion

**Example for 1920×1080 (16:9) video:**

| Pixel Position | Normalized Position | Location |
|---------------|---------------------|----------|
| (0, 0) | (-1.0, -0.5625) | Top-left |
| (960, 540) | (0.0, 0.0) | Center |
| (1920, 1080) | (1.0, 0.5625) | Bottom-right |
| (1920, 0) | (1.0, -0.5625) | Top-right |

### Temporal Receptive Field

VideoPose3D uses **temporal convolutions** to aggregate information across frames.

**Key Parameters:**
- Filter widths: `[3, 3, 3, 3, 3]` (5 layers, each with kernel size 3)
- Receptive field: 3⁵ = 243 frames
- Center frame: frame 122 (0-indexed: 121)

**What This Means:**
To predict the 3D pose at frame N, the model looks at frames [N-121, N+121] (243 frames total).

```
Frame:     0   1   2   ...  119  120  121  122  123  ...  241  242
           |   |   |   ...   |    |    |    |    |   ...   |    |
           └───┴───┴─────────┴────┴────┼────┴────┴─────────┴────┘
                                       ↑
                              Output for frame 122
                         (center of 243-frame window)
```

**Why 243 Frames?**
- More context = better depth estimation
- Smooth motions (walking) benefit from long-range patterns
- Trade-off: 243 frames @ 30fps = ~8 seconds of context

### Sliding Window Inference

For real-time or shorter videos, we use sliding window inference:

```python
def run_videopose3d_inference(keypoints_2d, model):
    """
    Run VideoPose3D inference using sliding window.
    
    Input: keypoints_2d (N, 17, 2) - normalized 2D keypoints
    Output: poses_3d (N, 17, 3) - 3D poses in root-relative coordinates
    """
    N = keypoints_2d.shape[0]
    pad = 121  # Half of receptive field (minus center)
    
    # Pad sequence by repeating first/last frames
    # This handles edge cases where we don't have 121 frames before/after
    left_pad = np.tile(keypoints_2d[0:1], (pad, 1, 1))   # Repeat first frame
    right_pad = np.tile(keypoints_2d[-1:], (pad, 1, 1))  # Repeat last frame
    keypoints_padded = np.concatenate([left_pad, keypoints_2d, right_pad])
    # Shape: (N + 242, 17, 2)
    
    poses_3d = []
    
    for i in range(N):
        # Extract 243-frame window centered at frame i
        window = keypoints_padded[i : i + 243]  # (243, 17, 2)
        
        # Add batch dimension: (243, 17, 2) → (1, 243, 17, 2)
        input_tensor = torch.from_numpy(window).unsqueeze(0).float()
        
        # Run inference
        with torch.no_grad():
            pred_3d = model(input_tensor)  # Output: (1, 1, 17, 3)
        
        # Extract prediction for center frame
        poses_3d.append(pred_3d[0, 0].numpy())  # (17, 3)
    
    return np.array(poses_3d)  # (N, 17, 3)
```

### What VideoPose3D Outputs

**CRITICAL UNDERSTANDING:** VideoPose3D outputs **root-relative 3D coordinates**, NOT screen coordinates.

```
Output shape per frame: (17, 3)
- X: Horizontal offset from hip center (in meters-like units)
- Y: Vertical offset from hip center (negative = above hip, positive = below)
- Z: Depth offset from hip center (positive = closer to camera)
```

**Important Properties:**
1. **Root-relative**: All coordinates are relative to joint 0 (hip_center), which is at (0, 0, 0)
2. **Metric-like scale**: Values represent approximate real-world distances (roughly in meters)
3. **NOT bounded**: Unlike normalized inputs, outputs can be any value (e.g., -1.5 to +1.5)
4. **Consistent orientation**: Same as camera view (right = +X, down = +Y, toward camera = +Z)

**Example output for a standing person:**
| Joint | X | Y | Z | Interpretation |
|-------|---|---|---|----------------|
| hip_center (0) | 0.0 | 0.0 | 0.0 | Origin |
| head (10) | 0.0 | -0.65 | 0.05 | 65cm above hip, slightly forward |
| left_wrist (13) | 0.35 | -0.30 | -0.10 | 35cm left, 30cm up, 10cm back |
| right_ankle (3) | -0.15 | 0.92 | 0.02 | 15cm right, 92cm below hip |

---

## Original Implementation Issues

### Issue 1: Wrong Input Format (CRITICAL - NEWLY DISCOVERED)

**Location:** `final/mediapipe_to_h36m.py` 

**Original Assumption:**
We were converting MediaPipe landmarks to H36M 17-joint format and feeding that to VideoPose3D.

**The Critical Discovery:**
VideoPose3D's `pretrained_h36m_detectron_coco.bin` model expects:
- **INPUT**: COCO 17-keypoint format (2D)
- **OUTPUT**: H36M 17-joint format (3D)

The model name `h36m_detectron_coco` means:
- Trained on **H36M** dataset for 3D ground truth
- Uses **Detectron2** for 2D detection
- Expects **COCO** format 2D keypoints as input

**COCO 17 keypoints** (what model expects):
```
0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
```

**H36M 17 joints** (what we were feeding, WRONG):
```
0:hip, 1:right_hip, 2:right_knee, 3:right_ankle, 4:left_hip,
5:left_knee, 6:left_ankle, 7:spine, 8:thorax, 9:neck, 10:head,
11:left_shoulder, 12:left_elbow, 13:left_wrist, 14:right_shoulder,
15:right_elbow, 16:right_wrist
```

Feeding H36M format to a model expecting COCO format caused completely wrong 3D predictions.

### Issue 2: 3D Projection Axis Flips Wrong

**Location:** `final/pipeline_3d.py` projection function

**Original Code:**
```python
# WRONG: Flipped both X and Y axes
px = -(x - bbox_3d_center[0]) * scale + bbox_2d_center[0]
py = -(y - bbox_3d_center[1]) * scale + bbox_2d_center[1]
```

**The Discovery:**
Through controlled experiments, we verified that VideoPose3D uses the **SAME coordinate system as screen coordinates**:
- **3D X**: +X = right (same as screen)
- **3D Y**: +Y = down (same as screen)
- **3D Z**: depth (not used for 2D projection)

No axis flips are needed - direct mapping works correctly.

### Issue 3: 3D Projection Math Wrong (Original Finding)

**Location:** `final/pipeline_3d.py` lines 449-453 (before fix)

**Original Code:**
```python
# WRONG: Treats 3D output as if it were screen-normalized
px = int((x + 1) * video_width / 2)
py = int((y + video_height / video_width) * video_width / 2)
```

**What This Does:**
- Assumes VideoPose3D output is in same space as input normalization
- Applies inverse of normalize_screen_coordinates()
- This is mathematically incorrect

**Evidence:**
- Measured 2D input range: X ∈ [-0.25, 0.29], Y ∈ [-0.27, 0.56]
- Measured 3D output range: X ∈ [-0.45, 0.13], Y ∈ [-0.77, 0.55]
- Output NOT bounded to input range
- Result: 4.54x skeleton area inflation

**Root Cause:**
VideoPose3D outputs root-relative metric coordinates, not normalized screen coordinates. The denormalization formula doesn't apply.

### Issue 2: Aspect Ratio Distortion (CRITICAL)

**Location:** `vidpose-amrita/kinemation/extract_keypoints.py` line 27-28

**Original Code:**
```python
frame_r = cv2.resize(frame, (1440, 810))  # Forces landscape aspect ratio
```

**What This Does:**
- Resizes ALL videos to 1440×810 regardless of original aspect ratio
- Portrait videos (e.g., vid3.mp4 at 2160×4096) get stretched 3.37x horizontally
- Detected poses have wrong proportions
- Detection rate drops 40% due to distorted humans

### Issue 3: No IoU Threshold in Tracking

**Location:** `final/pipeline_3d.py` batch_track_people function

**Original Behavior:**
- Hungarian algorithm assigns detections to tracks
- No minimum IoU requirement
- Bad assignments possible when IoU < 0.3 (no overlap)

**Risk:** Identity contamination in crowded scenes

---

## Changes Made

### Change 1: COCO Input Format (CRITICAL FIX)

**Files Modified:** `final/mediapipe_to_h36m.py` (complete rewrite)

**Problem:** We were feeding H36M-format keypoints to a model expecting COCO format.

**Solution:** Rewrote the adapter to output COCO 17-keypoint format:

```python
# MediaPipe landmark indices for each COCO keypoint
MEDIAPIPE_TO_COCO = [
    0,   # 0: nose
    2,   # 1: left_eye (MediaPipe uses inner eye, we use outer)
    5,   # 2: right_eye
    7,   # 3: left_ear
    8,   # 4: right_ear
    11,  # 5: left_shoulder
    12,  # 6: right_shoulder
    13,  # 7: left_elbow
    14,  # 8: right_elbow
    15,  # 9: left_wrist
    16,  # 10: right_wrist
    23,  # 11: left_hip
    24,  # 12: right_hip
    25,  # 13: left_knee
    26,  # 14: right_knee
    27,  # 15: left_ankle
    28,  # 16: right_ankle
]

def mediapipe_to_coco(landmarks, frame_width, frame_height):
    """Convert MediaPipe 33 landmarks to COCO 17 keypoints."""
    keypoints_2d = np.zeros((17, 2), dtype=np.float32)
    for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
        lm = landmarks[mp_idx]
        keypoints_2d[coco_idx] = [lm.x * frame_width, lm.y * frame_height]
    return keypoints_2d
```

**Result:** VideoPose3D now receives correctly formatted input and produces anatomically correct 3D skeletons.

### Change 2: Removed Incorrect Axis Flips

**Files Modified:** `final/pipeline_3d.py` (projection function)

**Problem:** Code assumed VideoPose3D used opposite coordinate conventions.

**Before:**
```python
px = -(x - bbox_3d_center[0]) * scale + bbox_2d_center[0]  # X flipped
py = -(y - bbox_3d_center[1]) * scale + bbox_2d_center[1]  # Y flipped
```

**After:**
```python
px = (x - bbox_3d_center[0]) * scale + bbox_2d_center[0]   # No flip
py = (y - bbox_3d_center[1]) * scale + bbox_2d_center[1]   # No flip
```

**Why:** Experimental verification showed VideoPose3D uses the SAME coordinate system as screen coordinates (+X right, +Y down).

### Change 3: Updated 2D Skeleton Connections

**Files Modified:** `final/pipeline_3d.py`, `final/mediapipe_to_h36m.py`

**Problem:** 2D visualization was using H36M_CONNECTIONS with COCO keypoints.

**Solution:** Added COCO_CONNECTIONS and updated draw_skeleton_2d to use them:

```python
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # Face
    (5, 6),                                # Shoulders
    (5, 7), (7, 9),                        # Left arm
    (6, 8), (8, 10),                       # Right arm
    (5, 11), (6, 12),                      # Torso
    (11, 12),                              # Hips
    (11, 13), (13, 15),                    # Left leg
    (12, 14), (14, 16),                    # Right leg
]
```

### Change 4: Bbox-Anchored 3D Projection

**Files Modified:** `final/pipeline_3d.py`

**New Function:** `project_3d_to_2d_anchored(keypoints_3d, keypoints_2d)`

**Algorithm:**
1. Compute bounding box of 2D keypoints (anchor)
2. Compute bounding box of 3D keypoints in XY plane
3. Scale 3D skeleton to match 2D bbox size
4. Translate 3D skeleton center to 2D bbox center
5. Use Z only for depth coloring (brightness)

**Why This Works:**
- 2D keypoints give us the correct screen position/size
- 3D keypoints give us the pose shape and depth
- We combine them: 3D shape at 2D location

**Important Note:** This is a **DISPLAY HEURISTIC**, not true geometric camera projection. True reprojection would require camera intrinsic/extrinsic parameters that we don't have.

### Change 2: Aspect-Preserving Resize

**Files Modified:** 
- `vidpose-amrita/kinemation/extract_keypoints.py`
- `vidpose-amrita/kinemation/run_vid3.py`

**New Function:** `resize_preserve_aspect(frame, max_dim=1440)`

**Algorithm:**
```python
def resize_preserve_aspect(frame, max_dim=1440):
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame, 1.0
    
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(frame, (new_w, new_h)), scale
```

**Result:**
- vid3.mp4 (2160×4096 portrait): 759×1440 (aspect preserved)
- vid1.mp4 (1920×1080 landscape): 1440×810 (same as before)

### Change 3: IoU Tracking Threshold

**Files Modified:** `final/pipeline_3d.py`, `vidpose-amrita/kinemation/run_vid3.py`

**New Parameter:** `batch_track_people(raw_keypoints, iou_threshold=0.3)`

**Logic:**
```python
for t_id, d_id in zip(t_ids, d_ids):
    iou = iou_matrix[t_id, d_id]
    
    # Reject assignments with IoU below threshold
    if iou < iou_threshold and prev_boxes[t_id] is not None:
        continue  # Detection becomes new track instead
```

**Effect:** Prevents identity swaps when tracked person doesn't overlap with detection

---

## Line-by-Line Code Explanations

### `final/pipeline_3d.py` - New Projection Functions

```python
def project_3d_to_2d_anchored(keypoints_3d, keypoints_2d):
    """
    Project 3D skeleton to 2D screen coordinates using bbox-anchored approach.
    
    NOTE: This is a PRACTICAL DISPLAY HEURISTIC, not true geometric camera projection.
    True reprojection would require camera intrinsics/extrinsics which we don't have.
    """
```
**Line 1-6:** Function docstring explaining what this function does and its limitations.

```python
    points_2d = []      # Will store projected (x,y) screen coordinates
    z_values = []       # Will store z values for depth coloring
```
**Line 7-8:** Initialize output lists.

```python
    # Get valid 2D keypoints for anchor bbox
    valid_2d_mask = (keypoints_2d[:, 0] > 0) & (keypoints_2d[:, 1] > 0)
    valid_3d_mask = np.abs(keypoints_3d).max(axis=1) > 0.001
```
**Line 10-12:** Create masks for valid keypoints. A 2D keypoint is valid if both x,y > 0. A 3D keypoint is valid if any coordinate > 0.001 (not all zeros).

```python
    if valid_2d_mask.sum() < 2 or valid_3d_mask.sum() < 2:
        return [None] * 17, [0] * 17
```
**Line 14-15:** Early return if fewer than 2 valid keypoints in either set. Can't compute meaningful bbox.

```python
    # Compute 2D anchor bbox from valid 2D keypoints
    valid_2d = keypoints_2d[valid_2d_mask]
    bbox_2d_min = valid_2d.min(axis=0)
    bbox_2d_max = valid_2d.max(axis=0)
    bbox_2d_center = (bbox_2d_min + bbox_2d_max) / 2
    bbox_2d_size = bbox_2d_max - bbox_2d_min
    bbox_2d_size = np.maximum(bbox_2d_size, 1)  # Avoid division by zero
```
**Line 17-23:** Compute 2D bounding box. This is our "anchor" - where the skeleton should appear on screen.

```python
    # Compute 3D bbox in XY plane from valid 3D keypoints
    valid_3d = keypoints_3d[valid_3d_mask]
    bbox_3d_min = valid_3d[:, :2].min(axis=0)  # Only X,Y
    bbox_3d_max = valid_3d[:, :2].max(axis=0)
    bbox_3d_center = (bbox_3d_min + bbox_3d_max) / 2
    bbox_3d_size = bbox_3d_max - bbox_3d_min
    bbox_3d_size = np.maximum(bbox_3d_size, 0.001)
```
**Line 25-31:** Compute 3D bounding box in XY plane (ignoring Z for positioning).

```python
    # Compute scale factor: map 3D bbox to 2D bbox size
    scale = min(bbox_2d_size[0] / bbox_3d_size[0], 
                bbox_2d_size[1] / bbox_3d_size[1])
```
**Line 33-35:** Compute uniform scale factor. Use minimum to preserve aspect ratio of skeleton.

```python
    for i in range(17):
        x, y, z = keypoints_3d[i]
        
        if abs(x) < 0.001 and abs(y) < 0.001 and abs(z) < 0.001:
            points_2d.append(None)
            z_values.append(0)
            continue
        
        # Transform: center 3D at origin, scale, then translate to 2D center
        px = (x - bbox_3d_center[0]) * scale + bbox_2d_center[0]
        py = (y - bbox_3d_center[1]) * scale + bbox_2d_center[1]
        
        points_2d.append((int(px), int(py)))
        z_values.append(z)
```
**Line 37-51:** For each joint:
1. Skip if all zeros (invalid)
2. Center 3D coordinates by subtracting 3D bbox center
3. Scale by computed scale factor
4. Translate to 2D bbox center
5. Store pixel coordinates and z value

### `vidpose-amrita/kinemation/extract_keypoints.py` - Aspect Preserving Resize

```python
def resize_preserve_aspect(frame, max_dim=1440):
    """
    Resize frame while preserving aspect ratio.
    """
    h, w = frame.shape[:2]
```
**Line 1-5:** Function signature and get current dimensions.

```python
    if max(h, w) <= max_dim:
        return frame, 1.0
```
**Line 7-8:** If already smaller than max_dim, return unchanged with scale=1.

```python
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
```
**Line 10-15:** 
1. Compute scale so largest dimension becomes max_dim
2. Apply same scale to both dimensions (preserves aspect ratio)
3. Resize with INTER_AREA (best for downscaling)
4. Return resized frame and scale factor

### `final/pipeline_3d.py` - IoU Tracking Threshold

```python
def batch_track_people(raw_keypoints, iou_threshold=0.3):
```
**Line 1:** Added iou_threshold parameter with default 0.3.

```python
        cost = np.ones((max_people, len(det_boxes)))
        iou_matrix = np.zeros((max_people, len(det_boxes)))
        
        for t in range(max_people):
            if prev_boxes[t] is None:
                continue
            for di, db in enumerate(det_boxes):
                iou = compute_iou(prev_boxes[t], db)
                iou_matrix[t, di] = iou
                cost[t, di] = 1.0 - iou
```
**New:** Store IoU values in separate matrix for threshold checking after Hungarian matching.

```python
        for t_id, d_id in zip(t_ids, d_ids):
            iou = iou_matrix[t_id, d_id]
            
            # IoU threshold gate: reject assignments with low IoU
            if iou < iou_threshold and prev_boxes[t_id] is not None:
                continue
```
**New:** After Hungarian matching, check if assignment has sufficient IoU. If not, reject it (detection will be assigned to empty slot as new track).

---

## Testing and Validation

### Test 1: Projection Function Unit Test

```
Input: Mock 3D skeleton + Mock 2D keypoints
Expected: Projected skeleton centered on 2D bbox
Result: Center difference = 14.5 pixels ✓ PASS
```

### Test 2: Detection Pipeline Integration

```
Input: vid1.mp4 (1920×1080 landscape)
Expected: Detections > 0
Result: 2.0 average detections over 10 frames ✓ PASS
```

### Test 3: Aspect Ratio Preservation

```
Portrait 1080×1920 → 810×1440: AR diff 0.00% ✓
Portrait 4096×2160 → 1440×759: AR diff 0.05% ✓
Landscape 1920×1080 → 1440×810: AR diff 0.00% ✓
Small 800×600 → 800×600: AR diff 0.00% ✓
```

### Test 4: IoU Tracking Robustness

```
Input: Synthetic sequence with teleporting person
Expected: Stationary person maintains ID
Result: 100% track continuity ✓ PASS
```

### Visual Test: Full Pipeline on vid1.mp4

```
Processed: 208 frames
Detected: 2-3 people per frame
Output: optimizing/vid1_fixed_60f.mp4
Status: 3D skeletons properly sized and positioned ✓
```

---

## Remaining Limitations

### 1. Bbox-Anchored Projection is a Heuristic

- NOT true geometric camera projection
- Ignores perspective/depth for positioning
- Skeleton shape comes from 3D, position from 2D
- For true 3D visualization, use separate 3D viewport (matplotlib)

### 2. MediaPipe-to-H36M Mapping Concerns

- Head (joint 10) uses nose as proxy; H36M expects top of head
- Spine (joint 7) is simple midpoint; real spine curves
- May affect lift quality (unquantified)

### 3. Resolution Trade-off

- final/ downscales to max_dim=800 (loses detail)
- vidpose-amrita uses max_dim=1440 (more detail, slower)
- Optimal setting depends on use case

### 4. Tracking Not Proven as Bottleneck

- IoU threshold added defensively
- No tracking failures observed in tested clips
- May not be necessary for current use cases

### 5. Fair Pipeline Comparison Still Needed

- After aspect ratio fix, vidpose-amrita may outperform final/
- Requires head-to-head evaluation with both fixes applied
- YOLO11-pose may produce better input for VideoPose3D

---

## Known Issues Requiring Further Work (Identified in planv3.md)

### Issue 1: Root Joint Skipped (Causes Torso-Leg Gap)

**Status:** ✅ FIXED

The hip joint (H36M index 0) is ALWAYS at (0,0,0) in VideoPose3D output because it outputs **root-relative coordinates**. This triggered the "invalid joint" check.

**Fix Applied:** Added special case for index 0 in `project_3d_to_2d_anchored()`:
```python
if joint_idx == 0:
    # Root joint at (0,0,0) is valid - use 2D hip center as anchor
    if kps_2d_coco[11] is not None and kps_2d_coco[12] is not None:
        hip_center = (np.array(kps_2d_coco[11]) + np.array(kps_2d_coco[12])) / 2
        points_2d.append(tuple(hip_center.astype(int)))
```

### Issue 2: Joint Angle Smoothing by Model

**Status:** 🟡 Model limitation (cannot fix without retraining)

VideoPose3D was trained on Human3.6M dataset which has limited pose variety. When lifting unusual poses (e.g., dynamic dancing), the model regresses toward "average" poses.

**Evidence:** Knee angle differs by 16° between 2D input and 3D output.

**Mitigation:** None available without model retraining. Documented as limitation.

### Issue 3: Bone Length Asymmetry

**Status:** ✅ FIXED

**Evidence (before fix):**
- Left thigh: 0.4979, Right thigh: 0.4468 → 10.3% asymmetry
- Left upper arm: 0.3451, Right upper arm: 0.3834 → 10.0% asymmetry

**Fix Applied:** Added `enforce_bone_constraints()` post-processing:
1. Compute median bone length across all frames
2. Enforce left-right symmetry (70% weight toward symmetric length)
3. Adjust child joint positions to match target lengths

```python
# Symmetric bone pairs (left → right)
H36M_SYMMETRIC_BONES = [
    (4, 5, 1, 2),     # thigh
    (5, 6, 2, 3),     # shin
    (11, 12, 14, 15), # upper arm  
    (12, 13, 15, 16), # forearm
]
```

### Issue 4: No 3D Trajectory Smoothing

**Status:** ✅ FIXED

**Fix Applied:** Added `smooth_3d_trajectory()` function applied after 3D lifting:
```python
all_3d_keypoints = smooth_all_3d_tracks(all_3d_keypoints, sigma=smoothing_sigma * 0.75)
```

Uses sigma=1.5 (75% of 2D smoothing sigma) since VideoPose3D already incorporates temporal context through its 243-frame receptive field.

---

## planv4.md Implementations: Visualization Improvements

### Overview

These changes improve the visual appearance of both 2D and 3D skeleton rendering to match the cleaner style seen in `samples/vid4_proto5_output.mp4`.

### Change 1: 2D Face Circle

**Before:** Wavy lines connecting nose→eyes→ears created messy face outline
**After:** Clean circle centered at nose with ear-based radius

**Implementation in `draw_skeleton_2d()`:**
```python
# Calculate radius from nose to ear distance (average of both ears)
ear_dist_l = np.sqrt((nose[0] - l_ear[0])**2 + (nose[1] - l_ear[1])**2)
ear_dist_r = np.sqrt((nose[0] - r_ear[0])**2 + (nose[1] - r_ear[1])**2)
radius = int(max((ear_dist_l + ear_dist_r) / 2, 10))

# Draw circle centered at nose
cv2.circle(canvas, nose, radius, color, 2)

# Draw neck connection (bottom of circle to shoulder center)
neck_bottom = (nose[0], nose[1] + radius)
```

**COCO indices used:**
- 0 = nose (circle center)
- 3 = left_ear, 4 = right_ear (for radius calculation)

### Change 2: 2D Spine Line

**Before:** Torso shown as rectangle (shoulder→hip lines on both sides)
**After:** Single spine line from mid-shoulder to mid-hip

**Implementation:**
```python
# Calculate midpoints
mid_shoulder = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

# Draw single spine line
cv2.line(canvas, mid_shoulder, mid_hip, color, 2)

# Keep shoulder bar and hip bar
cv2.line(canvas, l_sh, r_sh, color, 2)  # Shoulder bar
cv2.line(canvas, l_hip, r_hip, color, 2)  # Hip bar
```

**Note:** Joint dots are now only drawn for body joints (indices 5-16), skipping face landmarks (0-4).

### Change 3: 3D Face Circle

**Before:** Lines connecting neck→nose→head_top created awkward face appearance
**After:** Clean circle at head position with neck connection

**Implementation in `draw_skeleton_3d()`:**
```python
# H36M indices for face
H36M_NOSE = 9
H36M_HEAD_TOP = 10
H36M_NECK = 8

# Skip face connections in normal loop
face_connections = {(8, 9), (9, 10)}

# Draw face circle instead
nose_2d = points_2d[H36M_NOSE]
head_2d = points_2d[H36M_HEAD_TOP]

if nose_2d and head_2d:
    # Radius = distance from nose to head_top
    radius = int(np.sqrt((nose_2d[0] - head_2d[0])**2 + 
                          (nose_2d[1] - head_2d[1])**2))
    
    # Center = midpoint between nose and head_top
    center_x = (nose_2d[0] + head_2d[0]) // 2
    center_y = (nose_2d[1] + head_2d[1]) // 2
    
    cv2.circle(canvas, center, radius, circle_color, 2)
```

### Change 4: Enhanced Hip Smoothing

**Before:** All joints smoothed with same sigma (1.5)
**After:** Hip joints (0, 1, 4) get 1.5x extra smoothing to reduce jerky angles

**Implementation in `smooth_3d_trajectory()`:**
```python
# Hip-related joints that benefit from extra smoothing
hip_joints = {0, 1, 4}  # hip (root), r_hip, l_hip

for joint in range(17):
    # Use higher sigma for hip joints
    joint_sigma = sigma * hip_sigma_multiplier if joint in hip_joints else sigma
    
    # Apply smoothing...
```

**H36M hip indices:**
- 0 = hip (root/pelvis center)
- 1 = right_hip
- 4 = left_hip

The `hip_sigma_multiplier` defaults to 1.5, so hip joints get sigma=2.25 while other joints get sigma=1.5.

---

## Files Modified

| File | Changes |
|------|---------|
| `final/pipeline_3d.py` | Root joint fix, 3D trajectory smoothing, bone length constraints, `project_3d_to_2d_anchored()`, `draw_skeleton_3d()`, `draw_skeleton_2d()`, `render_frame()`, IoU threshold tracking, face circles, spine line, hip smoothing |
| `final/mediapipe_to_h36m.py` | COCO format conversion, H36M joint definitions |
| `vidpose-amrita/kinemation/extract_keypoints.py` | `resize_preserve_aspect()`, updated `extract_yolo_keypoints()` |
| `vidpose-amrita/kinemation/run_vid3.py` | `resize_preserve_aspect()`, `project_3d_to_2d_anchored()`, `draw_skeleton_3d()`, IoU threshold tracking |

## Backups Created

All original files backed up to:
- `optimizing/backups/final/` - Pre-implementation backup
- `optimizing/backups/final_v2/` - Post-COCO fix, pre-planv3 backup
- `optimizing/backups/final_v3_*` - Pre-planv4 backup (visualization changes)
- `optimizing/backups/final_v4_*` - Pre-planv5 backup (smoothing improvements)
- `optimizing/backups/vidpose-amrita/`
- `optimizing/backups/prototypes/`

---

## planv5.md Implementation: Joint-Adaptive Smoothing

### Problem Analysis

After planv4 visualization improvements, the user reported that 3D joint angles appeared "clunky" and inconsistent during movement. Quantitative analysis revealed:

| Joint | Jitter (°/frame) | Target |
|-------|------------------|--------|
| l_elbow | 10.23 | <5 |
| r_elbow | 9.85 | <5 |
| l_hip | 5.51 | <3 |
| r_hip | 7.40 | <3 |

### Root Causes Identified

1. **Uniform Smoothing**: Same σ=1.5 Gaussian applied to all joints
2. **No Velocity Limiting**: Outlier spikes propagated through
3. **Joint Role Ignored**: Arms need less smoothing than spine

### Solution: Two-Pass Joint-Adaptive Smoothing

**Pass 1: Joint-Specific Gaussian Smoothing**

Different joints require different smoothing levels based on their role:

```python
JOINT_SIGMA_3D = {
    # Core stability joints - moderate-high smoothing
    0: 2.2,   # hip (root)
    7: 2.2,   # spine
    8: 2.0,   # neck
    
    # Locomotion joints - moderate smoothing
    1: 1.9, 4: 1.9,   # hip angles
    2: 1.7, 5: 1.7,   # knees
    3: 1.5, 6: 1.5,   # ankles
    
    # Head - moderate smoothing
    9: 1.7, 10: 1.7,
    
    # Arm joints - low smoothing (preserve expressiveness)
    11: 1.4, 14: 1.4,  # shoulders
    12: 1.2, 15: 1.2,  # elbows
    13: 1.0, 16: 1.0,  # wrists
}
```

**Pass 2: Velocity Limiting**

Remove physically impossible movement spikes:

```python
VELOCITY_LIMITS_3D = {
    # Core - limited
    0: 0.025, 7: 0.025, 8: 0.035,
    # Hips - moderate
    1: 0.050, 4: 0.050,
    # Legs - cyclic motion
    2: 0.060, 5: 0.060, 3: 0.080, 6: 0.080,
    # Arms - fast expressive
    11: 0.080, 14: 0.080, 12: 0.120, 15: 0.120, 13: 0.150, 16: 0.150,
    # Head - moderate
    9: 0.050, 10: 0.050,
}
```

### Implementation

Updated function signature:
```python
def smooth_3d_trajectory(poses_3d, sigma=1.5, hip_sigma_multiplier=1.5, 
                         use_adaptive_sigma=True, apply_velocity_limit=True):
```

### Results

| Metric | Old (planv4) | New (planv5) | Improvement |
|--------|--------------|--------------|-------------|
| Overall smoothness | 0.0114 | 0.0065 | **43.4%** |
| L_elbow jitter | 10.2°/frame | 3.9°/frame | **62%** |
| L_hip jitter | 5.5°/frame | 1.7°/frame | **69%** |

### Tradeoff

Motion range is reduced by the smoothing:
- L_elbow: 153° → 56° (63% reduction)
- L_hip: 72° → 19° (74% reduction)

This is an inherent tradeoff - reducing jitter reduces perceived motion range. The current parameters balance smoothness vs expressiveness. For highly dynamic content, reduce sigma values.

### Files Changed

- `final/pipeline_3d.py`:
  - Added `JOINT_SIGMA_3D` constant (line ~337-362)
  - Added `VELOCITY_LIMITS_3D` constant (line ~365-378)
  - Rewrote `smooth_3d_trajectory()` with two-pass algorithm (line ~380-458)
  - Updated `smooth_all_3d_tracks()` to pass new parameters (line ~461-486)
  - Updated pipeline call to enable adaptive smoothing (line ~1268-1275)

### Testing

Validated on:
- `vid2_30f.mp4` - 30 frames, single person
- `vid2_90f.mp4` - 90 frames, single person (longer motion)
- `vid1_60f.mp4` - 60 frames, 2 people (multi-person)

All tests passed with visually smoother output while preserving overall motion characteristics.

---

## VideoPose3D vs MotionBERT: A Comprehensive Comparison

This section provides a detailed comparison between VideoPose3D (currently used in Kinemation) and MotionBERT (a potential future alternative) to help understand the tradeoffs and inform future decisions.

### What is VideoPose3D?

**VideoPose3D** is a 2D-to-3D pose lifting model developed by Facebook Research (FAIR) in 2019. It uses a **temporal convolutional network (TCN)** architecture to convert 2D pose sequences into 3D coordinates.

**Key Characteristics:**

1. **Architecture: Dilated Temporal Convolutions**
   - Uses 1D convolutions along the time axis
   - Dilated convolutions exponentially increase receptive field
   - Filter widths [3,3,3,3,3] create 243-frame receptive field (3^5)
   - Efficient: O(n) computation, no attention mechanism

2. **Input/Output:**
   - Input: Normalized 2D keypoints (17 joints, COCO format)
   - Output: Root-relative 3D coordinates (17 joints, H36M format)
   - Sliding window: Processes 243 frames to predict 1 center frame

3. **Training:**
   - Trained on Human3.6M dataset (indoor, controlled environment)
   - Uses MoCap ground truth for supervision
   - COCO-format 2D detections from Detectron2

4. **Inference Speed:**
   - Very fast (~1000+ FPS on GPU for single person)
   - Low memory footprint
   - Can run on CPU in reasonable time

### What is MotionBERT?

**MotionBERT** is a more recent (2022) transformer-based model that treats 3D pose estimation as a sequence modeling problem, similar to language models like BERT.

**Key Characteristics:**

1. **Architecture: Dual-Stream Transformer**
   - Spatial transformer: Models joint relationships within a frame
   - Temporal transformer: Models motion patterns across frames
   - Self-attention captures long-range dependencies
   - Pre-trained on large motion datasets, then fine-tuned

2. **Input/Output:**
   - Input: 2D keypoints (various formats supported)
   - Output: 3D coordinates with optional motion features
   - Variable sequence lengths (typically 243-frame windows)

3. **Training:**
   - Pre-trained on multiple datasets (AMASS, Human3.6M, 3DPW)
   - Uses masked motion modeling (like BERT's masked language modeling)
   - Self-supervised pretraining + supervised fine-tuning

4. **Inference Speed:**
   - Slower than VideoPose3D (~50-100 FPS on GPU)
   - Higher memory requirements
   - May need GPU for real-time performance

### Head-to-Head Comparison

| Aspect | VideoPose3D | MotionBERT |
|--------|-------------|------------|
| **Architecture** | Temporal CNN (TCN) | Dual-Stream Transformer |
| **Year Released** | 2019 | 2022 |
| **Accuracy (MPJPE)** | 46.8mm on H36M | 37.7mm on H36M |
| **Accuracy (P-MPJPE)** | 36.5mm on H36M | 30.2mm on H36M |
| **Receptive Field** | Fixed 243 frames | Variable/configurable |
| **Speed (GPU)** | ~1000+ FPS | ~50-100 FPS |
| **Memory** | ~500MB | ~2-4GB |
| **CPU Feasibility** | ✅ Yes | ⚠️ Slow |
| **Pretrained Models** | COCO→H36M only | Multiple configurations |
| **Real-time Ready** | ✅ Yes | ⚠️ Depends on hardware |

*MPJPE = Mean Per Joint Position Error (lower is better)*
*P-MPJPE = Procrustes-aligned MPJPE (measures pose structure accuracy)*

### Pros of VideoPose3D

1. **Speed:** 10-20x faster than MotionBERT, enabling real-time processing
2. **Simplicity:** Straightforward architecture, easy to understand and modify
3. **Low Resources:** Runs on CPU, works on older hardware
4. **Proven:** Widely used, well-documented, stable codebase
5. **Predictable:** Consistent behavior across different videos

### Cons of VideoPose3D

1. **Limited Accuracy:** ~20% higher error than MotionBERT
2. **Fixed Receptive Field:** Cannot adapt to different motion speeds
3. **Limited Generalization:** Trained only on H36M indoor dataset
4. **No Spatial Reasoning:** Doesn't model joint relationships explicitly
5. **Edge Artifacts:** First/last frames use repeated padding

### Pros of MotionBERT

1. **Higher Accuracy:** State-of-the-art results on benchmarks
2. **Better Generalization:** Pre-trained on diverse motion data
3. **Spatial Awareness:** Self-attention models joint dependencies
4. **Motion Understanding:** Captures subtle motion patterns
5. **Flexible Architecture:** Can adapt to different tasks

### Cons of MotionBERT

1. **Speed:** 10-20x slower, may not be real-time on all hardware
2. **Memory Hungry:** Needs GPU with 4+ GB VRAM
3. **Complexity:** Harder to understand, debug, and modify
4. **Setup:** More dependencies, larger model files
5. **Overkill:** For simple applications, may be unnecessary

### Feasibility Analysis for Kinemation

#### Current Setup (VideoPose3D)

| Factor | Assessment |
|--------|------------|
| Accuracy | Acceptable for visualization |
| Speed | Excellent (~30+ FPS processing) |
| Integration | ✅ Fully integrated |
| Maintenance | Low effort |

#### Potential Migration (MotionBERT)

| Factor | Assessment |
|--------|------------|
| Accuracy Gain | ~20% improvement expected |
| Speed Impact | Would drop to ~5-10 FPS |
| Integration Effort | 2-3 days work |
| Hardware Needs | Requires GPU |

#### Recommendation

**For the current project goals, VideoPose3D remains the better choice** because:

1. Speed is important for iterating on multiple videos
2. The accuracy difference is not critical for visualization purposes
3. The current pipeline is working well after planv1-v5 fixes
4. CPU-only deployment is possible

**Consider MotionBERT if:**
- Accuracy becomes critical (e.g., motion capture for animation)
- GPU is always available
- Processing speed is not a concern
- More complex pose understanding is needed

---

## final/ vs vidpose-amrita: Pipeline Comparison

This section explains the differences between the two 3D pose estimation pipelines developed for this project.

### Overview

| Aspect | final/ | vidpose-amrita/ |
|--------|--------|-----------------|
| **Purpose** | Production-ready pipeline | Alternative experimental pipeline |
| **2D Detection** | YOLOv8n + MediaPipe | YOLO11-pose |
| **Keypoint Format** | 33 (MediaPipe) → 17 (H36M) | 17 (COCO) → 17 (H36M) |
| **3D Lifting** | VideoPose3D | VideoPose3D |
| **Status** | Actively improved (planv1-v5) | Baseline with fixes |

### Architectural Differences

#### final/ Pipeline Flow

```
Input Video
    │
    ▼
┌───────────────────┐
│ Preprocessing     │ ← CLAHE in LAB space, Gaussian blur
│ (max_dim=800)     │   Aspect-preserving resize
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ YOLOv8n Detection │ ← Person bounding boxes
│ (confidence≥0.5)  │   Up to 6 people tracked
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ MediaPipe Pose    │ ← 33 landmarks per person
│ (per-person crop) │   Includes face/hand detail
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ MediaPipe→H36M    │ ← Custom mapping (33→17)
│ Conversion        │   Computed joints: hip_center, spine, thorax
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Batch Tracking    │ ← IoU-based Hungarian matching
│ (IoU threshold)   │   Consistent person IDs
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 2D Smoothing      │ ← Gaussian filter (σ=2)
│                   │   Detection-aware (no bleeding)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ VideoPose3D       │ ← 243-frame receptive field
│ (H36M→H36M)       │   Sliding window inference
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 3D Smoothing      │ ← Joint-adaptive Gaussian (planv5)
│ + Bone Constraints│   Velocity limiting + bone length enforcement
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Visualization     │ ← Face circle, spine line
│                   │   Bbox-anchored 3D projection
└───────────────────┘
```

#### vidpose-amrita/ Pipeline Flow

```
Input Video
    │
    ▼
┌───────────────────┐
│ Aspect-Preserving │ ← max_dim=1440
│ Resize            │   (Fixed after planv2)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ YOLO11-pose       │ ← Detection + pose in one step
│                   │   17 COCO keypoints directly
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ COCO→H36M         │ ← Standard mapping (17→17)
│ Conversion        │   Same computed joints
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Tracking          │ ← IoU-based matching
│                   │   Similar to final/
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ 2D Smoothing      │ ← Simple Gaussian (σ=2)
│                   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ VideoPose3D       │ ← Same model as final/
│                   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Visualization     │ ← Basic skeleton
│                   │   Bbox-anchored projection
└───────────────────┘
```

### Key Differences Explained

#### 1. 2D Detection Approach

**final/ (YOLOv8n + MediaPipe):**
- Two-stage: detection then pose estimation
- MediaPipe provides 33 landmarks including face mesh points
- Better detail for face, hands (if needed in future)
- Requires coordinate mapping from crop to full frame

**vidpose-amrita/ (YOLO11-pose):**
- Single-stage: detection and pose together
- Directly outputs 17 COCO keypoints
- Faster (one model instead of two)
- Less detailed but sufficient for body pose

**Which is Better?** `final/` for quality, `vidpose-amrita/` for speed.

#### 2. Preprocessing

**final/:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
- Improves visibility in low-light or uneven lighting
- Gaussian blur to reduce noise
- Resizes to max_dim=800 (smaller = faster)

**vidpose-amrita/:**
- Simple resize to max_dim=1440
- No contrast enhancement
- Larger processing size (more detail but slower)

**Which is Better?** `final/` handles varied lighting better; `vidpose-amrita/` preserves more original detail.

#### 3. The Initial Problems (Now Fixed)

##### Problem 1: Aspect Ratio Distortion in vidpose-amrita

**What Was Wrong:**
The original `extract_keypoints.py` used a fixed resize:
```python
# ORIGINAL (BROKEN)
frame_r = cv2.resize(frame, (1440, 810))  # Forces 16:9 aspect ratio
```

This caused severe distortion on portrait videos:
- A 1080x1920 video (9:16) was squashed to 1440x810 (16:9)
- 3.37x horizontal distortion
- 40% of detections lost

**The Fix (planv2):**
```python
# FIXED - Aspect preserving resize
def resize_preserve_aspect(frame, max_dim=1440):
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame, 1.0
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
```

Now the video maintains its original proportions.

##### Problem 2: Wrong 3D Projection (Both Pipelines)

**What Was Wrong:**
The original 3D→2D projection used broken denormalization:
```python
# ORIGINAL (BROKEN)
def denormalize_3d_to_screen(X, w, h):
    # This assumed 3D output was in [-1, 1] range
    # But VideoPose3D outputs are root-relative 3D coordinates!
    return (X + [1, h/w, 0]) * w / 2
```

This caused:
- 4.54x skeleton inflation
- Skeletons appearing in wrong positions
- Complete visual mismatch with 2D

**The Fix:**
```python
# FIXED - Bbox-anchored projection
def project_3d_to_2d_anchored(keypoints_3d, keypoints_2d):
    # Use 2D detection bbox as anchor
    # Scale 3D skeleton to match 2D bbox size
    # Position 3D skeleton at 2D bbox center
    
    # Get 2D bounding box
    valid_2d = keypoints_2d[keypoints_2d.max(axis=1) > 0]
    bbox_2d_center = (valid_2d.min(axis=0) + valid_2d.max(axis=0)) / 2
    bbox_2d_size = valid_2d.max(axis=0) - valid_2d.min(axis=0)
    
    # Get 3D bounding box (XY plane)
    valid_3d = keypoints_3d[np.abs(keypoints_3d).max(axis=1) > 0.001]
    bbox_3d_center = (valid_3d[:, :2].min(axis=0) + valid_3d[:, :2].max(axis=0)) / 2
    bbox_3d_size = valid_3d[:, :2].max(axis=0) - valid_3d[:, :2].min(axis=0)
    
    # Scale and translate
    scale = min(bbox_2d_size / bbox_3d_size)
    projected = (keypoints_3d[:, :2] - bbox_3d_center) * scale + bbox_2d_center
    
    return projected
```

**Important Note:** This is a **display heuristic**, not true geometric reprojection. It provides visually correct placement but doesn't represent actual camera projection.

##### Problem 3: Axis Orientation Errors in final/

**What Was Wrong:**
Early versions had incorrect axis flips:
```python
# ORIGINAL (BROKEN)
pred_3d = model(input_tensor)
pred_3d[:, :, 0] *= -1  # Flip X - WRONG!
pred_3d[:, :, 1] *= -1  # Flip Y - WRONG!
```

This caused:
- 90° rotated skeleton orientation
- Left/right confusion
- Upside-down poses

**The Fix:**
Simply remove the axis flips. VideoPose3D's output coordinate system matches screen coordinates:
- +X = right
- +Y = down
- +Z = towards camera

#### 4. Quality Improvements in final/ (Not in vidpose-amrita)

| Feature | final/ | vidpose-amrita/ |
|---------|--------|-----------------|
| Joint-adaptive smoothing | ✅ planv5 | ❌ |
| Velocity limiting | ✅ | ❌ |
| Bone length constraints | ✅ | ❌ |
| Face circle visualization | ✅ planv4 | ❌ |
| Spine line rendering | ✅ planv4 | ❌ |
| Enhanced hip smoothing | ✅ | ❌ |

### Which Pipeline Should You Use?

**Use final/ when:**
- Quality is the priority
- You need the latest improvements (planv1-v5)
- Multi-person tracking is important
- Visualization aesthetics matter

**Use vidpose-amrita/ when:**
- Speed is critical
- You want simpler code to understand
- You're doing quick experiments
- You prefer single-model pose estimation

### Summary of Changes Made

#### Changes to final/

1. **planv1:** Fixed COCO input format, removed axis flips
2. **planv2:** Confirmed aspect ratio was already correct
3. **planv3:** Added bone length constraints, fixed root joint gap
4. **planv4:** Added face circle, spine line, enhanced hip smoothing
5. **planv5:** Added joint-adaptive smoothing, velocity limiting

#### Changes to vidpose-amrita/

1. **planv2:** Fixed aspect-ratio distortion in `extract_keypoints.py`
2. **run_vid3.py:** Added bbox-anchored projection (same as final/)
3. **test_samples.py:** Created for testing (by Copilot)

The core VideoPose3D model and inference remain identical in both pipelines.

---

## planv6.md Implementation: True 3D Visualization

### The Problem: Flat Orthographic Projection

Before planv6, the 3D visualization used a **bbox-anchored 2D projection** which, while correctly placing the skeleton on the video, produced a **flat** appearance without true 3D perspective:

**Previous Rendering (Bbox-Anchored):**
```python
# Project 3D coordinates to 2D using bbox anchoring
def project_3d_to_2d_anchored(keypoints_3d, keypoints_2d):
    # Scale 3D to fit 2D bbox
    # Lost depth perception, hips looked weird
    projected_2d = (keypoints_3d[:, :2] - center) * scale + offset
    return projected_2d
```

**Issues:**
1. **No perspective** - Far limbs rendered same size as near limbs
2. **Hip angles looked unnatural** - The flat projection couldn't show the 3D hip rotation
3. **No depth cues** - Users couldn't tell which limb was in front

### The Solution: Matplotlib 3D Rendering

Planv6 implemented **true 3D visualization** using matplotlib's 3D projection capabilities, matching the official VideoPose3D visualization style.

**New Architecture:**
```
VideoPose3D Output (x, y, z)
        │
        ▼
┌────────────────────────┐
│ Coordinate Conversion  │  Y ↔ Z swap for "standing" orientation
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ matplotlib Axes3D      │  True 3D perspective projection
│ - elev=15°, azim=70°  │  Same camera angles as official VP3D
│ - Floor grid at z=-0.9│  Spatial reference plane
│ - Depth coloring       │  Purple=front, Green=back
└────────────────────────┘
        │
        ▼
┌────────────────────────┐
│ Canvas to OpenCV       │  buffer_rgba() → BGR conversion
└────────────────────────┘
```

### Coordinate System Conversion

**Critical Insight:** VideoPose3D and matplotlib use different coordinate systems:

| Axis | VideoPose3D | matplotlib 3D |
|------|-------------|---------------|
| X | Horizontal (right) | Horizontal (right) |
| Y | Vertical (down) | Depth (forward) |
| Z | Depth (forward) | Vertical (up) |

**Conversion Code:**
```python
# VideoPose3D: X=horizontal, Y=vertical(down), Z=depth
# matplotlib: X=horizontal, Y=depth, Z=vertical(up)
x_plot = keypoints_3d[:, 0]       # X stays X
y_plot = keypoints_3d[:, 2]       # VP3D Z → matplotlib Y (depth)
z_plot = -keypoints_3d[:, 1]      # -VP3D Y → matplotlib Z (flip for up=positive)
```

### Depth-Based Coloring

To enhance 3D perception, limbs are colored based on their depth (Y position in matplotlib space):

```python
def draw_3d_skeleton_matplotlib(ax, x, y, z, base_color):
    # Normalize depth for coloring
    y_min, y_max = y.min(), y.max()
    y_range = max(y_max - y_min, 0.1)
    
    for start, end in H36M_CONNECTIONS:
        # Average depth of this bone
        avg_depth = (y[start] + y[end]) / 2
        depth_normalized = (avg_depth - y_min) / y_range
        
        # Interpolate: green (back) → purple (front)
        color = (
            0.3 + 0.5 * depth_normalized,   # R: 0.3→0.8
            0.8 - 0.5 * depth_normalized,   # G: 0.8→0.3
            0.3 + 0.5 * depth_normalized    # B: 0.3→0.8
        )
        ax.plot3D([x[start], x[end]], 
                  [y[start], y[end]], 
                  [z[start], z[end]], 
                  color=color, linewidth=2)
```

### 3D Face Circle

The face circle now renders in true 3D space:

```python
def draw_3d_face_circle(ax, x, y, z, base_color, n_points=30):
    # H36M indices: nose=9, head_top=10
    nose_x, nose_y, nose_z = x[9], y[9], z[9]
    head_x, head_y, head_z = x[10], y[10], z[10]
    
    # Radius from nose to head_top
    radius = np.sqrt((head_x - nose_x)**2 + 
                     (head_y - nose_y)**2 + 
                     (head_z - nose_z)**2) * 0.6
    
    # Draw circle in XZ plane (facing camera)
    theta = np.linspace(0, 2*np.pi, n_points)
    circle_x = nose_x + radius * np.cos(theta)
    circle_z = nose_z + radius * np.sin(theta)
    circle_y = np.full_like(theta, nose_y)
    
    ax.plot3D(circle_x, circle_y, circle_z, 
              color=base_color, linewidth=2)
```

### Floor Grid

A reference grid helps users understand the 3D space:

```python
def draw_3d_grid(ax, size=1.5, num_lines=8):
    # Floor plane at z = -0.9 (below the person)
    z_floor = -0.9
    
    for i in range(num_lines + 1):
        pos = -size + (2 * size * i / num_lines)
        # Lines parallel to X axis
        ax.plot3D([-size, size], [pos, pos], [z_floor, z_floor], 
                  color='gray', alpha=0.5, linewidth=0.5)
        # Lines parallel to Y axis
        ax.plot3D([pos, pos], [-size, size], [z_floor, z_floor], 
                  color='gray', alpha=0.5, linewidth=0.5)
```

### New Render Modes

Planv6 added three new visualization modes:

| Mode | Description |
|------|-------------|
| `'matplotlib_3d'` | Pure true 3D perspective rendering (square output) |
| `'3d_only'` | Alias for `matplotlib_3d` |
| `'side_by_side_true3d'` | 2D on left, True 3D on right |

**Usage:**
```python
pipeline.process_video(
    input_path, 
    output_path,
    render_mode='matplotlib_3d'  # Use true 3D
)
```

### Before vs After

| Aspect | Before (planv5) | After (planv6) |
|--------|-----------------|----------------|
| Projection type | Flat orthographic | True 3D perspective |
| Hip angles | Unnatural, weird-looking | Anatomically correct |
| Depth perception | None | Grid + depth coloring |
| Reference plane | None | Floor grid |
| Rendering backend | OpenCV 2D drawing | matplotlib 3D + Agg |

### Performance Considerations

Matplotlib rendering is slower than direct OpenCV drawing:
- ~50-100ms per frame (vs ~5ms for OpenCV)
- Creates new figure per frame (could be optimized with figure reuse)
- Best for offline processing, not real-time

For real-time applications, the legacy `'side_by_side'` mode with bbox-anchored projection is still available.

---

*Document created: 2026-04-01*
*Last updated: 2026-04-03 (Added planv6 True 3D Visualization section)*
*Pipeline version: Post-optimization v6*
