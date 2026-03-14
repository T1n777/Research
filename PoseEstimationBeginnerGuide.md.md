# 🎯 The Complete Beginner's Guide to OpenCV, MediaPipe & OpenPose

> **Who is this for?** Absolute beginners who want to understand computer vision and human pose estimation from scratch. No prior experience needed — just basic Python knowledge.

---

## 📑 Table of Contents

1. [What is Computer Vision?](#1-what-is-computer-vision)
2. [OpenCV — The Swiss Army Knife of Vision](#2-opencv--the-swiss-army-knife-of-vision)
   - [Installation](#21-installation)
   - [Core Concepts](#22-core-concepts)
   - [All Key Functions & Use Cases](#23-all-key-functions--use-cases)
   - [Practice Examples](#24-practice-examples-opencv)
3. [MediaPipe — Google's AI Perception Toolkit](#3-mediapipe--googles-ai-perception-toolkit)
   - [Installation](#31-installation)
   - [Core Concepts](#32-core-concepts-1)
   - [All Key Functions & Use Cases](#33-all-key-functions--use-cases-1)
   - [Practice Examples](#34-practice-examples-mediapipe)
4. [OpenPose — Carnegie Mellon's Pose Powerhouse](#4-openpose--carnegie-mellons-pose-powerhouse)
   - [Installation](#41-installation)
   - [Core Concepts](#42-core-concepts-2)
   - [All Key Functions & Use Cases](#43-all-key-functions--use-cases-2)
   - [Practice Examples](#44-practice-examples-openpose)
5. [Comparison: OpenCV vs MediaPipe vs OpenPose](#5-comparison-opencv-vs-mediapipe-vs-openpose)
6. [Integration Project — Human Movement to 2D Stick Figure](#6-integration-project--human-movement-to-2d-stick-figure)
7. [Next Steps & Resources](#7-next-steps--resources)

---

## 1. What is Computer Vision?

**Computer Vision** is a field of Artificial Intelligence that teaches computers to "see" and understand images and videos — just like humans do with their eyes and brain.

Think of it this way:
- Your **eyes** capture light → Your **brain** interprets it (that's a dog, that person is waving).
- A **camera** captures pixels → A **computer vision algorithm** interprets them (that's a face, that hand is open).

### Why does it matter?

| Application | Example |
|---|---|
| Self-driving cars | Detecting pedestrians, lanes, traffic signs |
| Healthcare | Analyzing X-rays, tracking body movements for rehab |
| Security | Face recognition, suspicious activity detection |
| Gaming & AR | Motion capture, virtual try-on, gesture controls |
| Fitness | Counting reps, correcting exercise form |

In this guide, we focus on three powerful libraries that make computer vision accessible:

| Library | Created By | Best For |
|---|---|---|
| **OpenCV** | Intel (1999) | General image/video processing |
| **MediaPipe** | Google (2019) | Real-time AI perception (face, hands, pose) |
| **OpenPose** | CMU (2017) | Multi-person pose estimation |

---

## 2. OpenCV — The Swiss Army Knife of Vision

### What is OpenCV?

**OpenCV** (Open Source Computer Vision Library) is the most popular computer vision library in the world. It has **2,500+ algorithms** for image and video processing. Think of it as a giant toolbox — it can read images, apply filters, detect objects, track movements, and much more.

**Key facts:**
- Written in C++ but has excellent Python bindings
- Works on Windows, Mac, Linux, Android, iOS
- Used by NASA, Google, Microsoft, Toyota, and thousands of startups
- Completely free and open source

### 2.1 Installation

```bash
# Install OpenCV (this gives you the main + contrib modules)
pip install opencv-python
pip install opencv-contrib-python  # Extra features like advanced trackers

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

### 2.2 Core Concepts

Before diving into functions, let's understand how computers see images:

#### What is an Image to a Computer?

An image is just a **grid of numbers** (called pixels). Each pixel has a value that represents its color.

```
Grayscale image (1 channel):      Color image (3 channels - BGR):
┌───┬───┬───┐                     Each pixel has 3 values:
│ 0 │128│255│                     Blue  = [0-255]
├───┼───┼───┤                     Green = [0-255]
│ 64│200│100│                     Red   = [0-255]
├───┼───┼───┤
│180│ 50│220│                     Example: [255, 0, 0] = Pure Blue
└───┴───┴───┘                              [0, 255, 0] = Pure Green
                                           [0, 0, 255] = Pure Red
0   = Black
255 = White
```

> **Important:** OpenCV uses **BGR** (Blue-Green-Red) format, NOT RGB. This is different from most other libraries!

#### What is a Video?

A video is simply a **sequence of images** (called frames) shown rapidly one after another. Typically 24–60 frames per second (FPS).

### 2.3 All Key Functions & Use Cases

Below is a comprehensive reference of OpenCV's most important functions, grouped by category.

---

#### 📂 Category 1: Reading, Writing & Displaying

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.imread(path, flag)` | Loads an image from disk into memory | Reading a photo for processing |
| `cv2.imwrite(path, img)` | Saves an image to disk | Saving a processed/filtered photo |
| `cv2.imshow(name, img)` | Displays an image in a window | Previewing results |
| `cv2.waitKey(ms)` | Waits for a key press (`0` = forever) | Keeping the display window open |
| `cv2.destroyAllWindows()` | Closes all display windows | Cleanup after viewing |
| `cv2.VideoCapture(source)` | Opens a video file or webcam (`0` = default cam) | Reading webcam feed |
| `cap.read()` | Reads the next frame from video | Processing video frame by frame |
| `cv2.VideoWriter(path, codec, fps, size)` | Creates a video file writer | Saving processed video |
| `cap.release()` | Releases the video source | Freeing the camera |

```python
# EXAMPLE: Read, display, and save an image
import cv2

img = cv2.imread("photo.jpg")          # Read image
print(f"Shape: {img.shape}")            # (height, width, channels)
print(f"Size: {img.size}")              # Total number of pixels × channels
print(f"Data type: {img.dtype}")        # Usually uint8 (0-255)

cv2.imshow("My Photo", img)             # Display it
cv2.waitKey(0)                          # Wait for ANY key
cv2.destroyAllWindows()                 # Close windows

cv2.imwrite("copy.jpg", img)            # Save a copy
```

```python
# EXAMPLE: Read from webcam
import cv2

cap = cv2.VideoCapture(0)              # 0 = default webcam

while True:
    ret, frame = cap.read()            # ret = True/False, frame = image
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
```

---

#### 🎨 Category 2: Color Space Conversions

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.cvtColor(img, code)` | Converts image between color spaces | Converting BGR to Grayscale or HSV |

**Common conversion codes:**

| Code | Conversion | When to Use |
|---|---|---|
| `cv2.COLOR_BGR2GRAY` | Color → Grayscale | Edge detection, thresholding |
| `cv2.COLOR_BGR2RGB` | BGR → RGB | Displaying with matplotlib |
| `cv2.COLOR_BGR2HSV` | BGR → HSV | Color-based object detection |
| `cv2.COLOR_BGR2LAB` | BGR → LAB | Color correction |
| `cv2.COLOR_GRAY2BGR` | Grayscale → BGR | Adding color info back |

```python
import cv2

img = cv2.imread("photo.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Grayscale
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        # HSV color space
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # RGB for matplotlib
```

> **What is HSV?** HSV stands for Hue (color type), Saturation (color intensity), Value (brightness). It's much easier to detect specific colors in HSV than in BGR.

---

#### ✏️ Category 3: Drawing Functions

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.line(img, pt1, pt2, color, thickness)` | Draws a straight line | Drawing skeleton bones |
| `cv2.circle(img, center, radius, color, thickness)` | Draws a circle | Marking joint positions |
| `cv2.rectangle(img, pt1, pt2, color, thickness)` | Draws a rectangle | Bounding boxes around objects |
| `cv2.ellipse(img, center, axes, angle, start, end, color, thickness)` | Draws an ellipse | Drawing head shapes |
| `cv2.polylines(img, pts, isClosed, color, thickness)` | Draws polygon outlines | Drawing complex shapes |
| `cv2.fillPoly(img, pts, color)` | Draws filled polygons | Creating masks |
| `cv2.putText(img, text, pos, font, scale, color, thickness)` | Draws text on image | Adding labels and info |
| `cv2.arrowedLine(img, pt1, pt2, color, thickness)` | Draws an arrow | Showing direction of movement |

```python
import cv2
import numpy as np

# Create a blank black canvas (500×500 pixels, 3 color channels)
canvas = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw shapes
cv2.line(canvas, (50, 50), (450, 50), (0, 255, 0), 3)         # Green line
cv2.rectangle(canvas, (100, 100), (400, 300), (255, 0, 0), 2)  # Blue rectangle
cv2.circle(canvas, (250, 250), 80, (0, 0, 255), -1)            # Red filled circle (-1 = filled)
cv2.putText(canvas, "Hello OpenCV!", (100, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   # White text

cv2.imshow("Drawing", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

#### 🔧 Category 4: Image Transformations

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.resize(img, size, interpolation)` | Resizes an image | Scaling images up/down |
| `cv2.flip(img, flipCode)` | Flips image (0=vertical, 1=horizontal, -1=both) | Mirror effect for selfie view |
| `cv2.rotate(img, rotateCode)` | Rotates image by 90/180/270° | Fixing image orientation |
| `cv2.warpAffine(img, M, size)` | Applies affine transformation | Rotation by any angle, translation |
| `cv2.warpPerspective(img, M, size)` | Applies perspective transformation | Bird's-eye view, document scanning |
| `cv2.getRotationMatrix2D(center, angle, scale)` | Creates rotation matrix | Custom angle rotation |
| `cv2.getAffineTransform(src, dst)` | Creates affine transform matrix | Warping three points |
| `cv2.getPerspectiveTransform(src, dst)` | Creates perspective transform matrix | Four-point warping |

```python
import cv2

img = cv2.imread("photo.jpg")

# Resize to 300×300
resized = cv2.resize(img, (300, 300))

# Resize by scale factor (50% of original)
half = cv2.resize(img, None, fx=0.5, fy=0.5)

# Flip horizontally (mirror effect for webcam)
flipped = cv2.flip(img, 1)

# Rotate by custom angle
h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degrees, scale=1
rotated = cv2.warpAffine(img, M, (w, h))
```

---

#### 🌫️ Category 5: Image Filtering & Blurring

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.blur(img, ksize)` | Simple averaging blur | Basic noise reduction |
| `cv2.GaussianBlur(img, ksize, sigmaX)` | Gaussian (weighted) blur | Smooth, natural blurring |
| `cv2.medianBlur(img, ksize)` | Median filter blur | Removing salt-and-pepper noise |
| `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)` | Edge-preserving blur | Smoothing while keeping edges sharp |
| `cv2.filter2D(img, ddepth, kernel)` | Applies a custom kernel/filter | Custom sharpening, embossing |

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg")

# Different blur types
blur1 = cv2.blur(img, (5, 5))                              # Average blur
blur2 = cv2.GaussianBlur(img, (5, 5), 0)                   # Gaussian blur
blur3 = cv2.medianBlur(img, 5)                              # Median blur
blur4 = cv2.bilateralFilter(img, 9, 75, 75)                # Bilateral blur

# Custom sharpening kernel
sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
sharpened = cv2.filter2D(img, -1, sharpen_kernel)
```

> **What is a Kernel?** A kernel is a small matrix (like 3×3 or 5×5) that slides over the image. At each position, it performs a math operation with the pixel values underneath it. Different kernels produce different effects (blur, sharpen, detect edges).

---

#### 📐 Category 6: Edge Detection & Gradients

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.Canny(img, threshold1, threshold2)` | Detects edges using Canny algorithm | Finding object boundaries |
| `cv2.Sobel(img, ddepth, dx, dy, ksize)` | Computes Sobel gradients | Detecting horizontal/vertical edges |
| `cv2.Laplacian(img, ddepth)` | Computes Laplacian gradient | Finding all edges at once |
| `cv2.Scharr(img, ddepth, dx, dy)` | More accurate Sobel alternative | Precise edge detection |

```python
import cv2

img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)           # Canny edge detection
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Horizontal edges
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1) # Vertical edges
laplacian = cv2.Laplacian(img, cv2.CV_64F)  # All edges
```

---

#### ⬛ Category 7: Thresholding

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.threshold(img, thresh, maxval, type)` | Applies fixed threshold | Binary masks, document cleanup |
| `cv2.adaptiveThreshold(img, maxval, method, type, blockSize, C)` | Adaptive local threshold | Varying lighting conditions |

```python
import cv2

gray = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)

# Fixed threshold: pixels > 127 become 255 (white), rest become 0 (black)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Otsu's method (auto-finds the best threshold value)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive threshold (handles uneven lighting)
adaptive = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
```

---

#### 🔍 Category 8: Contour Detection

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.findContours(img, mode, method)` | Finds contours (outlines) in binary image | Detecting object shapes |
| `cv2.drawContours(img, contours, idx, color, thickness)` | Draws contours on image | Visualizing detected shapes |
| `cv2.contourArea(contour)` | Calculates contour area | Filtering small/large objects |
| `cv2.arcLength(contour, closed)` | Calculates contour perimeter | Measuring object boundaries |
| `cv2.approxPolyDP(contour, epsilon, closed)` | Approximates contour shape | Shape recognition |
| `cv2.boundingRect(contour)` | Gets bounding rectangle | Drawing boxes around objects |
| `cv2.minEnclosingCircle(contour)` | Gets smallest enclosing circle | Circular object detection |
| `cv2.convexHull(contour)` | Finds convex hull | Hand gesture recognition |
| `cv2.moments(contour)` | Calculates shape moments | Finding center of mass |

```python
import cv2

img = cv2.imread("shapes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw ALL contours in green
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Process each contour
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # Ignore tiny contours (noise)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

---

#### 🧮 Category 9: Morphological Operations

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.erode(img, kernel, iterations)` | Shrinks white regions | Removing thin noise |
| `cv2.dilate(img, kernel, iterations)` | Expands white regions | Filling small holes |
| `cv2.morphologyEx(img, op, kernel)` | Advanced morphological operations | Opening, closing, gradient |
| `cv2.getStructuringElement(shape, size)` | Creates kernel for morphology | Custom-shaped kernels |

```python
import cv2
import numpy as np

gray = cv2.imread("noisy.jpg", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)

eroded = cv2.erode(binary, kernel, iterations=1)    # Shrink white areas
dilated = cv2.dilate(binary, kernel, iterations=1)   # Expand white areas
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Erode then Dilate
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Dilate then Erode
```

---

#### 🎯 Category 10: Feature Detection

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.goodFeaturesToTrack(img, maxCorners, quality, minDist)` | Finds strong corner points | Tracking points in video |
| `cv2.cornerHarris(img, blockSize, ksize, k)` | Harris corner detection | Finding image corners |
| `cv2.HoughLines(img, rho, theta, threshold)` | Detects straight lines | Lane detection |
| `cv2.HoughLinesP(img, rho, theta, threshold, minLen, maxGap)` | Probabilistic line detection | More practical line detection |
| `cv2.HoughCircles(img, method, dp, minDist)` | Detects circles | Coin/ball detection |
| `cv2.matchTemplate(img, template, method)` | Template matching | Finding a pattern in image |

---

#### 👤 Category 11: Object Detection (Pre-trained)

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.CascadeClassifier(xml_path)` | Loads a Haar cascade classifier | Face/eye detection |
| `classifier.detectMultiScale(img, scaleFactor, minNeighbors)` | Detects objects in image | Finding faces in a photo |
| `cv2.dnn.readNetFromCaffe(proto, model)` | Loads a Caffe neural network | Deep learning-based detection |
| `cv2.dnn.readNetFromTensorflow(model, config)` | Loads a TensorFlow model | Using TF models in OpenCV |
| `cv2.dnn.readNetFromDarknet(cfg, weights)` | Loads a YOLO/Darknet model | Real-time object detection |
| `cv2.dnn.blobFromImage(img, scale, size, mean, swapRB)` | Prepares image for neural network | Pre-processing for DNN |

```python
import cv2

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

img = cv2.imread("group_photo.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces", img)
cv2.waitKey(0)
```

---

#### 🎥 Category 12: Video & Motion Analysis

| Function | What It Does | Example Use Case |
|---|---|---|
| `cv2.calcOpticalFlowFarneback(...)` | Computes dense optical flow | Motion visualization |
| `cv2.calcOpticalFlowPyrLK(...)` | Sparse optical flow (Lucas-Kanade) | Tracking specific points |
| `cv2.createBackgroundSubtractorMOG2()` | Background subtraction | Detecting moving objects |
| `cv2.createBackgroundSubtractorKNN()` | KNN background subtractor | Surveillance motion detection |
| `cv2.absdiff(frame1, frame2)` | Absolute difference between frames | Simple motion detection |

---

### 2.4 Practice Examples (OpenCV)

#### 🟢 Beginner Level: Photo Editor

```python
"""
BEGINNER PROJECT: Simple Photo Editor
Skills: imread, resize, flip, blur, Canny, imwrite
"""
import cv2
import numpy as np

# Load your image (replace with your own image path)
img = cv2.imread("photo.jpg")

# --- Apply various effects ---
# 1. Grayscale version
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Blurred version
blurred = cv2.GaussianBlur(img, (15, 15), 0)

# 3. Edge-detected version
edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)

# 4. Flipped version (mirror)
mirrored = cv2.flip(img, 1)

# 5. Resized to half
small = cv2.resize(img, None, fx=0.5, fy=0.5)

# Save all versions
cv2.imwrite("gray.jpg", gray)
cv2.imwrite("blurred.jpg", blurred)
cv2.imwrite("edges.jpg", edges)
cv2.imwrite("mirrored.jpg", mirrored)
cv2.imwrite("small.jpg", small)

print("All effects saved successfully!")
```

#### 🟡 Intermediate Level: Motion Detector

```python
"""
INTERMEDIATE PROJECT: Webcam Motion Detector
Skills: VideoCapture, absdiff, threshold, contours, drawing
"""
import cv2

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Find difference between current and previous frame
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill gaps
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:  # Ignore small movements
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            motion_detected = True

    status = "MOTION DETECTED!" if motion_detected else "No motion"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Motion Detector", frame)
    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### 🔴 Advanced Level: Color-Based Object Tracker

```python
"""
ADVANCED PROJECT: Track a colored object in real-time
Skills: HSV conversion, inRange, morphology, contours, minEnclosingCircle
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Define HSV range for blue color (adjust for your object)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Store trail points
trail = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural feel
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            trail.append(center)
            if len(trail) > 100:
                trail.pop(0)

    # Draw trail
    for i in range(1, len(trail)):
        cv2.line(frame, trail[i-1], trail[i], (0, 255, 0), 2)

    cv2.imshow("Color Tracker", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 3. MediaPipe — Google's AI Perception Toolkit

### What is MediaPipe?

**MediaPipe** is a framework built by Google that provides **pre-trained AI models** for understanding humans in images and videos. Unlike OpenCV (which gives you raw tools), MediaPipe gives you **ready-to-use solutions** for complex tasks like detecting faces, hands, and body poses.

Think of it this way:
- **OpenCV** = A toolbox with hammers, saws, and screwdrivers (you build everything yourself)
- **MediaPipe** = Pre-assembled gadgets (face detector, hand tracker, pose estimator — just plug and play)

**Key facts:**
- Created by Google in 2019
- Works in real-time (even on mobile devices!)
- Uses machine learning models behind the scenes
- Completely free and open source

### 3.1 Installation

```bash
pip install mediapipe
python -c "import mediapipe as mp; print(mp.__version__)"
```

> **Note:** MediaPipe requires OpenCV as a dependency, so it will be installed automatically.

### 3.2 Core Concepts

#### Solutions Architecture

MediaPipe is organized into **Solutions** — each solution solves a specific perception task:

```
MediaPipe Solutions
├── Face Detection      → Finding faces in images
├── Face Mesh           → 468 facial landmarks (detailed face mapping)
├── Hands               → 21 hand landmarks per hand
├── Pose                → 33 body landmarks (full body tracking)
├── Holistic            → Face + Hands + Pose combined
├── Selfie Segmentation → Separating person from background
├── Objectron           → 3D object detection
└── Drawing Utils       → Helper to visualize results
```

#### What are Landmarks?

A **landmark** is a specific point on a body part. Each landmark has:
- **x**: Horizontal position (0.0 = left edge, 1.0 = right edge)
- **y**: Vertical position (0.0 = top edge, 1.0 = bottom edge)
- **z**: Depth (how close/far from camera; smaller = closer)
- **visibility**: Confidence that the landmark is visible (0.0 to 1.0)

> **Important:** x, y, z are **normalized** (between 0 and 1). To get pixel coordinates, multiply by image width/height.

### 3.3 All Key Functions & Use Cases

---

#### 👤 Solution 1: Face Detection

| Function / Property | What It Does |
|---|---|
| `mp.solutions.face_detection.FaceDetection(min_detection_confidence)` | Creates face detector |
| `detector.process(rgb_image)` | Runs detection on an RGB image |
| `results.detections` | List of detected faces |
| `detection.location_data.relative_bounding_box` | Bounding box (xmin, ymin, width, height) |
| `detection.score` | Confidence score(s) |

**Detects 6 facial keypoints:** Right eye, Left eye, Nose tip, Mouth center, Right ear, Left ear.

```python
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(min_detection_confidence=0.5) as face_det:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_det.process(rgb)
        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

---

#### 🎭 Solution 2: Face Mesh (468 landmarks)

| Function / Property | What It Does |
|---|---|
| `FaceMesh(max_num_faces, min_detection_confidence, min_tracking_confidence)` | Creates face mesh detector |
| `results.multi_face_landmarks` | List of face landmarks per detected face |
| `face_landmarks.landmark[index]` | Access specific landmark (0-467) |

**Key landmark indices:** 1 = Nose tip, 33 = Right eye inner, 263 = Left eye outer, 61 = Right mouth corner, 291 = Left mouth corner, 10 = Forehead, 152 = Chin.

```python
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, face_lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

---

#### ✋ Solution 3: Hands (21 landmarks per hand)

| Function / Property | What It Does |
|---|---|
| `Hands(static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence)` | Creates hand detector |
| `results.multi_hand_landmarks` | Hand landmarks for each detected hand |
| `results.multi_handedness` | Whether each hand is left or right |

**21 Hand Landmarks:**
```
            MIDDLE_FINGER_TIP (12)
                    |
           MIDDLE_FINGER_DIP (11)
                    |
           MIDDLE_FINGER_PIP (10)
                    |
           MIDDLE_FINGER_MCP (9)
    INDEX(5-8)  |   RING(13-16)   PINKY(17-20)
        \       |       /
         WRIST (0)
    THUMB: TIP(4)—IP(3)—MCP(2)—CMC(1)
```

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                # Get index finger tip
                h, w, _ = frame.shape
                tip = hand_lm.landmark[8]
                cv2.circle(frame, (int(tip.x*w), int(tip.y*h)), 10, (255, 0, 255), -1)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

---

#### 🏃 Solution 4: Pose (33 body landmarks) — **Most Important for Our Project**

| Function / Property | What It Does |
|---|---|
| `Pose(static_image_mode, model_complexity, smooth_landmarks, min_detection_confidence, min_tracking_confidence)` | Creates pose detector |
| `results.pose_landmarks` | 33 pose landmarks |
| `results.pose_world_landmarks` | 3D world coordinates (in meters) |
| `landmark.x, .y, .z, .visibility` | Position and visibility |

**33 Pose Landmarks:**
```
                NOSE (0)
               /    \
    LEFT_EYE(2)      RIGHT_EYE(5)
    LEFT_EAR(7)      RIGHT_EAR(8)
              \      /
    LEFT_SHOULDER(11)——RIGHT_SHOULDER(12)
              |                    |
    LEFT_ELBOW(13)          RIGHT_ELBOW(14)
              |                    |
    LEFT_WRIST(15)          RIGHT_WRIST(16)
              \                   /
    LEFT_HIP(23)————RIGHT_HIP(24)
              |                    |
    LEFT_KNEE(25)           RIGHT_KNEE(26)
              |                    |
    LEFT_ANKLE(27)          RIGHT_ANKLE(28)
```

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            # Example: Get left wrist position
            h, w, _ = frame.shape
            lw = results.pose_landmarks.landmark[15]
            cv2.putText(frame, f"L.Wrist: ({int(lw.x*w)},{int(lw.y*h)})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

---

#### 🧑‍🤝‍🧑 Solution 5: Holistic (Face + Hands + Pose Combined)

| Function / Property | What It Does |
|---|---|
| `Holistic(min_detection_confidence, min_tracking_confidence)` | Creates holistic detector |
| `results.face_landmarks` | 468 face landmarks |
| `results.left_hand_landmarks` | 21 left hand landmarks |
| `results.right_hand_landmarks` | 21 right hand landmarks |
| `results.pose_landmarks` | 33 pose landmarks |

---

#### 🖼️ Solution 6: Selfie Segmentation

| Function / Property | What It Does |
|---|---|
| `SelfieSegmentation(model_selection)` | Creates segmenter (0=general, 1=landscape) |
| `results.segmentation_mask` | Mask: person ≈ 1.0, background ≈ 0.0 |

---

#### 🎨 Drawing Utilities

| Function | What It Does |
|---|---|
| `mp_draw.draw_landmarks(image, landmarks, connections, landmark_spec, connection_spec)` | Draws landmarks and connections |
| `mp_draw.draw_detection(image, detection)` | Draws face detection bounding box |
| `DrawingSpec(color, thickness, circle_radius)` | Customizes drawing appearance |

---

### 3.4 Practice Examples (MediaPipe)

#### 🟢 Beginner: Finger Counter

```python
"""
BEGINNER PROJECT: Count raised fingers using hand landmarks
"""
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        finger_count = 0
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            # Thumb (sideways check)
            if lm[4].x < lm[3].x:
                finger_count += 1
            # Other 4 fingers (tip above pip = raised)
            for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
                if lm[tip].y < lm[pip].y:
                    finger_count += 1
        cv2.putText(frame, f"Fingers: {finger_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

#### 🟡 Intermediate: Bicep Curl Counter

```python
"""
INTERMEDIATE PROJECT: Count bicep curl reps using pose angle calculation
"""
import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_angle(a, b, c):
    """Calculate angle at point b given points a, b, c."""
    a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
    cosine = np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
counter, stage = 0, None

with mp_pose.Pose(min_detection_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            angle = calculate_angle(
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                lm[mp_pose.PoseLandmark.LEFT_WRIST])
            if angle > 160:
                stage = "down"
            if angle < 40 and stage == "down":
                stage = "up"
                counter += 1
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.rectangle(frame, (0, 0), (250, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Reps: {counter}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Stage: {stage or '-'}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Bicep Curl Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
```

---

## 4. OpenPose — Carnegie Mellon's Pose Powerhouse

### What is OpenPose?

**OpenPose** is the first real-time **multi-person** pose estimation system, created by researchers at Carnegie Mellon University (CMU). Its special power is that it can detect the poses of **multiple people simultaneously** — even when they overlap or partially hide each other.

**Key facts:**
- Created by CMU's Perceptual Computing Lab in 2017
- First real-time multi-person keypoint detection system
- Can detect body, hand, foot, and face keypoints
- Uses deep learning (Convolutional Neural Networks)
- Open source (for non-commercial use)

> **Important:** OpenPose is more complex to set up than MediaPipe and requires more computing power. It's best for scenarios where you need multi-person detection with high accuracy.

### 4.1 Installation

OpenPose is more complex to install than pip packages. Here are your options:

**Option A: Build from source (Recommended for full features)**
```bash
# 1. Clone the repository
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# 2. Follow platform-specific build instructions at:
#    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md

# Requirements: CMake, CUDA (for GPU), cuDNN
```

**Option B: Use OpenCV's DNN module to load OpenPose models (Easier)**
```bash
# Just use OpenCV (already installed!)
pip install opencv-python

# Download the pre-trained models:
# - COCO model: pose_iter_440000.caffemodel
# - BODY_25 model: pose_iter_584000.caffemodel
# - Prototxt files from OpenPose GitHub releases
```

**Option C: Use tf-pose-estimation (TensorFlow wrapper)**
```bash
pip install tf-pose-estimation
```

> **Recommendation for beginners:** Use **Option B** (OpenCV's DNN module) — it's the easiest to set up and doesn't require building from source.

### 4.2 Core Concepts

#### How OpenPose Works (Bottom-Up Approach)

Unlike MediaPipe (which detects one person, then finds their joints), OpenPose uses a "bottom-up" approach:

```
Step 1: Detect ALL body parts in the entire image
        (all elbows, all knees, all shoulders, etc.)

Step 2: Group body parts into individual people
        (this elbow belongs to person A, that one to person B)

This is why it can handle multiple people efficiently!
```

#### Architecture Overview

```
Input Image
     ↓
[VGG-19 Neural Network]     ← Feature extraction
     ↓
┌─────────────────────────┐
│  Two-Branch CNN:        │
│  Branch 1: Confidence   │  ← Heatmaps (where are body parts?)
│  Branch 2: Part         │  ← PAFs - Part Affinity Fields
│  Affinity Fields        │     (which parts connect to which?)
└─────────────────────────┘
     ↓
[Bipartite Matching]         ← Group parts into people
     ↓
Poses for ALL people
```

#### Keypoint Models

OpenPose supports multiple keypoint models:

**BODY_25 Model (25 keypoints — Recommended):**
```
        NOSE (0)
       /    \
 NECK (1)
 /         \
L_SHOULDER(5)  R_SHOULDER(2)
|               |
L_ELBOW(6)    R_ELBOW(3)
|               |
L_WRIST(7)    R_WRIST(4)
|               |
L_HIP(12)     R_HIP(9)
|               |
L_KNEE(13)    R_KNEE(10)
|               |
L_ANKLE(14)   R_ANKLE(11)
|               |
L_BIGTOE(19)  R_BIGTOE(22)
L_SMALLTOE(20) R_SMALLTOE(23)
L_HEEL(21)    R_HEEL(24)

Also: L_EYE(15), R_EYE(16), L_EAR(17), R_EAR(18)
```

**COCO Model (18 keypoints):**

| Index | Keypoint |
|---|---|
| 0 | Nose |
| 1 | Neck |
| 2-4 | Right Shoulder, Elbow, Wrist |
| 5-7 | Left Shoulder, Elbow, Wrist |
| 8-10 | Right Hip, Knee, Ankle |
| 11-13 | Left Hip, Knee, Ankle |
| 14-17 | Eyes (R, L), Ears (R, L) |

### 4.3 All Key Functions & Use Cases

Since OpenPose is typically used through OpenCV's DNN module, here are the key functions:

#### Core OpenPose via OpenCV DNN

| Function | What It Does |
|---|---|
| `cv2.dnn.readNetFromCaffe(protoFile, weightsFile)` | Loads the OpenPose neural network model |
| `cv2.dnn.blobFromImage(img, scalefactor, size, mean, swapRB, crop)` | Converts image to format the network expects |
| `net.setInput(blob)` | Feeds the processed image into the network |
| `net.forward()` | Runs inference and gets heatmaps/PAFs |
| `cv2.minMaxLoc(heatmap)` | Finds the peak point in each body part's heatmap |

#### Working with Heatmaps

| Concept | Explanation |
|---|---|
| **Heatmap** | A grayscale image where bright spots = high chance of a body part being there |
| **Confidence threshold** | Minimum brightness in heatmap to count as a valid detection (typically 0.1-0.3) |
| **PAF (Part Affinity Field)** | A vector field that tells which body parts connect to which — used to assemble full skeletons |

#### Keypoint Pair Definitions (COCO Model)

```python
# Which keypoints connect to form the skeleton
POSE_PAIRS_COCO = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]
```

### 4.4 Practice Examples (OpenPose)

#### 🟢 Beginner: Single-Person Pose Detection with OpenCV DNN

```python
"""
BEGINNER PROJECT: Detect body pose using OpenPose model via OpenCV
Skills: DNN module, heatmaps, drawing skeleton
NOTE: Download models first (see instructions below)
"""
import cv2
import numpy as np

# ============================================================
# SETUP: Download these files before running:
# 1. pose_deploy_linevec.prototxt (from OpenPose GitHub)
# 2. pose_iter_440000.caffemodel (COCO model weights)
# Place them in a folder called 'models/'
# ============================================================

# Model paths
protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"

# COCO keypoint names and skeleton pairs
BODY_PARTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle",
    14: "REye", 15: "LEye", 16: "REar", 17: "LEar"
}

POSE_PAIRS = [
    [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [0, 14], [0, 15], [14, 16], [15, 17]
]

# Load the network
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Read image
img = cv2.imread("person.jpg")
h, w = img.shape[:2]

# Prepare input blob
inWidth, inHeight = 368, 368
blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

# Run inference
net.setInput(blob)
output = net.forward()  # Shape: (1, 57, height, width)

num_points = 18
points = []
threshold = 0.1

# Parse heatmaps to find keypoints
for i in range(num_points):
    heatmap = output[0, i, :, :]
    _, confidence, _, point = cv2.minMaxLoc(heatmap)

    # Scale point back to original image size
    x = int((w * point[0]) / output.shape[3])
    y = int((h * point[1]) / output.shape[2])

    if confidence > threshold:
        points.append((x, y))
        cv2.circle(img, (x, y), 8, (0, 255, 255), thickness=-1)
        cv2.putText(img, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 255), 1)
    else:
        points.append(None)

# Draw skeleton
for pair in POSE_PAIRS:
    partA, partB = pair
    if points[partA] and points[partB]:
        cv2.line(img, points[partA], points[partB], (0, 255, 0), 3)

cv2.imshow("OpenPose Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 🟡 Intermediate: Real-Time Webcam Pose with OpenPose

```python
"""
INTERMEDIATE PROJECT: Real-time pose detection from webcam using OpenPose
Skills: Video processing, DNN inference, performance optimization
"""
import cv2
import time

protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
num_points = 18

POSE_PAIRS = [
    [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [0, 14], [0, 15], [14, 16], [15, 17]
]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# Use GPU if available (much faster)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0/255, (192, 192),
                                  (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    points = []
    for i in range(num_points):
        heatmap = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = int((w * point[0]) / output.shape[3])
        y = int((h * point[1]) / output.shape[2])
        points.append((x, y) if conf > 0.1 else None)

    # Draw skeleton
    for pair in POSE_PAIRS:
        a, b = pair
        if points[a] and points[b]:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
            cv2.circle(frame, points[a], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[b], 5, (0, 0, 255), -1)

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("OpenPose Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5. Comparison: OpenCV vs MediaPipe vs OpenPose

| Feature | OpenCV | MediaPipe | OpenPose |
|---|---|---|---|
| **Purpose** | General vision toolkit | AI perception solutions | Multi-person pose estimation |
| **Created** | 1999 (Intel) | 2019 (Google) | 2017 (CMU) |
| **Ease of Setup** | ⭐⭐⭐⭐⭐ `pip install` | ⭐⭐⭐⭐⭐ `pip install` | ⭐⭐ Build from source |
| **Pose Detection** | Via DNN module only | Built-in (33 points) | Specialized (18/25 points) |
| **Multi-Person** | Depends on model | ❌ Single person only | ✅ Native multi-person |
| **Speed** | Very fast (basic ops) | ⚡ Real-time on CPU | Needs GPU for real-time |
| **Hand Tracking** | ❌ Not built-in | ✅ 21 landmarks | ✅ 21 landmarks |
| **Face Tracking** | Haar cascades (basic) | ✅ 468 landmarks | ✅ 70 landmarks |
| **3D Coordinates** | ❌ | ✅ (world landmarks) | ✅ |
| **GPU Required?** | No | No | Recommended |
| **Best For** | Image processing, drawing, video I/O | Single-person real-time apps | Multi-person research |
| **License** | Apache 2.0 (free) | Apache 2.0 (free) | Non-commercial use |

### When to Use What?

| Scenario | Best Choice | Why |
|---|---|---|
| Single person fitness app | **MediaPipe** | Fast, easy, accurate |
| Multi-person dance analysis | **OpenPose** | Handles multiple people |
| Processing images/video | **OpenCV** | Best for reading, writing, filtering |
| Drawing stick figures | **OpenCV** | Best drawing functions |
| Mobile app | **MediaPipe** | Optimized for mobile |
| Research paper | **OpenPose** | Industry standard for research |
| Quick prototype | **MediaPipe + OpenCV** | Fastest to implement |

### How They Work Together

```
┌────────────────────────────────────────────────────┐
│                   YOUR PROJECT                      │
│                                                     │
│   ┌──────────┐    ┌───────────┐    ┌────────────┐  │
│   │  OpenCV   │───▶│ MediaPipe │───▶│   OpenCV    │  │
│   │           │    │     OR    │    │             │  │
│   │ • Read    │    │ OpenPose  │    │ • Draw      │  │
│   │   video   │    │           │    │   stick     │  │
│   │ • Convert │    │ • Detect  │    │   figure    │  │
│   │   colors  │    │   pose    │    │ • Save      │  │
│   │ • Resize  │    │ • Get     │    │   output    │  │
│   │           │    │   joints  │    │ • Display   │  │
│   └──────────┘    └───────────┘    └────────────┘  │
│                                                     │
│   INPUT LAYER      DETECTION LAYER   OUTPUT LAYER   │
└────────────────────────────────────────────────────┘
```

---

## 6. Integration Project — Human Movement to 2D Stick Figure

This is the final project that brings **ALL THREE libraries together**. We will:
1. Use **OpenCV** to read webcam video and handle display/drawing
2. Use **MediaPipe Pose** to detect 33 body landmarks in real-time
3. Draw a clean **2D stick figure** on a separate canvas (like OpenPose-style output)
4. Support **recording** the stick figure animation as a video file

### 6.1 Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 STICK FIGURE MAPPER                      │
│                                                          │
│  STEP 1: Capture  ──▶  STEP 2: Detect  ──▶  STEP 3: Draw│
│                                                          │
│  cv2.VideoCapture      mp.solutions.pose     cv2.line()  │
│  cv2.cvtColor()        pose.process()        cv2.circle()│
│  cv2.flip()            landmark extraction   cv2.putText()│
│                                                          │
│  ──▶  STEP 4: Display & Save                             │
│       cv2.imshow() (side-by-side)                        │
│       cv2.VideoWriter() (save animation)                 │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Step-by-Step Explanation

**Step 1 — Capture:** OpenCV opens the webcam and reads frames.

**Step 2 — Detect:** MediaPipe's Pose model processes each frame and returns 33 landmark coordinates (x, y normalized between 0 and 1, plus visibility).

**Step 3 — Draw:** We create a blank canvas and use OpenCV's drawing functions to connect the landmarks into a clean stick figure. We define which joints connect to form bones (e.g., shoulder→elbow→wrist).

**Step 4 — Display:** We show the original webcam feed side-by-side with the stick figure, and optionally save as a video.

### 6.3 Complete Project Code

```python
"""
=================================================================
INTEGRATION PROJECT: Human Movement → 2D Stick Figure Mapper
=================================================================
Libraries Used:
  - OpenCV:   Video I/O, drawing, display
  - MediaPipe: Pose detection (33 landmarks)
  - NumPy:    Array operations

Features:
  - Real-time webcam pose detection
  - Clean 2D stick figure visualization on separate canvas
  - Side-by-side original + stick figure view
  - FPS counter
  - Joint angle display
  - Video recording option (press 'r')
  - Screenshot (press 's')
=================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ========================
# CONFIGURATION
# ========================
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
BG_COLOR = (20, 20, 20)           # Dark gray background
BONE_COLOR = (0, 255, 128)         # Green skeleton lines
JOINT_COLOR = (0, 200, 255)        # Orange-yellow joints
HEAD_COLOR = (255, 200, 0)         # Cyan head circle
TEXT_COLOR = (200, 200, 200)       # Light gray text
BONE_THICKNESS = 3
JOINT_RADIUS = 6
HEAD_RADIUS = 20


# ========================
# SKELETON DEFINITION
# ========================
# Define which landmarks connect to form bones (index pairs)
# Using MediaPipe Pose landmark indices
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12),   # Left shoulder → Right shoulder
    (11, 23),   # Left shoulder → Left hip
    (12, 24),   # Right shoulder → Right hip
    (23, 24),   # Left hip → Right hip

    # Left arm
    (11, 13),   # Left shoulder → Left elbow
    (13, 15),   # Left elbow → Left wrist

    # Right arm
    (12, 14),   # Right shoulder → Right elbow
    (14, 16),   # Right elbow → Right wrist

    # Left leg
    (23, 25),   # Left hip → Left knee
    (25, 27),   # Left knee → Left ankle

    # Right leg
    (24, 26),   # Right hip → Right knee
    (26, 28),   # Right knee → Right ankle
]

# Landmark indices for the head (we'll draw a circle here)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# ========================
# HELPER FUNCTIONS
# ========================

def get_pixel_coords(landmark, width, height):
    """
    Convert normalized MediaPipe landmark to pixel coordinates.

    MediaPipe gives us coordinates between 0.0 and 1.0.
    We multiply by image dimensions to get actual pixel positions.

    Args:
        landmark: MediaPipe landmark with .x, .y, .visibility
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        (x, y) tuple in pixels, or None if landmark is not visible
    """
    if landmark.visibility < 0.5:
        return None
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A-B-C.

    This is useful for measuring joint angles (e.g., elbow bend).
    Uses the dot product formula: cos(θ) = (BA · BC) / (|BA| × |BC|)

    Args:
        a, b, c: Each is a (x, y) tuple

    Returns:
        Angle in degrees (0-180)
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
    return angle


def draw_stick_figure(canvas, landmarks, width, height):
    """
    Draw a clean 2D stick figure on the canvas.

    This function:
    1. Converts all landmark coordinates to pixels
    2. Draws bones (lines between connected joints)
    3. Draws joints (circles at each landmark)
    4. Draws a head circle (calculated from shoulder midpoint and nose)
    5. Returns the list of pixel coordinates for further use

    Args:
        canvas: The image to draw on
        landmarks: MediaPipe pose landmarks
        width: Canvas width
        height: Canvas height

    Returns:
        Dictionary of landmark_index → (x, y) pixel coordinates
    """
    points = {}

    # Step 1: Convert all landmarks to pixel coordinates
    for idx in range(33):
        lm = landmarks.landmark[idx]
        coords = get_pixel_coords(lm, width, height)
        if coords:
            points[idx] = coords

    # Step 2: Draw bones (skeleton lines)
    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(canvas, points[start_idx], points[end_idx],
                    BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

    # Step 3: Draw joints (circles at each point)
    for idx, pt in points.items():
        # Skip face landmarks (0-10) except nose for head
        if idx > 10 or idx == NOSE:
            cv2.circle(canvas, pt, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

    # Step 4: Draw head circle
    if LEFT_SHOULDER in points and RIGHT_SHOULDER in points and NOSE in points:
        # Calculate neck position (midpoint of shoulders)
        neck_x = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) // 2
        neck_y = (points[LEFT_SHOULDER][1] + points[RIGHT_SHOULDER][1]) // 2

        # Draw neck-to-nose line
        cv2.line(canvas, (neck_x, neck_y), points[NOSE],
                BONE_COLOR, BONE_THICKNESS, cv2.LINE_AA)

        # Draw head circle centered at nose
        cv2.circle(canvas, points[NOSE], HEAD_RADIUS, HEAD_COLOR, 2, cv2.LINE_AA)

    return points


def draw_angle_info(canvas, points, mp_pose):
    """
    Calculate and display key joint angles on the canvas.

    Shows: Left elbow angle, Right elbow angle,
           Left knee angle, Right knee angle
    """
    angles = {}

    # Define angle calculations: (joint_name, point_A, point_B_vertex, point_C)
    angle_defs = [
        ("L.Elbow", 11, 13, 15),   # Shoulder → Elbow → Wrist
        ("R.Elbow", 12, 14, 16),
        ("L.Knee",  23, 25, 27),   # Hip → Knee → Ankle
        ("R.Knee",  24, 26, 28),
    ]

    y_offset = 30
    for name, a, b, c in angle_defs:
        if a in points and b in points and c in points:
            angle = calculate_angle(points[a], points[b], points[c])
            angles[name] = angle
            cv2.putText(canvas, f"{name}: {int(angle)}deg",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, TEXT_COLOR, 1, cv2.LINE_AA)
            y_offset += 25

    return angles


# ========================
# MAIN APPLICATION
# ========================

def main():
    """
    Main application loop.

    Controls:
      q     - Quit
      r     - Start/stop recording
      s     - Take screenshot
      +/-   - Adjust detection confidence
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return

    # Recording variables
    recording = False
    video_writer = None
    frame_count = 0
    screenshot_count = 0

    print("=" * 50)
    print("HUMAN MOVEMENT → 2D STICK FIGURE MAPPER")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  r - Start/Stop recording")
    print("  s - Screenshot")
    print("=" * 50)

    with mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam")
                break

            start_time = time.time()

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # --- STEP 1: DETECT POSE ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False   # Performance boost
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # --- STEP 2: CREATE STICK FIGURE ---
            # Create a blank dark canvas
            canvas = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)

            if results.pose_landmarks:
                # Draw MediaPipe skeleton on original frame (semi-transparent)
                mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)

                # Draw our clean stick figure on the canvas
                points = draw_stick_figure(canvas, results.pose_landmarks, w, h)

                # Draw angle information
                draw_angle_info(canvas, points, mp_pose)
            else:
                cv2.putText(canvas, "No person detected",
                           (w//2 - 120, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 0, 255), 2)

            # --- STEP 3: DISPLAY ---
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time + 1e-6)

            # Add FPS to both views
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add recording indicator
            if recording:
                cv2.circle(canvas, (w - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(canvas, "REC", (w - 70, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Add labels
            cv2.putText(frame, "ORIGINAL", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "STICK FIGURE", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Combine side by side
            combined = np.hstack([frame, canvas])

            cv2.imshow("Human Movement to Stick Figure", combined)

            # --- STEP 4: RECORDING ---
            if recording and video_writer:
                video_writer.write(combined)
                frame_count += 1

            # --- KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    # Start recording
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(
                        f'stick_figure_recording.avi',
                        fourcc, 20.0,
                        (combined.shape[1], combined.shape[0])
                    )
                    recording = True
                    frame_count = 0
                    print("Recording started...")
                else:
                    # Stop recording
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print(f"Recording saved! ({frame_count} frames)")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.png"
                cv2.imwrite(filename, combined)
                print(f"Screenshot saved: {filename}")

    # Cleanup
    if video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
```

### 6.4 How to Run This Project

```bash
# Step 1: Install dependencies
pip install opencv-python mediapipe numpy

# Step 2: Save the code above as "stick_figure_mapper.py"

# Step 3: Run it!
python stick_figure_mapper.py

# Controls:
#   q → Quit
#   r → Start/Stop recording
#   s → Take screenshot
```

### 6.5 Extending the Project — Ideas

Here are ways you can improve this project as you learn more:

| Enhancement | Difficulty | What You'll Learn |
|---|---|---|
| Add different color themes | 🟢 Easy | Color manipulation |
| Show movement trail (ghost effect) | 🟡 Medium | Frame buffering |
| Calculate body symmetry score | 🟡 Medium | Math, proportions |
| Add gesture recognition (wave, jump) | 🟡 Medium | Pose classification |
| Compare two poses (reference vs actual) | 🔴 Hard | Pose similarity metrics |
| Export stick figure animation as GIF | 🔴 Hard | Video encoding |
| Add OpenPose for multi-person tracking | 🔴 Hard | DNN module integration |
| Build a web interface with Flask | 🔴 Hard | Web development + streaming |

---

## 7. Next Steps & Resources

### Learning Path

```
Week 1-2: OpenCV Basics
  → Images, videos, drawing, filters, edge detection
  → Complete all OpenCV practice examples above

Week 3-4: MediaPipe Solutions
  → Hands, Pose, Face Mesh
  → Build the finger counter and curl counter

Week 5-6: Integration Project
  → Build the Stick Figure Mapper
  → Add custom features from the extension ideas

Week 7-8: Advanced Topics
  → Try OpenPose for multi-person scenarios
  → Explore 3D pose estimation
  → Build your own project!
```

### Official Documentation

| Resource | Link |
|---|---|
| OpenCV Docs | https://docs.opencv.org/ |
| OpenCV Python Tutorials | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html |
| MediaPipe Docs | https://developers.google.com/mediapipe |
| MediaPipe Python Solutions | https://google.github.io/mediapipe/solutions/solutions.html |
| OpenPose GitHub | https://github.com/CMU-Perceptual-Computing-Lab/openpose |
| OpenPose Paper | https://arxiv.org/abs/1812.08008 |

### Recommended Practice Order

1. ✅ Read through this entire guide
2. ✅ Run each code example one by one
3. ✅ Modify the examples (change colors, thresholds, etc.)
4. ✅ Complete the practice projects in order (🟢→🟡→🔴)
5. ✅ Build the Integration Project
6. ✅ Add your own features
7. ✅ Create something new!

### Key Takeaways

> **OpenCV** is your foundation — it handles everything related to images and videos (reading, writing, processing, drawing).

> **MediaPipe** is your AI engine — it gives you pre-trained models that detect human body parts in real-time with minimal code.

> **OpenPose** is your research tool — when you need to track multiple people or need the highest accuracy, OpenPose is the gold standard.

> **Together**, they form a powerful pipeline: OpenCV for I/O → MediaPipe/OpenPose for detection → OpenCV for visualization.

---

*Guide created for the AIEP Kinemation Project. Happy coding! 🚀*
