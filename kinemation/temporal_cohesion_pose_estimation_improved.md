# Temporal Cohesion in Pose Estimation with MediaPipe & OpenPose
### A Comprehensive Beginner-Friendly Guide

---

## Table of Contents
1. [What is Temporal Cohesion?](#what-is-temporal-cohesion)
2. [Why Does It Matter in Pose Estimation?](#why-does-it-matter)
3. [MediaPipe Pose — Quick Primer](#mediapipe-pose--quick-primer)
4. [OpenPose — Quick Primer](#openpose--quick-primer)
5. [Method 1: Phase-Based Temporal Cohesion](#method-1-phase-based-temporal-cohesion)
6. [Method 2: Event-Driven Temporal Cohesion](#method-2-event-driven-temporal-cohesion)
7. [Method 3: Timer-Based Temporal Cohesion](#method-3-timer-based-temporal-cohesion)
8. [Method 4: Pipeline-Stage Temporal Cohesion](#method-4-pipeline-stage-temporal-cohesion)
9. [Method 5: Windowed Temporal Cohesion](#method-5-windowed-temporal-cohesion)
10. [Method 6: Keyframe-Based Temporal Cohesion](#method-6-keyframe-based-temporal-cohesion)
11. [Comparing All Methods](#comparing-all-methods)
12. [Alternative Smoothing Filters (Beyond EMA)](#alternative-smoothing-filters)
13. [MediaPipe vs. OpenPose — Temporal Cohesion Comparison](#mediapipe-vs-openpose-comparison)
14. [Open Source References](#open-source-references)
15. [Common Mistakes to Avoid](#common-mistakes-to-avoid)

---

## What is Temporal Cohesion?

Imagine you're writing code for a MediaPipe pose estimation system. At some point, you'll notice that certain pieces of code always run **at the same time** — like when the app starts up, or every time a new frame arrives from the camera, or when a person walks out of frame.

**Temporal cohesion** is when you group pieces of code together **because they happen at the same time**, rather than because they logically belong together.

> **Simple analogy:** Think of a morning routine. You brush your teeth, make coffee, and check the news — not because these things are related, but because they all happen "in the morning." That's temporal cohesion in real life.

### Where Temporal Cohesion Sits in the Cohesion Hierarchy

In software engineering, "cohesion" measures how strongly the elements inside a module relate to each other. There are **seven types**, ranked from weakest (worst) to strongest (best):

| Rank | Type | Description | Pose Estimation Example |
|------|------|-------------|------------------------|
| 7 (weakest) | **Coincidental** | Elements grouped randomly, no relationship | A utility file with `draw_skeleton()`, `parse_config()`, and `send_email()` |
| 6 | **Logical** | Elements do similar *categories* of things | A module that handles all "input" — camera, keyboard, and file |
| 5 | **Temporal** | Elements executed at the *same time* | A `startup()` that inits MediaPipe, opens camera, AND creates log file |
| 4 | **Procedural** | Elements follow a fixed *sequence* | Preprocessing → Inference → Drawing all in one function |
| 3 | **Communicational** | Elements operate on the *same data* | Functions that all read/write the same landmark buffer |
| 2 | **Sequential** | Output of one element becomes *input* of the next | BGR→RGB, then RGB→landmarks, then landmarks→angles |
| 1 (strongest) | **Functional** | Every element contributes to *one single purpose* | A function that only computes the angle at a joint |

**Temporal cohesion (rank 5) is "okay but not great."** The goal of this guide is to show you how to **use temporal cohesion deliberately** — grouping code by *when* it runs, while keeping each individual piece of code functionally cohesive (rank 1).

### Good vs. Bad Temporal Cohesion

```
WEAK  (bad):   One function does 5 unrelated things because they all fire at startup
STRONG (good): 5 separate functions, each doing one thing, called by a startup coordinator
```

> **Key principle:** The coordinator handles the **"when"** (timing). Each individual function handles the **"what"** (a single responsibility). This way, the coordinator is temporally cohesive, but the individual functions are *functionally* cohesive — the strongest type.

---

## Why Does It Matter in Pose Estimation?

Pose estimation with MediaPipe (or OpenPose) is inherently **time-sensitive**. You're processing video frames one after another, tracking landmarks across time, smoothing jittery predictions, and reacting to events like people entering or leaving the scene.

### What Happens Without Deliberate Temporal Design

Without deliberate temporal cohesion design, you end up with:
- A `process_frame()` that calls `mp_pose.process()`, smooths landmarks, logs metrics, AND draws the skeleton — **impossible to test any one piece**
- A startup block that initializes MediaPipe, opens the camera, resets buffers, AND warms up the model — **if one step fails, everything is broken**
- Code that is nearly impossible to test one piece at a time

### What Happens With Deliberate Temporal Design

With good temporal cohesion design, each responsibility is isolated, and a thin coordinator handles the timing:
- **Startup coordinator** calls `init_pose()`, `open_camera()`, `create_buffer()` separately — each can be tested alone
- **Frame coordinator** calls `preprocess()`, `infer()`, `smooth()`, `classify()`, `render()` in sequence — swap any stage without touching others
- **Shutdown coordinator** ensures `pose.close()`, `cap.release()`, `save_log()` always run — even on crashes

---

## MediaPipe Pose — Quick Primer

MediaPipe Pose detects **33 body landmarks** per frame. Each landmark has:

| Field | Type | Description |
|---|---|---|
| `x` | float [0, 1] | Normalized horizontal position |
| `y` | float [0, 1] | Normalized vertical position |
| `z` | float | Depth relative to hips (smaller = closer to camera) |
| `visibility` | float [0, 1] | Confidence that the landmark is visible |

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Standard initialization
pose = mp_pose.Pose(
    static_image_mode=False,       # video mode: uses tracking between frames
    model_complexity=1,            # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,         # MediaPipe's built-in EMA smoother
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Per-frame usage
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(frame_rgb)

if results.pose_landmarks:
    for lm in results.pose_landmarks.landmark:
        print(lm.x, lm.y, lm.visibility)
```

> **Key MediaPipe landmark indices used throughout this guide:**
> - `0` = Nose, `11` = Left shoulder, `12` = Right shoulder
> - `13` = Left elbow, `14` = Right elbow, `15` = Left wrist, `16` = Right wrist
> - `23` = Left hip, `24` = Right hip, `25` = Left knee, `26` = Right knee
> - `27` = Left ankle, `28` = Right ankle

---

## OpenPose — Quick Primer

While MediaPipe is a **top-down** single-person tracker (detect person first → find landmarks inside), OpenPose uses a **bottom-up** approach (find all body parts in the entire image → assign them to people). Understanding both helps you see where temporal cohesion applies differently.

### How OpenPose Works (Simplified)

```
Input Image
   ↓
[Step 1] CNN outputs two things simultaneously:
         • Confidence Maps (heatmaps) — "where are body parts?"
         • Part Affinity Fields (PAFs) — "which parts connect to which?"
   ↓
[Step 2] Find peaks in confidence maps → body part candidates
   ↓
[Step 3] Use PAFs to connect candidates → assemble full skeletons
   ↓
Output: Multiple skeletons with keypoints
```

### What Are Part Affinity Fields (PAFs)?

Imagine you've detected three shoulders and three elbows. How do you know which shoulder connects to which elbow? **PAFs** answer this question.

- For every pixel that falls within a limb region (e.g., the upper arm), the PAF stores a **2D vector** pointing from one joint to the next (shoulder → elbow)
- To test if shoulder A connects to elbow B, you draw a line between them and check if the PAF vectors along that line are **pointing in the same direction** as the line itself
- A strong alignment score = these body parts likely belong to the same person

> **Simple analogy:** Confidence maps are like GPS pins showing "there's a shoulder here." PAFs are like arrows painted on a road showing "this shoulder connects to that elbow along this path."

### OpenPose Keypoint Format

OpenPose outputs **18 or 25 keypoints** per person (depending on the model), in **pixel coordinates** (not normalized like MediaPipe):

| Field | Type | Description |
|---|---|---|
| `x` | float (pixels) | Horizontal pixel position |
| `y` | float (pixels) | Vertical pixel position |
| `confidence` | float [0, 1] | Detection confidence for this keypoint |

**Key difference from MediaPipe:** OpenPose gives you pixel coordinates directly, so no normalization is needed for angle calculation. However, this means coordinates change with image resolution.

### OpenPose Key Keypoint Indices (COCO 18-point model)

> - `0` = Nose, `1` = Neck, `2` = Right shoulder, `5` = Left shoulder
> - `3` = Right elbow, `6` = Left elbow, `4` = Right wrist, `7` = Left wrist
> - `8` = Right hip, `11` = Left hip, `9` = Right knee, `12` = Left knee
> - `10` = Right ankle, `13` = Left ankle

### Temporal Cohesion Differences: MediaPipe vs. OpenPose

| Aspect | MediaPipe | OpenPose |
|---|---|---|
| **Built-in temporal tracking** | Yes — detector-tracker pipeline; detector runs only when tracking fails | No — each frame is processed independently |
| **Built-in smoothing** | Yes (`smooth_landmarks=True`) using One-Euro filter internally | No — you must add your own post-processing (Savgol filter, EMA, etc.) |
| **Multi-person** | Single person (primary design) | Multi-person natively via PAFs |
| **Temporal smoothing approach** | Internal filter + optional custom EMA | External: Spatial-Temporal Affinity Fields (STAF) or post-hoc filtering |
| **Output coordinates** | Normalized [0, 1] | Pixel coordinates |

> **Why this matters for temporal cohesion:** Because OpenPose has **no built-in temporal tracking**, every temporal cohesion method in this guide becomes _more_ critical when using OpenPose. You must implement your own smoothing (Method 4), event detection (Method 2), and windowed analysis (Method 5) — none of it comes for free.

---

## Method 1: Phase-Based Temporal Cohesion

### What Is It?

Phase-based temporal cohesion groups code by the **lifecycle phase** of your application. Every system has phases — it starts up, runs for a while, then shuts down. This method ensures each phase has its own coordinator, and within each phase, individual functions still have single responsibilities.

### The Three Core Phases

```
STARTUP  →  RUN (frame loop)  →  SHUTDOWN
```

### Problems This Fixes

**Problem 1 — Resource leaks:** Without a dedicated shutdown phase, MediaPipe's `pose.close()` and `cap.release()` often get skipped when the program exits unexpectedly (e.g., pressing Ctrl+C), leaving GPU/CPU memory allocated indefinitely.

**Problem 2 — Startup side effects:** When initialization, warmup, and the run loop are in one function, a crash during startup can leave the camera open with no way to recover without restarting the whole process.

**Problem 3 — Untestable startup:** You can't unit-test "does the MediaPipe model load correctly?" if model loading is buried inside a monolithic run function that also opens a camera and starts an infinite loop.

### When to Use It

- When your pose estimation system has a clear start and end (a script, a session, a video file)
- When you need clean resource management (camera, MediaPipe context, memory)
- When you want your system to be restartable without side effects

### How to Apply It

**Step 1:** Identify everything that happens at startup, during the run loop, and at shutdown.

**Step 2:** Group them into phase coordinators — but keep each individual task as its own function.

**Step 3:** The coordinator calls the individual functions. It doesn't do any real work itself.

### Code Example

```python
# ============================================================
# BAD: Everything in one monolithic function
# ============================================================
def run_pose_system():
    # Startup mixed with config mixed with run loop
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    landmark_buffer = deque(maxlen=30)
    session_log = {"frames": 0, "start": time.time()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmark_buffer.append(results.pose_landmarks)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        session_log["frames"] += 1
        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Easy to forget or skip if the loop crashes
    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    with open("log.json", "w") as f:
        json.dump(session_log, f)
```

```python
# ============================================================
# GOOD: Phase-based separation with MediaPipe
# ============================================================
import mediapipe as mp
import cv2, json, time
from collections import deque

mp_pose_module = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# --- Individual startup responsibilities ---

def create_pose_estimator(complexity: int = 1, detection_conf: float = 0.5,
                           tracking_conf: float = 0.5):
    """Initializes and returns a MediaPipe Pose instance."""
    return mp_pose_module.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    )

def open_camera(index: int = 0, width: int = 640, height: int = 480):
    """Opens a video capture device and sets resolution."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    return cap

def init_landmark_buffer(maxlen: int = 30) -> deque:
    """Creates a rolling buffer to store recent landmark sequences."""
    return deque(maxlen=maxlen)

def init_session_log() -> dict:
    """Creates a fresh session metadata record."""
    return {"start_time": time.time(), "frames_processed": 0, "detections": 0}

# --- Individual shutdown responsibilities ---

def release_camera(cap):
    """Releases the OpenCV camera resource."""
    cap.release()

def close_pose_estimator(pose):
    """Properly closes the MediaPipe Pose context (frees GPU/CPU memory)."""
    pose.close()

def save_session_log(log: dict, path: str = "session_log.json"):
    """Persists session metadata to disk."""
    log["end_time"] = time.time()
    log["duration_sec"] = log["end_time"] - log["start_time"]
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Session log saved to {path}")


# --- Phase coordinators ---

class MediaPipePoseSystem:

    def startup(self):
        """Coordinator: initializes all resources. Never does processing itself."""
        self.pose    = create_pose_estimator(complexity=1)
        self.cap     = open_camera(index=0)
        self.buffer  = init_landmark_buffer(maxlen=30)
        self.session = init_session_log()
        print("MediaPipe Pose system ready.")

    def run(self):
        """Coordinator: owns the frame loop. Delegates all frame work."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera stream ended.")
                break
            self._process_frame(frame)
            self.session["frames_processed"] += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _process_frame(self, frame):
        """Handles a single frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            self.buffer.append(results.pose_landmarks.landmark)
            self.session["detections"] += 1
            mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                   mp_pose_module.POSE_CONNECTIONS)
        cv2.imshow("MediaPipe Pose", frame)

    def shutdown(self):
        """Coordinator: cleans up all resources in reverse startup order."""
        close_pose_estimator(self.pose)
        release_camera(self.cap)
        cv2.destroyAllWindows()
        save_session_log(self.session)
        print("System shut down cleanly.")


# --- Entry point ---
if __name__ == "__main__":
    system = MediaPipePoseSystem()
    try:
        system.startup()
        system.run()
    finally:
        system.shutdown()   # always runs, even on crash or KeyboardInterrupt
```

### Key Takeaway

> The coordinator (`startup`, `run`, `shutdown`) handles **when**. The individual functions handle **what**. Wrapping the run loop in `try/finally` ensures `shutdown()` always fires — even on a crash or keyboard interrupt.

---

## Method 2: Event-Driven Temporal Cohesion

### What Is It?

Event-driven temporal cohesion groups code by **what triggers it** — a specific thing that happens during runtime. Instead of running on a timer or in a fixed sequence, code fires in response to state changes: a person enters the frame, a landmark becomes invisible, confidence drops below a threshold.

### Common MediaPipe Pose Events

| Event | Trigger condition using MediaPipe |
|---|---|
| `on_person_detected` | `results.pose_landmarks` transitions `None` → not `None` |
| `on_person_lost` | `results.pose_landmarks` transitions not `None` → `None` |
| `on_landmark_occluded` | `landmark.visibility < threshold` for a key joint |
| `on_visibility_restored` | `landmark.visibility` rises back above threshold |
| `on_pose_classified` | Angle/position logic matches a known pose |

### Problems This Fixes

**Problem 1 — Spaghetti conditionals:** Without event handlers, the frame loop accumulates `if/elif` chains — one for detection, one for occlusion, one for classification — all interleaved and impossible to test individually.

**Problem 2 — Missed transitions:** When "person detected" and "processing a detected person" share the same block, initialization logic accidentally runs on every frame instead of only once on entry.

**Problem 3 — No clear reaction model:** Adding a new reaction (e.g., send an alert on fall detection) requires surgically editing the core loop instead of simply adding a new handler function.

### How to Apply It

**Step 1:** Define events as clear boolean conditions on MediaPipe outputs.

**Step 2:** Write a dedicated handler for each event that does only what's needed for that event.

**Step 3:** Use a dispatcher in the main loop to detect transitions and route them.

### Code Example

```python
# ============================================================
# BAD: All event logic tangled into the main loop
# ============================================================
prev_detected = False
while True:
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detected = results.pose_landmarks is not None

    if detected and not prev_detected:
        landmark_buffer = deque(maxlen=60)   # init logic
        session_start = time.time()

    if detected:
        lms = results.pose_landmarks.landmark
        landmark_buffer.append(lms)
        if lms[mp_pose.PoseLandmark.LEFT_WRIST].visibility < 0.5:
            print("Left wrist occluded")   # mixed with frame logic

    if not detected and prev_detected:
        print(f"Session: {time.time() - session_start:.1f}s")
        landmark_buffer.clear()             # cleanup mixed in

    prev_detected = detected
```

```python
# ============================================================
# GOOD: Event-driven handlers with MediaPipe
# ============================================================
import mediapipe as mp
from collections import deque
import time

mp_pose_module = mp.solutions.pose
PoseLandmark = mp_pose_module.PoseLandmark

VISIBILITY_THRESHOLD = 0.5
MONITORED_LANDMARKS = [
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
]


class MediaPipeEventHandler:
    """Reacts to discrete state changes in MediaPipe Pose output."""

    def __init__(self):
        self.landmark_buffer = None
        self.session_start = None
        self.occluded_landmarks = set()

    def on_person_detected(self, landmarks):
        """Fires once when pose transitions None → detected."""
        self.landmark_buffer = deque(maxlen=60)
        self.session_start = time.time()
        self.occluded_landmarks.clear()
        print("Person detected — session started.")

    def on_person_lost(self):
        """Fires once when pose transitions detected → None."""
        duration = time.time() - self.session_start if self.session_start else 0
        print(f"Person lost — session duration: {duration:.1f}s")
        self._save_session()
        self.landmark_buffer = None
        self.session_start = None

    def on_landmark_occluded(self, landmark_id: int, visibility: float):
        """Fires when a monitored landmark's visibility drops below threshold."""
        if landmark_id not in self.occluded_landmarks:
            self.occluded_landmarks.add(landmark_id)
            print(f"Landmark {landmark_id} occluded (vis={visibility:.2f})")

    def on_visibility_restored(self, landmark_id: int, visibility: float):
        """Fires when a previously occluded landmark becomes visible again."""
        if landmark_id in self.occluded_landmarks:
            self.occluded_landmarks.discard(landmark_id)
            print(f"Landmark {landmark_id} restored (vis={visibility:.2f})")

    def on_pose_classified(self, pose_name: str, angle_deg: float):
        """Fires when a specific pose is recognized."""
        print(f"Pose detected: {pose_name} (angle={angle_deg:.1f}°)")

    def on_frame(self, landmarks):
        """Called every frame when a person is present. Buffers landmarks."""
        if self.landmark_buffer is not None:
            self.landmark_buffer.append([
                (lm.x, lm.y, lm.z, lm.visibility)
                for lm in landmarks.landmark
            ])

    def _save_session(self):
        if self.landmark_buffer:
            print(f"  Buffered {len(self.landmark_buffer)} frames.")


class MediaPipeEventDispatcher:
    """
    Monitors MediaPipe output each frame and fires events on state transitions.
    Owns the "when" logic; delegates all "what" to the handler.
    """

    def __init__(self, handler: MediaPipeEventHandler):
        self.handler = handler
        self.prev_detected = False
        self.prev_visibility = {}

    def dispatch(self, results):
        """Call every frame with the MediaPipe results object."""
        detected = results.pose_landmarks is not None

        if detected and not self.prev_detected:
            self.handler.on_person_detected(results.pose_landmarks)

        if not detected and self.prev_detected:
            self.handler.on_person_lost()

        if detected:
            self.handler.on_frame(results.pose_landmarks)
            self._check_occlusion(results.pose_landmarks.landmark)

        self.prev_detected = detected

    def _check_occlusion(self, landmarks):
        for lm_id in MONITORED_LANDMARKS:
            vis = landmarks[lm_id].visibility
            prev_vis = self.prev_visibility.get(lm_id, 1.0)

            if vis < VISIBILITY_THRESHOLD and prev_vis >= VISIBILITY_THRESHOLD:
                self.handler.on_landmark_occluded(lm_id, vis)

            if vis >= VISIBILITY_THRESHOLD and prev_vis < VISIBILITY_THRESHOLD:
                self.handler.on_visibility_restored(lm_id, vis)

            self.prev_visibility[lm_id] = vis


# --- Usage ---
handler    = MediaPipeEventHandler()
dispatcher = MediaPipeEventDispatcher(handler)

while True:
    ret, frame = cap.read()
    if not ret: break
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dispatcher.dispatch(results)   # one line in the main loop
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Key Takeaway

> The dispatcher detects **transitions** (None → detected, visible → occluded). The handler reacts to them. Never put transition detection and reaction logic in the same function.

---

## Method 3: Timer-Based Temporal Cohesion

### What Is It?

Timer-based temporal cohesion groups tasks that fire on a **regular schedule** — every N frames or every N seconds. These tasks don't depend on specific events; they run periodically to maintain system health, recalibrate thresholds, or log performance.

### Common Periodic Tasks with MediaPipe Pose

| Task | Typical Interval | Why periodic, not every frame |
|---|---|---|
| Recalibrate visibility threshold | Every 30 frames | EMA converges quickly; per-frame is wasteful |
| Flush landmark history buffer | Every 60 frames | Remove stale data for memory control |
| Log FPS and latency | Every 10 frames | Smoothed metric — per-frame is too noisy |
| Compute joint angle statistics | Every 30 frames | Aggregated statistic, not per-frame signal |
| Check confidence degradation | Every 100 frames | Slow drift — per-frame attention is overkill |

### Problems This Fixes

**Problem 1 — Per-frame waste:** Running calibration or statistics every frame is 30× more work than every second for identical results on a 30 FPS stream.

**Problem 2 — Shared interval rigidity:** Grouping all maintenance tasks under `if frame_idx % 30 == 0:` means you can't give individual tasks different cadences without duplicating the block.

**Problem 3 — Scattered interval logic:** Hardcoded `% N` checks scattered through the main loop are hard to locate, modify, or disable without touching core processing code.

### Mathematical Foundation — EMA for Threshold Recalibration

A common timer task is recalibrating the effective visibility threshold based on recent landmark quality. The standard approach is an **Exponential Moving Average (EMA)**:

$$\bar{v}_t = \alpha \cdot v_t + (1 - \alpha) \cdot \bar{v}_{t-1}$$

**What each symbol means (in plain English):**
- $\bar{v}_t$ = the **new smoothed value** we're calculating — our best estimate of "typical" visibility right now
- $v_t$ = the **raw measurement** this frame — the average visibility across all 33 landmarks
- $\bar{v}_{t-1}$ = the **previous smoothed value** — what we thought "typical" was last time
- $\alpha \in (0, 1)$ = **how much we trust the new measurement vs. the old estimate**. Think of it as a "trust dial":
  - $\alpha = 0.9$ → "I trust this new frame a lot" → reacts quickly to changes
  - $\alpha = 0.1$ → "I mostly trust my running average" → very stable, slow to adapt

**Why EMA and not a simple average?** A simple average treats all past frames equally. EMA gives **exponentially decreasing weight** to older frames — frame $t$ has weight $\alpha$, frame $t-1$ has weight $\alpha(1-\alpha)$, frame $t-2$ has weight $\alpha(1-\alpha)^2$, and so on. This means it automatically "forgets" old data without needing to store every past frame.

The threshold is then derived as:

$$\theta = \max\!\bigl(\theta_{\min},\ \bar{v}_t - \delta\bigr)$$

Where $\delta = 0.10$ is a safety margin below the mean, and $\theta_{\min} = 0.30$ is a hard floor to prevent the threshold from collapsing.

#### Worked Example — Step by Step

Let's trace through 3 frames with $\alpha = 0.2$, starting from $\bar{v}_0 = 0.50$:

| Frame | Raw visibility $v_t$ | Calculation | Smoothed $\bar{v}_t$ | Threshold $\theta$ |
|-------|---------------------|-------------|---------------------|--------------------|
| 1 | 0.80 (good light) | $0.2 \times 0.80 + 0.8 \times 0.50 = 0.16 + 0.40$ | **0.56** | $\max(0.30, 0.56-0.10) = 0.46$ |
| 2 | 0.75 | $0.2 \times 0.75 + 0.8 \times 0.56 = 0.15 + 0.45$ | **0.60** | $\max(0.30, 0.60-0.10) = 0.50$ |
| 3 | 0.40 (poor light) | $0.2 \times 0.40 + 0.8 \times 0.60 = 0.08 + 0.48$ | **0.56** | $\max(0.30, 0.56-0.10) = 0.46$ |

Notice how frame 3's poor value (0.40) only pulls the smoothed value down slightly (0.60 → 0.56) because $\alpha = 0.2$ is cautious. A single bad frame doesn't ruin the threshold.

**Intuition:** If the average visibility across all landmarks is 0.75 (good lighting), the threshold becomes $\max(0.30,\ 0.75 - 0.10) = 0.65$. In poor lighting where visibility drops to 0.45, it adapts down to $\max(0.30,\ 0.45 - 0.10) = 0.35$ — making the system more lenient rather than treating every landmark as occluded.

### Mathematical Foundation — Joint Angle for Statistics

Another periodic task is computing angle statistics over a buffer of frames. The angle at joint $B$ (formed by landmarks $A$, $B$, $C$) is computed using the **dot-product formula**.

#### Step-by-Step: How the Dot-Product Angle Formula Works

**Goal:** Find the angle at the elbow ($B$), given the shoulder ($A$) and wrist ($C$).

**Step 1 — Create vectors from the joint:**
$$\vec{BA} = A - B, \quad \vec{BC} = C - B$$

These are arrows pointing from the elbow *toward* the shoulder and *toward* the wrist. The angle between these two arrows is the elbow angle.

**Step 2 — Compute the dot product** (multiply matching components and add):
$$\vec{BA} \cdot \vec{BC} = (BA_x \times BC_x) + (BA_y \times BC_y)$$

The dot product tells you how much two vectors "point in the same direction." If they point the same way, it's a large positive number. If perpendicular, it's zero. If opposite, it's negative.

**Step 3 — Compute the magnitudes** (lengths of each vector):
$$|\vec{BA}| = \sqrt{BA_x^2 + BA_y^2}, \quad |\vec{BC}| = \sqrt{BC_x^2 + BC_y^2}$$

**Step 4 — Divide and take arccos:**
$$\theta = \arccos\!\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}|\,|\vec{BC}|}\right)$$

The fraction inside arccos is always between -1 and 1. It tells you the cosine of the angle. `arccos` converts it back to an angle in radians (multiply by $180/\pi$ for degrees).

#### Worked Example

Suppose shoulder $A = (0.4, 0.3)$, elbow $B = (0.5, 0.5)$, wrist $C = (0.6, 0.4)$ — all in MediaPipe's normalized [0,1] coordinates:

1. $\vec{BA} = (0.4 - 0.5,\ 0.3 - 0.5) = (-0.1,\ -0.2)$
2. $\vec{BC} = (0.6 - 0.5,\ 0.4 - 0.5) = (0.1,\ -0.1)$
3. Dot product: $(-0.1)(0.1) + (-0.2)(-0.1) = -0.01 + 0.02 = 0.01$
4. Magnitudes: $|\vec{BA}| = \sqrt{0.01 + 0.04} = 0.224$, $|\vec{BC}| = \sqrt{0.01 + 0.01} = 0.141$
5. $\cos(\theta) = 0.01 / (0.224 \times 0.141) = 0.316$
6. $\theta = \arccos(0.316) \approx 71.6°$ — the elbow is bent at about 72 degrees

Over a window of frames, the **mean** $\mu_\theta$ and **standard deviation** $\sigma_\theta$ of this angle reveal the range and consistency of motion:
- Low $\sigma_\theta$ → stable, repeatable movement (e.g., holding a yoga pose)
- High $\sigma_\theta$ → dynamic movement (e.g., bicep curls)

### Code Example

```python
# ============================================================
# BAD: All periodic tasks hardcoded at the same interval
# ============================================================
for frame_idx, frame in enumerate(camera_stream()):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if frame_idx % 30 == 0:
        lms = results.pose_landmarks.landmark if results.pose_landmarks else []
        avg_vis = np.mean([lm.visibility for lm in lms]) if lms else 0.5
        threshold = max(0.3, avg_vis - 0.1)   # recalibrate
        landmark_buffer.clear()               # flush (same interval, different need)
        print(f"FPS: {fps_counter.current():.1f}")   # logging (could be every 10)
        compute_joint_angle_stats(landmark_buffer)   # heavy (should be less frequent)
```

```python
# ============================================================
# GOOD: Scheduler with per-task intervals for MediaPipe
# ============================================================
import numpy as np
from dataclasses import dataclass
from typing import Callable
from collections import deque
import time

@dataclass
class ScheduledTask:
    name: str
    fn: Callable
    interval: int
    last_run: int = -1

class FrameScheduler:
    """Runs tasks at their own individual cadences."""

    def __init__(self):
        self.tasks: list[ScheduledTask] = []

    def register(self, name: str, fn: Callable, every_n_frames: int):
        self.tasks.append(ScheduledTask(name=name, fn=fn, interval=every_n_frames))

    def unregister(self, name: str):
        self.tasks = [t for t in self.tasks if t.name != name]

    def tick(self, frame_idx: int, ctx: dict):
        for task in self.tasks:
            if frame_idx % task.interval == 0:
                task.fn(ctx)
                task.last_run = frame_idx


# EMA state persists across frames
_ema_visibility = 0.5

def recalibrate_visibility_threshold(ctx: dict):
    """
    Recalibrates landmark visibility threshold using EMA.

    EMA:       v̄_t = α * v_t + (1-α) * v̄_{t-1}
    Threshold: θ   = max(θ_min, v̄_t - δ)
    """
    global _ema_visibility
    alpha = 0.2; delta = 0.10; theta_min = 0.30
    landmarks = ctx.get("landmarks")
    if not landmarks:
        return
    v_t = np.mean([lm.visibility for lm in landmarks])
    _ema_visibility = alpha * v_t + (1 - alpha) * _ema_visibility
    ctx["threshold"] = max(theta_min, _ema_visibility - delta)

def flush_landmark_buffer(ctx: dict):
    """Removes entries beyond max_age from the buffer."""
    buf: deque = ctx["landmark_buffer"]
    max_age = ctx.get("buffer_max_age", 90)
    while len(buf) > max_age:
        buf.popleft()

def log_fps(ctx: dict):
    """Logs current FPS."""
    fps = ctx["fps_tracker"].get_fps()
    print(f"[Frame {ctx['frame_idx']}] FPS: {fps:.1f}")

def compute_joint_angle_stats(ctx: dict):
    """
    Computes mean and std of joint angles over recent frames.

    Angle at joint B: θ = arccos( (BA · BC) / (|BA| * |BC|) )
    Stats:  μ_θ = mean(θ), σ_θ = std(θ)
    """
    buf = ctx["landmark_buffer"]
    if len(buf) < 5:
        return
    angles = []
    for frame_lms in buf:
        a, b, c = frame_lms[12], frame_lms[14], frame_lms[16]  # shoulder-elbow-wrist
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if n_ba > 1e-6 and n_bc > 1e-6:
            cos_t = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_t)))
    if angles:
        ctx["stats"]["elbow_angle_mean"] = np.mean(angles)
        ctx["stats"]["elbow_angle_std"]  = np.std(angles)


class FPSTracker:
    def __init__(self, window: int = 30):
        self.timestamps = deque(maxlen=window)
    def tick(self): self.timestamps.append(time.time())
    def get_fps(self) -> float:
        if len(self.timestamps) < 2: return 0.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])


# --- Wiring ---
scheduler    = FrameScheduler()
fps_tracker  = FPSTracker()
landmark_buf = deque(maxlen=120)
ctx = {
    "landmark_buffer": landmark_buf,
    "fps_tracker": fps_tracker,
    "threshold": 0.5,
    "buffer_max_age": 90,
    "stats": {},
}

scheduler.register("visibility_calibration", recalibrate_visibility_threshold, every_n_frames=30)
scheduler.register("buffer_flush",           flush_landmark_buffer,             every_n_frames=60)
scheduler.register("fps_logging",            log_fps,                           every_n_frames=10)
scheduler.register("angle_stats",            compute_joint_angle_stats,         every_n_frames=30)

for frame_idx, frame in enumerate(camera_stream()):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps_tracker.tick()
    ctx["frame_idx"] = frame_idx
    ctx["landmarks"] = results.pose_landmarks.landmark if results.pose_landmarks else None
    if ctx["landmarks"]:
        landmark_buf.append(list(ctx["landmarks"]))
    scheduler.tick(frame_idx, ctx)
```

### Key Takeaway

> Each task owns its own interval. The scheduler handles the "every N frames" logic so the main loop stays clean. EMA adapts the visibility threshold to real lighting conditions — no manual tuning required.

---

## Method 4: Pipeline-Stage Temporal Cohesion

### What Is It?

Pipeline-stage temporal cohesion structures your system as a **series of sequential stages**, where each stage takes an input, does exactly one transformation, and passes the result to the next stage. It fires in order, frame by frame.

### A MediaPipe Pose Pipeline

```
Raw BGR Frame
   ↓
[Stage 1] Preprocess     — BGR→RGB conversion
   ↓
[Stage 2] Inference      — mp_pose.process() → NormalizedLandmarkList
   ↓
[Stage 3] Extract        — landmarks → list of (x, y, z, visibility) floats
   ↓
[Stage 4] Smooth         — EMA per landmark (own smoother, not MediaPipe's)
   ↓
[Stage 5] Classify       — compute joint angles → detect pose/action
   ↓
[Stage 6] Render         — draw skeleton + label on BGR frame
   ↓
Annotated Output Frame
```

### Problems This Fixes

**Problem 1 — Untestable processing:** A single `process_frame()` function can't be unit-tested at the level of individual transformations. You can't test the smoother without also running the camera and MediaPipe.

**Problem 2 — Tight coupling to MediaPipe:** If you want to swap MediaPipe Pose for another model, you have to edit a monolithic function. With pipeline stages, you swap only `MediaPipeInferenceStage`.

**Problem 3 — Unclear data format:** When preprocessing, inference, and rendering are interleaved, it's hard to know "what format is the data in right now?" Each stage makes this explicit through its input/output contract.

### Mathematical Foundation — EMA Smoothing (Stage 4)

MediaPipe has a built-in smoother (`smooth_landmarks=True`), but it's all-or-nothing. The Temporal Smoother stage applies **per-landmark EMA** for fine-grained control:

$$\hat{p}_t^{(j)} = \alpha \cdot p_t^{(j)} + (1 - \alpha) \cdot \hat{p}_{t-1}^{(j)}$$

Where:
- $\hat{p}_t^{(j)}$ = smoothed position of landmark $j$ at frame $t$ (applied to $x$, $y$, $z$ independently)
- $p_t^{(j)}$ = raw MediaPipe output for landmark $j$ at frame $t$
- $\alpha \in (0, 1)$ = smoothing factor

**Choosing $\alpha$:**
| $\alpha$ | Effect |
|---|---|
| 0.9 | Very reactive, minimal lag — good for fast movement |
| 0.5 | Balanced — general purpose |
| 0.2 | Heavy smoothing, more lag — good for slow exercises like yoga |

### Mathematical Foundation — Joint Angle Classification (Stage 5)

The Classify stage uses the **dot-product angle formula** on MediaPipe's normalized (x, y) coordinates:

Given three landmarks $A$, $B$, $C$ (e.g., shoulder, elbow, wrist), the angle $\theta$ at joint $B$ is:

$$\vec{BA} = A - B, \quad \vec{BC} = C - B$$

$$\theta = \arccos\!\left(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}|\,|\vec{BC}|}\right)$$

This maps directly to MediaPipe landmark indices. Classification rules:
- $\theta > 160°$ at landmark 14 (elbow) → arm extended
- $\theta < 90°$ at landmark 26 (knee) → squat
- $\theta \approx 90°$ at landmark 12 (shoulder) → T-pose

### Code Example

```python
# ============================================================
# BAD: All stages crammed into one function
# ============================================================
def process_frame(frame, pose, prev_lms, alpha=0.7):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # preprocess
    results = pose.process(rgb)                     # inference
    if not results.pose_landmarks:
        return frame
    lms = results.pose_landmarks.landmark

    smoothed = []                                   # extract + smooth mixed
    for i, lm in enumerate(lms):
        if prev_lms:
            sx = alpha * lm.x + (1 - alpha) * prev_lms[i].x
            sy = alpha * lm.y + (1 - alpha) * prev_lms[i].y
        else:
            sx, sy = lm.x, lm.y
        smoothed.append((sx, sy, lm.visibility))

    sh = lms[12]; el = lms[14]; wr = lms[16]       # classify buried inside
    ba = np.array([sh.x - el.x, sh.y - el.y])
    bc = np.array([wr.x - el.x, wr.y - el.y])
    angle = np.degrees(np.arccos(np.dot(ba, bc) /
                       (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)))
    label = "arm_extended" if angle > 160 else "bent"

    mp.solutions.drawing_utils.draw_landmarks(      # render also mixed in
        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return frame
```

```python
# ============================================================
# GOOD: MediaPipe pipeline with composable stages
# ============================================================
from abc import ABC, abstractmethod
import mediapipe as mp, cv2, numpy as np

mp_pose_module = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


class PipelineStage(ABC):
    @abstractmethod
    def process(self, data): pass


class PreprocessStage(PipelineStage):
    """BGR frame → dict with 'bgr' and 'rgb' keys."""
    def process(self, frame: np.ndarray) -> dict:
        return {"bgr": frame, "rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)}


class MediaPipeInferenceStage(PipelineStage):
    """Runs MediaPipe Pose. Adds 'mp_results' to the dict."""
    def __init__(self, complexity=1, det_conf=0.5, track_conf=0.5):
        self.pose = mp_pose_module.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=False,   # we apply our own EMA in Stage 4
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
    def process(self, data: dict) -> dict:
        data["mp_results"] = self.pose.process(data["rgb"])
        return data


class LandmarkExtractStage(PipelineStage):
    """
    Converts NormalizedLandmarkList → list of (x, y, z, vis) tuples.
    Adds 'landmarks' key (None if no person detected).
    """
    def process(self, data: dict) -> dict:
        results = data["mp_results"]
        data["landmarks"] = (
            [(lm.x, lm.y, lm.z, lm.visibility)
             for lm in results.pose_landmarks.landmark]
            if results.pose_landmarks else None
        )
        return data


class EMALandmarkSmoother(PipelineStage):
    """
    Per-landmark Exponential Moving Average smoothing.

    Formula: p̂_t^(j) = α * p_t^(j) + (1-α) * p̂_{t-1}^(j)

    Applied independently to x, y, z. Visibility is not smoothed.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.prev: list | None = None

    def process(self, data: dict) -> dict:
        current = data["landmarks"]
        if current is None:
            self.prev = None
            return data
        if self.prev is None:
            self.prev = current
            return data

        a = self.alpha
        data["landmarks"] = [
            (a*cx + (1-a)*px, a*cy + (1-a)*py, a*cz + (1-a)*pz, cvis)
            for (cx, cy, cz, cvis), (px, py, pz, _)
            in zip(current, self.prev)
        ]
        self.prev = data["landmarks"]
        return data


class JointAngleClassifier(PipelineStage):
    """
    Classifies poses from joint angles.

    Formula: θ = arccos( (BA · BC) / (|BA| * |BC|) )
    where A, B, C are three consecutive landmark positions.
    """
    POSE_RULES = [
        ("arm_extended", (12, 14, 16), (160, 180)),
        ("squat",        (24, 26, 28), (70,  120)),
        ("t_pose",       (12, 11, 23), (85,  95)),
    ]

    def process(self, data: dict) -> dict:
        lms = data.get("landmarks")
        if lms is None:
            data["pose_label"] = "no_person"; data["angles"] = {}
            return data
        angles, poses = {}, []
        for label, (ia, ib, ic), (lo, hi) in self.POSE_RULES:
            ang = self._angle(lms[ia], lms[ib], lms[ic])
            angles[label] = ang
            if ang is not None and lo <= ang <= hi:
                poses.append(label)
        data["pose_label"] = poses[0] if poses else "neutral"
        data["angles"] = angles
        return data

    @staticmethod
    def _angle(a, b, c) -> float | None:
        ba = np.array([a[0]-b[0], a[1]-b[1]])
        bc = np.array([c[0]-b[0], c[1]-b[1]])
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6: return None
        return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(n1*n2),-1,1))))


class RenderStage(PipelineStage):
    """Draws MediaPipe skeleton and pose label on the BGR frame."""
    def process(self, data: dict) -> np.ndarray:
        frame = data["bgr"].copy()
        if data["mp_results"].pose_landmarks:
            mp_draw.draw_landmarks(
                frame, data["mp_results"].pose_landmarks,
                mp_pose_module.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2),
            )
        cv2.putText(frame, data.get("pose_label",""), (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)
        return frame


class StagedPipeline:
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
    def run(self, initial_input):
        data = initial_input
        for stage in self.stages:
            data = stage.process(data)
        return data


# --- Wiring ---
pipeline = StagedPipeline([
    PreprocessStage(),
    MediaPipeInferenceStage(complexity=1),
    LandmarkExtractStage(),
    EMALandmarkSmoother(alpha=0.5),
    JointAngleClassifier(),
    RenderStage(),
])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    output = pipeline.run(frame)
    cv2.imshow("MediaPipe Pipeline", output)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
```

### Key Takeaway

> Each stage has an explicit input format and output format. You can unit-test `EMALandmarkSmoother` by feeding it fake landmark lists — no camera required. Swapping MediaPipe for another model means only touching `MediaPipeInferenceStage`.

---

## Method 5: Windowed Temporal Cohesion

### What Is It?

Windowed temporal cohesion processes **a fixed-size window of consecutive frames** together, rather than one frame at a time. Instead of asking "what pose is in this frame?", it asks "what action is happening across these N frames?"

This is essential for action recognition on top of MediaPipe — a single frame's landmarks cannot tell you if someone is doing a squat, a bicep curl, or a jumping jack. You need the motion trajectory.

### How a Temporal Window Works

```
MediaPipe landmark sequence (one row = one frame, 33 landmarks each):

[frame t-15] [(x,y,z,vis) × 33 landmarks]
[frame t-14] [(x,y,z,vis) × 33 landmarks]
    ...
[frame t-1 ] [(x,y,z,vis) × 33 landmarks]
[frame t   ] [(x,y,z,vis) × 33 landmarks]  ← pushed in, oldest dropped
                                              → classify if buffer is full
```

The resulting tensor fed to a classifier: **(T, 33, 4)** — T frames × 33 landmarks × (x, y, z, visibility).

### Problems This Fixes

**Problem 1 — Single-frame ambiguity:** A standing person and the bottom of a squat look nearly identical in a single frame. The temporal trajectory (descent → hold → ascent) is what distinguishes them.

**Problem 2 — Flickering labels:** Classifying independently on every frame produces rapidly changing labels. A window produces stable classifications over a consistent time span.

**Problem 3 — Missing motion context:** Velocity and acceleration of joints are the most discriminative features for action recognition. These are only defined across frames, not within a single frame.

### Mathematical Foundation — Angular Velocity

A core feature computed over the window is the **angular velocity** of a joint — how fast the angle is changing:

$$\omega_t^{(j)} = \frac{\theta_t^{(j)} - \theta_{t-1}^{(j)}}{\Delta t}$$

**What each symbol means:**
- $\omega_t^{(j)}$ = angular velocity of joint $j$ at frame $t$ (degrees per second)
- $\theta_t^{(j)}$ = joint angle at frame $t$ (computed using the dot-product formula explained in Method 3 above)
- $\theta_{t-1}^{(j)}$ = joint angle at the previous frame
- $\Delta t = 1 / \text{FPS}$ = time between frames (e.g., $1/30 = 0.033$ seconds at 30 FPS)

> **Plain English:** Angular velocity is just "how many degrees did the angle change, divided by how much time passed." If the knee angle went from 160° to 140° in one frame at 30 FPS, that's $(140 - 160) / 0.033 = -606$ degrees/second — a fast downward bend.

**How it detects actions:** A squat is characterized by:
1. $\omega_{\text{knee}} < 0$ (bending down — angle decreasing) 
2. followed by $\omega_{\text{knee}} > 0$ (straightening up — angle increasing)

By looking at the *sequence* of angular velocities across a window, you can detect this "V-shaped" pattern.

### Mathematical Foundation — Cosine Similarity Between Frames

To measure how much a pose has changed between consecutive frames, use **cosine similarity** on the flattened (x, y) landmark vectors:

$$\text{sim}(f_t, f_{t+1}) = \frac{\vec{f}_t \cdot \vec{f}_{t+1}}{|\vec{f}_t|\,|\vec{f}_{t+1}|}$$

Where $\vec{f}_t \in \mathbb{R}^{66}$ is the flattened $(x, y)$ coordinates of all 33 landmarks at frame $t$.

> **What does "flattened" mean?** Take each landmark's (x, y) and lay them all out in a single long list: $[x_0, y_0, x_1, y_1, ..., x_{32}, y_{32}]$. That's 66 numbers. Cosine similarity then measures the angle between two such 66-dimensional vectors.

**Interpreting the result:**
- $\text{sim} \approx 1.0$ → the person barely moved (standing still) — like saying "these two pose snapshots point in the same direction"
- $\text{sim} \approx 0.9$ → moderate movement — some joints shifted
- $\text{sim} < 0.8$ → rapid, large motion — the person's pose changed substantially

> **Why cosine similarity and not just Euclidean distance?** Cosine similarity is **scale-invariant** — it measures the *direction* of change, not the *magnitude*. This means it works consistently regardless of how close the person is to the camera (which affects coordinate scale).

This is computed across the window to produce a motion profile of shape $(T-1,)$.

### Code Example

```python
# ============================================================
# BAD: Classifying per frame independently
# ============================================================
while True:
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        lms = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        action = action_classifier.predict(lms)  # meaningless from 1 frame
        cv2.putText(frame, action, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
```

```python
# ============================================================
# GOOD: Windowed temporal cohesion with MediaPipe
# ============================================================
import numpy as np
from collections import deque

class MediaPipeTemporalWindow:
    """
    Accumulates MediaPipe landmarks over T frames and classifies actions.

    Window tensor: (window_size, 33, 4) — T × landmarks × (x,y,z,vis)

    Args:
        window_size: Frames per classification window.
        stride:      Classify every N frames once window is full.
        fps:         Used to compute angular velocity (degrees/second).
    """

    def __init__(self, window_size: int = 30, stride: int = 10, fps: float = 30.0):
        self.window_size = window_size
        self.stride      = stride
        self.dt          = 1.0 / fps        # Δt in seconds
        self.buffer      = deque(maxlen=window_size)
        self.frame_count = 0

    def push(self, landmarks: list | None) -> dict | None:
        """
        Add one frame's landmarks. Returns classification result if ready.
        landmarks: list of 33 (x,y,z,vis) tuples, or None.
        """
        if landmarks is None:
            self.buffer.clear()    # reset on person loss
            return None
        self.buffer.append(landmarks)
        self.frame_count += 1
        if len(self.buffer) == self.window_size and self.frame_count % self.stride == 0:
            return self._classify()
        return None

    def _classify(self) -> dict:
        window = np.array(list(self.buffer), dtype=np.float32)  # (T, 33, 4)
        return {
            "action":             self._rule_based_classify(window),
            "angular_velocities": self._angular_velocities(window),
            "frame_similarities": self._cosine_similarities(window),
        }

    def _angular_velocities(self, window: np.ndarray) -> dict:
        """
        ω_t^(j) = (θ_t - θ_{t-1}) / Δt   for key joints.
        Returns dict of joint_name → ndarray of shape (T-1,).
        """
        triples = {
            "right_knee":  (24, 26, 28),
            "left_knee":   (23, 25, 27),
            "right_elbow": (12, 14, 16),
            "left_elbow":  (11, 13, 15),
        }
        result = {}
        for name, (ia, ib, ic) in triples.items():
            angles = [self._angle(window[t,ia], window[t,ib], window[t,ic])
                      for t in range(len(window))]
            result[name] = np.diff(np.array(angles, dtype=float)) / self.dt
        return result

    def _cosine_similarities(self, window: np.ndarray) -> np.ndarray:
        """
        sim(f_t, f_{t+1}) = (f_t · f_{t+1}) / (|f_t| * |f_{t+1}|)
        Uses x,y coordinates only. Returns shape (T-1,).
        """
        xy = window[:, :, :2].reshape(len(window), -1)   # (T, 66)
        sims = []
        for t in range(len(xy) - 1):
            dot = np.dot(xy[t], xy[t+1])
            n   = np.linalg.norm(xy[t]) * np.linalg.norm(xy[t+1])
            sims.append(dot / (n + 1e-8))
        return np.array(sims)

    def _rule_based_classify(self, window: np.ndarray) -> str:
        """Simple rule-based classifier. Replace with LSTM for production."""
        omega  = self._angular_velocities(window)
        sims   = self._cosine_similarities(window)
        knee_w = omega.get("right_knee", np.array([]))

        if len(knee_w) > 5:
            mid = len(knee_w) // 2
            if np.mean(knee_w[:mid]) < -30 and np.mean(knee_w[mid:]) > 30:
                return "squat"

        if np.mean(sims) > 0.997:
            return "standing_still"

        return "moving"

    @staticmethod
    def _angle(a, b, c) -> float:
        ba = np.array([a[0]-b[0], a[1]-b[1]])
        bc = np.array([c[0]-b[0], c[1]-b[1]])
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(n1*n2),-1,1))))


# --- Usage ---
window = MediaPipeTemporalWindow(window_size=30, stride=10, fps=30.0)

while True:
    ret, frame = cap.read()
    if not ret: break
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = (
        [(lm.x, lm.y, lm.z, lm.visibility)
         for lm in results.pose_landmarks.landmark]
        if results.pose_landmarks else None
    )
    result = window.push(landmarks)
    if result:
        cv2.putText(frame, result["action"], (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow("Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
```

### Key Takeaway

> The window gives your classifier a "memory" of recent motion. Angular velocity and cosine similarity are computed directly from MediaPipe's normalized coordinates — no pixels needed. Choose your window size based on the typical duration of the action you're classifying.

---

## Method 6: Keyframe-Based Temporal Cohesion

### What Is It?

Keyframe-based temporal cohesion runs **full MediaPipe inference only on selected frames** (keyframes) and **interpolates** landmarks on frames in between. This reduces CPU/GPU load while maintaining smooth visual output.

This is particularly useful on constrained hardware (Raspberry Pi, older laptops) where MediaPipe at full FPS is too slow.

### How It Works

```
Frame:    1    2    3    4    5    6    7    8    9    10
Type:     K    i    i    i    i    K    i    i    i    K

K = Keyframe → MediaPipe inference runs (~15–30ms on CPU)
i = Interpolated frame → instant lerp calculation (~0.1ms)

With interval=5: only 20% of frames require full inference.
```

### Problems This Fixes

**Problem 1 — CPU bottleneck:** MediaPipe Pose model complexity 2 takes 30–50ms per frame on CPU. At 30 FPS that's a 100% CPU budget with no room for anything else. Keyframing reduces inference calls by 60–80%.

**Problem 2 — Frozen or jumping landmarks when skipping frames:** Naively skipping frames (reusing the last result) produces frozen landmarks when the person moves. Interpolation keeps the skeleton visually smooth between keyframes.

**Problem 3 — Wasted inference on static scenes:** Fixed-interval keyframing with adaptive motion detection avoids running inference when the person is standing still — saving computation precisely when it's not needed.

### Mathematical Foundation — Linear Interpolation (Lerp)

Between two keyframes at times $t_a$ (previous) and $t_b$ (next), the landmark position at intermediate time $t$ is:

$$p_t = (1 - \lambda) \cdot p_{t_a} + \lambda \cdot p_{t_b}$$

Where the interpolation factor is:

$$\lambda = \frac{t - t_a}{t_b - t_a} \in [0, 1]$$

**Intuition:** When $\lambda = 0$, we're at the previous keyframe. When $\lambda = 1$, we're at the next keyframe. For a point halfway between ($\lambda = 0.5$), we take the average of both positions. This assumes uniform motion between keyframes, which is reasonable for short intervals (< 5 frames at 30 FPS, i.e., < 167ms).

This is applied **per landmark**, **per coordinate** (x, y, z, visibility) independently.

### Mathematical Foundation — Motion Magnitude (Adaptive Keyframes)

For adaptive keyframing, we compute the **mean per-landmark displacement** between the current MediaPipe output and the last keyframe landmarks:

$$\Delta = \frac{1}{N} \sum_{j=1}^{N} \sqrt{(x_t^{(j)} - x_K^{(j)})^2 + (y_t^{(j)} - y_K^{(j)})^2}$$

Where:
- $N = 33$ (MediaPipe landmarks)
- $(x_t^{(j)}, y_t^{(j)})$ = current raw landmark $j$
- $(x_K^{(j)}, y_K^{(j)})$ = landmark $j$ at the last keyframe
- Coordinates are normalized $\in [0, 1]$

A threshold of $\Delta > 0.02$ corresponds to ~2% of the frame width — enough displacement to warrant re-running inference. When $\Delta \leq 0.02$, we reuse the last keyframe result.

### Code Example

```python
# ============================================================
# BAD: Full MediaPipe inference on every frame (CPU bound)
# ============================================================
while True:
    ret, frame = cap.read()
    # On slow CPU: 30-50ms per call → 20-33 FPS ceiling with nothing else running
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose_module.POSE_CONNECTIONS)
    cv2.imshow("Pose", frame)
```

```python
# ============================================================
# GOOD: Keyframe-based MediaPipe with lerp interpolation
# ============================================================
import numpy as np, mediapipe as mp, cv2

mp_pose_module = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


class KeyframeMediaPipePose:
    """
    Runs MediaPipe only on keyframes; interpolates between them.

    Interpolation (lerp):
        λ   = frames_since_keyframe / keyframe_interval
        p_t = (1-λ) * p_prev_keyframe + λ * p_next_keyframe

    Args:
        keyframe_interval: Run full inference every N frames.
    """

    def __init__(self, keyframe_interval: int = 5, model_complexity: int = 1):
        self.interval = keyframe_interval
        self.pose = mp_pose_module.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=False,
        )
        self.prev_kf: list | None = None
        self.next_kf: list | None = None
        self.frames_since_kf = 0

    def process(self, frame: np.ndarray, frame_idx: int) -> list | None:
        if frame_idx % self.interval == 0:
            landmarks = self._infer(frame)
            self.prev_kf = self.next_kf
            self.next_kf = landmarks
            self.frames_since_kf = 0
            return landmarks
        else:
            self.frames_since_kf += 1
            return self._interpolate()

    def _infer(self, frame) -> list | None:
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            return [(lm.x, lm.y, lm.z, lm.visibility)
                    for lm in results.pose_landmarks.landmark]
        return None

    def _interpolate(self) -> list | None:
        """
        p_t = (1-λ) * prev + λ * next
        λ   = frames_since_kf / interval
        """
        if self.next_kf is None:
            return None
        if self.prev_kf is None:
            return self.next_kf
        lam = self.frames_since_kf / self.interval
        return [
            ((1-lam)*px + lam*nx,
             (1-lam)*py + lam*ny,
             (1-lam)*pz + lam*nz,
             (1-lam)*pv + lam*nv)
            for (px,py,pz,pv),(nx,ny,nz,nv) in zip(self.prev_kf, self.next_kf)
        ]


class AdaptiveKeyframeMediaPipePose:
    """
    Triggers a new keyframe only when pose displacement exceeds a threshold.

    Motion magnitude:
        Δ = (1/33) * Σ_j sqrt( (x_t^j - x_K^j)² + (y_t^j - y_K^j)² )

    A new keyframe fires when Δ > motion_threshold.
    """

    def __init__(self, motion_threshold: float = 0.02, model_complexity: int = 1):
        self.threshold = motion_threshold
        self.pose = mp_pose_module.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=False,
        )
        self.last_kf: list | None = None

    def process(self, frame: np.ndarray) -> list | None:
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            self.last_kf = None
            return None
        current = [(lm.x, lm.y, lm.z, lm.visibility)
                   for lm in results.pose_landmarks.landmark]
        if self.last_kf is None:
            self.last_kf = current
            return current
        delta = np.mean([
            np.sqrt((cx-px)**2 + (cy-py)**2)
            for (cx,cy,*_),(px,py,*_) in zip(current, self.last_kf)
        ])
        if delta > self.threshold:
            self.last_kf = current
            return current         # new keyframe
        return self.last_kf        # reuse — person hasn't moved enough


# --- Usage (fixed-interval) ---
estimator = KeyframeMediaPipePose(keyframe_interval=5, model_complexity=1)
cap = cv2.VideoCapture(0)

for frame_idx in range(100000):
    ret, frame = cap.read()
    if not ret: break
    landmarks = estimator.process(frame, frame_idx)
    if landmarks:
        h, w = frame.shape[:2]
        for (x, y, z, vis) in landmarks:
            if vis > 0.5:
                cv2.circle(frame, (int(x*w), int(y*h)), 4, (0,255,0), -1)
    cv2.imshow("Keyframe Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
```

### Key Takeaway

> Lerp assumes smooth motion between keyframes — valid for short intervals at typical video FPS. Adaptive keyframing is smarter: it uses the displacement formula to skip inference only when the person is genuinely not moving.

---

## Comparing All Methods

| Method | Trigger | Problems Fixed | Complexity | Best For |
|---|---|---|---|---|
| **Phase-Based** | App lifecycle | Resource leaks, untestable startup | Low | Every MediaPipe system |
| **Event-Driven** | State change | Spaghetti conditionals, missed transitions | Medium | Person tracking, occlusion |
| **Timer-Based** | Every N frames | Per-frame waste, scattered interval logic | Low | Calibration, FPS logging |
| **Pipeline-Stage** | Every frame, sequential | Monolithic functions, tight coupling | Medium | Core inference loop |
| **Windowed** | Buffer full | Single-frame ambiguity, no motion context | Medium-High | Action/activity recognition |
| **Keyframe-Based** | Every N frames or motion | CPU bottleneck, frozen/jumping landmarks | Medium | Edge / low-power devices |

### How All Six Methods Layer in a Real System

```
startup()                       ← Phase-Based: init pose, camera, buffers
    ↓
for each frame:
    ├── PreprocessStage
    │   → InferenceStage (MediaPipe)
    │   → ExtractStage
    │   → EMASmootherStage       ← Pipeline-Stage: sequential transforms
    │   → ClassifyStage
    │   → RenderStage
    │
    ├── dispatcher.dispatch()    ← Event-Driven: person in/out, occlusion
    │
    ├── window.push(landmarks)   ← Windowed: action recognition over 30 frames
    │
    ├── scheduler.tick()         ← Timer-Based: calibrate every 30, log every 10
    │
    └── estimator.process()      ← Keyframe-Based: full inference every 5 frames
                                                    on constrained hardware

shutdown()                      ← Phase-Based: close pose, release camera, save log
```

---

## Alternative Smoothing Filters (Beyond EMA)

This guide primarily uses **EMA** for smoothing because it's the simplest to implement and understand. However, real production systems often use more sophisticated filters. Here's a comparison to help you choose:

### 1. One-Euro Filter (Used Internally by MediaPipe)

The One-Euro Filter is the **gold standard** for interactive pose smoothing. It adapts its smoothing strength based on the **speed of the signal** — smooth heavily when still, react quickly when moving.

**Core idea:** The cutoff frequency changes with speed:

$$f_c = f_{c,\min} + \beta \cdot |\dot{x}_t|$$

Where:
- $f_{c,\min}$ = minimum cutoff frequency (controls jitter when stationary — **lower = smoother**)
- $\beta$ = speed coefficient (controls lag during fast movement — **higher = more responsive**)
- $|\dot{x}_t|$ = speed of the signal (computed as derivative of the input)

The filtered output is then computed as a low-pass filter with this adaptive cutoff.

**When to use:** Real-time interactive applications where both jitter reduction AND low lag are critical.

**Tuning recipe:**
1. Set $\beta = 0$, have the person stand still. Lower $f_{c,\min}$ until jitter disappears.
2. Then increase $\beta$ while the person moves quickly, until lag is acceptable.

### 2. Kalman Filter

A Kalman Filter **predicts** where a landmark *should* be (using a motion model) and then **corrects** using the actual measurement. It's optimal for linear systems with Gaussian noise.

**When to use:** When landmarks are frequently occluded and you need to predict their position during gaps. Also useful when you have a physical model of human motion (e.g., limbs can't teleport).

**Downside:** More complex to tune (requires setting process noise and measurement noise covariances).

### 3. Savitzky-Golay Filter (Common with OpenPose)

The Savgol filter fits a **polynomial** to a window of data points and uses the polynomial's value at the center as the smoothed output. Unlike EMA, it preserves the shape of peaks and valleys.

**When to use:** Post-processing recorded data (not real-time, because it needs future frames). Common for smoothing OpenPose output in offline video analysis.

### 4. Simple Moving Average (SMA)

Averages the last $N$ values:

$$\bar{x}_t = \frac{1}{N} \sum_{i=0}^{N-1} x_{t-i}$$

**When to use:** Quick-and-dirty smoothing when you don't need adaptive behavior. Introduces constant lag of $N/2$ frames.

### Smoothing Filter Comparison

| Filter | Adaptive? | Real-time? | Lag | Jitter Reduction | Complexity | Best For |
|--------|-----------|------------|-----|-------------------|------------|----------|
| **EMA** | No | Yes | Fixed (depends on α) | Good | Very Low | Simple prototypes |
| **One-Euro** | Yes (speed) | Yes | Low when moving | Excellent | Low | Production real-time |
| **Kalman** | Yes (prediction) | Yes | Minimal | Very Good | Medium | Occluded joints |
| **Savgol** | No | No (needs future) | None (centered) | Excellent | Low | Offline analysis |
| **SMA** | No | Yes | N/2 frames | Good | Very Low | Quick experiments |

---

## MediaPipe vs. OpenPose — Temporal Cohesion Comparison

Here's how each temporal cohesion method applies differently to MediaPipe vs. OpenPose:

| Method | MediaPipe | OpenPose |
|--------|-----------|----------|
| **Phase-Based** | Same pattern — startup/run/shutdown. MediaPipe uses `pose.close()` | Same pattern. OpenPose uses `net.forward()` and must free the DNN model |
| **Event-Driven** | `results.pose_landmarks is None` detects person loss. Built-in `visibility` field per landmark | Must check if keypoint confidence < threshold. No built-in visibility — use confidence score instead |
| **Timer-Based** | Same pattern. MediaPipe's internal smoother handles some calibration | More critical — no internal smoother means you must recalibrate your own filters periodically |
| **Pipeline-Stage** | Swap `MediaPipeInferenceStage` for `OpenPoseInferenceStage` — rest of pipeline unchanged | OpenPose outputs pixel coords, so `ExtractStage` must normalize to [0,1] if downstream stages expect it |
| **Windowed** | Window tensor: (T, 33, 4) — 33 landmarks × 4 values | Window tensor: (T, 18, 3) — 18 keypoints × 3 values (x, y, conf). Need person tracking across frames for multi-person |
| **Keyframe-Based** | Lerp between keyframes. MediaPipe's tracker helps between keyframes | No built-in tracker. Must run full inference on every keyframe. Interpolation even more important to save CPU |

> **Bottom line:** Every method in this guide works with both MediaPipe and OpenPose. The difference is that MediaPipe gives you some temporal features for free (tracking, smoothing), while OpenPose requires you to implement them all yourself — making careful temporal cohesion design even more valuable.

---

## Open Source References

| Repo | Methods Demonstrated | Key Files to Study |
|---|---|---|
| [MediaPipe official examples](https://github.com/google/mediapipe/tree/master/mediapipe/python) | Phase-Based, Pipeline-Stage | `solutions/pose.py` |
| [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | Phase-Based, Pipeline-Stage | `src/openpose/pose/` — PAF + confidence map pipeline |
| [MMPose](https://github.com/open-mmlab/mmpose) | Phase-Based, Pipeline-Stage, Timer-Based | `mmpose/models/pose_estimators/`, `tools/train.py` |
| [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) | Windowed | `common/model.py` (TemporalModelOptimized) |
| [VIBE](https://github.com/mkocabas/VIBE) | Keyframe, Windowed, Smoothing | `lib/models/vibe.py` |
| [pose_refinement](https://github.com/vegesm/pose_refinement) | Event-Driven (occlusion events) | Temporal smoothing for occluded joints |
| [1-Euro Filter](https://gery.casiez.net/1euro/) | Smoothing (used by MediaPipe internally) | Interactive demo + papers + implementations |

---

## Common Mistakes to Avoid

### 1. Forgetting `pose.close()` in the shutdown phase
MediaPipe holds CPU/GPU resources. Without `pose.close()`, memory is not freed between runs.
```python
# BAD — close() skipped if exception is thrown
pose = mp_pose_module.Pose(...)
run_system()
pose.close()

# GOOD — always runs
try:
    run_system()
finally:
    pose.close()
    cap.release()
```

### 2. Passing BGR frames directly to MediaPipe (silent accuracy loss)
MediaPipe expects RGB. BGR won't crash — it just silently reduces landmark accuracy.
```python
# BAD
results = pose.process(frame)                             # BGR!

# GOOD
results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

### 3. Double-smoothing: `smooth_landmarks=True` AND a custom EMA stage
This applies smoothing twice, causing excessive lag and unresponsive landmarks.
```python
# BAD — MediaPipe smooths, then EMALandmarkSmoother smooths again
pose = mp_pose_module.Pose(smooth_landmarks=True)

# GOOD — disable MediaPipe's smoother when using your own pipeline stage
pose = mp_pose_module.Pose(smooth_landmarks=False)
```

### 4. Hardcoding all timer intervals to the same value
```python
# BAD — FPS logging doesn't need the same cadence as buffer flushing
if frame_idx % 30 == 0:
    recalibrate()
    flush_buffer()
    log_fps()         # should be every 10 frames
    check_drift()     # should be every 100 frames
```

### 5. Classifying actions on a single frame
```python
# BAD — impossible to recognize movement from one pose snapshot
action = classify(results.pose_landmarks.landmark)

# GOOD — needs trajectory over time
result = temporal_window.push(landmarks)   # fires only when window is full
```

### 6. Choosing a window size that doesn't match the action duration

| Action | Recommended window at 30 FPS |
|---|---|
| Wave | 20–30 frames (~0.7–1 sec) |
| Squat | 30–60 frames (~1–2 sec) |
| Jumping jack | 30–45 frames |
| Fall detection | 15–20 frames |
| Bicep curl | 45–60 frames |

### 7. Using raw (x, y) coordinates without accounting for normalization in angle math
MediaPipe normalizes coordinates to [0,1], meaning the aspect ratio of the image affects the computed angles. For accurate angle calculation, always convert to pixel coordinates first:
```python
# CORRECTED angle computation accounting for image dimensions
h, w = frame.shape[:2]
ax_px = landmark_a.x * w;  ay_px = landmark_a.y * h
bx_px = landmark_b.x * w;  by_px = landmark_b.y * h
cx_px = landmark_c.x * w;  cy_px = landmark_c.y * h
# ... then compute angle using pixel coordinates
```

---

*This guide covers temporal cohesion patterns as applied to real-time pose estimation using **MediaPipe Pose** and **OpenPose**. All code examples use `mediapipe>=0.10`, `opencv-python>=4.8`, and `numpy>=1.24`. For OpenPose, the concepts apply to both the official C++ library and Python wrappers. Start with the MediaPipe official examples repository to see Phase-Based and Pipeline-Stage patterns in production, then study VideoPose3D for Windowed architecture. For OpenPose temporal extensions, research Spatial-Temporal Affinity Fields (STAF) and post-hoc Savitzky–Golay filtering.*
