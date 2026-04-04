# Kinemation Master Textbook

## Purpose

This document is a full technical briefing for the `AIEP KINEMATION` workspace. It is written for one practical reason: to help you explain the project clearly in a meeting, defend the design choices, answer technical questions, and show that you understand both the current working system and the path that led to it.

This is not a short summary. It is a teaching document. It covers:

- what Kinemation is
- what each top-level folder contains
- the full tech stack
- how each major component works
- why each component was chosen
- what was tested and later dropped
- how the project evolved over time
- what the final working code actually does
- where the limitations still are

The single most important repo fact is this:

`final/` contains the current working backend pipeline.

Everything else in the folder is still important, but for different reasons:

- `prototypes/` shows how the team learned and iterated
- `kinemation/` contains early image and video processing work
- `vidpose-amrita/` contains an older full 3D path based on YOLO pose plus VideoPose3D
- `optimizing/` contains the investigation, experiments, plans, and fixes that explain why `final/` looks the way it does now
- `project/` and `frontend/` wrap the pipeline into an application and UI
- `Reading Material/` contains the theory notes and research explanations that support the implementation choices

## How To Use This Document

If you need to study quickly:

1. Read `What Kinemation Is`
2. Read `Repository Map`
3. Read `Current Final Pipeline`
4. Read `Why These Choices`
5. Read `Known Limitations`
6. Read `Meeting Defense Notes`

If you need deep understanding:

1. Read `Full Tech Stack`
2. Read `Concept Primer`
3. Read `System Evolution and Timeline`
4. Read `Current Final Pipeline`
5. Read `Code Walkthrough`
6. Read `Model Comparisons`

---

## What Kinemation Is

Kinemation is a human pose estimation project focused on converting image or video input into skeletal motion data. In simple language, it tries to watch a person or multiple people in video, identify their body joints, and reconstruct that motion as a structured skeleton.

In this repository, Kinemation evolved through multiple stages:

- basic image and video processing experiments
- 2D stick-figure generation from body landmarks
- multi-person detection using YOLO plus MediaPipe
- 3D lifting using VideoPose3D
- tracking, smoothing, and visualization improvements
- Flask-based app wrapping for upload and webcam usage

The current active system in `final/` is a four-stage pipeline:

1. preprocess a video frame
2. detect people in the frame
3. estimate 2D body landmarks for each detected person
4. lift those 2D keypoints into 3D skeletal motion

Kinemation is therefore not one model. It is an engineered pipeline made by combining several tools, models, and supporting utilities.

---

## Repository Map

### `final/`

This is the authoritative current backend.

Important files:

- `pipeline_3d.py`
- `mediapipe_to_h36m.py`
- `person_tracker.py`
- `visualizer_3d.py`
- `VideoPose3D/`
- `models/`
- `Working pipeline.md`
- `test_separate_outputs.py`

Role:

- current working implementation
- best folder to study first for runtime behavior

### `project/`

This wraps the backend in a Flask app.

Important files:

- `app.py`
- `backend/main.py`
- `backend/mediapipe_to_h36m.py`
- `backend/person_tracker.py`
- `backend/visualizer_3d.py`
- `frontend-resources/`

Role:

- upload and webcam interface
- backend integration layer

### `frontend/`

This is UI-focused packaging and frontend resources.

Role:

- templates
- assets
- presentation layer

### `vidpose-amrita/`

This is an older but important 3D pipeline branch.

Role:

- predecessor architecture
- comparison point for optimization work

### `prototypes/`

This contains the main prototype progression:

- `prototype3.py`
- `prototype4.py`
- `prototype5.py`

Role:

- shows the path from simple 2D work to stronger multi-person processing

### `kinemation/`

This contains early image and video processing work.

Role:

- original OpenCV-centered learning and preprocessing foundation

### `optimizing/`

This is the investigation and engineering notebook of the repo.

Role:

- plans
- controlled experiments
- comparisons
- backups
- timeline evidence

### `Reading Material/`

This stores theory notes, tutorials, and paper summaries.

Role:

- explains the concepts behind the code
- helps defend design choices in a meeting

### `models/`

This contains model assets used or evaluated by the project.

Important files:

- `yolov8n.pt`
- `pose_landmarker_lite.task`
- `pose_deploy_linevec.prototxt`
- `pose_iter_440000.caffemodel`

### `samples/`

This contains input videos and output videos.

Role:

- runtime evidence
- visual examples

### `test/`

This contains simple early tests and sanity scripts.

---

## Full Tech Stack

## Python

Python is the main programming language of the repository. That sounds obvious, but in a project like this the language choice has architectural consequences, so it is worth explaining properly.

### What Python is

Python is a high-level programming language designed for readability, fast development, and strong library support. It is interpreted rather than compiled in the same way as C or C++, which makes it easier to prototype ideas quickly.

In machine learning and computer vision, Python became the default language because most major libraries provide first-class Python interfaces:

- OpenCV
- NumPy
- PyTorch
- MediaPipe
- Ultralytics YOLO
- SciPy
- Flask

### How Python works in this project

Python is the orchestration layer. That means it is not responsible for doing every heavy operation itself. Instead, it coordinates specialized lower-level libraries that do the heavy lifting.

In Kinemation, Python code:

- opens files and videos
- calls OpenCV functions for image operations
- calls Ultralytics APIs for YOLO inference
- calls MediaPipe APIs for pose landmarking
- calls PyTorch for VideoPose3D inference
- stores and transforms keypoints using NumPy
- runs tracking and smoothing logic
- serves the interface through Flask

So when you say "the project is written in Python," the accurate deeper meaning is:

Python is the control layer that wires the whole pipeline together.

### Why Python was chosen

Python is ideal when the project needs:

- rapid iteration
- experimentation
- readable code for a student or research workflow
- integration of many AI libraries

That is exactly the situation in this repo. The folder structure itself shows heavy experimentation, multiple prototypes, several branches, and a growing integration story. Python is the natural fit for that environment.

### Pros

- very fast development speed
- huge AI and CV ecosystem
- excellent for experimentation and teaching
- easy to combine many frameworks in one runtime

### Cons

- slower than compiled languages in pure-Python loops
- can become messy if project structure grows without cleanup
- deployment at scale may require more optimization work

### Meeting-ready explanation

Python was used because this project is a multi-library AI pipeline. The main engineering challenge was integration, experimentation, and iteration speed, and Python is the strongest practical language for that kind of work.

## OpenCV

OpenCV is the media-processing backbone of the project.

### What OpenCV is

OpenCV stands for Open Source Computer Vision Library. It is a large library of image-processing and computer-vision tools. In most practical CV systems, OpenCV is the toolbox that handles the media itself: reading it, writing it, converting it, resizing it, and drawing on it.

You can think of OpenCV as the "eyes and hands" infrastructure around the AI models:

- the models do inference
- OpenCV moves and edits the images those models need

### How OpenCV represents data

An image is stored as a matrix of pixels.

For a color image:

- height = number of rows
- width = number of columns
- channels = usually 3

OpenCV uses BGR channel order by default, not RGB.

That matters because:

- OpenCV reads images in BGR
- MediaPipe expects RGB
- converting between them correctly is essential

A video is handled as a sequence of such image frames.

### How OpenCV works in Kinemation

OpenCV is used in almost every stage:

- `cv2.VideoCapture` opens input videos
- `cap.read()` pulls frames one by one
- `cv2.resize` scales frames
- `cv2.GaussianBlur` reduces noise
- `cv2.cvtColor` converts between BGR, RGB, and LAB
- `cv2.createCLAHE` enhances local contrast
- cropping is done through array slicing on OpenCV images
- `cv2.line`, `cv2.circle`, and `cv2.putText` draw skeletons and labels
- `cv2.VideoWriter` saves output videos

It is also part of earlier OpenPose-oriented workflows in the repo, especially where blobs and DNN-style inputs are discussed in the reading material.

### Why OpenCV is necessary even with AI models

A common beginner mistake is to think that the "real work" is only in the ML models. In practice, raw images are rarely in the right shape or color format for direct use.

OpenCV handles:

- file access
- frame extraction
- preprocessing
- coordinate-space transformations
- visual debugging

Without that, the models cannot be used effectively.

### CLAHE in OpenCV

CLAHE appears repeatedly in this repo, so it deserves a specific explanation.

CLAHE stands for Contrast Limited Adaptive Histogram Equalization.

What it is:

- a contrast enhancement method

How it works:

1. the image is split into small tiles rather than treated as one global brightness distribution
2. contrast is enhanced inside each tile
3. clipping prevents noise from being over-amplified
4. tiles are blended together

Why use it:

- it improves local contrast in difficult lighting
- it is more stable than naive histogram equalization

Why LAB space is used:

- LAB separates lightness from color
- the code enhances only the lightness channel
- this avoids changing colors too aggressively

### Gaussian blur in OpenCV

Gaussian blur appears in the preprocessing functions.

What it is:

- a smoothing filter based on a Gaussian bell-curve kernel

How it works:

- each pixel is replaced by a weighted average of nearby pixels
- nearby pixels contribute more than far pixels

Why use it:

- suppresses noise
- can reduce small visual artifacts before detection or landmarking

Tradeoff:

- too much blur can erase useful detail

### Pros

- fast and mature
- industry standard
- excellent image/video I/O
- works naturally with NumPy

### Cons

- low-level enough that mistakes in coordinate systems, channel ordering, and resizing logic can easily create bugs

### Meeting-ready explanation

OpenCV is the infrastructure layer of Kinemation. It handles reading videos, preprocessing frames, converting color spaces, drawing skeletons, and writing outputs. The ML models depend on OpenCV to turn raw media into model-ready inputs and model outputs into viewable results.

## NumPy

NumPy is the numerical array layer used almost everywhere in the repo.

### What NumPy is

NumPy is a numerical computing library centered around arrays. It gives Python efficient matrix and vector operations, which is why almost every scientific and ML pipeline depends on it.

### How it works

Instead of storing data in plain Python lists, NumPy stores them in structured arrays. Those arrays support:

- fixed shapes
- efficient memory layout
- fast element-wise operations
- slicing and broadcasting

In pose estimation, this is extremely important because the project is constantly dealing with data shaped like:

- `(33, 2)` for one person’s 2D landmarks
- `(17, 2)` for converted 2D keypoints
- `(17, 3)` for one 3D skeleton
- `(N, P, 17, 3)` for a full multi-frame, multi-person 3D sequence

### How it works in Kinemation

NumPy is used to:

- store detected landmarks
- create empty tensors for missing detections
- convert between coordinate conventions
- compute box extents
- compute averages for derived joints
- smooth or interpolate tracks
- stack frame data into sequences

### Why it matters

Without NumPy, the repo would have to manipulate nested Python lists everywhere. That would be slower, harder to read, and much more error-prone.

### Meeting-ready explanation

NumPy is the data structure layer of the system. It stores the skeletons, the frame sequences, and the coordinate arrays that move through every stage of the pipeline.

## MediaPipe

MediaPipe is Google's perception framework. In this project it is used mainly for 2D body landmark estimation.

### What MediaPipe is

MediaPipe is a framework for real-time perception pipelines. It includes ready-made models and APIs for tasks like:

- face detection
- hand tracking
- pose estimation
- gesture analysis

In Kinemation, the important part is MediaPipe Pose.

### What MediaPipe Pose does

MediaPipe Pose predicts 33 body landmarks. These landmarks include:

- nose
- eyes
- ears
- shoulders
- elbows
- wrists
- hips
- knees
- ankles
- heel and foot points
- some hand-adjacent landmarks

So MediaPipe gives a richer body description than a minimal 17-joint format.

### How MediaPipe Pose works conceptually

At a high level:

1. it receives an image of a person
2. it runs a learned pose model
3. it outputs body landmark positions in normalized coordinates

Those normalized coordinates are typically in a range relative to the crop, not the original full-frame pixel space. That is why the repo remaps them after inference.

### Why MediaPipe is used in this repo

MediaPipe is well suited when:

- you want a reasonably detailed body landmark set
- you want practical runtime speed
- you are working in Python
- you can feed it a cleaner crop instead of a full crowded frame

That last point is the reason it is paired with YOLO in the final stack.

### Why not run MediaPipe directly on the whole frame?

Because in crowded or complex scenes, isolating each person first usually helps.

If you first detect each person and crop them:

- the landmarking model sees a more focused image
- overlapping people are less confusing
- multi-person support becomes easier to structure

### Output format

MediaPipe returns landmark coordinates relative to the crop.

That means:

- `x` and `y` are normalized
- the repo multiplies them by crop width and crop height
- then adds the crop’s top-left corner offset

This remaps them into full-frame pixel coordinates.

### Pros

- practical and fast
- detailed landmark set
- easy Python integration

### Cons

- not designed to solve full 3D temporal motion reconstruction by itself
- crop quality matters
- small or low-detail crops can reduce accuracy

### Meeting-ready explanation

MediaPipe is the body-landmark estimator in the final pipeline. YOLO first finds each person, then MediaPipe estimates 33 body landmarks for each detected crop. Those landmarks are later converted into the 17-joint format needed by the 3D stage.

## MediaPipe Tasks API

The repo uses the Tasks API with `.task` model files.

### What the Tasks API is

MediaPipe used to be taught mostly through older solution APIs. Newer versions rely more on the Tasks API, where you explicitly load a model file and create a configured task object.

### How it works

The process in the repo is:

1. create `BaseOptions` with a model path
2. create `PoseLandmarkerOptions`
3. choose a running mode
4. create the landmarker object
5. wrap image data in an `mp.Image`
6. call `.detect(...)`

### Why the `.task` file matters

The `.task` file contains the packaged model asset needed for the Tasks API. In this repo that file is:

- `pose_landmarker_lite.task`

### Why `RunningMode.IMAGE` matters

The final code uses `RunningMode.IMAGE`.

What that means:

- each crop is processed independently
- MediaPipe itself is not being asked to do temporal sequence reasoning in the final path

That is important because temporal logic is handled later through:

- tracking
- smoothing
- VideoPose3D temporal lifting

### Why the lite model matters

The repo uses the lite variant because it is smaller and faster. That makes sense in a pipeline that already also runs:

- YOLO
- VideoPose3D
- smoothing and rendering logic

Tradeoff:

- lighter models are usually faster but may sacrifice some accuracy compared to heavier variants

### Meeting-ready explanation

The final pipeline uses MediaPipe’s newer Tasks API with the `pose_landmarker_lite.task` model and `RunningMode.IMAGE`. That means MediaPipe is used as a per-crop landmark detector, while temporal consistency is handled later by the project’s own tracking, smoothing, and 3D lifting stages.

## YOLO

YOLO is the person detector in the final pipeline.

### What YOLO is

YOLO stands for "You Only Look Once." It is a family of object detectors designed to find objects in an image in a single forward pass.

### What object detection means

Object detection answers:

- what objects are present?
- where are they?

For Kinemation, the relevant object class is:

- person

### How YOLO works conceptually

The model processes the image once and predicts:

- bounding boxes
- class labels
- confidence scores

The project then filters those predictions so it keeps only:

- person detections
- above-threshold confidence

### Why YOLO is useful here

The project needs to know where each person is before estimating detailed body landmarks. YOLO handles that localization problem efficiently.

### Why YOLO comes before MediaPipe

Because the final architecture is:

1. detect each person
2. crop each person
3. run landmarking on the crop

This is more robust than asking the landmark model to interpret the entire multi-person scene by itself.

### Meeting-ready explanation

YOLO is the person detector in the final stack. It finds where each person is in the frame so that MediaPipe can focus on each person individually.

## YOLOv8n

`yolov8n.pt` is the final detector used in the working backend.

### What `yolov8n` means

The `n` stands for nano. It is the smallest YOLOv8 family variant among the common choices.

### How that choice affects the system

A smaller detector means:

- less computation
- faster inference
- easier CPU usage
- usually lower peak accuracy than larger variants

### Why it was a sensible choice here

The final pipeline is not only a detector. It also runs:

- MediaPipe landmarking for each person
- VideoPose3D temporal lifting
- tracking
- smoothing
- rendering

So detector speed matters. A heavier detector might help in some hard scenes, but it would also increase total pipeline cost.

### Meeting-ready explanation

`yolov8n` was chosen because it is lightweight and practical. In a multi-stage pipeline, using the smallest reasonable detector helps keep the whole system usable.

## YOLO11 Pose

The older branch also explored direct YOLO pose output.

### What YOLO pose means

A pose model does more than detection. It predicts body keypoints directly, often in a COCO-style 17-keypoint format.

### Why that is attractive

It simplifies the pipeline:

- one model can both detect and estimate pose

### Why it mattered in this repo

The `vidpose-amrita/` branch and related optimization notes show that the team explored direct pose-model output paths as part of the project’s evolution.

### Why it is not the active final path

The final code converged on a more modular approach:

- YOLO for detection
- MediaPipe for landmarking

That likely gave the team better control over:

- multi-person isolation
- landmark detail
- integration with the chosen 3D lifting path

### Meeting-ready explanation

Direct YOLO pose output was explored because it makes the pipeline simpler, but the final working implementation moved to a detector-plus-landmarker design that better matched the project’s needs.

## OpenPose

OpenPose is not part of the current active final runtime, but it is part of the project history and research base.

Evidence in repo:

- `pose_deploy_linevec.prototxt`
- `pose_iter_440000.caffemodel`
- `OpenPose-Multi-Person.zip`
- `Reading Material/OpenCV_MediaPipe_OpenPose_Guide.md`

### What OpenPose is

OpenPose is a landmark pose-estimation system originally associated with CMU. It became famous for multi-person pose estimation and for using heatmaps plus part affinity fields.

### How OpenPose works conceptually

At a simplified level:

1. the model predicts heatmaps for body joints
2. it predicts affinity fields that describe how joints belong together
3. a post-processing step assembles joints into full skeletons

This is why OpenPose is often described as a more structured classical multi-person pose approach.

### How it appears in this repo

The repo includes:

- Caffe architecture and weight files
- an OpenPose zip archive
- reading material about OpenCV, MediaPipe, and OpenPose

So even though OpenPose is not the active final runtime, it is clearly part of the project’s technical history.

### Why it matters

- it shows the team studied older, influential pose methods before converging on the current implementation
- it provides historical context for the project’s learning curve

### Why it was not the final active path

The repo ultimately moved toward:

- YOLO for person detection
- MediaPipe for 2D body landmarks
- VideoPose3D for temporal 3D lifting

That stack appears to have been the more practical integration path for the working final code.

### Meeting-ready explanation

OpenPose was part of the early exploration and theoretical foundation of the project, but the final active backend moved to a different architecture that was easier to integrate into the full multi-stage pipeline.

## PyTorch

PyTorch is used for the VideoPose3D model.

### What PyTorch is

PyTorch is a deep-learning framework used to build and run neural networks. It provides tensors, model layers, automatic differentiation, and model-loading utilities.

### How it works here

In this repo, PyTorch is not used as a general training framework for everything. Its main role is:

- loading the VideoPose3D network
- loading the pretrained checkpoint
- running inference on 2D pose sequences

### Why it matters

Without PyTorch, the temporal 3D lifting stage would not run, because VideoPose3D is implemented through PyTorch modules.

### Meeting-ready explanation

PyTorch is the neural-network runtime behind the 3D lifting stage. The repo uses it mainly to load and run VideoPose3D.

## VideoPose3D

VideoPose3D is the core 3D lifting model used in the final pipeline.

### What VideoPose3D is

VideoPose3D is a temporal 3D human pose estimation method. It does not directly read raw RGB and output 3D bodies from scratch in the way an end-to-end image model might. Instead, it works as a lifting model:

- input: 2D keypoint trajectories over time
- output: 3D joint trajectories

### What "lifting" means

Lifting means converting lower-dimensional pose data into higher-dimensional pose data.

In this repo:

- 2D joints are the input
- 3D joints are the output

### How VideoPose3D works conceptually

The key idea is that one frame is ambiguous, but motion over time is informative.

For example:

- if an arm appears short in one frame, that could mean it is bent or it is pointing toward the camera
- by watching nearby frames, the model can infer which interpretation is more likely

That is why temporal context matters.

### How it works in the repo

The repo uses:

- a temporal model class from the bundled VideoPose3D code
- a pretrained checkpoint
- a receptive field of 243 frames

The pipeline:

1. converts 2D landmarks to the expected joint convention
2. normalizes them
3. feeds them through the temporal model
4. receives 3D joint coordinates

### Why VideoPose3D was chosen

It is a strong fit when:

- you already have a working 2D pose estimator
- you want to add temporal reasoning
- you want a proven research model without rebuilding the whole system around a different architecture

### Pros

- uses time, not just single frames
- fits cleanly into a 2D-to-3D lifting pipeline
- proven and understandable

### Cons

- depends strongly on 2D input quality
- output interpretation is non-trivial
- wrong keypoint semantics or normalization can break results

### Meeting-ready explanation

VideoPose3D is the temporal lifting model. It takes the 2D skeleton motion over time and predicts a 3D skeleton, using long temporal context to resolve ambiguities that a single frame cannot resolve.

## MotionBERT

MotionBERT is present as a researched alternative, not the active implementation.

### What MotionBERT is

MotionBERT is a more modern motion-representation approach based on transformer-style learning and pretraining across motion tasks.

### Why it appears in this repo

The repo contains:

- MotionBERT reference material
- comparison writeups against VideoPose3D

That means the team was evaluating future-looking alternatives rather than assuming the current model was the only possible choice.

### Why it is not the active final implementation

There is no MotionBERT runtime path in `final/`. The working integrated backend is built on VideoPose3D.

### Why it is still important

It gives you a strong answer if someone asks:

- "Did you evaluate modern alternatives?"
- "Why this 3D model and not a transformer-based one?"

### Meeting-ready explanation

MotionBERT is a researched alternative and a possible future direction, but it was not the implemented working backend in this repository.

## SciPy

SciPy is used for:

- Hungarian matching via `linear_sum_assignment`
- Gaussian smoothing via `gaussian_filter1d`

### What SciPy is

SciPy is a scientific computing library that builds on NumPy and provides higher-level numerical algorithms.

### Why it matters here

The repo does not use all of SciPy. It uses specific tools that solve concrete engineering problems:

- assignment optimization
- temporal filtering

That is a good example of pragmatic library usage: only bring in the numerical tools that solve a clear problem.

## Flask

Flask is used in `project/app.py` to expose the backend through a web interface.

It handles:

- uploads
- webcam recordings
- job status polling
- downloads

### What Flask is

Flask is a lightweight Python web framework. It makes it easy to define routes, receive requests, render templates, and return responses.

### How it works here

The app:

- serves HTML pages for upload and webcam use
- receives posted files
- starts backend processing threads
- exposes status routes for polling
- lets the user download generated outputs

### Why Flask was a sensible choice

The project did not need a huge enterprise web framework. It needed a simple Python-friendly wrapper around an ML backend.

Flask is ideal for that.

## Hungarian Algorithm

This is one of the most important non-neural parts of the project.

### What it is

The Hungarian algorithm is an optimization algorithm for assignment problems. If you have a set of existing tracks and a set of current detections, it finds the best one-to-one matching that minimizes total cost.

### How it works conceptually

Imagine you have:

- 3 tracked people from the previous frame
- 3 detected people in the current frame

You can compute a cost for every possible match. In this repo, that cost is based on overlap:

- high overlap means low cost
- low overlap means high cost

The Hungarian algorithm then chooses the globally best assignment, not just a greedy local one.

That distinction matters. A greedy match can make one good early decision that causes worse later assignments. The Hungarian algorithm solves the full assignment problem more systematically.

### How it works in Kinemation

The pipeline:

1. computes a cost matrix using `1 - IoU`
2. passes it to `linear_sum_assignment`
3. gets optimal row/column matches
4. still rejects matches that fail the IoU threshold gate

### Why it matters

- person colors and IDs should stay stable across frames
- without a principled assignment step, the system can swap identities and flicker

## IoU

IoU means Intersection over Union.

### What it is

IoU is a number that measures how much two boxes overlap.

The formula is:

- overlap area divided by union area

If two boxes are identical:

- IoU = 1

If they do not overlap at all:

- IoU = 0

### How it works in tracking

The pipeline compares:

- a bounding box derived from a previous tracked person
- a bounding box derived from a current detection

If the overlap is high, they are probably the same person.

### Why IoU is a good choice here

It is:

- simple
- interpretable
- easy to compute
- often good enough for short-term frame-to-frame matching

### Limitation

IoU is geometry-only. It does not know anything about:

- clothing
- face identity
- long-term reappearance after occlusion

## Gaussian Smoothing

Gaussian smoothing reduces frame-to-frame jitter.

### What it is

Gaussian smoothing is a filtering method that replaces each point in a trajectory with a weighted average of nearby points, using a Gaussian bell-curve weighting.

### How it works

For one joint over time:

- nearby frames contribute more
- distant frames contribute less

This keeps the motion trend while suppressing high-frequency noise.

### Why used here

- raw 2D and 3D predictions are noisy
- humans move continuously, so trajectories should usually be smooth

### Why it appears in both 2D and 3D stages

2D smoothing:

- gives the 3D model cleaner input

3D smoothing:

- makes the final lifted skeleton more stable

### Limitation

Too much smoothing can oversoften fast motion, so the smoothing strength must be chosen carefully.

## Bone Constraints

The final backend also includes bone-length consistency logic.

### What they are

Bone constraints are post-processing rules that try to keep skeleton structure anatomically plausible.

### Why they are needed

A pose model may output slightly different effective limb lengths from frame to frame because of noise.

But a real human body does not suddenly gain a longer forearm for one frame and shrink it in the next.

### How they work here

The repo includes functions that:

- compute bone lengths
- compare symmetric bones
- enforce more stable lengths over time

### Why this helps

- improves visual stability
- reduces anatomically unrealistic fluctuations
- makes the final output look more coherent

## Matplotlib and `Axes3D`

Matplotlib appears in the project as part of true-3D visualization planning and the bundled VideoPose3D visualization utilities.

### What it is

Matplotlib is a plotting library. `Axes3D` is its 3D plotting interface.

### Why it matters here

The project discovered that a 3D pose estimate and a convincing 3D rendering are not the same thing.

Flat projection can make limbs and hips look wrong even when the underlying pose is better than it appears.

### How it would help

With true 3D plotting, you can:

- set camera elevation and azimuth
- render perspective
- use a grid or floor plane
- show front/back depth more naturally

That is why `planv6` focuses on true-3D rendering improvements.

## Human3.6M and COCO Conventions

These are body-joint conventions and datasets, not general-purpose libraries, but they are core parts of the technical stack.

### What they are

These are standard ways of representing body joints.

COCO:

- usually used as a 2D keypoint convention
- commonly 17 keypoints

Human3.6M:

- commonly used in 3D pose-estimation research
- uses a 17-joint skeleton convention

### Why they matter

Different frameworks do not all speak the same body-language format.

For example:

- MediaPipe gives 33 landmarks
- a YOLO pose model may output COCO-style 17 keypoints
- VideoPose3D output is interpreted in an H36M-style joint structure

That means conversion is not optional. It is a core integration task.

---

## Concept Primer

## 2D Pose Estimation

2D pose estimation means finding body joints in the image plane. Each joint gets an `(x, y)` location.

### What it is

It is the problem of locating body joints on a flat image.

Examples of such joints:

- nose
- shoulders
- elbows
- wrists
- hips
- knees
- ankles

### How it works conceptually

A pose model analyzes the image and predicts where each joint likely is. Different architectures do this differently:

- heatmap-based methods
- direct coordinate regression
- detector-plus-keypoint methods

In Kinemation’s final path, 2D pose estimation is achieved by:

- detecting a person with YOLO
- estimating landmarks on the crop with MediaPipe

### Why it matters

The entire 3D stage depends on this step. If the 2D pose is poor, the 3D lift will inherit those errors.

### Important limit

2D pose has no true depth. It knows where a joint appears on the image, not how far forward or backward it is in space.

## 3D Pose Estimation

3D pose estimation adds a depth dimension. Each joint becomes `(x, y, z)`.

### What it is

It is the task of reconstructing body pose in three-dimensional space rather than only on the flat image plane.

### Why it is difficult

The same 2D image can often come from multiple possible 3D poses. This is called depth ambiguity.

### How it works here

Kinemation uses a two-step design:

1. estimate 2D keypoints
2. lift them to 3D with VideoPose3D

### Why temporal information helps

Watching motion over time helps resolve ambiguities that are impossible to solve from one frame alone.

## Detection vs Landmarking

These are different tasks.

- detection answers: where is the person?
- landmarking answers: where are the joints?

In the final stack:

- YOLO performs detection
- MediaPipe performs landmarking

### Detection

Detection finds where the person is and returns a box.

### Landmarking

Landmarking finds detailed body points inside that box.

### Why splitting them helps

The detector first isolates each person, which gives the landmarker a cleaner input and improves multi-person handling.

## Multi-Person Processing

A multi-person system must:

- detect each person
- estimate a skeleton for each
- keep each identity stable over time

This last step is what tracking solves.

### Why this is hard

Multiple people can:

- overlap
- cross
- leave and re-enter
- change size based on distance from the camera

So multi-person processing is more than just running single-person pose estimation multiple times. It also needs identity management.

## Tracking

Tracking in this repo is practical geometry-based tracking.

It uses:

- keypoint-derived boxes
- IoU overlap
- Hungarian matching

This is simpler than full appearance-based re-identification, but it is transparent and useful.

### What it answers

Tracking answers:

"Which current detection corresponds to which previous person?"

### Why it matters

Without tracking:

- person colors can swap
- one person’s motion can jump between slots
- 3D sequence quality becomes worse

## Temporal Smoothing

Smoothing reduces jitter in both 2D and 3D trajectories.

The basic intuition:

- neighboring frames should usually look similar
- noise should not dominate the motion path

### How it works conceptually

A joint should move along a continuous path, not teleport randomly. Smoothing uses neighboring frames to reduce short-term noise while keeping the main motion pattern.

## Receptive Field

VideoPose3D reasons across time using a large receptive field.

In the final code:

- `RECEPTIVE_FIELD = 243`

Meaning:

- one prediction is informed by a wide temporal window rather than a single frame

### Why it matters

A wide receptive field lets the model understand motion context instead of guessing 3D pose from an isolated frame.

## Normalization

2D keypoints are normalized before entering the 3D model.

Why:

- models are easier to train and run when coordinates are standardized
- normalized inputs are less tied to one specific video resolution

### How it works

The raw pixel coordinates are transformed into a standardized coordinate system based on frame width and aspect ratio. This makes the numerical representation of pose more consistent across different resolutions.

## Root-Relative 3D

A major lesson from the optimization notes is that VideoPose3D outputs are root-relative.

Meaning:

- output joints are positioned relative to a root joint
- they are not raw image coordinates

Why this matters:

- you cannot visualize them by pretending they are ordinary 2D screen points
- bad projection logic can create apparently broken skeletons

### Why models use root-relative output

Root-relative output focuses the model on body shape and articulation rather than absolute camera-space translation.

## Projection vs Visualization

These are not the same thing.

The model may predict a reasonable 3D pose, but the drawing method can still make it look wrong.

That is why the repo spent so much time on:

- anchored projection
- axis interpretation
- true-3D rendering discussions

### Projection

Projection is the geometric act of mapping 3D points into a 2D view.

### Visualization

Visualization is the broader presentation layer: colors, perspective style, layout, and rendering choices.

### Why the distinction matters

You can have:

- correct model output but bad projection
- correct projection but poor visual presentation

Kinemation’s optimization history shows that both issues mattered.

---

## System Evolution and Timeline

## Stage 1: Basic Image and Video Processing

Evidence:

- `kinemation/projekt/img_processing.py`
- `kinemation/projekt/vid_processing.py`
- `test/test1.py`

What this stage did:

- learned how to load images and videos
- resize frames
- blur frames
- draw simple stick figures

Why it mattered:

- it established the OpenCV-first foundation for the whole repo

## Stage 2: OpenPose-Oriented Study

Evidence:

- `models/pose_deploy_linevec.prototxt`
- `models/pose_iter_440000.caffemodel`
- `OpenPose-Multi-Person.zip`
- `Reading Material/OpenCV_MediaPipe_OpenPose_Guide.md`
- `Reading Material/IMAGE_PROCESSING_WORKFLOW.md`

What happened:

- the team studied classic OpenPose workflows and body-joint extraction concepts

Why important:

- this provided the early theoretical framework for pose estimation
- it explains why OpenPose files still exist in the repo

## Stage 3: MediaPipe-Based 2D Prototype

Evidence:

- `prototypes/prototype3.py`

What happened:

- MediaPipe PoseLandmarker was used to create 2D stick-figure outputs from video

What this proved:

- 2D landmark extraction was working
- OpenCV plus MediaPipe was already a viable path

## Stage 4: YOLO + MediaPipe Multi-Person Upgrade

Evidence:

- `prototypes/prototype4.py`
- `Reading Material/mediapipe_pose_estimation.md`

What changed:

- YOLOv8 was added to detect each person
- MediaPipe then ran on the crop for each person

Why this was a major upgrade:

- it made the system properly multi-person
- it separated detection from landmarking

## Stage 5: Visual Refinement

Evidence:

- `prototypes/prototype5.py`

What changed:

- skeleton styling improved
- head-circle stick-figure rendering was added

Why it mattered:

- better visualization helps both demos and debugging

## Stage 6: Older 3D Branch in `vidpose-amrita/`

Evidence:

- `vidpose-amrita/kinemation/backend/main.py`
- `optimizing/plan.md`
- `optimizing/workflow.md`

What this branch represented:

- an older integrated 3D approach
- a strong comparison point for later optimization

Important learning:

- this branch exposed issues like aspect-ratio distortion and coordinate interpretation problems

## Stage 7: Consolidated Final Pipeline

Evidence:

- `final/pipeline_3d.py`
- `final/Working pipeline.md`

What happened:

- the stronger 2D stack from the prototype line was combined with VideoPose3D-based 3D lifting
- the current four-phase architecture was formalized

## Stage 8: Intensive Optimization

Evidence:

- `optimizing/plan.md`
- `optimizing/planv2.md`
- `optimizing/planv3.md`
- `optimizing/planv4.md`
- `optimizing/planv5.md`
- `optimizing/planv6.md`
- `workflow_improved.md`
- `optimizing/workflow.md`

What happened:

- geometry bugs were investigated
- projection and visualization were revised
- tracking and smoothing were improved
- backup snapshots were created during rapid iteration

## Stage 9: App Integration

Evidence:

- `project/app.py`
- `project/backend/main.py`
- `project/frontend-resources/templates/*.html`

What happened:

- the backend became accessible through upload and webcam pages
- background processing and job tracking were added

## Timeline Anchors

Evidence in the repo gives strong date anchors:

- `samples/final-outputs/README.md` marks output generation on `2026-04-03`
- `optimizing/planv6.md` is dated `2026-04-03`
- backup folders such as `final_v3_20260403_004947`, `final_v5_20260403_104023`, and `final_v6_20260404_091407` show rapid iteration on April 3 and April 4, 2026

The safest high-level timeline statement is:

- early work focused on image processing and pose-estimation fundamentals
- the project then matured into multi-person 2D pipelines
- older 3D branches were evaluated and compared
- the final working backend was consolidated and heavily optimized in early April 2026

---

## Current Final Pipeline

This is the most important section in the document.

The `final/pipeline_3d.py` backend is the current working implementation. The cleanest way to explain it is as four phases.

### High-level data flow

The final backend can be described as:

`video -> frame preprocessing -> person detection -> per-person landmarking -> identity tracking -> 2D conversion -> 2D smoothing -> 3D lifting -> 3D smoothing -> rendering`

That full chain matters because every later stage depends on the output quality of the earlier stage.

## Phase 1: Preprocessing and 2D Pose Extraction

Key functions:

- `apply_clahe`
- `preprocess_frame`
- `detect_persons`
- `estimate_pose_mediapipe`

### Step 1: Open the video and inspect metadata

The pipeline reads:

- frame count
- FPS
- width
- height

This is needed for:

- progress reporting
- output video creation
- normalization and rendering logic

This matters because the pipeline is temporal. It cannot just process one frame in isolation and forget the overall sequence structure.

### Step 2: Preprocess each frame

The final backend preprocesses each frame with:

- resize to `max_dim=800`
- CLAHE in LAB color space
- Gaussian blur

Why resize:

- lowers compute cost

Why CLAHE:

- improves contrast in difficult lighting

Why blur:

- reduces noise and small artifacts

Important engineering tradeoff:

The optimization docs point out that aggressive downscaling can hurt detail for small subjects. So this stage is a quality-vs-speed compromise, not an automatic win.

### How preprocessing actually changes the image

Resize:

- reduces pixel count
- makes model inference cheaper

CLAHE:

- changes local brightness contrast
- helps reveal body regions in darker or flatter parts of the frame

Gaussian blur:

- slightly smooths the frame
- can reduce small noise artifacts that do not help detection

The reason all three are here is that the team was trying to give the downstream detectors cleaner inputs, not just raw camera frames.

### Step 3: Detect persons with YOLOv8n

YOLO runs on the processed frame.

Logic:

- only class `0` is kept
- only sufficiently confident boxes are kept

Output:

- a list of person bounding boxes

### How the detector output is used

The boxes are not the final goal. They are intermediate structure.

Their job is to define:

- where each person is
- what crop should be sent to MediaPipe
- how multi-person processing will be organized

So in the final architecture, the detector is the localization stage, not the final pose stage.

### Step 4: Run MediaPipe on each detected person

For each bounding box:

- pad the box slightly
- crop the person region
- convert the crop to RGB
- wrap it as an `mp.Image`
- run PoseLandmarker
- convert normalized crop-relative landmarks back to full-frame coordinates

Output:

- `(33, 2)` landmark array per detected person

At the end of phase 1, the system has raw 2D landmarks for each frame, but identity is not yet stable over time.

### Why the coordinate remapping step matters

MediaPipe outputs crop-relative coordinates. But the rest of the pipeline wants all people described in the coordinate space of the full processed frame.

So the code:

- scales normalized landmark coordinates by crop size
- offsets them by the crop origin

Without that step, each person would exist in a different local coordinate system and could not be compared or tracked correctly.

## Phase 2: Batch Tracking

Key functions:

- `get_bbox_from_keypoints`
- `compute_iou`
- `batch_track_people`
- `filter_short_tracks`

### Why this stage exists

If the system simply stores detections in the order they appear each frame, person identity can swap constantly. That ruins:

- color consistency
- motion continuity
- multi-person 3D lifting quality

This is a key meeting point:

pose estimation and tracking are different problems. Even if the detector and landmarker are perfect, the system still needs a mechanism to say which person is which over time.

### How it works

1. derive a bounding box from each person’s keypoints
2. compare current detections to previous tracked boxes
3. compute IoU overlap
4. build a cost matrix using `1 - IoU`
5. run Hungarian assignment
6. reject poor matches using an IoU threshold
7. assign leftover detections to empty slots

### Why this is called batch tracking in the docs

The docs describe this stage as a batch process because the pipeline first extracts all raw 2D poses for the video and then performs identity organization over the sequence, rather than fully committing to final identities during the first frame-processing pass.

### Short-track filtering

After tracking, very short runs are removed because they are likely false positives.

This is a practical quality-control step.

Why it helps:

- removes tiny spurious detection bursts
- prevents noise tracks from being lifted to 3D and rendered as if they were real people

## Phase 3: Keypoint Conversion, Smoothing, and 3D Lifting

Key files and functions:

- `final/mediapipe_to_h36m.py`
- `smooth_all_tracks`
- `normalize_screen_coordinates`
- `VideoPose3DLifter`
- `smooth_all_3d_tracks`
- `enforce_all_bone_constraints`

### Step 1: Convert MediaPipe landmarks into model-compatible 2D joints

MediaPipe outputs 33 landmarks, but the 3D model path expects a 17-joint input convention.

The adapter file handles that semantic bridge.

Important point for the meeting:

This is not just "shrinking 33 to 17." It is a meaning-preserving conversion between different body-joint conventions.

### How conversion works conceptually

The conversion file contains explicit index mapping. That means the project is saying:

- MediaPipe landmark index X corresponds to semantic joint Y
- semantic joint Y should be placed into model input slot Z

This is one of the most failure-sensitive parts of the system, because a wrong mapping can create structurally wrong 3D outputs even if the detector and landmarker are working correctly.

### Step 2: Smooth the 2D tracks

The 2D trajectories are smoothed before 3D lifting.

Why:

- frame-level noise hurts temporal lifting
- smoother 2D input gives cleaner temporal patterns

### Why smoothing before lifting is important

The 3D model sees a motion sequence, not just individual frames.

If the 2D input contains random zig-zag noise, the model may interpret that noise as meaningful motion. Smoothing reduces that risk.

### Step 3: Normalize the 2D coordinates

Coordinates are normalized using frame dimensions before being fed into the model.

Why:

- standardizes input range
- makes inference less dependent on raw image size

### What the normalization function is really doing

It transforms pose coordinates from raw pixel space into a standardized coordinate system centered and scaled relative to the frame size.

The important conceptual point is:

- the model should see the shape of the pose, not arbitrary pixel magnitudes tied to one resolution

### Step 4: Lift 2D pose into 3D with VideoPose3D

The `VideoPose3DLifter` class:

- loads pretrained weights
- creates the temporal model
- performs sequence lifting

Important facts:

- receptive field is 243 frames
- the model uses temporal context rather than single-frame inference

### How temporal lifting works in practice

The model takes a window of 2D joint positions across time and predicts the 3D pose for the relevant frames. Because the model has context from neighboring frames, it can infer depth more plausibly than a purely single-frame method.

### Step 5: Smooth the 3D trajectories

The final backend includes:

- joint-adaptive smoothing
- velocity limiting

Why:

- not all joints move the same way
- wrists and ankles can move quickly
- torso joints should be more stable
- unrealistic spikes are often jitter rather than real motion

### Why joint-adaptive smoothing is smarter than one global filter

If one smoothing value is used for everything:

- torso may still jitter too much
- wrists may become too sluggish

So using different assumptions for different joints is more physically sensible.

### Step 6: Enforce bone constraints

The backend also enforces consistency in:

- bone lengths
- symmetry

Why:

- raw output can wobble in anatomically unrealistic ways

### How constraints help visually

Even when the model output is numerically plausible, small frame-to-frame variations in limb length can make the skeleton look unstable or rubber-like. Bone constraints reduce that effect.

## Phase 4: Rendering

Key functions:

- `draw_skeleton_2d`
- `project_3d_to_2d_anchored`
- `draw_skeleton_3d`
- `render_frame`

### What gets rendered

Depending on mode, the backend can render:

- 2D skeletons
- 3D skeletons
- side-by-side 2D and 3D outputs

### What rendering is doing mathematically

The render stage takes model-space or pose-space representations and turns them into visible lines and points on an output frame.

That includes:

- deciding how 3D joints should be projected into 2D output space
- deciding colors and line structure
- composing the final display layout

### Why side-by-side is useful

It helps compare:

- what the 2D detector saw
- what the 3D lifter inferred

That is extremely useful for debugging and presentation.

It also helps diagnose where a failure comes from:

- if 2D is already wrong, the problem started early
- if 2D is good but 3D is strange, the problem is later in the pipeline
- if both seem right but the rendering is odd, the visualization layer may be responsible

---

## Code Walkthrough

## `final/pipeline_3d.py`

This is the orchestration file for the current backend.

### What it contains

- preprocessing logic
- detection logic
- MediaPipe inference logic
- tracking logic
- 2D smoothing
- 3D smoothing
- bone constraint enforcement
- VideoPose3D loading and inference
- rendering
- CLI usage

### Important constants

- `MAX_PEOPLE = 6`
- `RECEPTIVE_FIELD = 243`
- per-joint 3D smoothing and velocity settings
- H36M bone and symmetry definitions

These constants matter because they show that the pipeline is tuned, not generic.

They are also evidence that the final backend is not a bare proof of concept. It includes practical runtime and quality-control tuning.

### Preprocessing

`apply_clahe`:

- converts BGR to LAB
- enhances the luminance channel
- converts back to BGR

`preprocess_frame`:

- resizes large frames
- optionally applies CLAHE
- optionally blurs

These functions show that the team did not treat raw input as automatically model-ready.

That is important because one of the major lessons of the repo is that data conditioning strongly affects downstream pose quality.

### Detection and MediaPipe inference

`detect_persons`:

- runs YOLO
- filters for person boxes

`estimate_pose_mediapipe`:

- crops around each detected person
- runs PoseLandmarker
- remaps coordinates to the full frame

This pair defines the final 2D extraction architecture.

In meeting language, these two functions are where the system turns "raw people in a frame" into "structured body landmarks."

### Tracking logic

`get_bbox_from_keypoints`
`compute_iou`
`batch_track_people`
`filter_short_tracks`

Together, these maintain consistent person IDs.

This subsystem is what makes the pipeline usable for multi-person video rather than only isolated per-frame pose snapshots.

### 2D smoothing logic

`smooth_track`
`smooth_all_tracks`

This stage reduces jitter before the 3D model sees the motion.

That is important because temporal models are sensitive to trajectory quality, not just single-frame accuracy.

### 3D post-processing logic

`smooth_3d_trajectory`
`smooth_all_3d_tracks`
`compute_bone_lengths`
`enforce_bone_constraints`
`enforce_all_bone_constraints`

This shows the final pipeline is not naive. It acknowledges that model output benefits from physically informed cleanup.

That point is often worth saying explicitly in a meeting: the project does not blindly trust raw model output. It adds classical post-processing where it makes engineering sense.

### Model-wrapping logic

`normalize_screen_coordinates`

This ensures 2D inputs match the model’s expected coordinate conventions.

`VideoPose3DLifter`

This wraps:

- temporal model creation
- checkpoint loading
- 3D sequence inference
- multi-person lifting logic

This class is the bridge between the repo’s custom pipeline logic and the bundled VideoPose3D model code.

### Rendering logic

`draw_skeleton_2d`
`project_3d_to_2d_anchored`
`draw_skeleton_3d`
`render_frame`

These functions are historically important because many optimization issues were discovered here, not only in the model itself.

That is one of the most important lessons in the repo: visual failure does not always mean model failure. Sometimes the projection or rendering interpretation is the actual problem.

### `PoseEstimationPipeline`

This class is the top-level runtime manager.

Responsibilities:

- load models
- process the video through all four phases
- save final outputs

If someone asks "what actually runs the whole system?" this is the class to mention.

It is the runtime coordinator for the full backend.

## `final/mediapipe_to_h36m.py`

This file is the semantic bridge between keypoint conventions.

It:

- maps MediaPipe landmarks to a 17-keypoint format suitable for the current VideoPose3D path
- defines COCO 2D connectivity
- defines H36M 3D connectivity

Why this file is critical:

- wrong keypoint semantics can break 3D lifting completely
- interoperability between frameworks is a real engineering problem

This file is one of the clearest examples of real-world ML integration work. Models are often individually strong, but the formats between them do not automatically line up.

## `final/person_tracker.py`

This file isolates the tracking subsystem.

It contains:

- keypoint-to-box conversion
- IoU computation
- Hungarian matching
- short-track filtering
- temporal smoothing for tracks
- a `PersonTracker` class

Why it matters:

- it makes the tracking strategy understandable as its own subsystem

It also demonstrates that the repo separated concerns: tracking is important enough to exist as its own logic unit.

## `final/visualizer_3d.py`

This file focuses on visualization.

It contains:

- depth-to-color mapping
- drawing of 3D skeletons onto a 2D canvas
- side-by-side rendering helpers
- a visualization wrapper class

Why it matters:

- perception of output quality depends strongly on rendering design

This file is where the project’s "technical output" becomes "human-readable output."

## `final/VideoPose3D/common/*`

These files are the bundled model internals and utilities from the VideoPose3D codebase.

They provide:

- temporal model architecture
- camera helpers
- skeleton definitions
- geometry helpers
- utility functions

Important meeting point:

The custom engineering in this repo is mainly in integration, conversion, tracking, smoothing, rendering, and app wrapping. The included VideoPose3D internals are reused model code.

That is not a weakness. It is standard engineering: reuse a proven model implementation, then build a working end-to-end system around it.

---

## Flask App Integration

The repository is not just scripts. `project/app.py` turns the backend into an actual user-facing application.

## What the Flask app does

- serves the landing page
- serves upload and webcam pages
- accepts video uploads
- accepts webcam recordings
- launches processing in a background thread
- stores in-memory job state
- exposes job status and download endpoints

## Important routes

- `/`
- `/upload`
- `/webcam`
- `/api/upload`
- `/api/webcam`
- `/api/status/<job_id>`
- `/api/download/<filename>`

## Why background threads are used

Video processing is slow compared with a normal web request.

If the pipeline ran directly inside the request/response cycle, the UI would freeze or timeout more easily.

Instead:

1. the app saves the input file
2. creates a job ID
3. starts a background thread
4. updates progress in memory
5. lets the frontend poll for completion

This is a practical design for a demo-style ML web application.

## Upload flow

For normal upload:

- file is validated by extension
- file is saved into `uploads/`
- output path is created in `outputs/`
- job is created and started

For webcam upload:

- recording may arrive as `.webm`
- if needed, the app converts it to `.mp4`
- then it runs the same backend pipeline

## Why this matters in the meeting

It proves the project is not only a model experiment. It was integrated into a real usage flow with:

- asynchronous processing
- progress tracking
- downloadable results

---

## Why These Choices Were Made

This is one of the most useful sections for answering "why" questions.

## Why OpenCV?

Because every stage of the pipeline needs dependable media handling.

OpenCV gives:

- image and video I/O
- preprocessing tools
- drawing tools
- compatibility with NumPy

Without OpenCV, the rest of the stack cannot be connected cleanly.

## Why YOLO + MediaPipe instead of only MediaPipe?

Because MediaPipe is better at estimating body landmarks when a single person is already isolated.

YOLO solves the detection problem first.
MediaPipe then solves the landmarking problem on the crop.

This separation improves multi-person handling.

## Why not stay with OpenPose?

Because the repo evolved toward a workflow that was more practical for its final needs:

- easier integration
- modern detector-plus-landmarker structure
- smoother bridge into the final 3D lifting path

OpenPose remains important as a research and early-development influence, but not as the active final runtime.

## Why VideoPose3D?

Because the project already had a strong 2D pipeline and needed temporal 3D lifting from those 2D joint trajectories.

VideoPose3D was a strong fit because it:

- uses time context
- is well known in research
- is practical to run with pretrained weights

## Why not MotionBERT in the final implementation?

Because the repo’s actual working integration is built around VideoPose3D.

MotionBERT was studied because it represents a more modern direction, but it was not wired into the final active backend.

## Why batch tracking?

Because stable person identities matter.

If IDs swap:

- colors flicker
- multi-person outputs become confusing
- 3D interpretation over time becomes weaker

Batch-style tracking makes the video-level identity story more coherent.

## Why smoothing?

Because pose estimates are noisy.

Smoothing is not a cosmetic trick. It is a standard way to reduce jitter and reflect the fact that motion should be temporally continuous.

## Why bone constraints?

Because human limbs should not change length from frame to frame.

Adding this prior improves physical plausibility and visual stability.

## Why side-by-side rendering?

Because showing 2D and 3D together makes the pipeline easier to debug and explain.

It helps distinguish:

- bad 2D input
- bad 3D lifting
- bad visualization

---

## Major Problems Found During Optimization

This section is based mainly on `optimizing/plan.md`, `optimizing/workflow.md`, and `workflow_improved.md`.

## Problem 1: 3D projection and display assumptions were tricky

Main lesson:

- root-relative 3D output cannot be treated like ordinary image-space points
- earlier display math could make skeletons look inflated or wrong

Why important:

- bad visualization can make the model look worse than it really is

## Problem 2: Aspect ratio distortion in the older branch

Main lesson:

- forcing fixed-size resize logic can distort the scene badly, especially for portrait video

Why important:

- pose estimation is sensitive to geometry

## Problem 3: Too much downscaling can erase useful detail

Main lesson:

- resizing improves speed
- but it can also make small people harder to analyze

This is a quality-vs-runtime tradeoff.

## Problem 4: Tracking needs gating, not just assignment

Main lesson:

- a match should not be accepted only because it minimizes cost
- overlap should still be good enough

That is why the IoU threshold matters.

## Problem 5: Keypoint ordering semantics are critical

Main lesson:

- frameworks do not all describe the body in the same way
- model compatibility depends on correct joint ordering and interpretation

## Problem 6: Flat 3D drawing can make good output look worse

Main lesson:

- 3D visualization is a technical problem, not just a cosmetic one
- this motivated the later true-3D visualization planning in `planv6`

---

## Model Comparisons

## OpenPose vs MediaPipe

OpenPose:

- historically influential
- strong research legacy in multi-person pose estimation
- heavier and older in style

MediaPipe:

- lighter and more practical
- strong landmark quality
- easier to integrate into the final modular pipeline

Why the project moved:

- MediaPipe fit better into the detector-plus-landmarker architecture chosen for the final path

## YOLO Pose vs YOLO Detection + MediaPipe

YOLO pose:

- simpler direct keypoint output
- attractive for a shorter pipeline

YOLO detection + MediaPipe:

- more modular
- lets each stage focus on its own task
- worked better for the final branch

Why the final repo favors the latter:

- better control over multi-person isolation and landmark quality

## VideoPose3D vs MotionBERT

VideoPose3D:

- temporal convolutional model
- simpler
- well matched to the current pipeline
- actually implemented in the final backend

MotionBERT:

- newer transformer-style motion representation approach
- potentially stronger and broader
- researched here, not implemented as the active backend

Meeting-safe answer:

VideoPose3D was the implemented choice because it fit the existing 2D pipeline and was practical to integrate. MotionBERT is a future-direction comparison, not the current delivered runtime.

---

## Strengths of the Current Final System

- complete end-to-end video pipeline exists
- multi-person handling is present
- 2D and 3D stages are separated clearly
- tracking and smoothing are included
- app-level integration exists
- optimization history is well documented

These are strong points to emphasize.

---

## Known Limitations

Strong technical communication includes honest limits.

## 1. 3D quality still depends heavily on 2D quality

If the 2D keypoints are poor, the 3D output will also be poor.

## 2. The pipeline is not absolute world-coordinate motion capture

The 3D output is root-relative and visualization-dependent. It is not a fully camera-calibrated motion-capture reconstruction system.

## 3. Small or distant people remain harder

Downscaling and crop quality matter. Small people can lose useful detail before landmarking.

## 4. Visualization can change perceived quality

Rendering choices can make the same underlying 3D output look more or less believable.

## 5. Tracking is practical, not perfect

IoU-based assignment is understandable and useful, but it is not advanced re-identification.

## 6. MotionBERT and other future alternatives are not implemented

They are research context in this repo, not active runtime features.

## 7. The repository is broad and contains duplicates

This is useful for evidence and timeline reconstruction, but it means the workspace is more like a research-and-integration archive than a minimal production repo.

---

## Test and Evidence Inventory

The repo contains multiple forms of evidence that the pipeline was actually developed, tested, and run.

## Sample outputs

`samples/` and `samples/final-outputs/` contain:

- raw example inputs
- generated outputs
- true-3D visualization examples

## Final backend testing

`final/test_separate_outputs.py` exists to validate separated outputs and pipeline behavior.

## Optimization experiments

The `optimizing/experiments/` and `optimizing/test_*.py` scripts exist to test:

- tracking behavior
- smoothing behavior
- output ranges
- projection assumptions

This is important evidence of engineering discipline.

---

## Meeting Defense Notes

## If asked: "What is Kinemation in one sentence?"

Kinemation is a modular human pose estimation system that converts video into structured 2D and 3D skeletal motion data using detection, landmark estimation, temporal lifting, tracking, smoothing, and visualization.

## If asked: "What is the current active stack?"

The current active backend in `final/` uses Python, OpenCV, NumPy, Ultralytics YOLOv8n for person detection, MediaPipe Pose Landmarker for 2D landmarks, SciPy for tracking and smoothing utilities, PyTorch VideoPose3D for 3D lifting, and Flask integration in `project/` for a web app interface.

## If asked: "Why use YOLO and MediaPipe together?"

YOLO locates each person in a multi-person frame, and MediaPipe then estimates detailed body landmarks on each isolated crop. That split makes the system more robust than asking a single landmarker to solve the whole crowded scene alone.

## If asked: "Why VideoPose3D?"

Because once the repo had a working 2D keypoint pipeline, VideoPose3D was a strong temporal model for lifting those 2D trajectories into 3D. It uses time context, which is essential because single-frame 3D pose is ambiguous.

## If asked: "Why not MotionBERT?"

MotionBERT was researched as a newer alternative, but the functioning integrated implementation in this repo is built around VideoPose3D. MotionBERT is comparison and future-direction material here, not the delivered final runtime.

## If asked: "Why is OpenPose still in the repo?"

Because it was part of the project’s earlier exploration and learning path. The final active implementation moved to a different stack, but the repository preserves the earlier research material and model assets.

## If asked: "What was the hardest technical issue?"

A defensible answer is:

The hardest issues were not only model choice, but also geometry and consistency: correct keypoint format conversion, stable multi-person tracking, correct interpretation of root-relative 3D output, and rendering that output in a way that matched the actual model behavior.

## If asked: "What are the main strengths?"

- full end-to-end pipeline
- multi-person support
- integrated 2D and 3D stages
- tracking and smoothing
- web app wrapper
- strong optimization documentation

## If asked: "What would you improve next?"

Good answers:

- improve small-person robustness by revisiting resize and crop strategy
- strengthen tracking beyond IoU-only matching
- evaluate MotionBERT or other newer 3D models
- improve camera-aware projection and visualization
- simplify the repository layout for maintainability

---

## Final Takeaway

Kinemation is not one standalone model. It is a layered computer-vision pipeline built through experimentation.

The repo began with image-processing and pose-estimation fundamentals, explored older approaches like OpenPose, evolved into stronger multi-person 2D pipelines using YOLO and MediaPipe, and then integrated temporal 3D lifting through VideoPose3D with tracking, smoothing, and rendering refinements.

The most accurate one-line description of the current system is:

Kinemation is a modular human pose estimation pipeline whose final working backend detects people with YOLO, estimates 2D body landmarks with MediaPipe, keeps identities consistent with tracking, lifts motion into 3D using VideoPose3D, and exposes the result through both direct scripts and a Flask-based application layer.
