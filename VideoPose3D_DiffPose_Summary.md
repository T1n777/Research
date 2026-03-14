# Summary: "3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training"
### Pavllo, Feichtenhofer, Grangier & Auli — CVPR 2019 (VideoPose3D)

---

## Table of Contents
1. [What Is This Paper About?](#1-what-is-this-paper-about)
2. [Background: The Same Old Problem](#2-background-the-same-old-problem)
3. [Key Idea: Learn Smoothness, Don't Enforce It](#3-key-idea-learn-smoothness-dont-enforce-it)
4. [The Architecture: Dilated Temporal Convolutions](#4-the-architecture-dilated-temporal-convolutions)
5. [Semi-Supervised Training: Back-Projection](#5-semi-supervised-training-back-projection)
6. [The Math: Loss Functions](#6-the-math-loss-functions)
7. [Experiments and Results](#7-experiments-and-results)
8. [Pros and Cons](#8-pros-and-cons)
9. [Key Terminology Glossary](#9-key-terminology-glossary)

---

## 1. What Is This Paper About?

This paper solves the same core problem as Arnab et al. — **estimating 3D human poses from video** — but takes a completely different approach.

Instead of running a heavy optimization process over the whole video (like bundle adjustment), the authors train a **convolutional neural network** that learns to infer 3D poses from sequences of 2D keypoints. The network looks at a window of frames at once and produces a smooth 3D pose for the center frame — no explicit optimization at inference time.

The second major contribution is a **semi-supervised training scheme** called back-projection, which lets the model train on unlabeled videos — videos where no 3D ground truth is available — by checking whether the predicted 3D poses project back cleanly to the original 2D detections.

---

## 2. Background: The Same Old Problem

The core challenges are identical to Arnab et al.:

- **Depth ambiguity** — a 2D image can't tell you how far away a limb is
- **Frame-by-frame jitter** — per-frame estimators give inconsistent poses even when the person barely moves
- **Lab data bias** — most 3D training data comes from motion capture labs, which don't reflect real-world video diversity

Previous attempts to fix temporal inconsistency used **RNNs (LSTMs)** — recurrent networks that process frames sequentially. The authors argue these are slow, hard to parallelize, and prone to drifting over long sequences. Their CNN-based approach is faster, simpler, and more accurate.

---

## 3. Key Idea: Learn Smoothness, Don't Enforce It

Arnab et al. explicitly enforce temporal smoothness through a penalty term ($E_T$) in an objective function. VideoPose3D takes a different route: it **lets the network learn smoothness implicitly** by training on sequences of frames.

The model takes as input a sequence of 2D keypoints across T frames and outputs the 3D pose for the **center frame**. Because the network sees neighboring frames, it naturally learns that poses should be consistent across time — without any explicit smoothness constraint.

This means:
- **No optimization step at inference** — just a single forward pass through the network
- **Much faster** — ~150,000 frames per second on a GPU (vs. ~8 minutes per clip for bundle adjustment)
- **Scalable** — can run on any video in real time

---

## 4. The Architecture: Dilated Temporal Convolutions

The network is a **fully convolutional model** — no recurrence, no attention, just 1D convolutions over the time dimension.

### Input
A sequence of 2D keypoint coordinates: $(x, y)$ for each of $J$ joints, across $T$ frames. Shape: $T \times J \times 2$.

No raw images — just skeleton coordinates.

### Dilated Convolutions

Regular convolutions look at adjacent frames. **Dilated convolutions** skip frames at exponentially increasing intervals (dilation 1, 3, 9, 27, 81...), allowing the network to see a very wide temporal window without needing many layers or heavy computation.

For example, with $B = 4$ residual blocks:
- Block 1: dilation 1 → sees 3 frames
- Block 2: dilation 3 → sees 9 frames
- Block 3: dilation 9 → sees 27 frames
- Block 4: dilation 27 → sees 81 frames
- Combined receptive field: **243 frames**

Each block doubles the receptive field exponentially while the number of parameters grows only linearly — very efficient.

### Residual Connections
The network uses **ResNet-style skip connections**, where the input to each block is added back to its output. This prevents vanishing gradients and helps training.

### Output
3D joint coordinates for the center frame: $J \times 3$.

---

## 5. Semi-Supervised Training: Back-Projection

One of the paper's most practically useful ideas. The problem: 3D ground truth is expensive (requires mocap equipment). Most real-world video has no labels.

### The Idea (Cycle Consistency)

Inspired by **unsupervised machine translation** (translate English → French → English, and check you get back what you started with), the authors apply a similar round-trip check to poses:

1. Take an unlabeled video
2. Detect 2D keypoints per frame
3. Predict 3D poses using the network
4. **Project the 3D poses back to 2D**
5. Check if the reprojected 2D positions match the original detections

If they don't match, the 3D prediction is probably wrong — penalize it.

### Trajectory Model

One subtlety: when you project 3D back to 2D, you need to know where the person is in 3D space globally (not just relative joint positions). The authors train a **separate trajectory model** — same architecture — that predicts the person's global 3D position in camera space. This is added to the pose before projecting.

### Bone Length Constraint

To prevent the model from "cheating" by producing degenerate poses that reproject well but look anatomically wrong, a **soft bone length constraint** is added: the mean bone lengths of unlabeled predictions should roughly match those of labeled predictions.

---

## 6. The Math: Loss Functions

### Supervised Loss (labeled data)

Standard **MPJPE loss** — mean Euclidean distance between predicted and ground truth 3D joint positions:

$$L_{\text{supervised}} = \frac{1}{J} \sum_{j=1}^{J} \| \hat{X}_j - X_j^* \|_2$$

### Back-Projection Loss (unlabeled data)

$$L_{\text{back}} = \frac{1}{J} \sum_{j=1}^{J} \| \Pi(\hat{X}_j + \hat{T}) - x_j^{\text{det}} \|_2$$

Where:
- $\Pi$ = projection from 3D to 2D (using camera intrinsics)
- $\hat{T}$ = predicted global trajectory
- $x_j^{\text{det}}$ = 2D keypoints from the detector

### Trajectory Loss (WMPJPE)

Weighted by inverse ground-truth depth — subjects far from the camera are harder to localize precisely, so their trajectory error is down-weighted:

$$L_{\text{traj}} = \frac{1}{y_z} \| f(x) - y \|$$

Where $y_z$ is the ground-truth depth.

### Combined Training

Both labeled and unlabeled data are used in the same batch:
- First half of batch: labeled data → supervised MPJPE loss
- Second half: unlabeled data → back-projection loss + bone length constraint

---

## 7. Experiments and Results

**Dataset**: Human3.6M (3.6M frames, 11 subjects, 15 actions)
**Metric**: MPJPE (lower = better)

### Supervised Results (Human3.6M, Protocol 1)

| Method | MPJPE (mm) |
|---|---|
| Martinez et al. 2017 (single-frame) | 62.9 |
| Lee et al. 2018 (LSTM, temporal) | 52.8 |
| **Ours, 243 frames, full conv** | **46.8** |

An 11% improvement over the previous best result.

### Velocity Error (Smoothness)

| Method | MPJVE (mm/frame) |
|---|---|
| Single-frame baseline | 11.6 |
| **Temporal model (ours)** | **2.8** |

The temporal model reduces velocity error by **76%** — poses are dramatically smoother.

### Semi-Supervised Results

When only 5,000 labeled frames are available (very scarce data), back-projection improves MPJPE by up to **14.7 mm** over the supervised baseline. With ground-truth 2D keypoints, the improvement reaches **22.6 mm**.

---

## 8. Pros and Cons

### Pros
- **Fast** — single forward pass at inference, ~150k FPS on GPU
- **Simple** — no optimization, no body model, just a CNN
- **Works with any 2D detector** — pipeline-agnostic
- **Semi-supervised** — can train on unlabeled video, reducing dependence on expensive mocap data
- **Smooth output** — 76% reduction in velocity error vs. single-frame baseline
- **State-of-the-art** — beats all prior methods on Human3.6M at the time

### Cons
- **No body model** — bone lengths are only implicitly constrained via training, not hard-enforced like SMPL
- **Requires a window of frames** — needs T frames around the target; can't do true single-frame inference without degraded performance
- **Causal convolution mode is weaker** — when only past frames are available (real-time), performance drops noticeably
- **Still trained on mocap data** — without the semi-supervised extension, still dependent on lab data
- **Camera intrinsics needed** for back-projection — not always available

---

## 9. Key Terminology Glossary

| Term | Plain-English Explanation |
|---|---|
| **Dilated Convolution** | A convolution where kernel points are spaced apart (skipping frames), allowing a large temporal window without heavy compute |
| **Receptive Field** | How many input frames the network can "see" when making a prediction for one frame |
| **Causal Convolution** | A convolution that only looks at past frames — enables real-time processing |
| **Back-Projection** | Projecting a predicted 3D pose back to 2D and comparing with the original 2D detections — used for self-supervision |
| **Cycle Consistency** | The idea that a round-trip transformation (3D → 2D → check) should return to the start — borrowed from unsupervised machine translation |
| **MPJPE** | Mean Per-Joint Position Error — average 3D distance in mm between predicted and true joints |
| **MPJVE** | Mean Per-Joint Velocity Error — same but for the first derivative (how much joints move between frames) |
| **Trajectory Model** | A second network that predicts the person's global 3D position, needed for correct back-projection |
| **Semi-Supervised** | Training using a mix of labeled (ground truth available) and unlabeled (no ground truth) data |
| **Residual Block** | A building block where the input is added back to the output — prevents vanishing gradients |
| **Human3.6M** | The largest standard 3D pose benchmark — 3.6M frames in a lab with 11 subjects |
| **HumanEva-I** | A smaller mocap benchmark used for cross-dataset evaluation |

---
---
---

# Summary: "DiffPose: Toward More Reliable 3D Pose Estimation"
### Gong, Foo, Fan, Ke, Rahmani & Liu — CVPR 2023

---

## Table of Contents
1. [What Is This Paper About?](#1-what-is-this-paper-about-1)
2. [Background: Why Uncertainty Is the Real Problem](#2-background-why-uncertainty-is-the-real-problem)
3. [Key Idea: Pose Estimation as Denoising](#3-key-idea-pose-estimation-as-denoising)
4. [Background on Diffusion Models](#4-background-on-diffusion-models)
5. [The DiffPose Framework](#5-the-diffpose-framework)
6. [The Math: Forward and Reverse Diffusion](#6-the-math-forward-and-reverse-diffusion)
7. [Architecture Details](#7-architecture-details)
8. [Experiments and Results](#8-experiments-and-results)
9. [Pros and Cons](#9-pros-and-cons)
10. [Key Terminology Glossary](#10-key-terminology-glossary)

---

## 1. What Is This Paper About?

Most 3D pose estimation papers (including Arnab et al. and VideoPose3D) treat pose estimation as a **regression problem** — given a 2D pose, predict the single best 3D pose. This works reasonably well, but it ignores a fundamental truth: **there is no single correct 3D pose for a given 2D observation**. Due to depth ambiguity and occlusion, many different 3D poses are plausible given the same 2D input.

DiffPose tackles this by treating 3D pose estimation as a **distribution problem**: instead of predicting one pose, the model maintains and refines a *distribution* over possible poses, progressively reducing uncertainty until it converges on a high-quality answer.

The tool it uses to do this is a **diffusion model** — the same class of models that powers modern image generation (Stable Diffusion, DALL-E, etc.) — repurposed for 3D pose estimation.

---

## 2. Background: Why Uncertainty Is the Real Problem

### Depth Ambiguity (Again)

As with all the previous papers, the fundamental challenge is that a 2D image doesn't contain depth information. A given 2D skeleton can correspond to many different 3D configurations.

### The Regression Problem

Previous methods deal with this by training a network to predict the *most likely* 3D pose — essentially averaging over the uncertainty. This works for common poses but fails for:
- **Unusual poses** — the average answer is often wrong when the pose is atypical
- **Occlusions** — when a joint is hidden, there's genuine ambiguity about where it is
- **Ambiguous body parts** — symmetric body parts (left/right hands) are frequently confused

### What DiffPose Does Differently

Instead of committing to one answer, DiffPose represents the 3D pose as a **distribution** and iteratively refines it — starting from a high-uncertainty cloud of poses and progressively narrowing it down to an accurate, low-uncertainty prediction.

---

## 3. Key Idea: Pose Estimation as Denoising

The core insight is a reframing of the problem:

> **The distance between "I have no idea where this person's joints are" and "I know exactly where they are" can be bridged step by step, just like diffusion models bridge noise and images.**

### The Forward Process (Training Time)
Start with a known ground truth 3D pose (low uncertainty) and gradually add noise to it, step by step, until it becomes a high-uncertainty cloud of possible poses. This generates intermediate distributions used as training supervision.

### The Reverse Process (Inference)
Start from an uncertain 3D pose distribution (initialized from the 2D input heatmaps) and progressively denoise it — step by step — guided by a trained neural network and conditioned on context from the input video. After K steps, you arrive at a sharp, accurate 3D pose distribution.

Take the **mean of N samples** from the final distribution as the actual predicted pose.

---

## 4. Background on Diffusion Models

Diffusion models are a class of generative models. The key mechanics:

### Forward Diffusion
Given a clean sample $h_0$, progressively add noise over $K$ steps:

$$h_k = \sqrt{\alpha_k} h_0 + \sqrt{1 - \alpha_k} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Where $\alpha_k$ is a decreasing sequence controlling how much noise is added at step $k$. As $\alpha_K \approx 0$, the sample $h_K$ becomes pure Gaussian noise.

### Reverse Diffusion
Train a neural network $g_\theta$ to undo this process step by step:

$$h_{k-1} = g_\theta(h_k)$$

After training, you can start from noise $h_K$ and run the reverse process to generate a clean sample $h_0$.

### The Key Difference in DiffPose

In image generation, the reverse process starts from **random Gaussian noise** — no prior information. In DiffPose, it starts from a **pose-specific uncertain distribution** derived from the input's 2D heatmaps. This means the model starts with much more useful information than random noise, leading to better convergence.

---

## 5. The DiffPose Framework

### Step 1: Initialize the Uncertain Distribution $H_K$

From the 2D pose detector, extract **heatmaps** — these are 2D probability maps showing where each joint likely is in the image. These directly encode the uncertainty of the 2D prediction.

- **X and Y distributions**: taken directly from the 2D heatmaps
- **Z distribution (depth)**: either computed from training set histograms or predicted by the Context Encoder

This gives a 3D pose distribution $H_K$ that is specific to the input — much more informative than random noise.

### Step 2: Forward Diffusion (Training Only)

Use a **Gaussian Mixture Model (GMM)** to fit the initialized uncertainty distribution $H_K$, then run the forward process to generate intermediate distributions $\hat{H}_1, ..., \hat{H}_K$ as supervisory signals for training.

Why GMM? Because the uncertainty distributions from heatmaps have **irregular, complex shapes** that a single Gaussian can't capture. A GMM with M components fits them much better.

### Step 3: Reverse Diffusion (Training & Inference)

The trained diffusion model $g$ takes the noisy pose distribution and progressively denoises it, conditioned on:
- **Spatial-temporal context** $f_{ST}$: extracted from the input 2D pose sequence by the Context Encoder
- **Step embedding** $f_D^k$: tells the model which denoising step it's currently on

After $K$ steps, sample $N$ poses from the final distribution $H_0$ and take their mean as the final prediction $h_s$.

---

## 6. The Math: Forward and Reverse Diffusion

### Standard Forward Diffusion (from background)

$$h_k = \sqrt{\alpha_k} h_0 + \sqrt{1 - \alpha_k} \cdot \epsilon$$

### DiffPose's GMM-Based Forward Diffusion

Because $H_K$ is not Gaussian, the standard formula doesn't apply. DiffPose modifies it:

$$\hat{h}_k = \mu^G + \sqrt{\alpha_k}(h_0 - \mu^G) + \sqrt{1 - \alpha_k} \cdot \epsilon^G$$

Where:
- $\mu^G = \sum_{m=1}^{M} \mathbf{1}_m \mu_m$ — weighted mean of the GMM components
- $\epsilon^G \sim \mathcal{N}(0, \sum_{m=1}^{M} \mathbf{1}_m \Sigma_m)$ — noise sampled from the selected GMM component
- $\mathbf{1}_m$ — binary indicator selecting which GMM component to use (sampled according to mixture weights $\pi_m$)

As $\alpha_K \approx 0$, $\hat{h}_K$ converges to the fitted GMM distribution — not random Gaussian noise.

### GMM Fitting (EM Algorithm)

$$\max_{\phi_{GMM}} \prod_{i=1}^{N_{GMM}} \sum_{m=1}^{M} \pi_m \mathcal{N}(h_K^i \mid \mu_m, \Sigma_m)$$

Fitted with $M = 5$ Gaussian components using Expectation-Maximization.

### Reverse Diffusion Step

$$\hat{h}_{k-1} = g_\theta(\hat{h}_k, f_{ST}, f_D^k)$$

### Training Loss

$$\mathcal{L} = \sum_{i=1}^{N} \sum_{k=1}^{K} \| g_\theta(\hat{h}_k^i, f_{ST}, f_D^k) - \hat{h}_{k-1}^i \|_2^2$$

At each reverse step, the model is penalized for how far its output is from the ground-truth intermediate distribution.

---

## 7. Architecture Details

### Diffusion Model $g$ (GCN-based)

The human skeleton is treated as a **graph** — joints are nodes, bones are edges. This allows the model to naturally encode the topological structure of the body.

The architecture:
- 3 stacked **GCN-Attention Blocks**, each containing:
  - 2 standard Graph Convolutional Network (GCN) layers
  - 1 Self-Attention layer (to capture global relationships between non-adjacent joints)
- Residual connections throughout
- Input: noisy pose $h_k \in \mathbb{R}^{J \times 3}$ + context $f_{ST}$ + step embedding $f_D^k$
- Output: denoised pose $h_{k-1} \in \mathbb{R}^{J \times 3}$

### Context Encoder $\phi_{ST}$

A **transformer-based** network that extracts spatial-temporal context features $f_{ST}$ from the input 2D pose sequence. It is pre-trained to predict 3D poses directly, then frozen during diffusion model training. Its features guide the reverse diffusion process.

### Key Hyperparameters
- $K = 50$ reverse diffusion steps
- $N = 5$ pose samples per inference
- $M = 5$ GMM components
- Accelerated with **DDIM** (only 5 steps needed at inference) → ~671 FPS

---

## 8. Experiments and Results

**Datasets**: Human3.6M (3.6M frames) and MPI-INF-3DHP (1.3M frames, indoor + outdoor)
**Metrics**: MPJPE and P-MPJPE

### Video-Based Results on Human3.6M (MPJPE, detected 2D poses)

| Method | MPJPE (mm) |
|---|---|
| Pavllo et al. [VideoPose3D] | 46.8 |
| Zheng et al. 2021 | 44.3 |
| Li et al. 2022 | 43.0 |
| Zhang et al. 2022 | 40.9 |
| **DiffPose (ours)** | **36.9** |

DiffPose outperforms the previous state-of-the-art by ~4 mm — a significant margin.

### Results on MPI-INF-3DHP (outdoor videos)

| Method | MPJPE (mm) | PCK ↑ |
|---|---|---|
| Pavllo et al. | 84.0 | 86.0 |
| Zhang et al. 2022 | 54.9 | 94.4 |
| **DiffPose (ours)** | **29.1** | **98.0** |

Huge improvement on outdoor/real-world videos — exactly the hardest setting.

### Ablation: Does the Diffusion Process Actually Help?

| Method | MPJPE (mm) |
|---|---|
| Baseline A (single step, same architecture) | 44.3 |
| Baseline B (stacked, same compute) | 41.1 |
| **DiffPose (full diffusion)** | **36.9** |

The iterative diffusion process is responsible for the improvement — not just the architecture.

### Inference Speed

| Method | MPJPE (mm) | FPS |
|---|---|---|
| Li et al. 2022 | 43.0 | 328 |
| Zhang et al. 2022 | 40.9 | 974 |
| DiffPose w/ DDIM | 36.9 | **671** |

With DDIM acceleration, DiffPose is competitive in speed while beating everyone on accuracy.

---

## 9. Pros and Cons

### Pros
- **State-of-the-art accuracy** — beats all prior methods on both Human3.6M and MPI-INF-3DHP
- **Handles uncertainty natively** — instead of committing to one answer, it maintains and refines a distribution
- **Better under occlusion** — the probabilistic approach naturally handles ambiguous/hidden joints
- **GMM initialization** — pose-specific uncertainty distribution is far more informative than random noise
- **Fast with DDIM** — 671 FPS with only 5 diffusion steps at inference
- **Works frame-based and video-based** — flexible pipeline

### Cons
- **More complex** — significantly harder to implement and understand than VideoPose3D
- **Slower without DDIM** — 50 reverse steps without acceleration (~173 FPS, still fine but heavier)
- **Still needs mocap training data** — no semi-supervised mechanism for unlabeled video like VideoPose3D
- **K reverse steps required** — even with DDIM, inference is a multi-step process (not a single forward pass)
- **Trained on 96 GPU hours** — expensive to train from scratch
- **Evaluated mainly on lab benchmarks** — real-world generalisation is somewhat less tested than e.g. the Kinetics work in Arnab et al.

---

## 10. Key Terminology Glossary

| Term | Plain-English Explanation |
|---|---|
| **Diffusion Model** | A model that learns to reverse a noise-adding process — starts from noise and progressively generates clean output |
| **Forward Diffusion** | Gradually adding noise to a clean sample over K steps — used to generate training supervision |
| **Reverse Diffusion** | The inference process — starting from noise/uncertainty and progressively denoising to a clean sample |
| **GMM (Gaussian Mixture Model)** | A probability distribution represented as a weighted sum of multiple Gaussians — used to model irregular uncertainty distributions |
| **Heatmap** | A 2D probability map from a 2D pose detector showing where each joint likely is in the image |
| **Indeterminacy** | The state of having multiple equally valid solutions — what DiffPose is designed to handle |
| **DDIM** | Denoising Diffusion Implicit Models — an acceleration technique that reduces 50 diffusion steps to just 5 without much accuracy loss |
| **GCN (Graph Convolutional Network)** | A neural network that operates on graph-structured data — here, the human skeleton graph |
| **Context Encoder** | A transformer network that extracts spatial-temporal features from the 2D pose sequence to guide the diffusion process |
| **Distribution-to-Distribution** | DiffPose's core framing — mapping a high-uncertainty distribution to a low-uncertainty one, rather than predicting a single point |
| **P-MPJPE** | MPJPE after Procrustes alignment (rotation + scale correction) — removes global orientation errors |
| **MPI-INF-3DHP** | A 3D pose benchmark with both indoor and outdoor scenes — harder and more realistic than Human3.6M |
| **Self-Attention** | A mechanism that lets every joint attend to every other joint — captures global body relationships |
| **Depth Ambiguity** | The fundamental problem that a 2D image can correspond to many different 3D configurations |
