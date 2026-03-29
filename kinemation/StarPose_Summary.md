# Summary: "StarPose: 3D Human Pose Estimation via Spatial-Temporal Autoregressive Diffusion"
### Yang et al. — IEEE Transactions on Circuits and Systems for Video Technology, 2025

---

## Table of Contents
1. [What Is This Paper About?](#1-what-is-this-paper-about)
2. [Background: The Landscape of 3D Pose Estimation](#2-background-the-landscape-of-3d-pose-estimation)
3. [What Are Diffusion Models? A Beginner's Explanation](#3-what-are-diffusion-models-a-beginners-explanation)
4. [The Core Problem with Existing Diffusion Methods](#4-the-core-problem-with-existing-diffusion-methods)
5. [StarPose: The Big Idea](#5-starpose-the-big-idea)
6. [Component 1: Autoregressive Pose Conditional Diffusion (AutoPCD)](#6-component-1-autoregressive-pose-conditional-diffusion-autopcd)
7. [Component 2: The Historical Pose Integration Module (HPIM)](#7-component-2-the-historical-pose-integration-module-hpim)
8. [Component 3: Spatial-Temporal Physical Guidance (STPG)](#8-component-3-spatial-temporal-physical-guidance-stpg)
9. [The Math: The Full Loss Function](#9-the-math-the-full-loss-function)
10. [Experiments and Results](#10-experiments-and-results)
11. [How StarPose Compares to the Arnab 2019 Paper](#11-how-starpose-compares-to-the-arnab-2019-paper)
12. [Key Terminology Glossary](#12-key-terminology-glossary)

---

## 1. What Is This Paper About?

This paper presents **StarPose**, a new deep learning system for **3D human pose estimation** — the task of figuring out where every joint in a person's body is in 3D space, using only a regular 2D video as input.

The unique angle here is that StarPose is built on top of **diffusion models** (the same kind of technology behind AI image generators like DALL-E and Stable Diffusion), but adapted specifically to generate 3D pose predictions that are:
1. **Accurate** — the joints end up in the right places
2. **Temporally consistent** — the pose doesn't jitter or jump around weirdly from frame to frame
3. **Anatomically plausible** — the body looks like a real human body (symmetrical limbs, consistent bone lengths, etc.)

The paper's main contribution is combining **autoregressive generation** (where each prediction depends on all previous predictions) with **physical guidance** (hard rules about how human bodies work) inside a diffusion framework.

---

## 2. Background: The Landscape of 3D Pose Estimation

### The Two-Stage Pipeline

Almost all modern 3D pose estimation methods work in two stages:

1. **Stage 1 — 2D Keypoint Detection**: Feed a video frame into a standard 2D pose detector (like CPN — Cascaded Pyramid Network). This gives you the pixel coordinates of joints (e.g., "the left knee is at pixel (245, 380)").

2. **Stage 2 — 2D-to-3D Lifting**: Take those 2D coordinates and try to figure out the 3D positions. This is the hard part and what StarPose focuses on.

The challenge in Stage 2 is the **depth ambiguity problem**: a 2D image has no depth (z-axis) information, so many different 3D poses can project to the same 2D skeleton. The system must use learned priors to pick the most likely 3D interpretation.

### Evolution of Methods

**CNN/LSTM-based methods** were first. These learned patterns from data but struggled to model long-range temporal dependencies efficiently.

**Transformer-based methods** followed. Transformers use attention mechanisms that can relate any two joints or frames to each other, regardless of distance. They became very popular and achieved strong results. However, they still do **deterministic regression** — for a given 2D input, they always output exactly one 3D pose. This is problematic because the problem is fundamentally ambiguous — there isn't just one answer.

**Diffusion-based methods** are the newest wave. They treat the problem probabilistically — instead of committing to one answer, they generate a *distribution* of possible answers and pick the best one. This inherently handles ambiguity better.

---

## 3. What Are Diffusion Models? A Beginner's Explanation

Diffusion models are a class of generative AI models. Here's the intuition:

### The Forward Process (Adding Noise)

Imagine you have a clear photograph. You slowly add random static/noise to it, step by step, until it's completely unrecognizable — just white noise. This is the **forward diffusion process**. Mathematically, at each step k, you're adding a small amount of Gaussian noise to the image, controlled by a noise schedule.

### The Reverse Process (Removing Noise = Generation)

Now, you train a neural network to learn the *reverse* of this: given a noisy image at step k, predict what it looked like at step k-1 (slightly less noisy). If the network learns this well, you can start from pure noise and iteratively run the reverse process K times to generate a clean, high-quality image.

**The key insight**: you can *condition* this reverse process on something — like a text description ("a cat sitting on a chair") or, in this paper's case, **2D keypoints**. The network learns to generate 3D poses that match the given 2D skeleton.

### Why Use Gaussian Mixture Models (GMM) Here?

Pure Gaussian noise isn't the best starting distribution for 3D pose estimation, because poses aren't randomly distributed in 3D space — they cluster around human-like configurations. DiffPose (a prior work this paper builds on) proposed starting the diffusion from a **Gaussian Mixture Model** fitted to real poses from the training set. This means the starting "noise" is already somewhat pose-like, which makes the denoising process more efficient and accurate.

A GMM is just a weighted sum of several bell curves (Gaussians), which together can approximate complex, multi-modal distributions. With M=5 components, you have 5 different clusters of typical poses, and the initial noisy distribution is drawn from this learned mixture.

**The forward process formula (Equation 1)**:
> h_k = μ + √α_k (h_0 − μ) + √(1 − α_k) · ε

- **h_0** = the clean, ground-truth 3D pose
- **h_k** = the corrupted pose after k noise steps
- **μ** = the mean of the GMM (a weighted average of typical poses)
- **α_k** = a noise scaling factor that decreases as k increases (α_K ≈ 0 means fully noisy)
- **ε** = a noise sample drawn from the GMM

In plain English: at each step, you pull the pose toward the GMM mean and add some GMM-distributed noise. By step K, the pose has converged to just a sample from the GMM.

---

## 4. The Core Problem with Existing Diffusion Methods

Previous diffusion-based pose estimators (DiffPose, FinePose, D3DP) had one major flaw: they treated each frame **in isolation**.

Here's what that means in practice:
- To predict the 3D pose for frame t, they look at the surrounding 2D poses (a window of frames)
- They generate a 3D pose from this 2D context
- For the next frame (t+1), they do the *exact same thing* from scratch — completely ignoring what they predicted for frame t

This is a problem because:
1. **No temporal memory**: The network has no recollection of what the previous 3D poses looked like
2. **Inconsistency**: Independently generated poses can be inconsistent — the hip might jump 5cm to the left between frames even if the person barely moved
3. **Missing causality**: Human motion has a cause-and-effect structure. Where your foot is now depends on where it was before. Ignoring this wastes valuable information.

The result: pose sequences that are jerky, physically implausible, and temporally incoherent.

---

## 5. StarPose: The Big Idea

StarPose fixes the above problems with two key innovations:

### Innovation 1: Autoregressive Generation

Instead of generating each frame independently, StarPose uses an **autoregressive** approach — each frame's prediction explicitly depends on the previous predictions.

"Autoregressive" means "self-regressing" — you feed the model's own outputs back as inputs for the next step. This is the same principle used by:
- Language models (each word predicted from all previous words)
- Video generation models (each frame generated from all previous frames)

In StarPose's case: to predict the 3D pose at time t, you feed in the 3D poses predicted at times t-L, t-L+1, ..., t-1 (the past L poses) alongside the current 2D keypoints.

### Innovation 2: Physical Guidance During Denoising

At every single denoising step, StarPose applies four physical constraints to "steer" the pose generation toward physically valid results. This is done using an **energy function** — a formula that measures how physically implausible the current pose estimate is, and then uses the gradient (derivative) of this formula to nudge the pose in a better direction.

Think of it like a GPS that doesn't just give you a destination, but continuously nudges you back onto a valid road if you drift into a field.

### The Overall Pipeline

Looking at Figure 2 in the paper:

1. Start with a video. For each time step t, you have the past L 2D pose sequences and the past L predicted 3D poses.
2. The **HPIM** (Historical Pose Integration Module) combines these past 2D and 3D poses into a rich historical embedding.
3. Sample N noisy poses from the GMM distribution.
4. Run K denoising iterations. At each iteration:
   - The denoising model refines the pose using the 2D features AND the historical embedding
   - The **STPG** (Spatial-Temporal Physical Guidance) computes an energy gradient and nudges the pose toward physical plausibility
5. Average the N denoised poses to get the final 3D prediction.
6. This prediction becomes part of the "historical poses" for the next frame.

---

## 6. Component 1: Autoregressive Pose Conditional Diffusion (AutoPCD)

The AutoPCD is the main denoising model with autoregressive conditioning.

The standard denoising process (without AutoPCD) is:
> h_{k-1} = D_θ(h_k, f_{2D}, k)

Where:
- **D_θ** = the denoising neural network (here, a GraFormer — a graph-based transformer)
- **h_k** = the current noisy pose hypothesis
- **f_{2D}** = the 2D pose features extracted by a pre-trained context encoder (MixSTE)
- **k** = the current diffusion step (encoded as a sinusoidal embedding, similar to positional embeddings in transformers)

With AutoPCD, this becomes:
> h_{k-1} = D_θ(h_k, f_{2D}, **f_his**, k)

Where **f_his** is the historical pose embedding produced by HPIM. This single addition is what enables the autoregressive behavior — past predictions influence current ones.

**The training loss (Equation 6)**:
> L_diff = Σ_{k=1}^{K} ‖ D_θ(h_k, f_{2D}, f_his, k) − h_{k-1} ‖²

This is a Mean Squared Error (MSE) loss: at each diffusion step k, the network tries to predict h_{k-1} from h_k, and we penalize the squared difference between the prediction and the true (less noisy) version.

**At inference time**: 
1. Sample N poses from the GMM: {h¹_K, h²_K, ..., hᴺ_K}
2. Run all N through K denoising steps (with STPG applied at each step)
3. Average the N final denoised poses to get the output

Why N hypotheses? Because there are multiple valid 3D interpretations of a 2D pose. Generating N candidates and averaging helps reduce variance and produce a more stable final estimate.

---

## 7. Component 2: The Historical Pose Integration Module (HPIM)

The HPIM is the "memory" of the system. It takes the past L 2D poses and past L predicted 3D poses and compresses them into a single rich feature vector f_his.

### Why Combine Both 2D AND 3D History?

- **3D history alone** tells you where the joints were in 3D space, but might accumulate errors over time
- **2D history alone** is accurate (it comes from a 2D detector) but lacks depth
- **Together**, they complement each other: the 2D observations ground the 3D predictions, and the 3D predictions provide depth context that 2D can't

### Architecture of HPIM (Figure 3)

HPIM has three parts:

**Part a: Multi-dimensional Spatial Encoders**

For the past L frames:
- The L 2D poses (each J joints × 2 coordinates) are linearly projected into a higher-dimensional embedding space (J × C), then fed through a **2D Spatial Transformer** to get Z_{2D} ∈ ℝ^{L×(J·C)}
- The L 3D poses (each J joints × 3 coordinates) are similarly projected and processed by a **3D Spatial Transformer** to get Z_{3D} ∈ ℝ^{L×(J·C)}

Both tensors have **learnable temporal positional embeddings** added, so the model knows which frame each feature came from.

**Part b: Skeleton Integration Graph**

The 2D and 3D features are fused into a single unified graph G. In this graph:
- **Nodes** (υ) = joints (2J total — J from the 2D pose and J from the 3D pose)
- **Edges** (ε) encode three types of relationships:
  - **Spatial edges within a frame**: neighboring joints in the skeleton (e.g., knee connected to ankle)
  - **Cross-dimensional edges**: each 2D joint connected to its 3D counterpart (e.g., 2D left knee → 3D left knee)
  - **Temporal edges**: the same joint across consecutive frames (tracking a joint over time)

This graph structure lets the model reason about how joints relate to each other spatially, across 2D/3D dimensions, and across time — all at once.

**Part c: Graph and Attention Integration Network**

Two parallel branches process the graph:
- **GCN branch**: Graph Convolutional Networks process the Skeleton Integration Graph to capture **short-distance associations** (nearby joints, direct connections)
- **Attention branch**: Self-attention mechanisms capture **long-distance associations** (distant joints like left hand and right foot, cross-temporal patterns)

These are combined:
> f' = f_G + f_A

Then fed into the next GCN+Attention layers. The final output is:
> f_his = FC(f'_G + f'_A)

Where FC is a linear layer that scales the dimensions to match what the denoising model expects.

**Why both GCN and Attention?**

GCN is great at local structure (knowing the knee is near the ankle) but has limited receptive fields — it can only see neighbors. Attention can see the whole skeleton at once but may miss fine structural details. Combining them gives the best of both worlds.

---

## 8. Component 3: Spatial-Temporal Physical Guidance (STPG)

STPG is a **plug-and-play** module that can be added to any diffusion-based pose estimator without retraining. It works by computing an **energy function** that measures how physically wrong the current pose estimate is, then uses the gradient of that energy to correct the estimate at each denoising step.

### The Energy Guidance Mechanism

The core idea (from Equations 7–9):

At each denoising step k, the denoising model produces an intermediate "clean" estimate of the pose:
> h_{0|k} = μ + (1/√α_k) × (h_k − μ − √(1−α_k) × D_θ(h_k, f_{2D}, f_his, k))

This is basically "what would the clean pose look like if we removed all the noise at this step?"

Then the energy function E measures how physically implausible h_{0|k} is. The gradient of this energy tells us in which direction to move h_k to make h_{0|k} more plausible:
> h_{k-1} = h_{k-1} − ρ_k × ∇_{h_k} E(c, h_{0|k})

Where ρ_k is a step-size (learning rate for the correction). This nudges the pose estimate at each step.

### The Four Physical Constraints

**Constraint 1: 2D Reprojection Consistency (L_p)**

The idea: if we project our 3D joint estimate back onto the 2D image, it should land on the observed 2D keypoint.

> L_p = Σ_k Σ_j ‖ R(h_{0|k}^(j)) − x^(j) ‖₂

- **R(·)** = the reprojection function (using known camera intrinsic parameters)
- **x^(j)** = the detected 2D position of joint j
- We're summing the Euclidean distance between the reprojected position and the detected position

In plain English: "Does the 3D pose, when projected into 2D, match where we actually observed the joints?"

**Constraint 2: Skeletal Symmetry Penalty (L_s)**

Human bodies are bilaterally symmetric — the left arm should be the same length as the right arm, the left leg as long as the right leg.

For each of P predefined left-right limb pairs p:
- Compute left bone length: Left(B_p(h_{0|k})) = ‖ h_{0|k}^(j1) − h_{0|k}^(j2) ‖₂
- Compute right bone length: Right(B_p(h_{0|k})) = similar

> L_s = Σ_k Σ_p ‖ Left(B_p(h_{0|k})) − Right(B_p(h_{0|k})) ‖₂

This penalizes asymmetry in the predicted 3D skeleton.

**Constraint 3: Bone Length Variance (L_b)**

Human bone lengths don't change during movement. The distance from your knee to your ankle is the same whether you're running or sitting.

For each of Q predefined bone connections q, compute the **variance** of that bone's length over the past L frames:
> V(B_q) = (1/L) Σ_l ‖ B_q(h_{0|k}^l) − B̄_q^L ‖₂

Where B̄_q^L is the mean bone length of bone q across L frames.

> L_b = Σ_k Σ_q V(B_q)

This penalizes bone lengths that fluctuate too much over time — a sign of physically implausible predictions.

**Constraint 4: Differential Sequence Variation (L_d)**

Different joints move at different speeds. Your head moves slowly; your feet can move quickly. This constraint penalizes unnatural motion patterns by comparing joints at time t to time t-1, weighted by how much each joint *typically* moves:

> L_d = Σ_k Σ_j w_j × (h_{0|k}^{t(j)} − h_{0|k}^{t-1(j)})²

Where w_j is a pre-defined per-joint weight. Joints that typically move slowly get high weights (so sudden movements are penalized more); joints that can move quickly get lower weights.

### The Full Energy Function

All four constraints are combined:
> E(c, h_{0|k}) = λ_p L_p + λ_s L_s + λ_b L_b + λ_d L_d

With default weights: λ_p = λ_s = 1, λ_b = λ_d = 0.01 (the bone/differential constraints are weaker since they're more approximate).

---

## 9. The Math: The Full Loss Function

During training, StarPose optimizes:
> L = L_diff + λ_p L_p + λ_s L_s + λ_b L_b + λ_d L_d

- **L_diff** = the core diffusion loss (MSE between predicted and actual denoised poses)
- The remaining four terms = STPG applied during training as regularization

During inference, STPG is applied via gradient-based guidance (Equation 9) — it doesn't require any extra learned parameters.

### Implementation Details

- **Sequence length**: f = 243 frames of 2D input context
- **Joints**: J = 17 (standard human skeleton)
- **Hypotheses**: N = 5 (5 candidate 3D poses are generated and averaged)
- **Diffusion steps**: K = 50 (but DDIM acceleration reduces this to just 5 at inference, making it much faster)
- **GMM components**: M = 5 (5 Gaussians to model the pose distribution)
- **History length**: L = 27 past frames for HPIM and bone variance
- **Optimizer**: Adam with learning rate 1e-4, trained for 50 epochs
- **Hardware**: NVIDIA RTX 4090 GPU

---

## 10. Experiments and Results

### Datasets

**Human3.6M**: The main benchmark. 3.6 million images of 11 people doing 15 daily activities (walking, sitting, eating, etc.) in a lab. Training on 5 subjects, testing on 2 unseen subjects.

**MPI-INF-3DHP**: More challenging — includes both indoor and outdoor scenes with much greater diversity. Tests generalization beyond lab conditions.

### Metrics

- **MPJPE** (mm): Mean Per-Joint Position Error — average 3D distance between predicted and true joint positions. Lower = better.
- **P-MPJPE** (mm): Same as MPJPE but after rigid alignment (rotation/scale correction). Tests pose shape accuracy, not absolute position.
- **MPJVE** (mm/s): Mean Per-Joint Velocity Error — how much the velocity of joints deviates from ground truth. Tests temporal smoothness.
- **ACC-ERR** (mm/s²): Acceleration Error — how much the acceleration of joints deviates. Also tests smoothness.

### Accuracy Results (Table II — Human3.6M)

| Method | MPJPE (mm) | P-MPJPE (mm) |
|---|---|---|
| DiffPose | 36.9 | 28.7 |
| FinePose | 31.9 | 28.0 |
| KTPFormer | 33.0 | 26.2 |
| **StarPose** | **29.9** | **24.6** |

StarPose achieves state-of-the-art, outperforming the previous best (FinePose) by 2.0mm in MPJPE.

### Temporal Consistency Results (Table III — Human3.6M)

This is where StarPose truly shines:

| Method | MPJVE (mm/s) | ACC-ERR (mm/s²) |
|---|---|---|
| DiffPose | 5.3 | 6.1 |
| FinePose | 5.7 | 6.6 |
| KTPFormer | 3.9 | 5.3 |
| **StarPose** | **1.3** | **1.6** |

StarPose is dramatically better at temporal consistency — roughly **3× better** than the next best method on velocity error, and **3× better** on acceleration error. This quantitatively confirms that the autoregressive design and STPG are extremely effective at producing smooth, physically plausible motion.

### Results on MPI-INF-3DHP (Table V)

| Method | PCK (%) | AUC (%) | MPJPE (mm) |
|---|---|---|---|
| FinePose | 98.9 | 80.0 | 26.2 |
| STCFormer | 98.7 | **83.9** | 23.1 |
| **StarPose** | **98.9** | 82.5 | **20.8** |

Best MPJPE on this challenging dataset, outperforming FinePose by 5.4mm.

### Ablation Study (Table VII)

The ablation study shows each component's individual contribution:

| Configuration | MPJPE (mm) | Improvement |
|---|---|---|
| Baseline (no HPIM, no STPG) | 39.5 | — |
| + STPG (training only) | 39.0 | −0.5mm |
| + STPG (inference only) | 36.6 | −2.9mm |
| + STPG (train + inference) | 36.3 | −3.2mm |
| + HPIM only | 33.4 | −6.1mm |
| + HPIM + STPG (full) | **29.9** | **−9.6mm** |

Key findings:
- STPG is most powerful at **inference time** (not training), suggesting it acts as a real-time corrective mechanism
- HPIM alone contributes the largest single improvement (6.1mm), confirming temporal context is critical
- They are **complementary** — together they provide 9.6mm improvement vs. just adding their individual gains

### Plug-and-Play Validation (Table VI)

STPG can be dropped into other diffusion models without retraining:

| Method | MPJPE (mm) | Speed (FPS) |
|---|---|---|
| DiffPose | 39.5 | 1918 |
| DiffPose + STPG | 36.3 (↓8.1%) | 1802 (↓6.0%) |
| D3DP | 39.1 | 102 |
| D3DP + STPG | 35.2 (↓10.0%) | 98 (↓3.9%) |

Both methods improve significantly with minimal speed cost, validating the plug-and-play design.

### Inference Speed (Table VIII)

| Method | MPJPE (mm) | FPS |
|---|---|---|
| D3DP | 39.5 | 102 |
| DiffPose | 36.9 | 1918 |
| **StarPose** | **29.9** | **1370** |

StarPose achieves 1370 FPS — fast enough for real-time use — while being significantly more accurate than all compared methods.

---

## 11. How StarPose Compares to the Arnab 2019 Paper

Both papers tackle 3D human pose estimation and both care deeply about **temporal consistency**. But they take very different approaches:

| Aspect | Arnab 2019 | StarPose 2025 |
|---|---|---|
| **Core approach** | Classical optimization (bundle adjustment) | Deep learning (diffusion model) |
| **Body representation** | SMPL 3D mesh model | Skeleton joint coordinates |
| **Temporal reasoning** | Global joint optimization over all frames | Autoregressive conditioning on past L frames |
| **Physical constraints** | Bone length consistency (via single β), joint angle prior (GMM) | Reprojection, symmetry, bone variance, differential variation |
| **Handles ambiguity** | Implicitly (by fitting mesh to all frames jointly) | Explicitly (by generating N hypotheses and averaging) |
| **Training required** | No (optimization at test time) | Yes (deep model must be trained) |
| **Speed** | ~2 sec/frame | 1370 FPS |
| **Scale** | Whole video processed jointly | Sliding window of L=27 past frames |
| **Output** | SMPL mesh (full 3D surface) | Skeleton joint positions |

The biggest conceptual difference: Arnab 2019 does optimization **at test time** (bundle adjustment), while StarPose does it **at training time** (learned diffusion) with only a lightweight guidance pass at inference. This is why StarPose is thousands of times faster.

---

## 12. Key Terminology Glossary

| Term | Plain-English Explanation |
|---|---|
| **Diffusion model** | A generative AI model that learns to gradually remove noise from random samples to produce high-quality outputs |
| **Forward diffusion** | Adding noise to a clean sample step by step until it's indistinguishable from noise |
| **Reverse diffusion / Denoising** | Iteratively removing noise to recover a clean sample (the actual generation step) |
| **Autoregressive** | A process where each output depends on all previous outputs (like predicting the next word using all previous words) |
| **GMM (Gaussian Mixture Model)** | A statistical distribution modeled as a weighted sum of bell curves; used here to initialize the noisy pose distribution |
| **HPIM** | Historical Pose Integration Module — the "memory" component that fuses past 2D and 3D poses |
| **STPG** | Spatial-Temporal Physical Guidance — the physics-based correction mechanism applied during denoising |
| **AutoPCD** | Autoregressive Pose Conditional Diffusion — the main diffusion model with autoregressive conditioning |
| **GCN** | Graph Convolutional Network — a neural network that operates on graph-structured data (like a skeleton) |
| **Energy function** | A formula that measures how "bad" (physically implausible) a pose is; lower energy = more plausible |
| **Energy guidance** | Using the gradient of the energy function to iteratively correct/improve a pose estimate |
| **DDIM** | Denoising Diffusion Implicit Model — a technique that reduces the required diffusion steps from 50 to 5 at inference, enabling real-time speed |
| **Plug-and-play** | A module that can be added to existing models without modifying their architecture or retraining them |
| **MPJPE** | Mean Per-Joint Position Error — average 3D distance (mm) between predicted and true joint locations |
| **MPJVE** | Mean Per-Joint Velocity Error — measures how smooth the predicted joint trajectories are over time |
| **ACC-ERR** | Acceleration Error — measures jitter in the predicted motion |
| **2D lifting** | The process of converting 2D joint coordinates to 3D coordinates |
| **Skeleton Integration Graph** | A graph where nodes are joints and edges encode spatial, cross-dimensional, and temporal relationships |
| **Temporal consistency** | The property that predicted poses change smoothly and naturally over time, without sudden jumps |
| **Depth ambiguity** | The fundamental problem that many different 3D poses can produce the same 2D projection |
| **Human3.6M** | A large-scale indoor motion capture benchmark dataset with 3.6 million labeled frames |
| **MPI-INF-3DHP** | A challenging indoor/outdoor 3D pose benchmark testing generalization |
| **PCK** | Percentage of Correct Keypoints — what fraction of joint predictions are within 150mm of the true position |
