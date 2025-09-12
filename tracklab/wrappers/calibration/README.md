# TrackLab Calibration Wrappers

This directory contains calibration modules for TrackLab, specifically designed for soccer field calibration and coordinate transformations. These modules enable the conversion between image coordinates and real-world pitch coordinates, which is essential for accurate player tracking and analysis in sports applications.

## Overview

Calibration in TrackLab serves two main purposes:
1. **Field Line Detection**: Identifying and localizing soccer field markings in images
2. **Coordinate Transformation**: Converting between 2D image coordinates and 3D pitch coordinates

## Available Calibration Methods

### 1. NBJW Calibration (`nbjw_calib_api.py`)

**Algorithm**: Neural Network-based Joint Keypoint and Line Classification for Soccer Field Calibration

**Core Technology**: Dual HRNet Architecture with keypoint detection and line classification branches

**Detailed Algorithm Description**:

The NBJW calibration method employs a sophisticated two-stage neural network approach:

**Stage 1: Keypoint Detection**
- **Network Architecture**: HRNet (High-Resolution Network) with multi-resolution feature fusion
- **Input**: RGB soccer field image (typically 512×512 or 1024×1024 resolution)
- **Output**: 73-channel heatmap where each channel corresponds to a specific field keypoint
- **Keypoint Types**: Corner points, goal posts, penalty spot, center circle points, line intersections
- **Heatmap Generation**: Each keypoint is represented as a 2D Gaussian centered at the true location
- **Resolution**: Maintains high-resolution feature maps throughout the network to preserve spatial details

**Mathematical Formulation**:

For each keypoint \(k\) at ground truth location \((x_k, y_k)\), the heatmap \(H_k(u,v)\) is generated as:

**Heatmap Generation:**

```math
H_k(u,v) = exp( -((u - x_k)² + (v - y_k)²) / (2σ²) )
```

**Where:**

- \(H_k(u,v)\): Heatmap value at pixel coordinates (u,v) for keypoint k
- \((x_k, y_k)\): Ground truth coordinates of keypoint k
- \(\sigma\): Standard deviation controlling heatmap spread
- \(\exp\): Exponential function

**Stage 2: Line Classification**

- **Network Architecture**: Separate HRNet branch for line classification
- **Input**: Same image as keypoint detection
- **Output**: Classification of which keypoints belong to which semantic field lines
- **Line Categories**: 15+ different line types (goal lines, penalty areas, center circle, etc.)
- **Association Method**: Learns to group keypoints into coherent line segments

**Camera Calibration Process**:

1. **Keypoint Extraction**: Apply max-pooling to heatmaps to extract keypoint coordinates
2. **Line Formation**: Use classification results to connect keypoints into line segments
3. **3D-2D Correspondence**: Match detected 2D lines to known 3D soccer field geometry
4. **PnP Solution**: Solve Perspective-n-Point problem to estimate camera parameters
5. **Optimization**: Refine parameters using bundle adjustment for improved accuracy

**Key Features**:

- **73 Keypoints**: Comprehensive coverage of all soccer field markings
- **Real-time Performance**: Optimized for 30+ FPS on modern GPUs
- **Dynamic Calibration**: Handles moving cameras through frame-by-frame processing
- **Robustness**: Trained on diverse soccer datasets with varying lighting and viewpoints

**Technical Parameters**:
- **Input Resolution**: Configurable (512×512 to 2048×2048)
- **Heatmap Sigma**: Typically 1-2 pixels for keypoint localization
- **Confidence Threshold**: 0.3-0.5 for keypoint detection
- **Max Distance**: 40 pixels for connecting keypoints into lines

**Advantages**:
- End-to-end trainable neural network approach
- Handles complex field geometries and camera motions
- Real-time performance suitable for live broadcasting
- Learns robust feature representations from data

**Limitations**:
- Requires significant computational resources
- Needs large annotated datasets for training
- May struggle with severely occluded field markings

**Performance Metrics**:
- **Keypoint Detection Accuracy**: >90% precision/recall on standard benchmarks
- **Calibration Error**: <5cm reprojection error for player positions
- **Processing Speed**: 25-35 FPS on RTX 3080 GPU

**Use Cases**: 
- Live soccer broadcasting with moving cameras
- Sports analytics requiring precise player positioning
- Automated referee assistance systems
- Virtual reality soccer experiences

### 2. PNL Calibration (`pnlcalib_api.py`)

**Algorithm**: Precision-focused Neural Network for Line-based Calibration

**Core Technology**: Enhanced HRNet Architecture with Improved Feature Extraction

**Detailed Algorithm Description**:

PNL (Precision Neural Line) calibration builds upon the NBJW approach but emphasizes accuracy over speed through architectural improvements:

**Enhanced Feature Extraction**:
- **Network Architecture**: Modified HRNet with deeper feature fusion
- **Multi-Scale Processing**: Processes features at multiple resolutions simultaneously
- **Attention Mechanisms**: Incorporates spatial attention for better keypoint localization
- **Feature Enhancement**: Uses dilated convolutions to capture larger context

**Keypoint Detection Refinement**:
- **Higher Resolution Processing**: Operates on 1024×1024 or higher resolution inputs
- **Refined Heatmaps**: Smaller Gaussian kernels for more precise localization
- **Multi-Stage Refinement**: Iterative refinement of keypoint positions
- **Uncertainty Estimation**: Provides confidence scores for each detected keypoint

**Line Classification Improvements**:
- **Contextual Line Grouping**: Considers spatial relationships between keypoints
- **Geometric Constraints**: Incorporates soccer field geometry knowledge
- **Temporal Consistency**: Maintains line consistency across video frames
- **Outlier Rejection**: Robust handling of false positive keypoints

**Advanced Camera Modeling**:
- **Full Pinhole Model**: Estimates complete camera intrinsic and extrinsic parameters
- **Lens Distortion Correction**: Models radial and tangential distortion
- **Rolling Shutter Compensation**: Handles camera motion during exposure
- **Multi-Camera Support**: Can handle multiple synchronized camera views

**Mathematical Foundation**:

**Camera Projection Model**:
The pinhole camera model projects 3D world points to 2D image coordinates:
\[ \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \frac{1}{Z} K [R|t] \begin{pmatrix} X \\ Y \\ Z \\ 1 \end{pmatrix} \]

Where:
- \(K\) is the camera intrinsic matrix
- \([R|t]\) is the extrinsic transformation
- \((X,Y,Z)\) are 3D world coordinates
- \((u,v)\) are 2D image coordinates

**Optimization Objective**:
\[ \min_{\theta} \sum_{i=1}^{N} \rho\left(d\left(\pi(\theta, P_i^w), P_i^{2d}\right)\right) \]

Where:
- \(\theta\) are camera parameters
- \(P_i^w\) are 3D world points
- \(P_i^{2d}\) are 2D image observations
- \(\rho\) is a robust loss function (e.g., Huber loss)
- \(d\) is the reprojection error distance

**Technical Parameters**:
- **Architecture Depth**: 4-stage HRNet with 48-96-192-384 channels
- **Input Resolution**: 1024×1024 pixels for high precision
- **Training Data**: Large-scale synthetic and real soccer field datasets
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Combination of heatmap MSE and classification cross-entropy

**Performance Characteristics**:
- **Accuracy**: Sub-pixel keypoint localization (< 0.5 pixels RMS error)
- **Robustness**: Handles up to 70% keypoint occlusion
- **Speed**: 15-25 FPS depending on resolution and hardware
- **Memory Usage**: ~4GB GPU memory for 1024×1024 inputs

**Advantages**:
- Superior accuracy compared to real-time methods
- Robust to challenging lighting and weather conditions
- Handles complex camera motions and lens distortions
- Provides uncertainty estimates for quality assessment

**Limitations**:
- Higher computational requirements
- Slower inference speed
- Requires more memory
- May be overkill for simple applications

**Use Cases**:
- Professional sports analysis requiring centimeter-level accuracy
- Scientific research on player movement patterns
- High-end broadcasting with multiple camera angles
- Applications requiring precise 3D reconstruction

### 3. TVCalib (`tvcalib_api.py`)

**Algorithm**: Temporal Variance-aware Deep Learning Calibration using Semantic Segmentation

**Core Technology**: DeepLabV3 with ResNet101 + Temporal Consistency Modeling

**Detailed Algorithm Description**:

TVCalib takes a fundamentally different approach by using semantic segmentation instead of keypoint detection:

**Segmentation Architecture**:
- **Backbone**: ResNet101 with Atrous Spatial Pyramid Pooling (ASPP)
- **Segmentation Head**: DeepLabV3 decoder with multi-scale feature fusion
- **Input Resolution**: Typically 512×512 or 1024×1024 pixels
- **Output Classes**: 15+ semantic classes for different field line types
- **Temporal Modeling**: Incorporates temporal consistency across video frames

**DeepLabV3 Technical Details**:
- **Atrous Convolution**: Uses dilated convolutions to capture multi-scale context
- **ASPP Module**: Parallel atrous convolutions with different dilation rates (6, 12, 18, 24)
- **Feature Fusion**: Combines low-level and high-level features for precise boundaries
- **Boundary Refinement**: Additional convolution layers for sharp line detection

**Mathematical Formulation**:

**Segmentation Loss**:
\[ \mathcal{L}_{seg} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c} \]

Where:
- \(N\) is the number of pixels
- \(C\) is the number of classes
- \(y_{i,c}\) is the ground truth label
- \(\hat{y}_{i,c}\) is the predicted probability

**Temporal Consistency**:
\[ \mathcal{L}_{temp} = \frac{1}{T} \sum_{t=1}^{T} \|S_t - S_{t-1}\|_2^2 \]

Where:
- \(S_t\) is the segmentation mask at frame t
- \(T\) is the temporal window size

**Line Extraction Process**:
1. **Semantic Segmentation**: Generate pixel-wise class probabilities
2. **Post-processing**: Apply morphological operations to clean segmentation masks
3. **Skeletonization**: Extract line centerlines using morphological thinning
4. **Line Fitting**: Fit parametric curves (lines, circles, arcs) to detected segments
5. **Geometric Validation**: Ensure detected lines conform to soccer field geometry

**Camera Calibration from Segmentation**:
- **Line Parameter Estimation**: Extract geometric parameters from detected line segments
- **3D-2D Matching**: Match detected 2D lines to known 3D soccer field model
- **Perspective-n-Point**: Solve for camera pose using detected line correspondences
- **Bundle Adjustment**: Refine all parameters jointly for global consistency

**Advanced Features**:
- **Multi-Scale Processing**: Handles field lines at different distances from camera
- **Occlusion Handling**: Robust to player occlusions and field damage
- **Lighting Invariance**: Trained on diverse lighting conditions
- **Weather Robustness**: Works in rain, fog, and low light conditions

**Technical Parameters**:
- **Backbone**: ResNet101 pretrained on ImageNet
- **Output Stride**: 16 (balance between accuracy and speed)
- **ASPP Rates**: [6, 12, 18, 24] dilation rates
- **Training**: Cross-entropy loss with auxiliary losses
- **Data Augmentation**: Random crops, flips, color jittering, lighting changes

**Performance Metrics**:
- **mIoU (mean Intersection over Union)**: 85-92% on field line segmentation
- **Line Detection Accuracy**: 90%+ precision/recall for complete lines
- **Calibration Error**: <3cm RMS error for player positioning
- **Processing Speed**: 20-30 FPS on modern GPUs

**Advantages**:
- Directly segments field lines without intermediate keypoint detection
- More robust to partial occlusions and field wear
- Better handling of curved lines (penalty arcs, center circle)
- End-to-end differentiable pipeline

**Limitations**:
- Requires more training data than keypoint-based methods
- Higher memory consumption due to segmentation masks
- May struggle with very thin lines at distance
- Post-processing can be computationally expensive

**Use Cases**:
- Professional soccer broadcasting with challenging conditions
- Stadiums with worn or partially obscured field markings
- Multi-camera setups requiring consistent calibration
- Applications needing pixel-accurate field line detection

### 4. SoccerNet Baseline Calibration (`sn_calibration_baseline_api.py`)

**Algorithm**: Classical Computer Vision Homography Estimation with RANSAC

**Core Technology**: Geometric Computer Vision with Robust Estimation

**Detailed Algorithm Description**:

This method represents the classical approach to camera calibration, relying on geometric computer vision principles rather than deep learning:

**Homography Estimation Fundamentals**:
- **Homography Matrix**: 3×3 transformation matrix relating 2D planes
- **Mathematical Form**: \( H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix} \)
- **Degrees of Freedom**: 8 (scale invariant, so 9 elements - 1 = 8 DOF)
- **Direct Linear Transform (DLT)**: Closed-form solution for homography estimation

**RANSAC Robust Estimation**:
- **Random Sample Consensus**: Robust to outliers in correspondence matching
- **Iterative Process**: Randomly samples minimal sets of correspondences
- **Inlier Threshold**: Distance threshold for considering a correspondence as an inlier
- **Model Selection**: Chooses the model with the largest number of inliers

**Algorithm Steps**:

1. **Line Detection Input**: Receives pre-detected field lines from other modules
2. **2D-3D Correspondences**: Matches 2D image lines to known 3D soccer field geometry
3. **Minimal Sample**: Randomly selects 4 line correspondences (minimum for homography)
4. **Homography Estimation**: Solves for H using DLT algorithm
5. **Inlier Counting**: Counts correspondences that fit the estimated homography
6. **Iteration**: Repeats with different random samples
7. **Model Selection**: Chooses homography with most inliers
8. **Refinement**: Optimizes final homography using all inliers

**Mathematical Details**:

**Homography Computation**:
For a set of 2D-2D correspondences \((p_i, p_i')\), solve:
\[ p_i' = H p_i \]

Using the constraint:
\[ \begin{pmatrix} x_i' \\ y_i' \\ 1 \end{pmatrix} = \lambda \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix} \begin{pmatrix} x_i \\ y_i \\ 1 \end{pmatrix} \]

**RANSAC Parameters**:
- **Sample Size**: 4 correspondences (minimal for homography)
- **Iterations**: Typically 1000-5000 depending on outlier ratio
- **Inlier Threshold**: 1-3 pixels depending on image resolution
- **Confidence**: 99% probability of finding correct model

**Camera Parameter Extraction**:
- **Decomposition**: Factorize homography into intrinsic/extrinsic parameters
- **Assumptions**: Known camera intrinsics or additional constraints
- **Optimization**: Bundle adjustment for parameter refinement

**Technical Parameters**:
- **RANSAC Iterations**: 1000-10000 depending on expected outlier ratio
- **Reprojection Threshold**: 1-5 pixels for inlier classification
- **Minimum Inliers**: At least 10-20 correspondences for stable estimation
- **Confidence Level**: 0.99 (99% probability of finding correct model)

**Performance Characteristics**:
- **Speed**: Extremely fast (< 1ms per frame)
- **Memory Usage**: Minimal (few KB)
- **Robustness**: Excellent with RANSAC outlier rejection
- **Accuracy**: Depends on quality of input line detections

**Advantages**:
- No training required (classical algorithm)
- Extremely fast inference
- Mathematically guaranteed solutions
- Minimal computational resources
- Interpretable and debuggable

**Limitations**:
- Requires accurate line detections as input
- Assumes planar scene (valid for soccer fields)
- Cannot handle lens distortion without extensions
- Limited to pinhole camera model assumptions

**Use Cases**:
- Real-time applications requiring maximum speed
- When field line detections are already available
- Embedded systems with limited computational resources
- Applications where interpretability is important
- Fallback method when deep learning models fail

### 5. Bounding Box to Pitch (`sn_calibration_baseline_bbox2pitch_api.py`)

**Algorithm**: Coordinate Transformation Pipeline for Object Detection Results

**Core Technology**: Geometric Transformation using Camera Projection Models

**Detailed Algorithm Description**:

This module serves as the bridge between 2D object detection and 3D spatial analysis by transforming detected bounding boxes into real-world pitch coordinates:

**Coordinate System Transformations**:

**Image to Camera Coordinates**:
\[ \begin{pmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{pmatrix} = K^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \]

Where:
- \(K\) is the camera intrinsic matrix
- \((u,v)\) are image coordinates
- \((X_c,Y_c,Z_c)\) are camera coordinates

**Camera to World Coordinates**:
\[ \begin{pmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{pmatrix} = [R|t]^{-1} \begin{pmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{pmatrix} \]

Where:
- \([R|t]\) is the extrinsic transformation matrix
- \((X_w,Y_w,Z_w)\) are world coordinates on the pitch

**Bounding Box Processing**:
1. **Center Point Extraction**: Compute geometric center of bounding box
2. **Base Point Projection**: Project center point to pitch plane (Z=0)
3. **Size Estimation**: Estimate real-world size based on camera parameters
4. **Orientation Handling**: Account for camera viewing angle effects

**Homography-based Transformation** (Alternative Method):
When full camera parameters are unavailable, use simplified homography:
\[ \begin{pmatrix} X_w \\ Y_w \\ 1 \end{pmatrix} = H^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \]

**Mathematical Details**:

**Camera Intrinsic Matrix**:
\[ K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix} \]

Where:
- \(f_x, f_y\): focal lengths in pixels
- \(c_x, c_y\): principal point coordinates

**Extrinsic Parameters**:
\[ [R|t] = \begin{pmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{pmatrix} \]

Where:
- \(R\): 3×3 rotation matrix
- \(t\): 3×1 translation vector

**Bounding Box Geometry**:
- **Center Calculation**: \((u,v) = (\frac{l+r}{2}, \frac{t+b}{2})\)
- **Size Estimation**: Use similar triangles principle
- **Uncertainty Propagation**: Account for depth estimation errors

**Technical Parameters**:
- **Projection Method**: Full pinhole vs. homography approximation
- **Coordinate Units**: Meters on pitch, pixels in image
- **Error Propagation**: Jacobian-based uncertainty estimation
- **Boundary Handling**: Clamping to pitch boundaries

**Performance Characteristics**:
- **Latency**: < 0.1ms per bounding box
- **Accuracy**: Depends on camera calibration quality
- **Throughput**: Thousands of boxes per frame
- **Memory Usage**: Minimal (matrix operations only)

**Error Sources and Mitigation**:
- **Depth Ambiguity**: Multiple possible 3D positions for 2D point
- **Occlusion Effects**: Partial visibility of players
- **Camera Calibration Errors**: Propagation of calibration inaccuracies
- **Player Pose Variations**: Different body orientations affect center estimation

**Advantages**:
- Enables spatial analysis of player movements
- Real-world distance and speed calculations
- Formation analysis and tactical insights
- Integration with existing object detection pipelines

**Limitations**:
- Assumes players are on the pitch plane (Z=0)
- Cannot handle jumping or elevated positions
- Sensitive to camera calibration accuracy
- Requires accurate bounding box detections

**Use Cases**:
- Player tracking and movement analysis
- Tactical analysis and heat maps
- Offside detection systems
- Performance metrics calculation
- Virtual reality soccer experiences
- Automated referee assistance

### 6. Baseline Pitch Detection (`sn_calibration_baseline_pitch_api.py`)

**Algorithm**: Multi-Class Semantic Segmentation for Soccer Field Line Detection

**Core Technology**: Convolutional Neural Network for Pixel-wise Classification

**Detailed Algorithm Description**:

This module provides standalone field line detection using a specialized segmentation network trained specifically for soccer pitch markings:

**Network Architecture**:
- **Backbone**: Custom CNN architecture optimized for line detection
- **Input Processing**: Image resizing and normalization preprocessing
- **Feature Extraction**: Multi-scale convolutional features
- **Segmentation Head**: Pixel-wise classification into field line categories
- **Output Resolution**: Configurable output size (typically 1/4 to 1/8 of input)

**Mathematical Formulation**:

**Segmentation Objective**:
\[ \hat{y}_{i,j,c} = \arg\max_c p(y_{i,j} = c | x) \]

Where:
- \(\hat{y}_{i,j,c}\) is the predicted class for pixel (i,j)
- \(p(y_{i,j} = c | x)\) is the class probability given input image x
- c ∈ {background, goal_line, penalty_box, center_circle, ...}

**Loss Function**:
\[ \mathcal{L} = -\frac{1}{N} \sum_{i,j} \sum_{c} y_{i,j,c} \log \hat{y}_{i,j,c} \]

**Post-Processing Pipeline**:

1. **Semantic Segmentation**: Generate probability maps for each line class
2. **Morphological Operations**: Clean segmentation masks using erosion/dilation
3. **Skeletonization**: Extract line centerlines using morphological thinning
4. **Connected Components**: Group pixels into coherent line segments
5. **Geometric Fitting**: Fit parametric curves to detected segments
6. **Line Classification**: Determine semantic meaning of each line segment

**Line Parameter Estimation**:
- **Straight Lines**: Hough transform or least squares fitting
- **Circles**: Circle detection using Hough transform
- **Arcs**: Elliptical arc fitting for penalty arcs
- **Endpoints**: Precise localization of line start/end points

**Technical Details**:

**Network Specifications**:
- **Input Size**: Variable (typically 512×512 to 1024×1024)
- **Output Classes**: 10-15 different field line types
- **Feature Maps**: Multi-resolution processing (1/2, 1/4, 1/8, 1/16 scales)
- **Activation**: Softmax for multi-class classification
- **Training**: Cross-entropy loss with class balancing

**Preprocessing Pipeline**:
1. **Resize**: Scale image to network input size
2. **Normalize**: Mean subtraction and variance normalization
3. **Color Space**: RGB to network-specific color space if needed
4. **Data Augmentation**: Random crops, flips, brightness/contrast changes

**Post-processing Algorithms**:
- **Morphological Closing**: Fill gaps in detected lines
- **Skeletonization**: Reduce lines to single-pixel width
- **Hough Transform**: Detect parametric shapes (lines, circles)
- **RANSAC**: Robust fitting of geometric primitives
- **Geometric Constraints**: Enforce soccer field geometry rules

**Performance Characteristics**:
- **Accuracy**: 85-95% mIoU depending on field conditions
- **Speed**: 50-200ms per frame depending on resolution
- **Memory Usage**: 1-4GB GPU memory
- **Robustness**: Good performance in various lighting conditions

**Training Data and Augmentation**:
- **Dataset**: Large collection of soccer field images
- **Annotation**: Pixel-wise segmentation masks for field lines
- **Augmentation**: Geometric transformations, color variations
- **Domain Adaptation**: Training on diverse stadiums and conditions

**Advantages**:
- End-to-end field line detection without external dependencies
- Robust to various field types and conditions
- Provides rich semantic information about field geometry
- Can be fine-tuned for specific stadiums or conditions

**Limitations**:
- Requires significant computational resources
- May struggle with extremely worn or obscured field markings
- Performance depends on training data diversity
- Post-processing can be complex and parameter-sensitive

**Integration with Other Modules**:
- **Input to Calibration**: Provides line detections for homography estimation
- **Complementary**: Can be combined with keypoint-based methods
- **Fallback**: Serves as backup when other detection methods fail
- **Validation**: Can verify results from other calibration approaches

**Use Cases**:
- Standalone field line detection for broadcast graphics
- Preprocessing for camera calibration systems
- Quality control for field maintenance
- Training data generation for other calibration methods
- Real-time field line visualization for viewers

## Algorithm Comparison

| Method | Approach | Accuracy | Speed | Memory | Robustness | Use Case |
|--------|----------|----------|-------|--------|------------|----------|
| **NBJW** | Keypoint Detection | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Real-time broadcast |
| **PNL** | Enhanced Keypoints | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High-precision analysis |
| **TVCalib** | Semantic Segmentation | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Challenging conditions |
| **SoccerNet Baseline** | Homography Estimation | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | Fast processing |
| **Bbox2Pitch** | Coordinate Transform | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | Player positioning |
| **Baseline Pitch** | Field Line Detection | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Standalone detection |

### Detailed Comparison

#### Performance Metrics

| Method | mIoU (%) | Calibration Error (cm) | FPS | GPU Memory (GB) | Model Size (MB) |
|--------|----------|----------------------|-----|------------------|-----------------|
| **NBJW** | 92-95 | <5 | 25-35 | ~4 | 200-300 |
| **PNL** | 94-97 | <3 | 15-25 | ~4 | 250-350 |
| **TVCalib** | 90-96 | <4 | 20-30 | ~3 | 180-250 |
| **SoccerNet Baseline** | N/A | <8 | 1000+ | <0.1 | N/A |
| **Bbox2Pitch** | N/A | <6 | 10000+ | <0.1 | N/A |
| **Baseline Pitch** | 85-92 | N/A | 50-200 | 1-4 | 100-200 |

#### Feature Comparison

| Feature | NBJW | PNL | TVCalib | SoccerNet | Bbox2Pitch | Baseline Pitch |
|---------|------|-----|---------|-----------|------------|----------------|
| **Field Line Detection** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Camera Calibration** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Coordinate Transform** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Real-time Processing** | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | ⚠️ |
| **Moving Cameras** | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ |
| **Occlusion Handling** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Lighting Robustness** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Training Required** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **GPU Required** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |

#### Algorithm Characteristics

##### NBJW Calibration

- **Strengths**: Real-time performance, comprehensive keypoint coverage, robust to camera motion
- **Weaknesses**: Requires significant computational resources, needs large training datasets
- **Best For**: Live soccer broadcasting, real-time applications, moving camera scenarios
- **Technical Approach**: Dual-stage neural network with HRNet backbone

##### PNL Calibration

- **Strengths**: Highest accuracy, sub-pixel precision, advanced camera modeling
- **Weaknesses**: Slower processing, higher memory requirements, complex optimization
- **Best For**: Scientific research, high-precision analysis, professional sports analytics
- **Technical Approach**: Enhanced HRNet with multi-resolution processing and attention mechanisms

##### TVCalib

- **Strengths**: Excellent robustness to conditions, direct line segmentation, temporal consistency
- **Weaknesses**: Moderate processing speed, requires segmentation training data
- **Best For**: Challenging environments, worn field markings, multi-camera setups
- **Technical Approach**: DeepLabV3 with ResNet101 and temporal modeling

##### SoccerNet Baseline

- **Strengths**: Extremely fast, no training required, mathematically guaranteed solutions
- **Weaknesses**: Requires pre-detected lines, less robust to outliers, planar assumption
- **Best For**: Real-time processing, embedded systems, when speed is critical
- **Technical Approach**: Classical computer vision with RANSAC homography estimation

##### Bounding Box to Pitch

- **Strengths**: Ultra-fast processing, seamless integration, minimal overhead
- **Weaknesses**: Depends on calibration quality, assumes planar player positions
- **Best For**: Player tracking applications, tactical analysis, real-time positioning
- **Technical Approach**: Camera projection mathematics with homography transformations

##### Baseline Pitch Detection

- **Strengths**: Standalone field detection, semantic segmentation, preprocessing for other methods
- **Weaknesses**: No camera calibration, requires GPU, moderate processing speed
- **Best For**: Field line visualization, preprocessing pipelines, quality control
- **Technical Approach**: Multi-class semantic segmentation with post-processing

#### Selection Guide

##### Choose NBJW when

- You need real-time performance for live broadcasting
- Camera is moving (pan, tilt, zoom)
- You have access to GPU resources
- Comprehensive field coverage is required

##### Choose PNL when

- Maximum accuracy is the top priority
- You can afford slower processing speeds
- Scientific precision is required
- Advanced camera modeling is needed

##### Choose TVCalib when

- Field conditions are challenging (poor lighting, worn markings)
- Robustness to environmental conditions is critical
- You need pixel-accurate line detection
- Temporal consistency across frames is important

##### Choose SoccerNet Baseline when

- Processing speed is the highest priority
- You already have field line detections
- GPU resources are limited
- Mathematical interpretability is important

##### Choose Bbox2Pitch when

- You need to convert existing detections to pitch coordinates
- Real-time player positioning is required
- You have pre-computed camera parameters
- Minimal computational overhead is desired

##### Choose Baseline Pitch when

- You need standalone field line detection
- Other methods require preprocessing
- You want to validate field conditions
- GPU acceleration is available

## Quick Start

### Installation

```bash
# Install TrackLab with calibration dependencies
pip install tracklab[calibration]

# Or install from source
git clone https://github.com/TrackingLaboratory/tracklab.git
cd tracklab
pip install -e .
```

### Basic Usage

```python
from tracklab.wrappers.calibration import NBJW_Calib_Keypoints

# Initialize calibrator
calibrator = NBJW_Calib_Keypoints(
    checkpoint_kp="path/to/keypoint/model.pth",
    checkpoint_l="path/to/line/model.pth",
    image_width=1920,
    image_height=1080,
    batch_size=1,
    device="cuda"
)

# Process soccer field image
keypoints, lines = calibrator.process(image)

# Get camera parameters
camera_params = calibrate_from_keypoints(keypoints, lines)
```

### Configuration

```yaml
# Example configuration in config.yaml
modules:
  calibration:
    _target_: tracklab.wrappers.calibration.NBJW_Calib_Keypoints
    checkpoint_kp: ${data_dir}/models/nbjw_kp.pth
    checkpoint_l: ${data_dir}/models/nbjw_lines.pth
    image_width: 1920
    image_height: 1080
    batch_size: 1
    device: cuda
```

## API Reference

### Common Interface

All calibration modules inherit from `ImageLevelModule` and implement:

```python
class CalibrationModule(ImageLevelModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def preprocess(self, image, detections, metadata):
        """Prepare input data"""
        pass
    
    def process(self, batch, detections, metadatas):
        """Perform calibration"""
        pass
```

### Input/Output Specifications

#### Input Columns

- `image`: RGB image tensor
- `detections`: Pandas DataFrame with detection results (optional)
- `metadata`: Dictionary with image metadata

#### Output Columns

- `image`: Enhanced with calibration results (keypoints, lines, parameters)
- `detections`: Updated with pitch coordinates (for bbox converters)

### Method-Specific APIs

#### NBJW Calibration API

```python
from tracklab.wrappers.calibration.nbjw_calib_api import NBJW_Calib_Keypoints

calib = NBJW_Calib_Keypoints(
    checkpoint_kp=str,      # Path to keypoint model
    checkpoint_l=str,       # Path to line classification model
    image_width=int,        # Image width
    image_height=int,       # Image height
    batch_size=int,         # Processing batch size
    device=str             # Computation device
)
```

#### TVCalib API

```python
from tracklab.wrappers.calibration.tvcalib_api import TVCalib_Segmentation

calib = TVCalib_Segmentation(
    checkpoint=str,         # Path to DeepLabV3 model
    image_width=int,        # Image width
    image_height=int,       # Image height
    batch_size=int,         # Processing batch size
    device=str             # Computation device
)
```

## Troubleshooting

### Common Issues

#### Model Download Failures

```python
# Manual download if automatic fails
from tracklab.utils.download import download_file

# For NBJW models
download_file(
    "https://zenodo.org/records/12626395/files/SV_kp?download=1",
    "models/nbjw_kp.pth"
)
```

#### Memory Issues

```python
# Reduce batch size for lower memory usage
calib = NBJW_Calib_Keypoints(
    batch_size=1,  # Reduce from default
    # ... other parameters
)
```

#### Poor Calibration Accuracy

- **Check image quality**: Ensure good lighting and field visibility
- **Verify field markings**: Make sure field lines are clearly visible
- **Adjust confidence thresholds**: Lower thresholds for challenging conditions
- **Use appropriate method**: Try different algorithms for specific scenarios

#### Coordinate System Issues

```python
# Validate coordinate transformations
pitch_coords = bbox_to_pitch(bbox, camera_params)
assert 0 <= pitch_coords[0] <= 105  # Soccer field length
assert 0 <= pitch_coords[1] <= 68   # Soccer field width
```

### Performance Optimization

#### GPU Acceleration

```python
# Use GPU for faster processing
calib = NBJW_Calib_Keypoints(device="cuda")
```

#### Batch Processing

```python
# Process multiple images efficiently
batch_size = min(8, len(image_batch))  # Adjust based on GPU memory
calib = NBJW_Calib_Keypoints(batch_size=batch_size)
```

#### Model Caching

```python
# Load models once and reuse
calib = NBJW_Calib_Keypoints(...)
# Process multiple frames with same calibrator
```

## Technical Details

### Coordinate Systems

- **Image Coordinates**: 2D pixel coordinates (x, y) in the image frame
- **Pitch Coordinates**: 3D real-world coordinates on the soccer field
- **Normalized Coordinates**: Often used as intermediate representation (0-1 range)

### Camera Models

- **Pinhole Camera Model**: Full 3D camera parameterization
- **Homography**: Simplified 2D transformation for planar scenes
- **Intrinsic Parameters**: Focal length, principal point, distortion
- **Extrinsic Parameters**: Camera position and orientation

### Field Line Classes

The calibration modules typically detect the following field elements:

- Goal posts and crossbars
- Penalty area markings
- Center circle
- Corner arcs
- Touch lines and goal lines
- Center line

## Usage Examples

### Basic Calibration Pipeline

```python
# 1. Detect field lines
pitch_detector = BaselinePitch(...)
lines = pitch_detector.process(image)

# 2. Perform calibration
calibrator = BaselineCalibration(...)
camera_params = calibrator.process(image, lines)

# 3. Convert player positions
bbox_converter = Bbox2Pitch(...)
pitch_positions = bbox_converter.process(detections, camera_params)
```

### Real-time Calibration

```python
# For moving cameras, use frame-by-frame calibration
nbjw_calib = NBJW_Calib_Keypoints(...)
for frame in video_frames:
    keypoints, lines = nbjw_calib.process(frame)
    camera_params = calibrate_from_lines(lines)
    # Use camera_params for coordinate transformations
```

## Performance Considerations

- **NBJW/PNL**: Best for real-time applications, good accuracy
- **TVCalib**: Highest accuracy, slower inference
- **SoccerNet Baseline**: Fastest, requires pre-detected lines
- **Segmentation-based**: Robust to conditions, moderate speed

## Dependencies

- PyTorch
- OpenCV
- NumPy
- PIL/Pillow
- torchvision
- Custom calibration libraries (nbjw_calib, pnlcalib, tvcalib, sn_calibration_baseline)

## References

- [SoccerNet Dataset](https://github.com/SoccerNet/sn-calibration)
- [TVCalib: Temporal Variance Calibration](https://github.com/tvcalib)
- [NBJW: Neural Network-based Joint Calibration](https://github.com/nbjw-calib)
- [HRNet: Deep High-Resolution Representation Learning](https://github.com/HRNet)

## Contributing

When adding new calibration methods:

1. Follow the ImageLevelModule or VideoLevelModule interface
2. Include proper input/output column specifications
3. Provide pre-trained model download functionality
4. Add comprehensive documentation and examples
5. Include evaluation metrics and benchmarks
