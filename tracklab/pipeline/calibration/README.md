# TrackLab Camera Calibration Methods Comparison

This directory contains multiple camera calibration methods for soccer video analysis. Each method has different strengths, requirements, and use cases. This guide helps you choose the right calibration approach for your needs.

## Available Methods

### 1. NBJW Calib (Recommended for Most Users)
**Type**: Deep Learning + Geometric Optimization  
**Best For**: Professional soccer videos, high accuracy requirements

#### Key Features:
- **HRNet-based keypoint detection** (58 keypoints)
- **PnP + RANSAC** for robust parameter estimation
- **Iterative refinement** with line constraints
- **Temporal consistency** via homography reuse

#### Requirements:
- **Models**: `SV_kp`, `SV_lines` (HRNet checkpoints)
- **GPU Memory**: High (batch_size: 32 recommended)
- **Accuracy**: Highest among all methods
- **Speed**: Medium-fast

#### Configuration:
```yaml
pipeline:
  pitch: nbjw_calib
  calibration: nbjw_calib
```

#### Pros:
- ✅ Most accurate calibration
- ✅ Robust to various camera angles
- ✅ Good temporal stability
- ✅ Well-documented and maintained

#### Cons:
- ❌ Requires large model files (~500MB)
- ❌ Higher computational cost
- ❌ Complex setup

### 2. PNLCalib (High Accuracy Alternative)
**Type**: Deep Learning + Line-based Geometric  
**Best For**: When line detection is more reliable than keypoints

#### Key Features:
- **HRNet line detection** with geometric constraints
- **Line-to-keypoint mapping** for structured calibration
- **Iterative line refinement** (optional)
- **Multi-hypothesis voting** for robustness

#### Requirements:
- **Models**: `pnl_SV_kp`, `pnl_SV_lines` (HRNet checkpoints)
- **GPU Memory**: High (batch_size: 256)
- **Accuracy**: Very high, especially for line-rich scenes
- **Speed**: Fast with batch processing

#### Configuration:
```yaml
pipeline:
  pitch: pnlcalib
  calibration: pnlcalib
```

#### Pros:
- ✅ Excellent accuracy on line detection
- ✅ Fast batch processing
- ✅ Good for broadcast-quality videos

#### Cons:
- ❌ Requires specific HRNet models
- ❌ May struggle with poor line visibility

### 3. TVCalib (State-of-the-Art Deep Learning)
**Type**: Deep Learning Segmentation + Optimization  
**Best For**: Research applications, maximum accuracy needed

#### Key Features:
- **DeepLabV3+ segmentation** (28 pitch element classes)
- **Learned parameter distributions** for different camera types
- **Iterative optimization** with AdamW
- **Multi-hypothesis refinement**

#### Requirements:
- **Models**: `train_59.pt` (DeepLabV3+ model)
- **GPU Memory**: Very high
- **Accuracy**: State-of-the-art
- **Speed**: Slow (2000 optimization steps per frame)

#### Configuration:
```yaml
pipeline:
  pitch: tvcalib
  calibration: tvcalib
```

#### Pros:
- ✅ Highest possible accuracy
- ✅ Learns from large datasets
- ✅ Handles complex camera geometries

#### Cons:
- ❌ Very slow processing
- ❌ High computational requirements
- ❌ Complex optimization may fail

### 4. Baseline Calibration (Traditional CV Approach)
**Type**: Traditional Computer Vision  
**Best For**: Lightweight applications, when deep learning models unavailable

#### Key Features:
- **Homography estimation** from detected lines
- **No deep learning models required**
- **Simple geometric approach**
- **Fast processing**

#### Requirements:
- **Models**: `soccer_pitch_segmentation.pth` (lightweight segmentation)
- **GPU Memory**: Low
- **Accuracy**: Moderate
- **Speed**: Fast

#### Configuration:
```yaml
pipeline:
  pitch: baseline
  calibration: baseline
```

#### Pros:
- ✅ No large model downloads needed
- ✅ Fast processing
- ✅ Simple and reliable
- ✅ Good for CPU-only systems

#### Cons:
- ❌ Lower accuracy than DL methods
- ❌ May struggle with complex camera angles
- ❌ Limited robustness to occlusions

## Method Selection Guide

### Choose NBJW Calib If:
- You want the best balance of accuracy and speed
- You have access to GPU resources
- You're working with professional soccer videos
- You need reliable calibration for tracking applications

### Choose PNLCalib If:
- Line detection quality is high in your videos
- You need fast batch processing
- You prefer line-based geometric constraints

### Choose TVCalib If:
- Maximum accuracy is absolutely critical
- You have significant computational resources
- You're doing research or need state-of-the-art performance
- Processing time is not a constraint

### Choose Baseline If:
- You have limited computational resources
- You want the simplest setup possible
- Model download size is a concern
- You're working with amateur videos where perfect accuracy isn't required

## Performance Comparison

| Method | Accuracy | Speed | GPU Memory | Model Size | Setup Complexity |
|--------|----------|-------|------------|------------|------------------|
| NBJW Calib | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High | ~500MB | Medium |
| PNLCalib | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High | ~500MB | Medium |
| TVCalib | ⭐⭐⭐⭐⭐ | ⭐⭐ | Very High | ~200MB | High |
| Baseline | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | ~150MB | Low |

## Configuration Tips

### Memory Optimization:
```yaml
pipeline:
  calibration:
    batch_size: 1  # Reduce if GPU memory is limited
    image_width: 1280  # Match your video resolution
    image_height: 720
```

### Speed Optimization:
```yaml
pipeline:
  calibration:
    use_prev_homography: True  # Reuse previous frame results
    refine_lines: False  # Skip iterative refinement
    ransac_iter: 300  # Balance between speed and robustness
```

### Accuracy Optimization:
```yaml
pipeline:
  calibration:
    refine_lines: True  # Enable iterative refinement
    use_prev_homography: True  # Temporal consistency
    ransac_iter: 600  # More robust outlier rejection
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce `batch_size` to 1
2. **Poor Calibration**: Try different `ransac_iter` values (300-1000)
3. **Slow Processing**: Disable `refine_lines` or reduce `optim_steps`
4. **Model Download Issues**: Check internet connection and disk space

### Model Availability:
- All required models should be automatically downloaded on first use
- Check `/pretrained_models/calibration/` for downloaded files
- Manual download may be needed in some environments

## Contributing

When adding new calibration methods:
1. Follow the existing module structure
2. Include comprehensive documentation
3. Provide configuration examples
4. Add performance benchmarks
5. Update this comparison guide</content>
<parameter name="filePath">/workspaces/tracklab/tracklab/pipeline/calibration/README.md