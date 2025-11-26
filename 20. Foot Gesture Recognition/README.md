# Foot Gesture Recognition using 360° Camera

## Project Overview

This project focuses on capturing and recognizing foot gestures using a 360° camera. The documentation addresses key challenges related to camera placement, motion blur, and speed variations during data capture.

---

## Table of Contents

1. [Addressing Speed Variations (Walking vs Running)](#1-addressing-speed-variations-walking-vs-running)
2. [Handling Motion Blur from Camera Movement](#2-handling-motion-blur-from-camera-movement)
3. [Camera Placement Recommendations](#3-camera-placement-recommendations)
4. [Step-by-Step Implementation Guide](#4-step-by-step-implementation-guide)
5. [Data Collection Best Practices](#5-data-collection-best-practices)
6. [Model Architecture Recommendations](#6-model-architecture-recommendations)

---

## 1. Addressing Speed Variations (Walking vs Running)

### Problem
Different movement speeds (walking, jogging, running) significantly affect the appearance and timing of foot gestures, making it challenging for the model to generalize across speeds.

### Solutions

#### A. Speed-Normalized Data Collection
- **Capture data at multiple speeds**: Collect gesture data at slow walk, normal walk, fast walk, jog, and run speeds
- **Label speed categories**: Tag each sample with its speed category for stratified training
- **Balanced dataset**: Ensure equal representation of all speed categories

#### B. Frame Rate Adjustment
- **High frame rate recording**: Use 60fps or higher to capture smooth motion at all speeds
- **Dynamic frame sampling**: For slower movements, sample fewer frames; for faster movements, sample more frames to normalize gesture duration

#### C. Temporal Normalization Techniques
- **Dynamic Time Warping (DTW)**: Align gesture sequences to a standard temporal length regardless of speed
- **Speed-invariant feature extraction**: Extract features that are independent of movement speed (e.g., joint angles, relative positions)
- **Optical flow normalization**: Normalize optical flow vectors by the overall body movement speed

#### D. Data Augmentation
- **Time stretching**: Artificially speed up or slow down captured gestures
- **Frame interpolation**: Generate intermediate frames to create speed variations
- **Temporal jittering**: Randomly skip or duplicate frames during training

#### E. Model Architecture Considerations
- **Temporal Convolutional Networks (TCN)**: Use dilated convolutions to capture patterns at different time scales
- **Attention mechanisms**: Allow the model to focus on key frames regardless of speed
- **LSTM with speed encoding**: Include speed as an additional input feature

---

## 2. Handling Motion Blur from Camera Movement

### Problem
When the camera moves with the person (body-mounted), images become blurry, making it difficult for the model to identify clear foot gestures.

### Solutions

#### A. Hardware Solutions
- **High shutter speed**: Use 1/500s or faster to freeze motion
- **Gimbal stabilization**: Use a 3-axis gimbal to minimize camera shake
- **Electronic Image Stabilization (EIS)**: Enable EIS on the 360° camera if available
- **Higher frame rate**: 60fps+ reduces motion blur per frame

#### B. Software Solutions

##### Pre-processing Techniques
- **Deblurring algorithms**: Apply motion deblurring using Wiener filter or deep learning-based deblurring
- **Frame selection**: Automatically select the sharpest frames from a sequence
- **Optical flow-based motion compensation**: Stabilize video using optical flow

##### Model Training Strategies
- **Train with blurry data**: Include motion-blurred samples in training to improve robustness
- **Data augmentation with blur**: Apply artificial motion blur to sharp images during training
- **Multi-frame input**: Use multiple consecutive frames to help the model infer gesture despite blur

#### C. Camera Settings Optimization
```
Recommended Settings for 360° Camera:
- Frame Rate: 60fps minimum (120fps preferred)
- Shutter Speed: 1/500s or faster
- ISO: Auto (but prefer lower ISO to reduce noise)
- Resolution: 4K or higher for cropping flexibility
- Stabilization: EIS enabled
```

---

## 3. Camera Placement Recommendations

### Analysis: Lower Body vs Head Mount

Based on your assumption, **mounting the 360° camera lower on the body is indeed better for foot gesture capture**. Here's a detailed comparison:

### Lower Body Placement (Recommended) ✓

**Advantages:**
| Aspect | Benefit |
|--------|---------|
| **Proximity to feet** | Closer view = higher resolution capture of foot details |
| **Less motion** | Lower body moves less than head during walking/running |
| **Better angles** | Captures foot movements from optimal viewing angles |
| **Reduced occlusion** | Less likely to have body parts blocking the view |
| **Consistent framing** | Feet remain in a more consistent position relative to camera |

**Recommended Mounting Positions:**
1. **Waist level** (belt mount): Good balance of stability and foot visibility
2. **Upper thigh** (thigh strap): Closer to feet, may have more movement
3. **Chest level** (harness): Alternative if waist mount is uncomfortable

### Head/Upper Body Placement

**Disadvantages:**
- Greater distance from feet results in lower resolution
- More camera shake due to head movement
- Feet appear smaller in frame
- More body occlusion issues
- Significant perspective changes during movement

### Mounting Recommendations

```
RECOMMENDED SETUP:
┌─────────────────────────────────────────┐
│                                         │
│    360° Camera Position: WAIST LEVEL    │
│    Height from ground: 80-100 cm        │
│    Orientation: Lens facing downward    │
│    at 30-45° angle toward feet          │
│                                         │
└─────────────────────────────────────────┘
```

---

## 4. Step-by-Step Implementation Guide

### Phase 1: Hardware Setup (Week 1)

#### Step 1.1: Camera Selection and Configuration
1. Choose a 360° camera with:
   - Minimum 4K resolution
   - 60fps capability
   - Good low-light performance
   - Electronic stabilization
   - (Examples: Insta360 ONE X2, GoPro MAX, Ricoh Theta Z1)

2. Configure camera settings:
   ```
   Resolution: 4K or higher
   Frame Rate: 60fps
   Shutter Speed: 1/500s or faster
   Field of View: Full 360° or 270° (to focus on lower hemisphere)
   ```

#### Step 1.2: Mounting System
1. Acquire a waist-level mounting system:
   - Belt-mounted camera clip
   - Or custom 3D-printed mount for specific camera
   
2. Ensure secure attachment:
   - Test stability during movement
   - Minimize vibration transmission

### Phase 2: Data Collection Protocol (Week 2-4)

#### Step 2.1: Define Gesture Categories
1. List all foot gestures to recognize:
   ```
   Examples:
   - Tap (single, double)
   - Slide (forward, backward, left, right)
   - Stomp
   - Kick (forward, backward)
   - Rotate (clockwise, counter-clockwise)
   - Shuffle
   - Cross-over step
   ```

#### Step 2.2: Collection Environment Setup
1. **Controlled environment first**:
   - Indoor space with uniform lighting
   - Non-reflective flooring
   - Markers for consistent positioning

2. **Multiple conditions**:
   - Various lighting conditions
   - Different floor surfaces
   - Indoor and outdoor

#### Step 2.3: Speed Variation Protocol
```
For each gesture, collect data at:
┌────────────────┬─────────────────┬─────────────┐
│ Speed Category │ Approx. Speed   │ Samples     │
├────────────────┼─────────────────┼─────────────┤
│ Stationary     │ 0 km/h          │ 50+ samples │
│ Slow Walk      │ 1.5-3 km/h      │ 50+ samples │
│ Normal Walk    │ 3-5.5 km/h      │ 50+ samples │
│ Fast Walk      │ 5.5-7.5 km/h    │ 50+ samples │
│ Jog            │ 7.5-10 km/h     │ 50+ samples │
│ Run            │ 10+ km/h        │ 50+ samples │
└────────────────┴─────────────────┴─────────────┘
```

#### Step 2.4: Multi-Person Data Collection
1. Collect data from multiple subjects:
   - At least 10-20 different people
   - Various foot sizes and walking styles
   - Different footwear types

### Phase 3: Data Processing (Week 5-6)

#### Step 3.1: Video Pre-processing Pipeline
```python
# Recommended processing pipeline
1. Extract frames at consistent interval
2. Apply motion stabilization
3. Crop to region of interest (feet area)
4. Apply deblurring if needed
5. Normalize brightness/contrast
6. Resize to model input dimensions
```

#### Step 3.2: Annotation
1. Label each clip with:
   - Gesture type
   - Start/end timestamps
   - Speed category
   - Subject ID
   - Environmental conditions

#### Step 3.3: Dataset Organization
```
dataset/
├── train/
│   ├── tap/
│   │   ├── stationary/
│   │   ├── slow_walk/
│   │   ├── normal_walk/
│   │   └── ...
│   ├── slide_forward/
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Phase 4: Model Development (Week 7-10)

#### Step 4.1: Model Architecture Selection

**Recommended Architectures:**

1. **3D CNN (C3D, I3D)**:
   - Good for spatiotemporal feature extraction
   - Can capture both foot appearance and motion

2. **Two-Stream Networks**:
   - RGB stream for appearance
   - Optical flow stream for motion
   - Fusion layer for final prediction

3. **Transformer-based (Video Swin Transformer)**:
   - State-of-the-art for video understanding
   - Good at capturing long-range dependencies

4. **Temporal Segment Networks (TSN)**:
   - Efficient for action recognition
   - Works well with variable-length inputs

#### Step 4.2: Training Strategy
```python
# Recommended training approach
1. Pre-train on large action recognition dataset (Kinetics, UCF101)
2. Fine-tune on foot gesture dataset
3. Use speed-aware data augmentation
4. Apply temporal augmentations
5. Use class-balanced sampling
```

#### Step 4.3: Speed-Invariance Training
```python
# Include speed as auxiliary task
model_output = {
    'gesture_class': gesture_prediction,
    'speed_class': speed_prediction  # Auxiliary task
}

# Use adversarial training for speed invariance
# Or gradient reversal layer to ignore speed features
```

### Phase 5: Evaluation and Optimization (Week 11-12)

#### Step 5.1: Evaluation Metrics
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Speed-specific accuracy (accuracy at each speed category)
- Latency (inference time)

#### Step 5.2: Cross-Speed Validation
```
Test scenarios:
- Train on all speeds, test on all speeds
- Train on walking, test on running (transfer test)
- Leave-one-speed-out validation
```

#### Step 5.3: Real-time Optimization
- Model quantization
- TensorRT/ONNX optimization
- Edge device deployment (if needed)

---

## 5. Data Collection Best Practices

### Lighting Considerations
- **Minimum illumination**: Ensure at least 500 lux for indoor settings
- **Light positioning**: Position light sources at 45-degree angles to minimize shadows
- **Diffused lighting**: Use softboxes or bounce lighting to reduce harsh shadows on feet
- **Consistency**: Maintain consistent lighting throughout recording sessions
- **Multi-condition testing**: Test with various lighting (indoor fluorescent, natural daylight, low-light)

### Subject Guidelines
- Diverse footwear (shoes, barefoot, sandals)
- Various clothing (pants, shorts)
- Natural gesture execution

### Quality Checks
- Review samples for blur, occlusion, and clarity
- Discard unusable samples immediately
- Maintain balanced class distribution

---

## 6. Model Architecture Recommendations

### For Real-time Applications
```
Recommended: MobileNet-based 3D CNN or EfficientNet with temporal modeling
- Lightweight architecture
- Good accuracy-speed trade-off
- Suitable for edge deployment
```

### For Maximum Accuracy
```
Recommended: Video Swin Transformer or I3D with attention
- State-of-the-art accuracy
- Higher computational requirements
- Suitable for server-side processing
```

### Sample Model Pipeline
```python
import torch
import torch.nn as nn

class FootGestureModel(nn.Module):
    def __init__(self, num_classes, num_speeds=6):
        super().__init__()
        # Backbone for spatial features
        self.backbone = EfficientNet3D()  # or I3D, C3D
        
        # Temporal modeling
        self.temporal = nn.LSTM(512, 256, bidirectional=True)
        
        # Classification heads
        self.gesture_head = nn.Linear(512, num_classes)
        self.speed_head = nn.Linear(512, num_speeds)  # Auxiliary
        
    def forward(self, x):
        # x: (batch, channels, frames, height, width)
        features = self.backbone(x)
        temporal_features, _ = self.temporal(features)
        
        gesture_out = self.gesture_head(temporal_features)
        speed_out = self.speed_head(temporal_features)
        
        return gesture_out, speed_out
```

---

## Summary of Key Recommendations

| Challenge | Solution |
|-----------|----------|
| Speed variation | Multi-speed data collection + temporal normalization + speed-invariant training |
| Motion blur | High shutter speed + stabilization + blur-aware training |
| Camera position | Waist-level mount for optimal foot capture |
| Model robustness | Diverse training data + augmentation + multi-task learning |

---

## Next Steps

1. **Hardware Acquisition**: Get appropriate 360° camera and mounting equipment
2. **Pilot Study**: Small-scale data collection to validate setup
3. **Protocol Refinement**: Adjust based on pilot study findings
4. **Full Data Collection**: Collect comprehensive dataset
5. **Model Development**: Build and train the recognition model
6. **Iteration**: Continuous improvement based on evaluation results

---

## Contact and Contributions

For questions or contributions to this project, please open an issue or submit a pull request.
