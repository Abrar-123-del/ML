# Foot Gesture Recognition using 360° Camera

## Project Overview

This project focuses on capturing and recognizing foot gestures using a 360° camera. The documentation addresses key challenges related to camera placement, motion blur, and speed variations during data capture.

---

## Table of Contents

1. [Working with 360° Camera Footage on Local Computer](#1-working-with-360-camera-footage-on-local-computer)
2. [Addressing Speed Variations (Walking vs Running)](#2-addressing-speed-variations-walking-vs-running)
3. [Handling Motion Blur from Camera Movement](#3-handling-motion-blur-from-camera-movement)
4. [Camera Placement Recommendations](#4-camera-placement-recommendations)
5. [Step-by-Step Implementation Guide](#5-step-by-step-implementation-guide)
6. [Data Collection Best Practices](#6-data-collection-best-practices)
7. [Model Architecture Recommendations](#7-model-architecture-recommendations)

---

## 1. Working with 360° Camera Footage on Local Computer

### Problem
After capturing footage with a 360° camera, you need to transfer, view, and process the video files on your local computer for creating the foot gesture dataset.

### Solution Overview

360° cameras typically store footage in special formats that require specific software and processing steps to work with on a standard computer.

---

### Step 1: Transferring 360° Video from Camera to Computer

#### A. Direct Connection Methods

**USB Cable Transfer (Recommended for most cameras):**
```bash
# Connect camera via USB cable
# Camera usually appears as external storage device
# Navigate to DCIM folder and copy video files

# Example locations:
# Windows: D:\DCIM\Camera\
# Mac: /Volumes/CAMERA_NAME/DCIM/
# Linux: /media/username/CAMERA_NAME/DCIM/
```

**WiFi/Bluetooth Transfer:**
- Most 360° cameras have companion mobile apps (Insta360, GoPro, etc.)
- Transfer files from camera to phone first
- Then transfer from phone to computer via cloud or cable

**SD Card Reader:**
- Remove SD card from camera
- Insert into computer's SD card reader
- Copy files directly

---

### Step 2: Installing Required Software

#### A. Camera Manufacturer's Software (Essential)

Different 360° cameras require their specific software:

**For Insta360 Cameras:**
```
Download: Insta360 Studio (Windows/Mac)
Website: https://www.insta360.com/download
Features:
- View and edit 360° footage
- Export specific viewing angles
- Stabilization and stitching
- Free to use
```

**For GoPro MAX:**
```
Download: GoPro Player (Windows/Mac)
Website: https://gopro.com/en/us/shop/softwareandapp/gopro-player/GoPro-Player.html
Features:
- Play and edit 360° videos
- Reframe to standard video
- Export clips
```

**For Ricoh Theta:**
```
Download: Ricoh Theta Desktop App
Website: https://theta360.com/en/support/download/
Features:
- View 360° images and videos
- Basic editing
- Export formats
```

#### B. General 360° Video Players

**VLC Media Player (Free, Cross-platform):**
```bash
# Install VLC
# Windows: Download from videolan.org
# Mac: brew install --cask vlc
# Linux: sudo apt-get install vlc

# VLC can play many 360° formats directly
# Use mouse to navigate the 360° view
```

**Potplayer (Windows):**
- Supports 360° video playback
- Download from potplayer.daum.net

---

### Step 3: Converting 360° Footage to Usable Format

#### A. Understanding 360° Video Formats

360° videos are typically stored in two projection formats:

1. **Equirectangular** (Most common):
   - Looks like a distorted panoramic image
   - Full 360° sphere mapped to rectangular frame
   - Standard video codec (H.264, H.265)

2. **Dual Fisheye**:
   - Two circular images side by side
   - Requires stitching software

#### B. Extracting Specific View Angles

For foot gesture recognition, you don't need the full 360° - just the foot area. Here's how to extract it:

**Method 1: Using Camera Manufacturer Software**

```
Example with Insta360 Studio:
1. Import 360° video file
2. Use "Reframe" or "FreeCapture" tool
3. Set viewing angle to look down at feet
   - Adjust pitch (tilt) to -30° to -45°
   - Set direction to straight ahead
4. Export as standard MP4 video (1920x1080 or higher)
5. This gives you a normal video focused on foot area
```

**Method 2: Using FFmpeg (Command Line)**

```bash
# Install FFmpeg
# Windows: Download from ffmpeg.org
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg

# Extract a specific view from equirectangular 360° video
ffmpeg -i input_360video.mp4 \
  -vf "v360=e:rectilinear:yaw=0:pitch=-40:roll=0:w=1920:h=1080" \
  -c:v libx264 -crf 18 -preset slow \
  output_feet_view.mp4

# Parameters explained:
# e:rectilinear - Convert from equirectangular to flat view
# yaw=0 - Direction (0=forward, 90=right, -90=left, 180=back)
# pitch=-40 - Tilt down toward feet (negative = down)
# w=1920:h=1080 - Output resolution
```

**Method 3: Using Python with OpenCV**

```python
import cv2
import numpy as np
import py360convert  # Import at module level

def extract_view_from_360(input_video, output_video, pitch=-40, yaw=0):
    """
    Extract a specific viewing angle from 360° video
    
    Args:
        input_video: Path to 360° equirectangular video
        output_video: Path to save extracted view
        pitch: Vertical angle (-90 to 90, negative = down)
        yaw: Horizontal angle (0-360, 0 = forward)
    """
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (1920, 1080))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract specific view using py360convert library
        # Install: pip install py360convert
        view = py360convert.e2p(frame, 
                                fov_deg=(90, 90),  # Field of view
                                u_deg=yaw,          # Horizontal direction
                                v_deg=pitch,        # Vertical direction (down for feet)
                                out_hw=(1080, 1920))
        
        out.write(view)
    
    cap.release()
    out.release()

# Usage
extract_view_from_360('360_video.mp4', 'feet_view.mp4', pitch=-40, yaw=0)
```

---

### Step 4: Processing Pipeline for Foot Gesture Dataset

#### Complete Workflow

```python
# Install required libraries
# pip install opencv-python py360convert ffmpeg-python numpy

import cv2
import os
import py360convert
import numpy as np

class FootGestureDataProcessor:
    """Process 360° camera footage for foot gesture recognition"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_360_video(self, video_path, output_name, 
                         pitch=-40, yaw=0, fov=90):
        """
        Extract foot view from 360° video
        
        Args:
            video_path: Path to 360° video file
            output_name: Name for output video
            pitch: Vertical angle (negative = look down at feet)
            yaw: Horizontal angle (0 = forward)
            fov: Field of view in degrees
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Output setup
        output_path = os.path.join(self.output_dir, output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
        
        frame_count = 0
        print(f"Processing {video_path}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract foot perspective from 360° frame
            foot_view = py360convert.e2p(
                frame,
                fov_deg=(fov, fov),
                u_deg=yaw,
                v_deg=pitch,
                out_hw=(1080, 1920),
                mode='bilinear'
            )
            
            # Optional: Apply stabilization or enhancement here
            
            out.write(foot_view)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        print(f"Completed: {output_path} ({frame_count} frames)")
        
        return output_path
    
    def extract_frames(self, video_path, output_folder, 
                      frame_interval=1):
        """
        Extract individual frames from processed video
        
        Args:
            video_path: Path to processed video
            output_folder: Folder to save frames
            frame_interval: Save every Nth frame (1=all frames)
        """
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(
                    output_folder, 
                    f"frame_{saved_count:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames to {output_folder}")

# Usage Example
processor = FootGestureDataProcessor(
    input_dir='raw_360_videos',
    output_dir='processed_foot_views'
)

# Process a 360° video to extract foot view
processor.process_360_video(
    video_path='raw_360_videos/walking_tap_gesture.mp4',
    output_name='tap_gesture_feet.mp4',
    pitch=-40,  # Look down at feet
    yaw=0,      # Forward direction
    fov=90      # Field of view
)

# Extract frames for dataset
processor.extract_frames(
    video_path='processed_foot_views/tap_gesture_feet.mp4',
    output_folder='dataset/frames/tap_gesture',
    frame_interval=2  # Save every 2nd frame
)
```

---

### Step 5: Recommended Software Stack

#### Essential Tools

| Tool | Purpose | Installation |
|------|---------|-------------|
| **Camera App** | View & export 360° videos | Download from camera manufacturer |
| **FFmpeg** | Video conversion & processing | `pip install ffmpeg-python` or download binary |
| **Python OpenCV** | Video frame extraction | `pip install opencv-python` |
| **py360convert** | 360° format conversion | `pip install py360convert` |
| **VLC Player** | Preview 360° videos | Download from videolan.org |

#### Optional Tools

| Tool | Purpose | Installation |
|------|---------|-------------|
| **DaVinci Resolve** | Advanced video editing | Free download from blackmagicdesign.com |
| **Blender** | 360° video compositing | Free download from blender.org |
| **Insta360 SDK** | Direct camera integration | Developer portal from camera manufacturer |

---

### Step 6: Quick Start Guide

**For Beginners - Simple Workflow:**

1. **Transfer video to computer**
   - Connect camera via USB
   - Copy all video files to folder: `raw_videos/`

2. **Install Insta360 Studio or equivalent**
   - Download from camera manufacturer website
   - Install and open

3. **Extract foot view**
   - Import 360° video
   - Use "Reframe" tool
   - Point camera view down at feet (about -40° tilt)
   - Export as standard MP4

4. **Process with Python (optional)**
   ```bash
   pip install opencv-python
   ```
   ```python
   import cv2
   
   # Extract frames from video
   video = cv2.VideoCapture('foot_view.mp4')
   frame_num = 0
   
   while True:
       ret, frame = video.read()
       if not ret:
           break
       cv2.imwrite(f'frames/frame_{frame_num:04d}.jpg', frame)
       frame_num += 1
   
   video.release()
   print(f"Extracted {frame_num} frames")
   ```

**For Advanced Users - Automated Pipeline:**

```bash
# Batch process all 360° videos
for video in raw_videos/*.mp4; do
    # Extract foot view using FFmpeg
    ffmpeg -i "$video" \
        -vf "v360=e:rectilinear:yaw=0:pitch=-40:w=1920:h=1080" \
        -c:v libx264 -crf 18 \
        "processed/$(basename $video)"
done

# Extract frames using Python script
python extract_frames.py --input_dir processed/ --output_dir dataset/frames/
```

---

### Troubleshooting Common Issues

#### Issue 1: Video file won't open
**Solution:** 
- Ensure camera firmware is updated
- Use manufacturer's software first to convert
- Check if file is fully transferred (not corrupted)

#### Issue 2: Computer is slow processing 360° video
**Solution:**
- Extract lower resolution view first (1280x720)
- Use hardware acceleration: `ffmpeg -hwaccel auto`
- Process videos in batches overnight

#### Issue 3: Foot area is distorted
**Solution:**
- Adjust pitch angle in conversion (-30° to -50°)
- Increase field of view if feet are cut off
- Check camera mounting angle during recording

#### Issue 4: Files are too large
**Solution:**
```bash
# Compress video while maintaining quality
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset medium output.mp4

# Or reduce resolution
ffmpeg -i input.mp4 -vf "scale=1280:720" output.mp4
```

---

## 2. Addressing Speed Variations (Walking vs Running)

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

## 3. Handling Motion Blur from Camera Movement

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

## 4. Camera Placement Recommendations

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

## 5. Step-by-Step Implementation Guide

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

## 6. Data Collection Best Practices

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

## 7. Model Architecture Recommendations

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
