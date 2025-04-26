# BubuNet: A Dual-Head Segmentation Model for Lane and Drivable Area Segmentation

## Demo
Below is a demonstration of BubuNet performing real-time drivable area and lane segmentation:

![BubuNet Demo](demo.gif)

### Sample Results
| Drivable Area Segmentation | Lane Segmentation |
|----------------------------|-------------------|
| ![Drivable Area](output_images/drivable_sample1.jpg) | ![Lane](output_images/0ace96c3-48481887_overlay.png) |
| ![Drivable Area](output_images/drivable_sample2.jpg) | ![Lane](output_images/0a98248b-de4df1d4_overlay.png) |

## Overview
BubuNet is a novel segmentation model inspired by YOLOP and TwinLiteNet, designed for autonomous driving tasks. It leverages a ResNet-50 backbone with a single decoder that simultaneously performs **drivable area segmentation** and **lane segmentation**. The model is trained on the [BDD100K dataset](https://www.bdd100k.com/) with a batch size of 10.

## Features
- **Dual-Head Architecture**: One head for drivable area segmentation and another for lane segmentation.
- **Backbone**: ResNet-50 encoder for robust feature extraction.
- **Decoder**: Lightweight decoder with upsampling to match input resolution (360x640).
- **Tasks**:
  - Drivable area segmentation (2 classes)
  - Lane segmentation (2 classes)
- **Applications**: Real-time inference on images and videos for autonomous driving scenarios.

## Architecture Diagram
Below is the architecture diagram of BubuNet:

![BubuNet Architecture](architecture_diagram.png)

## Requirements
To run BubuNet, install the following dependencies:
```bash
pip install torch torchvision opencv-python pillow
