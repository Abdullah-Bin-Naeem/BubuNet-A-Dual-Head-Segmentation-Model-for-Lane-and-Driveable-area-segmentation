# BuduNet: A Dual-Head Segmentation Model for Lane and Drivable Area Segmentation

## Demo
Below is a demonstration of BuduNet performing on unseen scenerio for drivable area and lane segmentation:

![BuduNet Demo](demo.gif)

### Sample Results
| Index    | Result |
|----------------------------|-------------------|
| Sample 1 | ![Lane](output_images/0ace96c3-48481887_overlay.png) |
| Sample 2 | ![Lane](output_images/0a98248b-de4df1d4_overlay.png) |


## Overview
BuduNet is a novel segmentation model inspired by YOLOP and TwinLiteNet, designed for autonomous driving tasks. It leverages a ResNet-50 backbone with a single decoder that simultaneously performs **drivable area segmentation** and **lane segmentation**. The model is trained on the [BDD100K dataset](https://www.bdd100k.com/) with a batch size of 10.

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

![BuduNet Architecture](architecture_diagram.png)

## Requirements
To run BuduNet, install the following dependencies:
```bash
pip install torch torchvision opencv-python pillow
```

## Directory Structure
```
BubuNet/
├── input_images/           # Folder for input images for inference
├── input_videos/           # Folder for input videos for inference
├── output_images/          # Folder for output images after inference
├── output_videos/          # Folder for output videos after inference
├── model/
│   └── budu.py            # Model definition
├── train.py               # Training script
├── infer_images.py        # Inference script for images
├── infer_videos.py        # Inference script for videos
├── architecture_diagram.png # Architecture diagram
├── demo.gif                # Demo GIF for README
├── output_images/
│   ├── drivable_sample1.jpg # Sample drivable area result
│   ├── lane_sample1.jpg     # Sample lane result
│   ├── drivable_sample2.jpg # Sample drivable area result
│   ├── lane_sample2.jpg     # Sample lane result
└── README.md               # This file
```

## Usage

### 1. Training
To train the model on the BDD100K dataset:
```bash
python train.py
```
- Ensure the BDD100K dataset is properly set up and accessible.
- The model is trained with a batch size of 10.

### 2. Inference on Images
To perform inference on images:
1. Place input images in the `input_images/` folder.
2. Run the inference script:
```bash
python infer_images.py
```
3. Results will be saved in the `output_images/` folder.

### 3. Inference on Videos
To perform inference on videos:
1. Place input videos in the `input_videos/` folder.
2. Run the inference script:
```bash
python infer_videos.py
```
3. Results will be saved in the `output_videos/` folder.

### 4. Model Modifications
To modify the BuduNet model:
1. Navigate to the `model/` directory.
2. Edit `budu.py` to make changes to the model architecture or parameters.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Abdullah-Bin-Naeem/BuduNet.git
cd BubuNet
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure the BDD100K dataset is downloaded and configured for training.

## Results
- **Drivable Area Segmentation**: Accurately identifies drivable regions in diverse driving conditions.
- **Lane Segmentation**: Detects lane markings with high precision.
- Outputs are saved as segmented images/videos in the respective output folders.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open an issue on GitHub or contact [abdullahbinnaeempro@gmail.com](mailto:abdullahbinnaeempro@gmail.com).
