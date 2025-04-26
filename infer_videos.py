from model.budu import SegmentationModel
import torch
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import glob
import time
from utils import overlay_masks_on_image

# Unnormalize function
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize a tensor image with given mean and std.
    
    Args:
        tensor: Tensor of shape (C, H, W)
        mean: List of mean values for each channel
        std: List of std values for each channel
    
    Returns:
        Unnormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Inference script for videos
def run_video_inference(model, input_folder, output_folder, image_size=(512, 512)):
    """
    Run inference on all videos in input_folder and save overlaid videos to output_folder.
    
    Args:
        model: Trained PyTorch model
        input_folder: Path to folder containing input videos
        output_folder: Path to folder to save overlaid videos
        image_size: Tuple of (height, width) for resizing frames
    
    Returns:
        fps: Frames per second for processing all frames across all videos
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get list of video files
    video_paths = glob.glob(os.path.join(input_folder, '*.mp4')) + \
                  glob.glob(os.path.join(input_folder, '*.avi'))
    
    if not video_paths:
        print("No videos found in the input folder.")
        return 0.0
    
    total_frames = 0
    start_time = time.time()
    
    with torch.no_grad():
        for video_path in video_paths:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(image_size[1])
            height = int(image_size[0])
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Define output video path
            output_filename = os.path.splitext(os.path.basename(video_path))[0] + '_overlay.mp4'
            output_path = os.path.join(output_folder, output_filename)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to RGB and PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Preprocess frame
                input_tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, C, H, W)
                
                # Run inference
                drivable_out, lane_out = model(input_tensor)
                drivable_pred = torch.softmax(drivable_out, dim=1).argmax(dim=1)[0]  # Shape: (H, W)
                lane_pred = torch.softmax(lane_out, dim=1).argmax(dim=1)[0]  # Shape: (H, W)
                
                # Unnormalize input for visualization
                input_display = unnormalize(input_tensor[0].clone(), 
                                          mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])
                
                # Overlay masks
                overlaid_image = overlay_masks_on_image(input_display, drivable_pred, lane_pred, alpha=0.5)
                
                # Convert to BGR for OpenCV
                overlaid_image_bgr = cv2.cvtColor((overlaid_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Write frame to output video
                out.write(overlaid_image_bgr)
            
            # Release resources
            cap.release()
            out.release()
            print(f"Saved overlaid video: {output_path}")
    
    # Calculate FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = total_frames / total_time if total_time > 0 else 0.0
    
    print(f"Processed {total_frames} frames across {len(video_paths)} videos in {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    return fps

# Main execution
if __name__ == "__main__":

    model = SegmentationModel(num_classes_drivable=2, num_classes_lane=2)
    # Note: If the saved file is only state_dict, use:
    model.load_state_dict(torch.load('/home/abdullah/Documents/expADAS/checkpoints/SegmentationModel_epoch_31.pth'))

    # Define input and output folders
    input_folder = '/home/abdullah/Documents/expADAS/input_videos'
    output_folder = '/home/abdullah/Documents/expADAS/output_videos'
    
    # Run inference
    run_video_inference(model, input_folder, output_folder, image_size=(360, 640))