from model.budu import SegmentationModel
import torch
import numpy as np
import matplotlib.pyplot as plt
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

# Inference script
def run_inference(model, input_folder, output_folder, image_size=(512, 512)):
    """
    Run inference on all images in input_folder and save overlaid images to output_folder.
    
    Args:
        model: Trained PyTorch model
        input_folder: Path to folder containing input images
        output_folder: Path to folder to save overlaid images
        image_size: Tuple of (height, width) for resizing images
    
    Returns:
        fps: Frames per second for processing all images
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
    
    # Get list of image files
    image_paths = glob.glob(os.path.join(input_folder, '*.jpg')) + \
                  glob.glob(os.path.join(input_folder, '*.png'))
    
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found in the input folder.")
        return 0.0
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for img_path in image_paths:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
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
            
            # Save overlaid image
            output_filename = os.path.splitext(os.path.basename(img_path))[0] + '_overlay.png'
            output_path = os.path.join(output_folder, output_filename)
            plt.imsave(output_path, overlaid_image)
            print(f"Saved overlaid image: {output_path}")
    
    # Calculate FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_images / total_time if total_time > 0 else 0.0
    
    print(f"Processed {num_images} images in {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    return fps

# Main execution
if __name__ == "__main__":
    # Load model
    # model = torch.load('/home/abdullah/Documents/expADAS/checkpoints/SegmentationModel_epoch_10.pth')
    model = SegmentationModel(num_classes_drivable=2, num_classes_lane=2)
    # Note: If the saved file is only state_dict, use:
    model.load_state_dict(torch.load('/home/abdullah/Documents/expADAS/checkpoints/SegmentationModel_epoch_36.pth'))
    
    # Define input and output folders
    input_folder = '/home/abdullah/Documents/expADAS/input_images'
    output_folder = '/home/abdullah/Documents/expADAS/output_images'
    
    # Run inference
    run_inference(model, input_folder, output_folder, image_size=(360, 640))