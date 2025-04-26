import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize a batch of images that were normalized with given mean and std.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [batch_size, C, H, W].
        mean (list): Mean values used for normalization (default: ImageNet mean).
        std (list): Standard deviation values used for normalization (default: ImageNet std).
    
    Returns:
        torch.Tensor: Unnormalized tensor of the same shape.
    """
    tensor = tensor.clone()  # Avoid modifying the original tensor
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    tensor.mul_(std).add_(mean)  # Reverse: (x * std) + mean
    return tensor

def save_checkpoint(model, model_name, epoch, save_path='checkpoints'):
    """
    Save the model's state dictionary with model name and epoch number.
    
    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Name of the model (e.g., 'SegmentationModel').
        epoch (int): Current epoch number.
        save_path (str): Directory to save the checkpoint (default: 'checkpoints').
    """
    import os
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Construct the filename with model name and epoch
    filename = f"{model_name}_epoch_{epoch + 1}.pth"
    filepath = os.path.join(save_path, filename)
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), filepath)
    print(f"Saved checkpoint: {filepath}")

# Define the overlay function
def overlay_masks_on_image(image, drivable_mask, lane_mask, alpha=0.5):
    """
    Overlay drivable area (green) and lane (red) masks on the input image, with lane mask dominant in overlaps.
    
    Args:
        image: Tensor or ndarray of shape (H, W, 3), RGB image normalized to [0,1]
        drivable_mask: Tensor of shape (H, W), binary mask for drivable area
        lane_mask: Tensor of shape (H, W), binary mask for lanes
        alpha: Transparency of the overlay (0 to 1)
    
    Returns:
        overlaid_image: ndarray of shape (H, W, 3), image with overlaid masks
    """
    # Convert image to numpy if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Ensure image is in [0,1]
    image = np.clip(image, 0, 1)
    
    # Convert masks to numpy if they are tensors
    drivable_mask = drivable_mask.cpu().numpy() if isinstance(drivable_mask, torch.Tensor) else drivable_mask
    lane_mask = lane_mask.cpu().numpy() if isinstance(lane_mask, torch.Tensor) else lane_mask
    
    # Create an empty overlay
    overlay = np.zeros_like(image)
    
    # Add green for drivable area (RGB: [0, 1, 0])
    overlay[drivable_mask == 1, 1] = 1
    
    # Add red for lanes (RGB: [1, 0, 0]), overwriting any overlapping areas
    overlay[lane_mask == 1, :] = [1, 0, 0]
    
    # Blend the overlay with the original image
    overlaid_image = image * (1 - alpha) + overlay * alpha
    overlaid_image = np.clip(overlaid_image, 0, 1)
    
    return overlaid_image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0



@torch.no_grad()
def val(val_loader, model):

    model.eval()


    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    total_batches = len(val_loader)
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().float() / 255.0
            # target = target.cuda()

        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)

        out_da,out_ll=output
        target_da,target_ll=target

        _,da_predict=torch.max(out_da, 1)
        _,da_gt=torch.max(target_da, 1)

        _,ll_predict=torch.max(out_ll, 1)
        _,ll_gt=torch.max(target_ll, 1)
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())


        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))


        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())


        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    return da_segment_result,ll_segment_result


def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
