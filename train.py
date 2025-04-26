from budu import SegmentationModel
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import dataset as myDataLoader
import cv2
from utils import unnormalize , save_checkpoint

train_batch_size= 10
test_batch_size= 10
num_epochs= 10


# Training Function
def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(31, num_epochs):
        model.train()
        train_loss = 0.0
        
        # # Visualize one batch at the start of each epoch
        # for data in train_loader:
        #     if data is None:
        #         continue
        #     image_names, images, (seg_da, seg_ll) = data
        #     print(f"Epoch {epoch+1} - Visualizing batch shapes:")
        #     print(f"Image shape: {images.shape}, Drivable mask shape: {seg_da.shape}, Lane mask shape: {seg_ll.shape}")
        #     myDataLoader.visualize_batch(image_names, images, (seg_da, seg_ll), num_samples=4)
        #     break

        # Training loop
        for i, data in enumerate(train_loader):
            if data is None:
                continue
            image_names, inputs, (seg_da, seg_ll) = data
            inputs = inputs.to(device)
            seg_da = seg_da.to(device).argmax(dim=1)  # [64, 360, 640]
            seg_ll = seg_ll.to(device).argmax(dim=1)

            optimizer.zero_grad()
            drivable_out, lane_out = model(inputs)
            # import pdb; pdb.set_trace()
            loss_drivable = criterion(drivable_out, seg_da)
            loss_lane = criterion(lane_out, seg_ll)
            loss = loss_drivable + loss_lane
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 200 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")


        print(f"Epoch {epoch+1}, Avg Train Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue
                image_names, inputs, (seg_da, seg_ll) = data
                inputs = inputs.to(device)
                seg_da = seg_da.to(device).argmax(dim=1)
                seg_ll = seg_ll.to(device).argmax(dim=1)

                drivable_out, lane_out = model(inputs)
                loss_drivable = criterion(drivable_out, seg_da)
                loss_lane = criterion(lane_out, seg_ll)
                val_loss += (loss_drivable + loss_lane).item()
            inputs = unnormalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # myDataLoader.visualize_batch(image_names, inputs.cpu(), (drivable_out.cpu(), lane_out.cpu()), num_samples=4)
        print(f"Epoch {epoch+1}, Avg Val Loss: {val_loss/len(val_loader):.4f}")
        # Save checkpoint
        save_checkpoint(model, "SegmentationModel", epoch, save_path='checkpoints')
        
if __name__ == "__main__":

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(),
        batch_size=train_batch_size, shuffle=True, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=test_batch_size, shuffle=False, pin_memory=True)

    # Model
    model = SegmentationModel(num_classes_drivable=2, num_classes_lane=2)
    model.load_state_dict(torch.load('/home/abdullah/Documents/expADAS/checkpoints/SegmentationModel_epoch_31.pth'))

    train_model(model, trainLoader, valLoader, num_epochs=100, device='cuda')

