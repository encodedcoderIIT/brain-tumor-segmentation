import base64
import glob
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional

matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

# Define the UNet model
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => Dropout) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.5):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


model = UNet(n_channels=1, n_classes=4)
model.load_state_dict(torch.load('static/model/Amit_model_weights_t2f.pkl', map_location=torch.device('cpu'), weights_only=True))
model.eval()

def analyze(image):
    # print(image)
    # Assume `model` is your trained U-Net model
    # Move the model to GPU if available, otherwise use CPU
    device = torch.device('cpu')
    # model.to(device)  # Move the model to the correct device

    # Load the image from the provided file
    # img = nib.load(image)
    print("***************************************************************")
    # print(img)
    data = image.get_fdata()
    print(data.shape)

    # Select a 2D slice from the middle of the 3D image
    slice_idx = data.shape[2] // 2  # Middle slice
    t2ce_slice = data[:, :, slice_idx]

    # Convert the 2D slice to a tensor and add a batch dimension (shape: [1, 1, H, W])
    input_tensor = torch.tensor(t2ce_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Move input tensor to the same device as the model
    input_tensor = input_tensor.to(device)

    # Run the model to get the output (predicted mask)
    with torch.no_grad():  # No gradient calculation needed for inference
        output = model(input_tensor)  # Model output shape: [1, num_classes, H, W]

    # Assuming the output has multiple classes, use argmax to get the predicted class
    segmentation_map = torch.argmax(output.squeeze(0), dim=0)  # Shape: [H, W]

    # Convert the segmentation map to a NumPy array for visualization
    segmentation_map = segmentation_map.cpu().numpy()  # Ensure it's on CPU

    # Visualize the predicted mask overlay on the T2ce slice
    fig, ax = plt.subplots()
    ax.imshow(t2ce_slice.T, cmap='gray')  # Display the original image in grayscale
    ax.imshow(segmentation_map.T, cmap='jet', alpha=0.5)  # Overlay the predicted mask with transparency (alpha)
    # ax.set_title(f"Predicted Mask Overlay - Slice {slice_idx}")
    ax.axis('off')  # Hide axis for cleaner view
    plt.colorbar(ax.imshow(segmentation_map.T, cmap='jet', alpha=0.5))  # Color bar for segmentation

    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # print(img_base64)
    return {'prediction': img_base64}


