import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import models
from torch.nn import functional as F
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import streamlit as st
from PIL import Image
from io import BytesIO
from src.imagecolorization.conponents.model_building import Generator, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ImageColorizationSystem:
    def __init__(self, generator_path, critic_path):
        self.generator = Generator(1, 2)  # Expecting 1 channel input, 2 channel output
        self.critic = Critic()  # Initialize your critic model
        self.generator.load_state_dict(torch.load(generator_path, map_location=device), strict=False)
        self.critic.load_state_dict(torch.load(critic_path, map_location=device), strict=False)
        self.generator.to(device)
        self.critic.to(device)
        self.generator.eval()
        self.critic.eval()

    def load_image(self, image):
        image = image.convert("L")  # Convert to grayscale (1 channel)
        image = image.resize((224, 224))  # Resize to the expected input size
        return image

    def colorize(self, bw_image):
        bw_tensor = transforms.ToTensor()(bw_image).unsqueeze(0).to(device)  # Move tensor to the correct device
        with torch.no_grad():
            colorized = self.generator(bw_tensor)
        colorized = colorized.cpu()  # Move tensor back to CPU for processing
        return self.lab_to_rgb(bw_tensor.squeeze(), colorized.squeeze())

    def lab_to_rgb(self, L, ab):
        # Ensure both tensors are on CPU
        L = L.cpu() * 100
        ab = (ab.cpu() * 2 - 1) * 128
        # Concatenate on CPU
        Lab = torch.cat([L.unsqueeze(0), ab], dim=0).numpy()  # Move to numpy for conversion
        Lab = np.moveaxis(Lab, 0, -1)
        rgb_img = lab2rgb(Lab)
        return rgb_img
