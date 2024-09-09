import torch 
import torch.nn as nn
from pathlib import Path
from torchsummary import summary
from src.imagecolorization.entity.config_entity import ModelBuildingConfig
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.identity_map = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = inputs.clone().detach()
        out = self.layer(x)
        residual  = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)
    
    
class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)
    
    
    
class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate = 0.2):
        super().__init__()
        self.encoding_layer1_ = ResBlock(input_channel,64)
        self.encoding_layer2_ = DownSampleConv(64, 128)
        self.encoding_layer3_ = DownSampleConv(128, 256)
        self.bridge = DownSampleConv(256, 512)
        self.decoding_layer3_ = UpSampleConv(512, 256)
        self.decoding_layer2_ = UpSampleConv(256, 128)
        self.decoding_layer1_ = UpSampleConv(128, 64)
        self.output = nn.Conv2d(64, output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, inputs):
        ###################### Enocoder #########################
        e1 = self.encoding_layer1_(inputs)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)
        
        ###################### Bridge #########################
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)
        
        ###################### Decoder #########################
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)
        
        ###################### Output #########################
        output = self.output(d1)
        return output
    
class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic, self).__init__()

        def critic_block(in_filters, out_filters, normalization=True):
            """Returns layers of each critic block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *critic_block(in_channels, 64, normalization=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, ab, l):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((ab, l), 1)
        output = self.model(img_input)
        return output
    
from torchsummary import summary
import torch
import os

class ModelBuilding:
    def __init__(self, config: ModelBuildingConfig):
        self.config = config
        self.root_dir = self.config.root_dir
        self.create_root_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_root_dir(self):
        os.makedirs(self.root_dir, exist_ok=True)
        print(f"Created directory: {self.root_dir}")

    def get_generator(self):
        return Generator(
            input_channel=self.config.INPUT_CHANNELS,  
            output_channel=self.config.OUTPUT_CHANNELS,  
            dropout_rate=self.config.DROPOUT_RATE
        ).to(self.device)

    def get_critic(self):
        return Critic(in_channels=self.config.IN_CHANNELS).to(self.device)

    def build(self):
        generator = self.get_generator()
        critic = self.get_critic()
        return generator, critic

    def save_model(self, model, filename):
        path = self.root_dir / filename
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def display_summary(self, model, input_size):
        print(f"\nModel Summary:")
        summary(model, input_size)

    def build_and_save(self):
        generator, critic = self.build()

        # Display summaries
        print("\nGenerator Summary:")
        self.display_summary(generator, (self.config.INPUT_CHANNELS, 224, 224))  # Assuming input size is 224x224

        print("\nCritic Summary:")
        self.display_summary(critic, [(2, 224, 224), (1, 224, 224)])  # Critic takes two inputs: ab and l

        self.save_model(generator, "generator.pth")
        self.save_model(critic, "critic.pth")
        return generator, critic
