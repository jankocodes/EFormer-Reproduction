import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet= resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        self.layer1= resnet.layer1
        
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        
        self.layer2 = resnet.layer2 #high-resolution features
        self.layer3 = resnet.layer3 #low-resolution features
        
        
    def forward(self, x):
        x = self.layer1(x) #(B,256,H/4,W/4)

        f_hr = self.layer2(x) #(B,512,H/8,W/8)
        f_lr = self.layer3(f_hr) #(B,1024,H/16,W/16)

        return f_hr, f_lr
    
    
