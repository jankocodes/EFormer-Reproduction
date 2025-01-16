import torch.nn as nn
import torch
import torchvision

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet= torchvision.models.resnet50(pretrained= True)
        
        self.layer1= resnet.layer1
        
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        
        self.layer2 = resnet.layer2 #high-resolution features
        self.layer3 = resnet.layer3 #low-resolution features
        
        
    def forward(self, x):
        x = self.layer1(x) #(B,256,H/4,W/4)

        f_hr = self.layer2(x) #(B,512,H/8,W/8)
        f_lr = self.layer3(f_hr) #(B,1024,H/16,W/16)

        return f_hr, f_lr
    
    
if __name__ == "__main__":
    # Example input: Batch of 2 images, each 3x224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Initialize backbone
    backbone = Backbone()
    
    # Forward pass
    f_hr, f_lr = backbone(dummy_input)
    
    print(f"High-resolution feature map shape: {f_hr.flatten(2).permute(2, 0, 1).shape}")  # Expected: [2, 1024, H/8, W/8]
    print(f"Low-resolution feature map shape: {f_lr.reshape([-1, 2, 256]).shape}")   # Expected: [2, 2048, H/16, W/16]
