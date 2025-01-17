import torch.nn  as nn
from backbone import Backbone
import torch

class EFormer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone= Backbone()
        
        # Projection layers
        self.proj_hr = nn.Conv2d(512, 256, kernel_size=1)
        self.proj_lr = nn.Conv2d(1024, 256, kernel_size=1)
    
    def forward(self, x):
        #resnet50 backbone
        f_hr, f_lr= self.backbone(x) #(B,512,H/8,W/8), (B,1024,H/16,W/16)
        
        f_hr = self.proj_hr(f_hr) #(B,256,H/8,W/8)
        f_lr = self.proj_lr(f_lr) #(B,256,H/16,W/16)
        
        f_lr_upsampled = nn.functional.interpolate(
            f_lr, size=f_hr.shape[2:], mode='bilinear', align_corners=False #(B,256,H/8,W/8)
        )
        
        f_hr_flattened = f_hr.flatten(2).permute(2, 0, 1)  #(N,B,256)
        f_lr_flattened = f_lr_upsampled.flatten(2).permute(2, 0, 1)  #(N,B,256)

        print(f_hr_flattened.shape)
        print(f_lr_flattened.shape)
        

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    model = EFormer()
    model(dummy_input)
    
