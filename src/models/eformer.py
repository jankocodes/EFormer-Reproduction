import torch.nn  as nn
from backbone import Backbone
from transformer import TransformerBlock
import torch

class EFormer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.backbone= Backbone()
        
        self.proj_hr = nn.Conv2d(512, 256, kernel_size=1)
        self.proj_lr = nn.Conv2d(1024, 256, kernel_size=1)
        
        # 4 transformer blocks
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock()
        )
        
    
    def forward(self, x):
        #resnet50 backbone
        f_hr, f_lr= self.backbone(x) #(B,512,H/8,W/8), (B,1024,H/16,W/16)
        
        f_hr = self.proj_hr(f_hr) #(B,256,H/8,W/8)
        f_lr = self.proj_lr(f_lr) #(B,256,H/16,W/16)
        
        f_lr_upsampled = nn.functional.interpolate(
            f_lr, size=f_hr.shape[2:], mode='bilinear', align_corners=False #(B,256,H/8,W/8)
        )

        #change dimension
        f_hr_emb = f_hr.flatten(2).permute(2, 0, 1)  #(N,B,256)
        f_lr_emb = f_lr_upsampled.flatten(2).permute(2, 0, 1)  #(N,B,256)
        
        #transformer
        f_contour, f_semantic = f_hr_emb, f_lr_emb
        for block in self.transformer_blocks:
            f_contour, f_semantic = block(f_contour, f_semantic)
        
        
        
        

if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    model = EFormer()
    model(dummy_input)
    
