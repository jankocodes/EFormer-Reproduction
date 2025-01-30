import torch.nn  as nn
from backbone import Backbone
from transformer import TransformerBlock
import torch

class EFormer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #backbone
        self.backbone= Backbone()
        
        self.proj_hr = nn.Conv2d(512, 256, kernel_size=1)
        self.proj_lr = nn.Conv2d(1024, 256, kernel_size=1)
        
        # transformer
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock(),
            TransformerBlock()
        )
        
        #prediction stage
        self.conv_semantic = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv_contour = nn.Conv2d(256, 256, kernel_size=3,padding=1)

        self.conv_fuse = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        
        self.head= nn.Conv2d(256, 1, kernel_size=3, padding=1 )
        
        self.sigmoid= nn.Sigmoid()
        
    
    def forward(self, x):
 
        B,_,H,W= x.shape
        
        #resnet50 backbone
        f_enc, f_hr, f_lr= self.backbone(x) #(B,512,H/8,W/8), (B,1024,H/16,W/16)
        
        f_hr = self.proj_hr(f_hr) #(B,256,H/8,W/8)
        f_lr = self.proj_lr(f_lr) #(B,256,H/16,W/16)
        
        f_lr_upsampled = nn.functional.interpolate(
            f_lr, size=f_hr.shape[2:], mode='bilinear', align_corners=False #(B,256,H/8,W/8)
        )

        #(B,256,H/8,W/8) -> (N,B,256)
        f_hr_emb = f_hr.flatten(2).permute(2, 0, 1)  
        f_lr_emb = f_lr_upsampled.flatten(2).permute(2, 0, 1) 
        
        #transformer
        f_contour, f_semantic = f_hr_emb, f_lr_emb
        for block in self.transformer_blocks:
            f_contour, f_semantic = block(f_contour, f_semantic)
        
        #(N,B,256) -> (B,256,N) -> (B,256,H//8,W//8)
        f_contour= f_contour.permute(1,2,0).view(B,256,H//8,W//8)
        f_semantic= f_semantic.permute(1,2,0).view(B,256,H//8,W//8)
        
        f_semantic= self.conv_semantic(f_semantic+f_lr_upsampled)
        f_contour= self.conv_contour(f_contour+f_hr)
        
        #prediction stage
        f_semantic_contour= self.conv_fuse(f_semantic+f_contour)
        
        fused_features= f_enc + nn.functional.interpolate(
            f_semantic_contour, size= f_enc.shape[2:], mode= 'bilinear',align_corners=False)
        
        matte= self.head(fused_features)
        
        #upsample to original size
        matte= nn.functional.interpolate(matte, (H,W))
        
        matte= self.sigmoid(matte)
        print(matte)

        return matte
            
            
        
        
        
        
        

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224) 
    model = EFormer()
    model(dummy_input)
    
    
