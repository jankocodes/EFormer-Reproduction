import torch.nn  as nn
from models.backbone import Backbone
from models.transformer import TransformerBlock
import torch

class EFormer(nn.Module):
    def __init__(self, use_ca= True, use_sa= True, hr_dim= "1_8", lr_dim= "1_16", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hr_dim= hr_dim
        self.lr_dim= lr_dim
        
        #backbone
        self.backbone= Backbone()
        
        channels = {"1_4": 256, "1_8": 512, "1_16": 1024}
        self.proj_hr = nn.Conv2d(channels[hr_dim], 256, kernel_size=1)
        self.proj_lr = nn.Conv2d(channels[lr_dim], 256, kernel_size=1)

        
        # transformer
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(use_ca=use_ca, use_sa=use_sa),
            TransformerBlock(use_ca=use_ca, use_sa=use_sa),
            TransformerBlock(use_ca=use_ca, use_sa=use_sa),
            TransformerBlock(use_ca=use_ca, use_sa=use_sa)
        )
        
        #prediction stage
        self.conv_semantic = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv_contour = nn.Conv2d(256, 256, kernel_size=3,padding=1)

        self.conv_fuse = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        
        self.head= nn.Conv2d(256, 1, kernel_size=3, padding=1 )
        
        self.sigmoid= nn.Sigmoid()
        
    
    def forward(self, x):
 
        B,_,H,W= x.shape
        
        # resnet50 backbone 
        features= self.backbone(x) #(B,512,hr_dim,hr_dim), (B,1024,lr_dim,lr_dim)
        f_enc= features["1_4"]
        f_hr= features[self.hr_dim]
        f_lr= features[self.lr_dim]
        
        f_hr = self.proj_hr(f_hr) #(B,256,hr_dim,hr_dim)
        f_lr = self.proj_lr(f_lr) #(B,256,lr_dim,lr_dim)
        
        #upsample lr_features to hr_dim
        f_lr_upsampled = nn.functional.interpolate(
            f_lr, size=f_hr.shape[2:], mode='bilinear', align_corners=False #(B,256,hr_dim,hr_dim)
        )

        #(B,256, hr_dim, hr_dim) -> (N,B,256)
        f_hr_emb = f_hr.flatten(2).permute(2, 0, 1)  
        f_lr_emb = f_lr_upsampled.flatten(2).permute(2, 0, 1) 
        
        #transformer
        f_contour, f_semantic = f_hr_emb, f_lr_emb
        for block in self.transformer_blocks:
            f_contour, f_semantic = block(f_contour, f_semantic)
        
        #(N,B,256) -> (B,256,N) -> (B,256,hr_dim,hr_dim)
        hr_h, hr_w = f_hr.shape[2:]
        f_contour= f_contour.permute(1,2,0).view(B,256,hr_h, hr_w)
        f_semantic= f_semantic.permute(1,2,0).view(B,256,hr_h, hr_w)
        
        f_semantic= self.conv_semantic(f_semantic+f_lr_upsampled)
        f_contour= self.conv_contour(f_contour+f_hr)
        
        #prediction stage
        f_semantic_contour= self.conv_fuse(f_semantic+f_contour)
        
        #upsample transformer output if hr_dim != 1_4
        if f_semantic_contour.shape[2:] != f_enc.shape[2:]:
            f_semantic_contour_up = nn.functional.interpolate(
                f_semantic_contour, size=f_enc.shape[2:], mode='bilinear', align_corners=False) #adapt to transConv
        else:
            f_semantic_contour_up= f_semantic_contour
                
        fused_features = f_enc + f_semantic_contour_up
        
        matte= self.head(fused_features)
        
        #upsample to original size
        matte= nn.functional.interpolate(matte, size=(H,W), mode='bilinear', align_corners=False)
        
        matte= self.sigmoid(matte)

        return matte
            
            
        
        
        
        
        

