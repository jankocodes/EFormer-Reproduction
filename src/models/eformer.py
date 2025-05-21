import torch.nn  as nn
from models.backbone import Backbone
from models.transformer import TransformerBlock

class EFormer(nn.Module):
    def __init__(self,
                 use_ca= True,
                 use_sa= True,
                 use_seb= True,
                 use_ceeb= True,
                 hr_dim= "1_8",
                 lr_dim= "1_16",
                 first_upsampling= "bilinear",
                 second_upsampling= "transConv",
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.hr_dim= hr_dim
        self.lr_dim= lr_dim
        self.first_upsampling= first_upsampling
        self.second_upsampling= second_upsampling
        
        #backbone
        self.backbone= Backbone()
        
        channels = {"1_4": 256, "1_8": 512, "1_16": 1024}
        self.proj_hr = nn.Conv2d(channels[hr_dim], 256, kernel_size=1)
        self.proj_lr = nn.Conv2d(channels[lr_dim], 256, kernel_size=1)
        
        if self.first_upsampling=="transconv":
            self.first_transconv= nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        
        # transformer
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(use_ca=use_ca, use_sa=use_sa, use_seb= use_seb, use_ceeb=use_ceeb),
            TransformerBlock(use_ca=use_ca, use_sa=use_sa, use_seb= use_seb, use_ceeb=use_ceeb),
            TransformerBlock(use_ca=use_ca, use_sa=use_sa, use_seb= use_seb, use_ceeb=use_ceeb),
            TransformerBlock(use_ca=use_ca, use_sa=use_sa, use_seb= use_seb, use_ceeb=use_ceeb)
        )
        
        #prediction stage
        self.conv_semantic = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv_contour = nn.Conv2d(256, 256, kernel_size=3,padding=1)

        self.conv_fuse = nn.Conv2d(256, 256, kernel_size=1)
        
        if second_upsampling== "transconv":
            self.second_transconv = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        
        self.head= nn.Conv2d(256, 1, kernel_size=3, padding=1 )
        
        self.sigmoid= nn.Sigmoid()
        
    def upsample_lr_features(self, x, size):
        if self.first_upsampling == 'bilinear':
            return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
       
        elif self.first_upsampling == 'transconv':     
            out = self.first_transconv(x)
            return out
        
        else:
            raise ValueError(f"Unsupported upsampling method: {self.first_upsampling}")
        
    
    def upsample_semantic_contour_features(self, x, size):
        if self.second_upsampling == 'bilinear':
            return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
        elif self.second_upsampling == 'transconv':
            out = self.second_transconv(x)
            return out
        
        else:
            raise ValueError(f"Unsupported upsampling method: {self.second_upsampling}")
    
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
        f_lr_upsampled = self.upsample_lr_features(f_lr, size=f_hr.shape[2:])

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
        
            f_semantic_contour_up= self.upsample_semantic_contour_features(f_semantic_contour, size=f_enc.shape[2:])
        else:
            f_semantic_contour_up= f_semantic_contour
                
        fused_features = f_enc + f_semantic_contour_up
        
        matte= self.head(fused_features)
        
        #upsample to original size
        matte= nn.functional.interpolate(matte, size=(H,W), mode='bilinear', align_corners=False)
        
        matte= self.sigmoid(matte)

        return matte
            
            
        
        
        
        
        

