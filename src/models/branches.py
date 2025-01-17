import torch
import torch.nn as nn

#Semantic & Contour Detector
class SCD(nn.Module):
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        
        self.hr_layernorm= nn.LayerNorm(256)
        self.lr_layernorm= nn.LayerNorm(256)
        self.hr_lr_layernorm= nn.LayerNorm(256)
        
        self.positional_enc1= nn.Parameter(torch.zeros(1, 1, 256))
        
        self.cross_attention= nn.MultiheadAttention(256, 4)
        
        self.enhance_layernorm= nn.LayerNorm(256)
        
        self.positional_enc2= nn.Parameter(torch.zeros(1,1,256))
        
        self.self_attention= nn.MultiheadAttention(256, 4)
        
        
    
    def forward(self, f_hr_emb, f_lr_emb): 
        
        #dim: (N,B,256)
        f_hr_lr_emb= f_hr_emb+f_lr_emb
        
        #applying layernorm + adding positional encoding for k,q       
        k_ca= self.hr_layernorm(f_hr_emb) + self.positional_enc1 
        q_ca= self.lr_layernorm(f_lr_emb) + self.positional_enc1 
        v_ca= self.hr_lr_layernorm(f_hr_lr_emb) 
        
        #perform cross-attention
        f_contour_edge,_= self.cross_attention(k_ca,q_ca,v_ca) 
        
        f_enhance= f_contour_edge + v_ca 
        
        #applying layernorm + adding positional encoding for k,q
        f_enhance_ln= self.enhance_layernorm(f_enhance)
        
        k_sa= q_sa= f_enhance_ln + self.positional_enc2 
        v_sa= f_enhance_ln 
        
        #perform self-attention
        self_attention,_= self.self_attention(k_sa,q_sa,v_sa)
        f_semantic_contour= self_attention + v_sa 
        
        return f_semantic_contour
        
        
#Contour-Edge Extraction Branch
class CEEB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm= nn.LayerNorm(256)
        
        self.mlp= nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        ceeb_ln= self.layer_norm(x)
        
        f_contour= self.mlp(ceeb_ln)
        
        return f_contour+ceeb_ln
        
#Semantic Extraction Branch
class SEB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm= nn.LayerNorm(256)
        
        self.mlp= nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        seb_ln= self.layer_norm(x)
        
        f_semantic= self.mlp(seb_ln)
        
        return f_semantic+seb_ln
    

    