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
        
        
    
    def forward(self, f_hr, f_lr):
        
        f_hr_emb = f_hr.flatten(2).permute(2, 0, 1)  #(N,B,256)
        f_lr_emb = f_lr.flatten(2).permute(2, 0, 1)  #(N,B,256)
        f_hr_lr_emb= f_hr_emb+f_lr_emb
        
        #applying layernorm + adding positional encoding for k,q       
        k_ca= self.hr_layernorm(f_hr_emb) + self.positional_enc1 #(2)
        q_ca= self.lr_layernorm(f_lr_emb) + self.positional_enc1 #(3)
        v_ca= self.hr_lr_layernorm(f_hr_lr_emb) #(4)
        
        #perform cross-attention
        f_contour_edge,_= self.cross_attention(k_ca,q_ca,v_ca) #(5)
        
        f_enhance= f_contour_edge + v_ca #(6)
        
        #applying layernorm + adding positional encoding for k,q
        f_enhance_ln= self.enhance_layernorm(f_enhance)
        
        k_sa= q_sa= f_enhance_ln + self.positional_enc2 #(7)
        v_sa= f_enhance_ln #(8)
        
        #perform self-attention
        self_attention,_= self.self_attention(k_sa,q_sa,v_sa)
        f_semantic_contour= self_attention + v_sa #(9)
        
        
        
        print(f_semantic_contour.shape)
        
        
        

#Contour-Edge Extraction Branch
class CEEB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

#Semantic Extraction Branch
class SEB(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
if __name__=='__main__':
    scd= SCD()
    dummy= torch.randn(2, 256, 18, 18)
    
    scd(dummy,dummy)
    