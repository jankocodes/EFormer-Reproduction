import torch
import torch.nn as nn

#Semantic & Contour Detector
class SCD(nn.Module):
    def __init__(self, use_ca= True, use_sa= True, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        
        #for ablation study 
        self.use_ca= use_ca
        self.use_sa= use_sa
        
        self.hr_layernorm= nn.LayerNorm(256)
        self.lr_layernorm= nn.LayerNorm(256)
        self.hr_lr_layernorm= nn.LayerNorm(256)
        
        self.max_seq_len = 3136  # depends on input resolution
        self.positional_enc1 = nn.Parameter(torch.zeros(self.max_seq_len, 1, 256))  # (N, 1, D)
        nn.init.trunc_normal_(self.positional_enc1, std=0.02)     
           
        self.cross_attention= nn.MultiheadAttention(256, 8)
        
        self.enhance_layernorm= nn.LayerNorm(256)
        
        self.positional_enc2 = nn.Parameter(torch.zeros(self.max_seq_len, 1, 256))  # (N, 1, D)
        nn.init.trunc_normal_(self.positional_enc2, std=0.02)
        
        self.self_attention= nn.MultiheadAttention(256, 8)
        
        
    
    def forward(self, f_hr_emb, f_lr_emb): 
        
        #dim: (N,B,256)
        f_hr_lr_emb= f_hr_emb+f_lr_emb
        
        if self.use_ca:
            #applying layernorm + adding positional encoding for k,q       
            seq_len = f_hr_lr_emb.size(0)  # N
            k_ca= self.hr_layernorm(f_hr_emb) + self.positional_enc1[:seq_len]
            q_ca= self.lr_layernorm(f_lr_emb) + self.positional_enc1[:seq_len]
            v_ca= self.hr_lr_layernorm(f_hr_lr_emb) 
            
            #perform cross-attention
            f_contour_edge,_= self.cross_attention(k_ca,q_ca,v_ca) 
            
            f_enhance= f_contour_edge + v_ca 
        else:
            # Skip CA: use HR-LR mix directly
            f_enhance= self.hr_lr_layernorm(f_hr_lr_emb) + f_hr_lr_emb
        
        
        if self.use_sa:
            #applying layernorm + adding positional encoding for k,q
            f_enhance_ln= self.enhance_layernorm(f_enhance)
            
            seq_len = f_enhance_ln.size(0)  # N
            k_sa= q_sa= f_enhance_ln + self.positional_enc2[:seq_len] 
            v_sa= f_enhance_ln 
            
            #perform self-attention
            att_out,_= self.self_attention(k_sa,q_sa,v_sa)
            f_semantic_contour= att_out + v_sa 
        else:
            f_semantic_contour= self.enhance_layernorm(f_enhance) + f_enhance
            
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
    

    