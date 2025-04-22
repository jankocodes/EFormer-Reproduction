import torch.nn as nn
from models.branches import SCD, CEEB, SEB
import torch

class TransformerBlock(nn.Module):
    def __init__(self, use_ca= True, use_sa= True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.scd= SCD(use_ca=use_ca, use_sa=use_sa)
        self.ceeb= CEEB()
        self.seb= SEB()
    
    def forward(self, f_hr_emb, f_lr_emb):
        
        f_semantic_contour= self.scd(f_hr_emb, f_lr_emb)
        
        ceeb_input= f_semantic_contour+f_hr_emb
        seb_input= f_semantic_contour+f_lr_emb
        
        f_contour= self.ceeb(ceeb_input)
        f_semantic= self.seb(seb_input)
        
        return f_contour,f_semantic
        
        

        
        