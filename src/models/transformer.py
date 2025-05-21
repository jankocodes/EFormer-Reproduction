import torch.nn as nn
from models.branches import SCD, CEEB, SEB

class TransformerBlock(nn.Module):
    def __init__(self, use_ca= True, use_sa= True, use_seb= True, use_ceeb=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sa= use_sa
        self.use_ca= use_ca
        self.use_seb= use_seb
        self.use_ceeb= use_ceeb
        
        self.scd= SCD(use_ca=use_ca, use_sa=use_sa)
        
        if self.use_ceeb:
            self.ceeb= CEEB()
        if self.use_seb:    
            self.seb= SEB()
    
    def forward(self, f_hr_emb, f_lr_emb):
        
        f_semantic_contour= self.scd(f_hr_emb, f_lr_emb)
        
        ceeb_input= f_semantic_contour+f_hr_emb
        seb_input= f_semantic_contour+f_lr_emb
        
        f_contour= self.ceeb(ceeb_input) if self.use_ceeb else ceeb_input
        f_semantic= self.seb(seb_input) if self.use_seb else seb_input
        
        return f_contour,f_semantic
        
        

        
        