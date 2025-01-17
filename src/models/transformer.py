import torch.nn as nn
from branches import SCD, CEEB, SEB
import torch

class TransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.scd= SCD()
        self.ceeb= CEEB()
        self.seb= SEB()
    
    def forward(self, f_hr, f_lr):
        f_hr_emb = f_hr.flatten(2).permute(2, 0, 1)  #(N,B,256)
        f_lr_emb = f_lr.flatten(2).permute(2, 0, 1)  #(N,B,256)
        
        f_semantic_contour= self.scd(f_hr_emb, f_lr_emb)
        
        ceeb_input= f_semantic_contour+f_hr_emb
        seb_input= f_semantic_contour+f_lr_emb
        
        f_contour= self.ceeb(ceeb_input)
        f_semantic= self.seb(seb_input)
        
        return f_contour,f_semantic
        
        
        
if __name__=="__main__":
    t= TransformerBlock()
    x= torch.randn(24, 256, 18,18)
    
    fc,fs= t(x,x)
    
    print(fc.shape)
    print(fs.shape)
        
        
        
        
        