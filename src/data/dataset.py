from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
from glob import glob
import numpy as np

class EFormerDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        
        self.com_dir= os.path.join(root_dir, 'composites')
        self.pha_dir= os.path.join(root_dir, 'pha')
        
        com_files = set(os.listdir(self.com_dir))
        pha_files = set(os.listdir(self.pha_dir))
        
        self.filenames = sorted(com_files.intersection(pha_files))  # Ensure matching pairs
        

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        composite_path= os.path.join(self.com_dir, name)
        pha_path = os.path.join(self.pha_dir, name)

        composite = Image.open(composite_path).convert("RGB")
        pha = Image.open(pha_path).convert("L")  # Alpha is grayscale
        
        

        if self.transform:
            composite = self.transform(composite)
            pha = self.transform(pha)

        return composite, pha

