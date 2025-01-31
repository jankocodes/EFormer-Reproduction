from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path

class EFormerDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        
        self.fgr_dir= os.path.join(root_dir, 'fgr')
        self.pha_dir= os.path.join(root_dir, 'pha')
        
        fgr_files = set(os.listdir(self.fgr_dir))
        pha_files = set(os.listdir(self.pha_dir))
        
        self.filenames = sorted(fgr_files.intersection(pha_files))  # Ensure matching pairs

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        fgr_path = os.path.join(self.fgr_dir, name)
        pha_path = os.path.join(self.pha_dir, name)

        fgr = Image.open(fgr_path).convert("RGB")
        pha = Image.open(pha_path).convert("L")  # Alpha is grayscale

        if self.transform:
            fgr = self.transform(fgr)
            pha = self.transform(pha)

        return fgr, pha

