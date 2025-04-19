from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

import os

import numpy as np
import random

class EFormerDataset(Dataset):
    def __init__(self, root_dir, size, p_flip, device):
        self.root_dir = root_dir
        self.size = size
        self.pairs = []
        self.device= device
        
        self.com_dir= os.path.join(root_dir, 'composites')
        self.pha_dir= os.path.join(root_dir, 'pha')
        
        com_files = set(os.listdir(self.com_dir))
        pha_files = set(os.listdir(self.pha_dir))
        
        self.filenames = sorted(com_files.intersection(pha_files))  # Ensure matching pairs
        
        self.p_flip= p_flip
        
        
        

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        composite_path = os.path.join(self.com_dir, name)
        pha_path = os.path.join(self.pha_dir, name)

        # Load images
        composite = Image.open(composite_path).convert("RGB")
        pha = Image.open(pha_path).convert("L")  # Alpha is grayscale

        # Perform flipping and resizing on the CPU (efficient)
        if random.random() < self.p_flip:  
            composite = F.hflip(composite)
            pha = F.hflip(pha)
        
        composite = F.resize(composite, self.size)
        pha = F.resize(pha, self.size)

        # Convert images to tensors (on CPU by default)
        composite = F.to_tensor(composite)
        pha = F.to_tensor(pha)
        
        # Pin memory (on CPU tensors) before moving to device (GPU)
        composite = composite.pin_memory()
        pha = pha.pin_memory()

        # Move tensors to device (GPU)
        composite = composite.to(self.device, non_blocking=True)
        pha = pha.to(self.device, non_blocking=True)
        
        return composite, pha


