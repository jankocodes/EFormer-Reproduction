from pathlib import Path
from preprocessing_utils import unpack_bg20k, unpack_videomatte240k, sample_bg10k
import os 

if __name__=='__main__':
    root= Path(__file__).parent.parent.parent
    
    #assuming Videomatte240k and BG20K are installed in root/datasets/raw:
    raw_path= os.path.join(root, 'datasets', 'raw')
    bg20k_path= os.path.join(raw_path, 'bg20k')
    videomatte240k_path= os.path.join(raw_path, 'VideoMatte240K_JPEG_SD')
    
    #unpack datasets
    unpack_bg20k(bg20k_path)
    unpack_videomatte240k(videomatte240k_path) 
    
    #create bg10k folder    
    bg10k_path= os.path.join(raw_path, 'bg10k')
    
    os.mkdir(bg10k_path)
    os.mkdir(os.path.join(bg10k_path, 'train'))
    os.mkdir(os.path.join(bg10k_path, 'test'))
    
    #sample 10k images from bg20k
    sample_bg10k(bg20k_path, bg10k_path)
    