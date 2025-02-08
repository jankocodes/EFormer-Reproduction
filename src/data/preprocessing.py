from pathlib import Path
from preprocessing_utils import unpack_bg20k, unpack_videomatte240k, sample_bg10k, split_train_set, blend_foreground_with_background
import os 

if __name__=='__main__':
    root= Path(__file__).parent.parent.parent
    
    #assuming Videomatte240k, AIM and BG20K are installed in root/datasets/raw:
    raw_path= os.path.join(root, 'datasets', 'raw')
    bg20k_path= os.path.join(raw_path, 'bg20k')
    videomatte240k_path= os.path.join(raw_path, 'VideoMatte240K_JPEG_SD')
    aim_path= os.path.join(raw_path, 'AIM') 
    
    #unpack datasets
    unpack_bg20k(bg20k_path)
    unpack_videomatte240k(videomatte240k_path) 
    
    #create bg10k folder    
    bg10k_path= os.path.join(raw_path, 'bg10k')
    
    os.makedirs(bg10k_path, exist_ok=True)
    os.makedirs(os.path.join(bg10k_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(bg10k_path, 'test'), exist_ok=True)
    
    #sample 10k images from bg20k
    sample_bg10k(bg20k_path, bg10k_path)
    
    #split train_set to train/val, (train/test already split)
    videomatte240k_val_path= os.path.join(videomatte240k_path, 'val')
    
    os.makedirs(videomatte240k_val_path, exist_ok=True)
    os.makedirs(os.path.join(videomatte240k_val_path, 'fgr'), exist_ok=True)
    os.makedirs(os.path.join(videomatte240k_val_path, 'pha'), exist_ok=True)
    
    split_train_set(videomatte240k_path)
    
    #composition stage
    composite_dataset_path= os.path.join(root,'datasets', 'composite_dataset')
    splits= ['train', 'val', 'test']
    
    os.makedirs(composite_dataset_path, exist_ok=True)

    #create composite-split and subfolders for VideoMatte240K
    for split in splits:
        composite_split_path= os.path.join(composite_dataset_path, split)
        os.makedirs(composite_split_path, exist_ok=True)
        os.makedirs(os.path.join(split, 'fgr'), exist_ok=True)
        os.makedirs(os.path.join(split, 'pha'), exist_ok=True)

        #compose 
        blend_foreground_with_background(os.path.join(videomatte240k_path, split),
                                        os.path.join(bg10k_path, split),
                                        composite_split_path)
    
    #compose AIM images
    blend_foreground_with_background(aim_path,
                                  os.path.join(bg10k_path, 'train'),
                                  os.path.join(composite_dataset_path, 'train')
                                  )