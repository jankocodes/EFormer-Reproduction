import os
from os import path
from random import sample, shuffle
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def unpack_bg20k(bg20k: str):  
    """
    Unpacks the BG20K dataset, which is a tarball of folders containing images.
    The folders are named '<number>/BG-20K', where <number> is a number from 1-7.
    Each folder contains two subfolders, 'train' and 'test', which contain the images.
    This function moves the images out of the subfolders and into the root of the dataset folder.
    
    Parameters
    ----------
    bg20k : str
        The path to the root of the BG20K dataset.
    
    Returns
    -------
    None
    """

    n_img= 0
    
    #bg20k/
    for f in os.listdir(bg20k):
        
        #bg20k/number/BG-20K
        f_path= path.join(bg20k, f, 'BG-20K')
        
        if path.isdir(f_path):
            for split in os.listdir(f_path):
                
                #bg20k/number/BG-20K/train
                split_path= path.join(f_path, split)
                
                #images
                if path.isdir(split_path):
                    for bg in os.listdir(split_path):  
                        
                        #bg20k/number/BG-20K/train/bg.jpg
                        src= path.join(split_path, bg)
                        
                        #bg20k/number/BG-20K/bg.jpg
                        dest= path.join(bg20k, bg)
                        
                        shutil.move(src, dest)           
                        n_img+=1
        
        f_num=path.join(bg20k, f)       
        
        if path.isdir(f_num):
            shutil.rmtree(f_num)
                                
    print(f'Unpacked {n_img} images from {bg20k}.')


        

def unpack_videomatte240k(videomatte240k):
    
    """
    Unpacks the Videomatte240k dataset, which is a tarball of folders containing scenes.
    Each scene has a subfolder 'fgr' containing the foreground images and a subfolder 'pha' containing the alpha mattes.
    This function moves the images out of the subfolders and into the root of the dataset folder.
    
    Parameters
    ----------
    videomatte240k : str
        The path to the root of the Videomatte240k dataset.
    
    Returns
    -------
    None
    """
    def unpack_split(split_path):
    
        #source    
        fgr_path= path.join(split_path, 'fgr')
        pha_path= path.join(split_path, 'pha')
        
        #destination
        n_img= 0
        
        #scenes
        for scene_fgr, scene_pha in zip(os.listdir(fgr_path),os.listdir(pha_path)):
            scene_fgr_path=path.join(fgr_path, scene_fgr)
            scene_pha_path= path.join(pha_path, scene_pha)
            
            #images
            if path.isdir(scene_fgr_path):
                for img_fgr, img_pha in zip(os.listdir(scene_fgr_path), os.listdir(scene_pha_path)):
                                
                    name= scene_fgr[-4:]+'_'+img_fgr #provide a unique file_name (scene_img.jpg) 
                    
                    #copy foreground and alpha matt
                    shutil.move(path.join(scene_fgr_path, img_fgr), path.join(fgr_path, name))            
                    shutil.move(path.join(scene_pha_path, img_pha), path.join(pha_path, name))
                    n_img+=2
                os.rmdir(scene_fgr_path)
                os.rmdir(scene_pha_path)
                                    
                    
            
        print(f'Unpacked {n_img} images from {split_path}.')

    videomatte_train= path.join(videomatte240k, "train")
    videomatte_test= path.join(videomatte240k, "test")
    
    unpack_split(videomatte_train)
    unpack_split(videomatte_test)
    
def sample_bg10k(src, dest):
    
    """
    Sample 10000 images from src and move them to dest.
    
    The images are split into a train and test set, with 9000 images in the train set and 1000 images in the test set.
    
    Parameters
    ----------
    src : str
        The source directory containing all the images.
    dest : str
        The destination directory where the sampled images will be moved.
    """
    all_images = [f for f in os.listdir(src) if f.endswith('.jpg')]
    
    sampled_images= sample(all_images, 10000)
    train_images= sample(sampled_images, 9000)
    test_images= [img for img in sampled_images if  img not in train_images]
    
    
    for img in train_images:
        shutil.copy(os.path.join(src, img), os.path.join(dest, 'train', img))
    for img in test_images:
        shutil.copy(os.path.join(src, img), os.path.join(dest, 'test', img))

        
    print(f'Sampled 10000 images from {src}')
    
def split_train_set(videomatte240k):
    """
    Split the training set of the Videomatte240k dataset into a train and val set.
    
    The val set will contain 3007 images, and the train set will contain 234982 images.
    
    Parameters
    ----------
    videomatte240k : str
        The path to the Videomatte240k dataset.
    """
    
    #train input
    train_fgr_folder= os.path.join(videomatte240k, 'train', 'fgr')
    train_pha_folder= os.path.join(videomatte240k, 'train', 'pha')
    
    train_fgr_set= os.listdir(train_fgr_folder)
    train_pha_set= os.listdir(train_pha_folder)
    
    #sample 3007 val images (fgr and pha names are equal)
    _,val_set= train_test_split(train_fgr_set, test_size=3007/237989, random_state= 42)
    
    #val output
    val_fgr_folder= os.path.join(videomatte240k, 'val', 'fgr')
    val_pha_folder= os.path.join(videomatte240k,'val', 'pha')

    
    #move fgr and pha
    for name in val_set:
        shutil.move(os.path.join(train_fgr_folder,name), os.path.join(val_fgr_folder, name))
        shutil.move(os.path.join(train_pha_folder, name),  os.path.join(val_pha_folder, name))
    
    print(f'Sampled 3007 images into {val_fgr_folder}')
    print(f'Sampled 3007 images into {val_pha_folder}')
    
def blend_foreground_with_background(split_src, background_folder, split_dest):
    
    """
    Blend foreground images with background images.
    
    The function will blend each foreground image with a randomly selected background image,
    and save the blended images to the split_dest folder.
    
    Parameters
    ----------
    split_src : str
        The path to the folder containing the foreground images and alpha mattes.
    background_folder : str
        The path to the folder containing the background images.
    split_dest : str
        The path to the folder where the blended images will be saved.
    """
    
    backgrounds= [f for f in os.listdir(background_folder) if f.endswith('.jpg')]
    
    shuffle(backgrounds)
    
    fgr_folder= os.path.join(split_src, 'fgr')
    pha_folder= os.path.join(split_src, 'pha')
    
    foregrounds= [f for f in os.listdir(fgr_folder) if f.endswith('.jpg')]
    alpha_mattes= [f for f in os.listdir(pha_folder) if f.endswith('.jpg')]
     
    for i, (fgr_path, pha_path) in enumerate(zip(foregrounds, alpha_mattes)):
        composite_path= os.path.join(split_dest,'composites', fgr_path)
        

        bg_path = backgrounds[i%len(backgrounds)]
        
        bg = Image.open(os.path.join(background_folder, bg_path)).convert("RGB")
        fgr= Image.open(os.path.join(fgr_folder, fgr_path)).convert('RGB')
        pha= Image.open(os.path.join(pha_folder, pha_path)).convert('L')
        
        #resize bg to fgr_size in order to avoid pha mismatch after composition
        bg= bg.resize(fgr.size, Image.BILINEAR)

        # Convert to NumPy arrays
        fgr_np = np.array(fgr)
        pha_np = np.array(pha) / 255.0  # Normalize alpha to [0,1]
        bg_np = np.array(bg)

        # Alpha blending: Composite = Foreground * Alpha + Background * (1 - Alpha)
        composite_np = (fgr_np[..., :3] * pha_np[..., None] + bg_np * (1 - pha_np[..., None])).astype(np.uint8)
        composite = Image.fromarray(composite_np)
        
        composite.save(composite_path)
        shutil.copy(os.path.join(pha_folder, pha_path), os.path.join(split_dest, 'pha', pha_path))
        
        
    print(f'Composed {len(foregrounds)} images and moved them to {split_dest}.')
