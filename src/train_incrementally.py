def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to composite dataset')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the current training run')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory of logs/checkpoints')
    parser.add_argument('--checkpoint_path', type= str, default="", help='Path of checkpoint used for training, if none is given new model is trained.')
    parser.add_argument('--use_sa', type= lambda x: str(x).lower()=="true", default=True, help='Use self-attention layers')
    parser.add_argument('--use_ca', type= lambda x: str(x).lower()=="true", default=True, help='Use cross-attention layers')
    parser.add_argument('--first_upsampling', type= str, default='bilinear', choices=['bilinear', 'transconv'], help='First upsampling method')
    parser.add_argument('--second_upsampling', type= str, default='transconv', choices=['bilinear', 'transconv'], help='Second upsampling method')
    parser.add_argument('--hr_resolution', type= str, default='1_8', choices=['1_4', '1_8'], help='Resolution of HR-embedding.')
    parser.add_argument('--lr_resolution', type= str, default='1_16', choices=[ '1_8', '1_16'], help='Resolution of LR-embedding.')
    
    args = parser.parse_args()

    data_root = args.data_root
    out_dir= args.out_dir
    checkpoint_path= args.checkpoint_path
    use_sa= args.use_sa
    use_ca= args.use_ca
    first_upsampling= args.first_upsampling
    second_upsampling= args.second_upsampling
    hr_res= args.hr_resolution
    lr_res= args.lr_resolution

    #create logging dirs 
    json_log = {}
    checkpoint_dir = f"{out_dir}/checkpoints/"
    log_dir= f"{out_dir}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # Load dataset with augmentation
    train_dataset = EFormerDataset(root_dir=data_root+'/train',
                                size=(224, 224),
                                p_flip=0.5)

    val_dataset= EFormerDataset(root_dir=data_root+'/val',
                                size=(224,224),
                                p_flip=0)


    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True, num_workers=8,
    pin_memory=True,
    persistent_workers=True  
    )
    
    val_loader= DataLoader(val_dataset, batch_size=18, shuffle=False, num_workers=8,
    pin_memory=True,
    persistent_workers=True 
    )
    
    model = EFormer(use_sa=use_sa,
                    use_ca= use_ca,
                    first_upsampling=first_upsampling,
                    second_upsampling=second_upsampling,
                    hr_dim=hr_res,
                    lr_dim=lr_res).to(device)

    # AdamW optimizer with lr decaying by 0.8 every 5 epochs
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    best_val_loss= float('inf')

    #resume training from checkpoint
    if checkpoint_dir:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from checkpoint: {args.checkpoint_path} at epoch {start_epoch}")
        

    criterion= torch.nn.BCELoss()

    # Training loop 
    num_epochs = 25

    # Debugging ############################################################################
    print("Start training: ", flush=True)
    
    print(f"Model on device: {next(model.parameters()).device}", flush=True)
    
    ########################################################################################


    for epoch in range(start_epoch, num_epochs):
        
        results= train(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device)    
        
        #log metrics
        json_log[epoch] = results
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {results['train_loss']:.4f} | "
        f"Val Loss: {results['val_loss']:.4f} | MAD: {results['mad']*1e3:.3f} | "
        f"MSE: {results['mse']*1e3:.3f} | Grad: {results['grad']*1e-3:.3f} | Conn: {results['conn']*1e-3:.3f}", flush=True)
        
        #save best model
        val_loss= results["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print(f"New best model saved (Epoch {epoch+1})", flush=True)

        #save every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
            }, f"{checkpoint_dir}/eformer_epoch{epoch+1}.pth")
            
        #save results    
        with open(f"{log_dir}/train_metrics.json", "w") as f:
            json.dump(json_log, f, indent=4)
            
        scheduler.step()  # Apply learning rate decay
        
    print("Training finished.", flush=True)



if __name__=="__main__":
    import torch.multiprocessing as mp

    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    import json
    import os
    import argparse
    import torch 
    import torchvision.transforms as transforms
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from data.dataset import EFormerDataset
    from models.eformer import EFormer
    from torch.optim.lr_scheduler import StepLR
    from utils.metrics import *
    from utils.training import train
    
    main()