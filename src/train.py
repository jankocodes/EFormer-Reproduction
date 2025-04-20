def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to composite dataset')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the current training run')
    args = parser.parse_args()

    data_root = args.data_root
    run_name = args.run_name

    #create logging dirs 
    json_log = {}
    checkpoint_dir = f"experiments/checkpoints/{run_name}"
    log_dir= f"experiments/logs/{run_name}"
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


    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=8,
    pin_memory=True,
    persistent_workers=True  # optional
    )
    
    val_loader= DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=8,
    pin_memory=True,
    persistent_workers=True  # optional
    )

    model = EFormer().to(device)  

    criterion= torch.nn.BCELoss()

    # AdamW optimizer with lr decaying by 0.8 every 5 epochs
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    # Training loop 
    num_epochs = 25

    best_val_loss= float('inf')

    # Debugging ############################################################################
    print("Start training: ", flush=True)
    
    print(f"[DEBUG] Model on device: {next(model.parameters()).device}", flush=True)
    
    for name, param in model.named_parameters():
        print(f"{name} -> {param.device}")


    for images, labels in train_loader:
        print(f"[DEBUG] Images on: {images.device}, Labels on: {labels.device}", flush=True)
        break
    
    for images, labels in val_loader:
        print(f"[DEBUG] Images on: {images.device}, Labels on: {labels.device}", flush=True)
        break
    ########################################################################################


    for epoch in range(num_epochs):
        print(f"[DEBUG] Model still on: {next(model.parameters()).device}")

        
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

        #save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/eformer_epoch{epoch+1}.pth")
        
        #save results    
        with open(f"{log_dir}/metrics_log.json", "w") as f:
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