import torch 
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import EFormerDataset
from models.eformer import EFormer
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import *
from utils.training import train
import json

data_root= "$TMPDIR/composite_dataset"

json_log = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),  
    transforms.ToTensor(),  
])

# Load dataset with augmentation
train_dataset = EFormerDataset(root_dir=data_root+'/train',
                               transform=transform,
                               p_flip=0.5)

val_dataset= EFormerDataset(root_dir=data_root+'/val',
                            transform=transform,
                            p_flip=0)


train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader= DataLoader(val_dataset, batch_size=24, shuffle=False)

model = EFormer().to(device)  

criterion= torch.nn.BCELoss()

# AdamW optimizer with lr decaying by 0.8 every 5 epochs
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

# Training loop 
num_epochs = 25

best_val_loss= float('inf')

for epoch in range(num_epochs):
    
    results= train(model=model,
             train_loader=train_loader,
             val_loader=val_loader,
             criterion=criterion,
             optimizer=optimizer,
             device=device)    
    
    #log metrics
    json_log[epoch] = results
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {results['train_loss']:.4f} | "
    f"Val Loss: {results['val_loss']:.4f} | MAD: {results['mad']:.3f} | "
    f"MSE: {results['mse']:.3f} | Grad: {results['grad']:.3f} | Conn: {results['conn']:.3f}")
    
    #save best model
    val_loss= results["val_loss"]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "experiments/checkpoints/best_model.pth")
        print(f"New best model saved (Epoch {epoch+1})")

    #save every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"experiments/checkpoints/eformer_epoch{epoch+1}.pth")
    
    scheduler.step()  # Apply learning rate decay
    

#save results    
with open("experiments/logs/metrics_log.json", "w") as f:
    json.dump(json_log, f, indent=4)

