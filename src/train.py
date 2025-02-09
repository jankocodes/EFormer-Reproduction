import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import EFormerDataset
from models.eformer import EFormer
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter



root_dir= Path(__file__).parent.parent

writer = SummaryWriter(root_dir/'experiments/logs') 

device = "mps" if torch.mps.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),  
    transforms.ToTensor(),  
])

# Load dataset with augmentation
train_dataset = EFormerDataset(root_dir=root_dir/'datasets/composite_dataset/train',
                               transform=transform,
                               p_flip=0.5)

val_dataset= EFormerDataset(root_dir=root_dir/'datasets/composite_dataset/val',
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
    model.train()
    train_loss = 0.0

    print('Train')
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        break
    
    writer.add_scalar("Loss/train", train_loss , epoch)
    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
    
    print('Val')
    with torch.no_grad():
        val_loss= 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            val_loss += criterion(outputs, labels).item()
            break
    
    writer.add_scalar("Loss/val", val_loss , epoch)
    
    #safe best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), root_dir/"experiments/checkpoints/best_model.pth")
        print(f"New best model saved (Epoch {epoch+1})")

    # Save every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), root_dir/f"experiments/checkpoints/eformer_epoch{epoch+1}.pth")
        break
    
    scheduler.step()  # Apply learning rate decay
    

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
writer.close()

print("Training complete!")
