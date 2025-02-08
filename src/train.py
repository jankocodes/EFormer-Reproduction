import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import EFormerDataset
from models.eformer import EFormer
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

root_dir= Path(__file__).parent.parent

device = "mps" if torch.mps.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),  
    transforms.ToTensor(),  
])

# Load dataset with augmentation
train_dataset = EFormerDataset(root_dir=root_dir/'datasets/composite_dataset/train',
                               transform=transform,
                               p_flip=0.5)

val_dataset= EFormerDataset(root_dir=root_dir/'datasets/composite_dataset/train',
                            transform=transform,
                            p_flip=0)


train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader= DataLoader(val_dataset, batch_size=24, shuffle=False)

model = EFormer().to(device)  

criterion= torch.nn.CrossEntropyLoss()

# AdamW optimizer with lr decaying by 0.8 every 5 epochs
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

# Training loop
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        break

    with torch.no_grad():
        val_loss= 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            val_loss += criterion(outputs, labels).item()
            break
            
    scheduler.step()  # Apply learning rate decay
    

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    break

print("Training complete!")
