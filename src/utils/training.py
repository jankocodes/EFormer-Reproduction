import torch
from utils.metrics import *
from models.eformer import EFormer
from torch.utils.data import DataLoader

def train(model: EFormer, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer, device):
 
    
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    
    with torch.no_grad():
        val_loss = 0.0
        total_mad = 0.0
        total_mse = 0.0
        total_grad = 0.0
        total_conn = 0.0

        model.eval()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            val_loss += criterion(outputs, labels).item()
            total_mad += mean_absolute_deviation(outputs, labels).item()
            total_mse += mean_squared_error(outputs, labels).item()
            total_grad += gradient_loss(outputs, labels).item()
            total_conn += connectivity_loss(outputs, labels).item()


        N = len(val_loader)
        avg_train_loss = train_loss / N
        avg_val_loss = val_loss / N
        avg_mad = total_mad / N
        avg_mse = total_mse / N
        avg_grad = total_grad / N
        avg_conn = total_conn / N

    results = {}
    results["train_loss"] = avg_train_loss
    results["val_loss"] = avg_val_loss
    results["mad"] = avg_mad
    results["mse"] = avg_mse
    results["grad"] = avg_grad
    results["conn"] = avg_conn
    

    return results
        