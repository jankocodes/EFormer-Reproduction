import torch
from utils.metrics import *
from models.eformer import EFormer
from torch.utils.data import DataLoader

def train(model: EFormer, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer, device):
 
    
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking= True), labels.to(device, non_blocking= True)


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        train_loss += loss.item()

    
    with torch.no_grad():
        mad= MetricMAD()
        mse= MetricMSE()
        conn= MetricCONN()
        grad= MetricGRAD()
        
        val_loss = 0.0
        total_mad = 0.0
        total_mse = 0.0
        total_grad = 0.0
        total_conn = 0.0

        model.eval()
        n_images= 0
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking= True), labels.to(device, non_blocking= True)
            outputs = model(images)
            
            val_loss += criterion(outputs, labels).item()
            
            # Compute image-wise metrics
            for i in range(images.size(0)):
                total_mad += mad(outputs[i:i+1], labels[i:i+1]).item()
                total_mse += mse(outputs[i:i+1], labels[i:i+1]).item()
                total_grad += grad(outputs[i:i+1], labels[i:i+1]).item()
                total_conn += conn(outputs[i:i+1], labels[i:i+1]).item()
                n_images+=1
                


        avg_val_loss = val_loss / len(val_loader)
        avg_train_loss= train_loss/ len(train_loader)
        avg_mad = total_mad / n_images
        avg_mse = total_mse / n_images
        avg_grad = total_grad / n_images
        avg_conn = total_conn / n_images

    results = {}
    results["train_loss"] = avg_train_loss
    results["val_loss"] = avg_val_loss
    results["mad"] = avg_mad
    results["mse"] = avg_mse
    results["grad"] = avg_grad
    results["conn"] = avg_conn
    

    return results
        