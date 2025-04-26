import torch
from utils.metrics import *
from models.eformer import EFormer
from torch.utils.data import DataLoader

def evaluate(model: EFormer, test_loader: DataLoader, criterion, device):

    with torch.no_grad():
        loss = 0.0
        total_mad = 0.0
        total_mse = 0.0
        total_grad = 0.0
        total_conn = 0.0

        model.eval()
        n_images= 0
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking= True), labels.to(device, non_blocking= True)
            outputs = model(images)
            
            loss += criterion(outputs, labels).item()
            
            # Compute image-wise metrics
            for i in range(images.size(0)):
                total_mad += mean_absolute_deviation(outputs[i:i+1], labels[i:i+1]).item()
                total_mse += mean_squared_error(outputs[i:i+1], labels[i:i+1]).item()
                total_grad += gradient_loss(outputs[i:i+1], labels[i:i+1]).item()
                total_conn += connectivity_loss(outputs[i:i+1], labels[i:i+1]).item()
                n_images+=1
                


        avg_loss = loss / len(test_loader)
        avg_mad = total_mad / n_images
        avg_mse = total_mse / n_images
        avg_grad = total_grad / n_images
        avg_conn = total_conn / n_images

    results = {}
    results["loss"] = avg_loss
    results["mad"] = avg_mad
    results["mse"] = avg_mse
    results["grad"] = avg_grad
    results["conn"] = avg_conn
    

    return results
        