# Mean Absolute Difference 
import torch
import torch.nn.functional as F
import numpy as np
import cv2

#MAD
def mean_absolute_deviation(pred: torch.Tensor, target: torch.Tensor):

    """
    Compute Mean Absolute Deviation (MAD) between predicted and target alpha matte.
    
    Args:
    - pred (Tensor): Predicted matte (1, 1, H, W)
    - target (Tensor): Target matte (1, 1, H, W)
    
    Returns:
    - Tensor: Mean absolute deviation value
    """

    return torch.mean(torch.abs(pred - target))

    
#MSE
def mean_squared_error(pred: torch.Tensor, target: torch.Tensor):
    
    
    
    """
    Compute Mean Squared Error (MSE) between predicted and ground truth alpha matte.
    
    Args:
    - pred (Tensor): Predicted alpha matte (1, 1, H, W)
    - target (Tensor): Ground truth alpha matte (1, 1, H, W)
    
    Returns:
    - Tensor: MSE value
    """
    return torch.mean((pred-target)**2)
    
#Grad  
def gradient_loss(pred: torch.Tensor, target: torch.Tensor):
    
  
    """
    Compute Gradient Loss between predicted and ground truth alpha mattes.

    Gradient Loss measures the difference of gradient magnitudes between predicted
    and ground truth alpha mattes. It is defined as the mean absolute difference
    between the gradient magnitudes of the predicted and target alpha mattes.

    Args:
        pred (Tensor): Predicted alpha matte (1, 1, H, W)
        target (Tensor): Ground truth alpha matte (1, 1, H, W)

    Returns:
        Tensor: Gradient Loss value
    """
    def compute_gradient(image: torch.Tensor):
        
        #sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)  

        sobel_y = torch.tensor([[-1, -2, -1], 
                                [0,  0,  0], 
                                [1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)  
        
        sobel_x, sobel_y= sobel_x.to(image.device), sobel_y.to(image.device)
        
        #gradient computation through convolution with sobel filters
        grad_x= F.conv2d(image, sobel_x, padding=1)
        grad_y= F.conv2d(image, sobel_y, padding=1)
        
        grad_magnitude= torch.sqrt(grad_x**2 + grad_y**2 + 1e-6) #avoid sqrt(0)
        
        return grad_magnitude
    
    
    grad_pred = compute_gradient(pred)
    grad_target = compute_gradient(target)

    return torch.sum(torch.abs(grad_pred - grad_target))


#Conn
def connectivity_loss(pred: torch.Tensor, true: torch.Tensor, step=0.1, threshold=0.15):
    """
    Simplified connectivity loss with lambda=1 and no distance weighting.
    """
    pred = pred.squeeze().cpu().detach().numpy()
    true = true.squeeze().cpu().detach().numpy()

    step = step
    thresh_steps = np.arange(0, 1 + step, step)

    round_down_map = -np.ones_like(true)

    for i in range(1, len(thresh_steps)):
        true_thresh = true >= thresh_steps[i]
        pred_thresh = pred >= thresh_steps[i]
        intersection = (true_thresh & pred_thresh).astype(np.uint8)

        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(intersection, connectivity=4)
        size = stats[1:, -1]

        omega = np.zeros_like(true)
        if len(size) != 0:
            max_id = np.argmax(size)
            omega[output == max_id + 1] = 1

        mask = (round_down_map == -1) & (omega == 0)
        round_down_map[mask] = thresh_steps[i-1]

    round_down_map[round_down_map == -1] = 1

    true_diff = true - round_down_map
    pred_diff = pred - round_down_map

    true_phi = 1 - true_diff * (true_diff >= threshold)
    pred_phi = 1 - pred_diff * (pred_diff >= threshold)

    connectivity_error = np.sum(np.abs(true_phi - pred_phi))
    
    return connectivity_error 
                
                