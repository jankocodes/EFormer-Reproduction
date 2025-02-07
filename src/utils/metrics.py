# Mean Absolute Difference 
import torch
import torch.nn.functional as F

#MAD
def mean_absolute_deviation(pred: torch.Tensor, target: torch.Tensor):
    """
    Compute Mean Absolute Deviation (MAD) between predicted and ground truth alpha mattes.
    
    Args:
    - pred (Tensor): Predicted alpha matte (B, 1, H, W)
    - target (Tensor): Ground truth alpha matte (B, 1, H, W)
    
    Returns:
    - Tensor: MAD value
    """
    return torch.mean(torch.abs(pred - target))

    
#MSE
def mean_squared_error(pred: torch.Tensor, targets: torch.Tensor):
    
    """
    Compute Mean Squared Error (MSE) between predicted and ground truth alpha mattes.

    Args:
    - pred (Tensor): Predicted alpha matte (B, 1, H, W)
    - targets (Tensor): Ground truth alpha matte (B, 1, H, W)

    Returns:
    - Tensor: MSE value
    """
    
    return torch.mean((pred-targets)**2)
    
#Grad  
def gradient_loss(pred: torch.Tensor, target: torch.Tensor):
    
    """
    Compute the gradient loss between the predicted and ground truth alpha mattes.
    
    This loss uses the Sobel operator to compute the gradient magnitude of both the predicted and ground truth alpha mattes.
    The difference between the two gradient magnitudes is then computed and the mean absolute difference is returned.
    
    Args:
    - pred (Tensor): Predicted alpha matte (B, 1, H, W)
    - target (Tensor): Ground truth alpha matte (B, 1, H, W)
    
    Returns:
    - Tensor: Gradient loss value
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

    return torch.mean(torch.abs(grad_pred - grad_target))


#Conn
def connectivity_loss(pred: torch.Tensor, target: torch.Tensor, step= 0.1):
        
        """
        Compute connectivity loss between predicted and ground truth alpha mattes.
        
        This loss evaluates the predicted alpha matte at multiple threshold levels and computes the difference in connectivity between the predicted and ground truth binary masks at each threshold.
        
        Args:
        - pred (Tensor): Predicted alpha matte (B, 1, H, W)
        - target (Tensor): Ground truth alpha matte (B, 1, H, W)
        - step (float): Step size for threshold values
        
        Returns:
        - Tensor: Connectivity loss value
        """
        batch_size= pred.shape[0]
        loss= 0.0
        
        for i in range(batch_size):
            pred_pha= pred[i, 0]
            target_pha= target[i, 0]    
                        
            for threshold in torch.arange(start=step, end=1.0, step=step, device=pred.device):
                
                #generate binary masks 
                pred_mask = (pred_pha >= threshold).float()
                target_mask = (target_pha >= threshold).float()
                
                #compute connectivity difference
                loss+= torch.sum(torch.abs(pred_mask, target_mask)) / pred.numel()
        
        return loss/batch_size
                
                
                
                