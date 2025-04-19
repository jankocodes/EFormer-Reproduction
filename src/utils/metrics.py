# Mean Absolute Difference 
import torch
import torch.nn.functional as F

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
def connectivity_loss(pred: torch.Tensor, target: torch.Tensor, step= 0.1):
        
 
        """
        Compute connectivity loss between predicted and ground truth alpha mattes.

        This loss measures the difference in connectivity between the predicted and ground truth alpha mattes.
        Connectivity is measured by thresholding the alpha matte values and computing the absolute difference between the two binary masks.

        Args:
        - pred (Tensor): Predicted alpha matte (1, 1, H, W)
        - target (Tensor): Ground truth alpha matte (1, 1, H, W)
        - step (float, optional): Incremental step for thresholding the alpha matte values. Defaults to 0.1.

        Returns:
        - Tensor: Connectivity loss value
        """
        pred_pha = pred[0, 0]  # Shape (H, W) for the first and only sample
        target_pha = target[0, 0]  # Shape (H, W) for the first and only sample
        
        loss = 0.0
        
        # Compute connectivity loss for each threshold value
        for threshold in torch.arange(start=step, end=1.0, step=step, device=pred.device):
            
            # Generate binary masks
            pred_mask = (pred_pha >= threshold).float()
            target_mask = (target_pha >= threshold).float()
            
            # Compute connectivity difference
            loss += torch.sum(torch.abs(pred_mask - target_mask))  # Sum the absolute differences
        
        return loss
                
                
                
                