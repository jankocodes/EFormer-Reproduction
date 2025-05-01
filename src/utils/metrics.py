# Mean Absolute Difference 
import torch
import torch.nn.functional as F
import numpy as np
import cv2

#MAD
class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred.detach().cpu().numpy() - true.detach().cpu().numpy()).mean() 


#MSE
class MetricMSE:
    def __call__(self, pred, true):
        return ((pred.detach().cpu().numpy() - true.detach().cpu().numpy()) ** 2).mean()
    
#Grad  
class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
    
    def __call__(self, pred: torch.Tensor, true: torch.Tensor):
        pred= pred.squeeze().detach().cpu().numpy()
        true= true.squeeze().detach().cpu().numpy()
        
        pred_normed = np.zeros_like(pred)
        true_normed = np.zeros_like(true)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss 
    
    def gauss_gradient(self, img):
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int64(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2


#Conn
class MetricCONN:
    def __call__(self, pred, true):
        step=0.1
        threshold=0.15
  
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
                    
                    