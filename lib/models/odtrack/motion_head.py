import torch
import torch.nn as nn
import torch.nn.functional as F

class StateSpaceHead(nn.Module):
    """ A simple MLP to predict motion delta and uncertainty. """
    # MODIFICATION START
    def __init__(self, history_len=3, hidden_dim=64):
    # MODIFICATION END
        """
        Initializes the motion head.
        Args:
            history_len (int): Number of historical bounding boxes to use as input.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super().__init__()
        # MODIFICATION START
        input_dim = history_len * 4  # Each box is (cx, cy, w, h)
        output_dim = 8               # 4 for delta mean + 4 for log variance
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # MODIFICATION END

    def forward(self, history_boxes_flat: torch.Tensor):
    # MODIFICATION END
        """
        Predicts the state update.
        Args:
            history_boxes_flat (torch.Tensor): A flattened tensor of historical bounding boxes.
                                               Shape (B, history_len * 4).
        
        Returns:
            torch.Tensor: Predicted delta (Δcx, Δcy, Δw, Δh). Shape (B, 4).
            torch.Tensor: Predicted log variance for each coordinate. Shape (B, 4).
        """
        # MODIFICATION START
        pred = self.mlp(history_boxes_flat)
        # MODIFICATION END
        delta_pred = pred[:, :4]
        log_sigma_pred = pred[:, 4:]
        return delta_pred, log_sigma_pred


def create_gaussian_attention_bias(center_xy, feat_sz, device, sigma_factor=5.0):
    """
    Generates a 2D Gaussian heatmap to be used as an attention bias.
    Args:
        center_xy (torch.Tensor): The predicted center of the target (x, y), normalized to [0, 1]. Shape (B, 2).
        feat_sz (int): The spatial size of the feature map (e.g., 24 for a 384 search area).
        device: The torch device.
        sigma_factor (float): Controls the "peakiness" of the Gaussian. Smaller is more peaky.
    
    Returns:
        torch.Tensor: A Gaussian heatmap. Shape (B, feat_sz*feat_sz).
    """
    if center_xy is None:
        return None
        
    B = center_xy.shape[0]
    
    # Denormalize center to feature map coordinates
    center_coord = center_xy * (feat_sz - 1)
    mu_x = center_coord[:, 0].view(B, 1, 1)
    mu_y = center_coord[:, 1].view(B, 1, 1)

    # Create coordinate grid
    x = torch.arange(0, feat_sz, device=device, dtype=torch.float32)
    y = torch.arange(0, feat_sz, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y, x)
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

    # Calculate Gaussian
    sigma = feat_sz / sigma_factor
    # The exponent term
    exponent = -((x_grid - mu_x)**2 + (y_grid - mu_y)**2) / (2 * sigma**2)
    # The heatmap
    heatmap = torch.exp(exponent)
    
    return heatmap.flatten(1) # (B, H*W)

def gaussian_nll_loss(gt_delta, pred_delta, pred_log_sigma):
    """
    Computes the Gaussian Negative Log-Likelihood loss.
    """
    # Clamp log_sigma to avoid numerical instability (exp(large_neg_val) -> 0)
    pred_log_sigma = torch.clamp(pred_log_sigma, -10, 10)
    
    inv_sigma_sq = torch.exp(-pred_log_sigma)
    loss = 0.5 * ((gt_delta - pred_delta)**2 * inv_sigma_sq + pred_log_sigma)
    return loss.mean()