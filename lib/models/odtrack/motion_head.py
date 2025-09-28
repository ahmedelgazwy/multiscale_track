# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class StateSpaceHead(nn.Module):
#     """ A simple MLP to predict motion delta and uncertainty. """
#     # MODIFICATION START
#     def __init__(self, history_len=3, hidden_dim=64, use_appearance=False, appearance_dim=768, use_time=False, time_embedding_dim=128):
#     # MODIFICATION END
#         """
#         Initializes the motion head.
#         Args:
#             history_len (int): Number of historical bounding boxes to use as input.
#             hidden_dim (int): Dimension of the hidden layer.
#             # MODIFICATION START
#             use_appearance (bool): Whether to use appearance features.
#             appearance_dim (int): Dimension of the appearance feature vector.
#             use_time (bool): Whether to use time gap features.
#             time_embedding_dim (int): Dimension of the time gap embedding.
#             # MODIFICATION END
#         """
#         super().__init__()
#         # MODIFICATION START
#         self.use_appearance = use_appearance
#         self.use_time = use_time
        
#         input_dim = history_len * 4  # Each box is (cx, cy, w, h)
#         if self.use_appearance:
#             input_dim += appearance_dim
#         if self.use_time:
#             input_dim += time_embedding_dim
        
#         output_dim = 4 + 2 # Predict (dcx, dcy, dw, dh) and log_var for (x, y)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#         # MODIFICATION END

#     # MODIFICATION START
#     def forward(self, history_boxes_flat: torch.Tensor, appearance_feature: torch.Tensor = None, time_gap_embedding: torch.Tensor = None):
#     # MODIFICATION END
#         """
#         Predicts the state update.
#         Args:
#             history_boxes_flat (torch.Tensor): A flattened tensor of historical bounding boxes.
#                                                Shape (B, history_len * 4).
#             # MODIFICATION START
#             appearance_feature (torch.Tensor, optional): Pooled appearance feature. Shape (B, appearance_dim).
#             time_gap_embedding (torch.Tensor, optional): Time gap embedding. Shape (B, time_embedding_dim).
#             # MODIFICATION END
        
#         Returns:
#             torch.Tensor: Predicted delta (Δcx, Δcy, Δw, Δh). Shape (B, 4).
#             torch.Tensor: Predicted log variance for center coords (x, y). Shape (B, 2).
#         """
#         # MODIFICATION START
#         inputs = [history_boxes_flat]
#         if self.use_appearance and appearance_feature is not None:
#             inputs.append(appearance_feature)
#         if self.use_time and time_gap_embedding is not None:
#             inputs.append(time_gap_embedding)
        
#         combined_input = torch.cat(inputs, dim=1)
#         pred = self.mlp(combined_input)
#         # MODIFICATION END

#         delta_pred = pred[:, :4]
#         log_sigma_pred = pred[:, 4:] # Only for x and y
#         return delta_pred, log_sigma_pred


# def create_gaussian_attention_bias(center_xy, variance_xy, feat_sz, device):
#     """
#     Generates a 2D Gaussian heatmap to be used as an attention bias.
#     The variance of the Gaussian is now controlled by the motion model's prediction.
#     Args:
#         center_xy (torch.Tensor): The predicted center of the target (x, y), normalized to [0, 1]. Shape (B, 2).
#         variance_xy (torch.Tensor): The predicted variance for the center (var_x, var_y). Shape (B, 2).
#         feat_sz (int): The spatial size of the feature map (e.g., 20 for a 320 search area).
#         device: The torch device.
    
#     Returns:
#         torch.Tensor: A Gaussian heatmap. Shape (B, feat_sz*feat_sz).
#     """
#     if center_xy is None:
#         return None
        
#     B = center_xy.shape[0]
    
#     # Denormalize center to feature map coordinates
#     center_coord = center_xy * (feat_sz - 1)
#     mu_x = center_coord[:, 0].view(B, 1, 1)
#     mu_y = center_coord[:, 1].view(B, 1, 1)

#     # Create coordinate grid
#     x = torch.arange(0, feat_sz, device=device, dtype=torch.float32)
#     y = torch.arange(0, feat_sz, device=device, dtype=torch.float32)
#     y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
#     x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
#     y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)
    
#     # MODIFICATION START - Use predicted variance
#     # The variance is in normalized coordinate space, scale it to feature map space.
#     # A base variance is added for stability when predicted variance is very small.
#     base_variance = (feat_sz / 5.0)**2 # Heuristic base size
#     var_x = (variance_xy[:, 0] * (feat_sz**2)).view(B, 1, 1) + base_variance
#     var_y = (variance_xy[:, 1] * (feat_sz**2)).view(B, 1, 1) + base_variance
    
#     # The exponent term for a 2D Gaussian
#     exponent = -(((x_grid - mu_x)**2 / (2 * var_x)) + ((y_grid - mu_y)**2 / (2 * var_y)))
#     # MODIFICATION END
    
#     # The heatmap
#     heatmap = torch.exp(exponent)
    
#     return heatmap.flatten(1) # (B, H*W)

# def gaussian_nll_loss(gt_delta, pred_delta, pred_log_sigma):
#     """
#     Computes the Gaussian Negative Log-Likelihood loss.
#     We only compute it for the center coordinates (cx, cy).
#     """
#     # Clamp log_sigma to avoid numerical instability
#     pred_log_sigma = torch.clamp(pred_log_sigma, -7, 7)
    
#     # Target only the center coordinates for the NLL loss
#     gt_delta_xy = gt_delta[:, :2]
#     pred_delta_xy = pred_delta[:, :2]

#     sigma_sq = torch.exp(pred_log_sigma)
#     loss = 0.5 * (((gt_delta_xy - pred_delta_xy)**2 / sigma_sq) + pred_log_sigma)
    
#     return loss.mean()



import torch
import torch.nn as nn
import torch.nn.functional as F

# MODIFICATION START
class StateSpaceHead(nn.Module):
    """ A GRU-based sequence model to predict motion delta and uncertainty. """
    def __init__(self, hidden_dim=128, rnn_dim=256, num_rnn_layers=2,
                 use_appearance=True, appearance_dim=768, 
                 use_time=True, time_embedding_dim=128):
        """
        Initializes the motion head.
        Args:
            hidden_dim (int): Dimension of the MLP hidden layers.
            rnn_dim (int): Dimension of the GRU hidden state.
            num_rnn_layers (int): Number of layers in the GRU.
            use_appearance (bool): Whether to use appearance features.
            appearance_dim (int): Dimension of the appearance feature vector.
            use_time (bool): Whether to use time gap features.
            time_embedding_dim (int): Dimension of the time gap embedding.
        """
        super().__init__()
        self.use_appearance = use_appearance
        self.use_time = use_time

        # GRU layer to process the motion sequence of bounding boxes
        self.rnn = nn.GRU(
            input_size=4,           # Input is (cx, cy, w, h)
            hidden_size=rnn_dim,
            num_layers=num_rnn_layers,
            batch_first=True        # Crucial for (B, seq_len, features) input
        )

        # Calculate the size of the combined feature vector after the GRU
        combined_dim = rnn_dim
        if self.use_appearance:
            combined_dim += appearance_dim
        if self.use_time:
            combined_dim += time_embedding_dim

        # MLP prediction head
        output_dim = 4 + 2  # Predict (dcx, dcy, dw, dh) and log_var for (x, y)
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, history_boxes: torch.Tensor,
                appearance_feature: torch.Tensor = None, 
                time_gap_embedding: torch.Tensor = None):
        """
        Predicts the state update.
        Args:
            history_boxes (torch.Tensor): A tensor of historical bounding boxes.
                                          Shape (B, history_len, 4).
            appearance_feature (torch.Tensor, optional): Pooled appearance feature. Shape (B, appearance_dim).
            time_gap_embedding (torch.Tensor, optional): Time gap embedding. Shape (B, time_embedding_dim).
        
        Returns:
            torch.Tensor: Predicted delta (Δcx, Δcy, Δw, Δh). Shape (B, 4).
            torch.Tensor: Predicted log variance for center coords (x, y). Shape (B, 2).
        """
        
        # 1. Process the motion sequence with the GRU
        # The first output is all hidden states for each timestep, the second is the final hidden state
        _, final_hidden_state = self.rnn(history_boxes)
        
        # final_hidden_state has shape (num_rnn_layers, B, rnn_dim).
        # We take the hidden state from the last layer.
        motion_summary = final_hidden_state[-1] # Shape becomes (B, rnn_dim)

        # 2. Prepare for feature fusion
        inputs = [motion_summary]
        if self.use_appearance and appearance_feature is not None:
            inputs.append(appearance_feature)
        if self.use_time and time_gap_embedding is not None:
            inputs.append(time_gap_embedding)
        
        # 3. Fuse features and predict
        combined_features = torch.cat(inputs, dim=1)
        pred = self.prediction_head(combined_features)

        delta_pred = pred[:, :4]
        log_sigma_pred = pred[:, 4:]
        return delta_pred, log_sigma_pred
# MODIFICATION END

def create_gaussian_attention_bias(center_xy, variance_xy, feat_sz, device):
    """
    Generates a 2D Gaussian heatmap to be used as an attention bias.
    The variance of the Gaussian is now controlled by the motion model's prediction.
    Args:
        center_xy (torch.Tensor): The predicted center of the target (x, y), normalized to [0, 1]. Shape (B, 2).
        variance_xy (torch.Tensor): The predicted variance for the center (var_x, var_y). Shape (B, 2).
        feat_sz (int): The spatial size of the feature map (e.g., 20 for a 320 search area).
        device: The torch device.
    
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
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)
    
    # Use predicted variance
    # The variance is in normalized coordinate space, scale it to feature map space.
    # A base variance is added for stability when predicted variance is very small.
    base_variance = (feat_sz / 5.0)**2 # Heuristic base size
    var_x = (variance_xy[:, 0] * (feat_sz**2)).view(B, 1, 1) + base_variance
    var_y = (variance_xy[:, 1] * (feat_sz**2)).view(B, 1, 1) + base_variance
    
    # The exponent term for a 2D Gaussian
    exponent = -(((x_grid - mu_x)**2 / (2 * var_x)) + ((y_grid - mu_y)**2 / (2 * var_y)))
    
    # The heatmap
    heatmap = torch.exp(exponent)
    
    return heatmap.flatten(1) # (B, H*W)

def gaussian_nll_loss(gt_delta, pred_delta, pred_log_sigma):
    """
    Computes the Gaussian Negative Log-Likelihood loss.
    We only compute it for the center coordinates (cx, cy).
    """
    # Clamp log_sigma to avoid numerical instability
    pred_log_sigma = torch.clamp(pred_log_sigma, -7, 7)
    
    # Target only the center coordinates for the NLL loss
    gt_delta_xy = gt_delta[:, :2]
    pred_delta_xy = pred_delta[:, :2]

    sigma_sq = torch.exp(pred_log_sigma)
    loss = 0.5 * (((gt_delta_xy - pred_delta_xy)**2 / sigma_sq) + pred_log_sigma)
    
    return loss.mean()