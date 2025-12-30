import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

class EventFlowEstimator(nn.Module):
    def __init__(self, in_channels, num_multi_flow=9):
        super(EventFlowEstimator, self).__init__()
        self.num_multi_flow = num_multi_flow
        self.event_voxel_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flow_estimator = nn.Conv3d(128, num_multi_flow * 2, kernel_size=3, padding=1)  # Predicting flow offsets
        
    def forward(self, event_voxel):
        # Encoding the event voxel grid
        encoded_features = self.event_voxel_encoder(event_voxel)
        
        # Estimate flow from encoded features
        flow_offsets = self.flow_estimator(encoded_features)  # (Batch, num_multi_flow * 3, T, H, W)
        flow_across_time = torch.mean(flow_offsets, dim=2)  # (Batch, num_multi_flow, 3, H, W)
        return flow_across_time.squeeze(1)

# # Usage Example
# event_voxel = torch.rand(2, 4, 5, 440, 640)  # Batch size of 2, 4 channels, voxel grid of 5x440x640
# event_flow_estimator = EventFlowEstimator(in_channels=4)
# flow_information = event_flow_estimator(event_voxel)