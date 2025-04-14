import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    """
    Example 2.5D RNN that processes a full subvolume (72x72x72) 
    slice-by-slice and produces a (num_classes x 72 x 72 x 72) output.
    """
    def __init__(self, cfg, num_classes=6):
        super().__init__()
        
        # Just an example encoder (2D):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Recurrent cell (ConvLSTM, single layer):
        self.hidden_dim = 64
        self.conv_lstm_cell = ConvLSTMCell(input_channels=64, hidden_channels=self.hidden_dim)
        
        # Final segmentation head to produce classes:
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
                
    def forward(self, x):
        """
        x: shape (B, 1, D, H, W) = (B, 1, 72, 72, 72)
        returns: shape (B, num_classes, D, H, W)
        """
        B, C, D, H, W = x.shape
        # Permute to (B, D, C, H, W) => time dimension is D
        x = x.permute(0, 2, 1, 3, 4)  # (B, D, 1, H, W)
        
        # Initialize hidden states
        device = x.device
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=device)
        
        # We'll store each slice's segmentation output in a list
        seg_slices = []
        
        for t in range(D):  # unroll across 72 slices
            # x[:, t] => shape (B, 1, H, W)
            slice_t = x[:, t]
            
            # 1) 2D CNN encoder
            feat_t = self.encoder(slice_t)   # shape (B, 64, H, W)
            
            # 2) ConvLSTM step
            h, c = self.conv_lstm_cell(feat_t, h, c)  # shape (B, hidden_dim, H, W)
            
            # 3) produce segmentation for slice t
            out_t = self.seg_head(h)         # shape (B, num_classes, H, W)
            seg_slices.append(out_t)
        
        # stack them => (B, D, num_classes, H, W)
        seg_3d = torch.stack(seg_slices, dim=2)  # shape (B, num_classes, D, H, W) ?
        # Actually, we used .append(out_t) as a list, so when we stack with dim=2, 
        # we get shape (B, num_classes, D, H, W).
        # We need to reorder to keep the channels dimension in the right place:
        # Currently out_t is (B, num_classes, H, W), so stacking on dim=1 or 2 can vary.
        # Let’s do it carefully:
        seg_3d = torch.stack(seg_slices, dim=1)  # (B, D, num_classes, H, W)
        seg_3d = seg_3d.permute(0, 2, 1, 3, 4)    # => (B, num_classes, D, H, W)
        
        return seg_3d


class ConvLSTMCell(nn.Module):
    """
    Simple ConvLSTM cell for demonstration.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
        
    def forward(self, x, h, c):
        # x shape: (B, input_channels, H, W)
        # h, c shape: (B, hidden_channels, H, W)
        combined = torch.cat([x, h], dim=1)  # => (B, input+hidden, H, W)
        gates = self.conv(combined)          # => (B, 4*hidden, H, W)
        
        # slice gates
        chunk = self.hidden_channels
        i = torch.sigmoid(gates[:, 0:chunk])
        f = torch.sigmoid(gates[:, chunk:2*chunk])
        o = torch.sigmoid(gates[:, 2*chunk:3*chunk])
        g = torch.tanh(gates[:, 3*chunk:4*chunk])
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next