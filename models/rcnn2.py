import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --------------------------------------------------
# 1) Bi-ConvLSTM cell definitions
# --------------------------------------------------

class ConvLSTMCell(nn.Module):
    """
    Standard ConvLSTM cell: (X_t, H_{t-1}, C_{t-1}) -> (H_t, C_t).
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
        # x:     (B, input_channels, H, W)
        # h,c:   (B, hidden_channels, H, W)
        combined = torch.cat([x, h], dim=1)  # => (B, in+hidden, H, W)
        gates = self.conv(combined)          # => (B, 4*hidden, H, W)
        
        chunk = self.hidden_channels
        i = torch.sigmoid(gates[:, 0:chunk])
        f = torch.sigmoid(gates[:, chunk:2*chunk])
        o = torch.sigmoid(gates[:, 2*chunk:3*chunk])
        g = torch.tanh(gates[:, 3*chunk:4*chunk])
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class BiConvLSTM(nn.Module):
    """
    Bidirectional ConvLSTM over the "time" dimension (Z-slices).
    We run forward LSTM and backward LSTM, then combine states.
    """
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.forward_cell = ConvLSTMCell(input_channels, hidden_channels)
        self.backward_cell = ConvLSTMCell(input_channels, hidden_channels)
        self.hidden_channels = hidden_channels

    def forward(self, x_seq):
        """
        x_seq: (B, T, C, H, W)  [T = number of slices, C = input_channels]
        Returns: 
          fwd_outs:  (B, T, hidden_channels, H, W)
          bwd_outs:  (B, T, hidden_channels, H, W)
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        
        # --- Forward pass ---
        h_fwd = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c_fwd = torch.zeros(B, self.hidden_channels, H, W, device=device)
        fwd_outs = []
        for t in range(T):
            x_t = x_seq[:, t]
            h_fwd, c_fwd = self.forward_cell(x_t, h_fwd, c_fwd)
            fwd_outs.append(h_fwd)
        fwd_outs = torch.stack(fwd_outs, dim=1)  # (B, T, hidden_dim, H, W)
        
        # --- Backward pass ---
        h_bwd = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c_bwd = torch.zeros(B, self.hidden_channels, H, W, device=device)
        bwd_outs = []
        for t in range(T-1, -1, -1):
            x_t = x_seq[:, t]
            h_bwd, c_bwd = self.backward_cell(x_t, h_bwd, c_bwd)
            bwd_outs.append(h_bwd)
        bwd_outs.reverse()  # so indices match the forward direction
        bwd_outs = torch.stack(bwd_outs, dim=1)  # (B, T, hidden_dim, H, W)
        
        return fwd_outs, bwd_outs


# --------------------------------------------------
# 2) The main "SOTA" 2.5D model
# --------------------------------------------------

class RCNN(nn.Module):
    """
    2.5D model with:
      - ConvNeXt 2D backbone (from timm) extracting multi-scale features.
      - Bi-ConvLSTM over the slice dimension on one of the deeper feature maps.
      - Feature-pyramid style skip connections for final segmentation per slice.
      - Output shape: (B, num_classes, D, H, W).
    """
    def __init__(self, cfg, num_classes=6, backbone_name='convnext_base', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # 1) Load a timm backbone (ConvNeXt, Swin, etc.)
        #    We'll freeze its classification head, keep the feature extraction.
        self.backbone = timm.create_model(backbone_name, 
                                          pretrained=pretrained, 
                                          features_only=True,
                                          in_chans=cfg.in_channels)
        # features_only=True => returns a list of feature maps at different stages.
        # e.g., for convnext_base, we typically get 4 feature maps at strides 4,8,16,32.
        
        # Let's see how many channels come out from the final stage
        backbone_channels = self.backbone.feature_info.channels()  # e.g. [128, 256, 512, 1024] for some convnext
        self.out_ch = backbone_channels[-1]  # the deepest feature map channel
        
        # 2) Projection or bridging layer (optional)
        #    E.g. project the final feature map to some smaller dimension before LSTM
        hidden_dim = 256
        self.proj = nn.Conv2d(self.out_ch, hidden_dim, kernel_size=1)
        
        # 3) Bi-ConvLSTM to process the final-level feature map across T slices
        self.bilstm = BiConvLSTM(input_channels=hidden_dim, hidden_channels=hidden_dim)
        
        # 4) (Optional) A Feature Pyramid or top-down approach
        #    For example, we can also gather mid-level features from the backbone
        #    and incorporate them in a 2D decoder. We'll keep it simpler below,
        #    but you could do an FPN or U-Net decoder with skip connections.
        
        # 5) A final 2D segmentation head that fuses forward & backward hidden states
        #    For each slice, we combine (fwd[t] + bwd[t]) => pass through conv => output
        self.seg_head = nn.Sequential(
            nn.Conv2d(2*hidden_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        """
        x: (B, 1, D, H, W) volume sub-block 
        Returns: (B, num_classes, D, H, W) probability/logit map
        """
        B, C, D, H, W = x.shape
        device = x.device
        
        # We'll produce a 2D segmentation map for each slice => 3D stack.
        outputs_3d = []
        
        # We can store for all slices at once, but to handle multi-scale features from timm,
        # we'll do it slice-by-slice. (That’s simpler but slower. 
        # If you have memory, you can batch more slices at once.)
        
        # Step 1: build a list of features for each slice
        #         features[t] = final-level feature map
        all_features = []
        for t in range(D):
            slice_t = x[:, :, t]  # shape (B, 1, H, W)
            # pass through backbone
            feats = self.backbone(slice_t)  
            # feats is a list: [level1, level2, level3, level4], each shape => (B, channels_l, H_l, W_l)
            
            final_map = feats[-1]         # the deepest level
            proj_map = self.proj(final_map)  # (B, hidden_dim, H4, W4)
            all_features.append(proj_map)
        
        # We should ensure each final_map has the same spatial size for BiLSTM if we want to stack them.
        # Typically, for input (H, W) multiple of 32, final_map might be (H/32, W/32).
        # all_features => list of length D, each shape => (B, hidden_dim, H/32, W/32).
        
        # Stack => (B, D, hidden_dim, H/32, W/32)
        feat_tensor = torch.stack(all_features, dim=1)
        
        # Step 2: Bi-ConvLSTM over this feature sequence
        fwd_outs, bwd_outs = self.bilstm(feat_tensor)  
          # each => (B, D, hidden_dim, H/32, W/32)
        
        # Step 3: For each slice, combine fwd and bwd hidden states => segmentation
        out_slices = []
        for t in range(D):
            cat_t = torch.cat([fwd_outs[:, t], bwd_outs[:, t]], dim=1)  # shape (B, 2*hidden_dim, H/32, W/32)
            seg_t = self.seg_head(cat_t)  # shape (B, num_classes, H/32, W/32)
            out_slices.append(seg_t)
        
        # Now, each seg_t is only (H/32, W/32) in spatial dimension. 
        # We must upsample to (H, W) if we want full resolution. Let's do that:
        out_full_slices = []
        for seg_t in out_slices:
            up_seg = F.interpolate(seg_t, size=(H, W), mode='bilinear', align_corners=False)
            out_full_slices.append(up_seg)
        
        # Combine => (B, num_classes, D, H, W)
        out_stack = torch.stack(out_full_slices, dim=2)  # shape (B, num_classes, D, H, W)
        
        return out_stack
