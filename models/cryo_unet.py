import torch.nn as nn
from torch.nn import functional as F
import torch
from functools import partial
import sys
from torch.nn import Conv3d, Module, Linear, BatchNorm3d, ReLU
from torch.nn.modules.utils import _pair, _triple

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class cSEBlock(nn.Module):
    """
    Channel Squeeze & Excitation: 
    1) Global Average Pool each channel
    2) Use small MLP to get channel-wise weights
    3) Multiply input by these channel-wise weights
    """
    def __init__(self, in_channels, reduction=16):
        super(cSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, d, h, w = x.size()
        # Squeeze: Global Average Pool
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn channel-wise weights
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Scale
        return x * y

class sSEBlock(nn.Module):
    """
    Spatial Squeeze & Excitation:
    1) Convolve across channels -> single 3D activation map
    2) Multiply input by this spatial mask
    """
    def __init__(self, in_channels):
        super(sSEBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: across channel dimension => 1 channel mask
        y = self.conv(x)
        y = self.sigmoid(y)
        # Excitation: multiply the mask
        return x * y

class scSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    The output is a sum of:
      x * cSE(x) + x * sSE(x)
    """
    def __init__(self, in_channels, reduction=16):
        super(scSEBlock, self).__init__()
        self.cSE = cSEBlock(in_channels, reduction=reduction)
        self.sSE = sSEBlock(in_channels)

    def forward(self, x):
        x_c = self.cSE(x)
        x_s = self.sSE(x)
        return x_c + x_s

class CryoResUNet3D(nn.Module):
    def __init__(self, cfg):
        super(CryoResUNet3D, self).__init__()
        self.use_IP = cfg.use_IP
        self.feature_maps = cfg.feature_maps
        encoders, decoders = [], []
        
        pool_layer = nn.AvgPool3d

        if self.use_IP:
            pools = []
            for _ in range(len(self.feature_maps) - 1):
                pools.append(pool_layer(2))
            self.pools = nn.ModuleList(pools)
        
        for i, out_channels in enumerate(self.feature_maps):
            in_channels = cfg.in_channels if i == 0 else self.feature_maps[i - 1]
            use_IP = False if i == 0 else cfg.use_IP
            apply_pooling = False if i == 0 else True
            encoders.append(Encoder(in_channels, out_channels, apply_pooling=apply_pooling, use_IP=use_IP,
                                    use_coord=cfg.use_coord,pool_layer=pool_layer,norm=cfg.norm,
                                    activation=cfg.activation, use_attention=cfg.use_attention,
                                    dropout_p=cfg.dropout_p, use_scSE=cfg.use_scSE))
            
        self.encoders = nn.ModuleList(encoders)
        
        reversed_feature_maps = list(reversed(self.feature_maps))
        for i in range(len(reversed_feature_maps) - 1):
            in_channels = reversed_feature_maps[i]
            out_channels = reversed_feature_maps[i + 1]
            decoders.append(Decoder(in_channels, out_channels, use_coord=cfg.use_coord,
                                    norm=cfg.norm, activation=cfg.activation, 
                                    dropout_p=cfg.dropout_p, use_scSE=cfg.use_scSE))
        self.decoders = nn.ModuleList(decoders)
        
        self.final_conv = nn.Conv3d(self.feature_maps[0], cfg.out_channels, 1)
        
    def forward(self, x):
        if self.use_IP:
            img_pyramid = []
            img_d = x
            for pool in self.pools:
                img_d = pool(img_d)
                img_pyramid.append(img_d)
                
        encoders_features = []
        for idx, encoder in enumerate(self.encoders):
            if self.use_IP and idx > 0:
                x = encoder(x, img_pyramid[idx - 1])
            else:
                x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        
        out = self.final_conv(x)
        #out = torch.softmax(out, dim=1)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, use_IP=False, use_coord=False,
                 pool_layer=nn.MaxPool3d, norm='bn', activation='relu', use_attention=False, input_channels=1,
                 dropout_p=0.0, use_scSE=False):
        super(Encoder, self).__init__()
        
        self.pooling = pool_layer(kernel_size=2) if apply_pooling else None
        self.use_IP = use_IP
        self.use_coord = use_coord
        inplaces = in_channels + input_channels if self.use_IP else in_channels
        inplaces = inplaces + 3 if self.use_coord else inplaces
        
        self.basic_module = ExtResNetBlock(inplaces, 
                                           out_channels, 
                                           norm=norm, 
                                           activation=activation, 
                                           dropout_p=dropout_p,
                                           use_scSE=use_scSE)
        if self.use_coord:
            self.coord_conv = AddCoords(rank=3, with_r=False)
            
    def forward(self, x, scaled_img=None):
        if self.pooling is not None:
            x = self.pooling(x)
        if self.use_IP:
            x = torch.cat([x, scaled_img], dim=1)
        if self.use_coord:
            x = self.coord_conv(x)
        x = self.basic_module(x)
        return x
    
class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 norm='bn', 
                 activation='relu', 
                 dropout_p=0.0,
                 use_scSE=False,
                 reduction=16):
        super(ExtResNetBlock, self).__init__()
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, norm=norm, activation=activation)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, norm=norm, activation=activation)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, norm=norm, activation=activation)
        self.non_linearity = nn.ELU(inplace=False)
        
        self.dropout = nn.Dropout3d(p=dropout_p)
        
        self.use_scSE = use_scSE
        if self.use_scSE:
            self.scse = scSEBlock(out_channels, reduction=reduction)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out = out + residual
        
        if self.use_scSE:
            out = self.scse(out)

        out = self.dropout(out)
        out = self.non_linearity(out)
        return out
    
class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm='bn', activation='relu'):
        super(SingleConv, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('batchnorm', normalization(out_channels, norm=norm))
        if activation == 'relu':
            self.add_module('relu', nn.ReLU(inplace=False))
        elif activation == 'lrelu':
            self.add_module('lrelu', nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif activation == 'elu':
            self.add_module('elu', nn.ELU(inplace=False))
        elif activation == 'gelu':
            self.add_module('elu', nn.GELU(inplace=False))
            
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2), mode='nearest', 
                 use_coord=False, norm='bn', activation='relu', dropout_p=0.0,use_scSE=False):
        super(Decoder, self).__init__()
        self.use_coord = use_coord
        if self.use_coord:
            self.coord_conv = AddCoords(rank=3, with_r=False)

        # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
        self.upsampling = Upsampling(transposed_conv=True, in_channels=in_channels, out_channels=out_channels,
                                     scale_factor=scale_factor, mode=mode)
        # sum joining
        self.joining = partial(self._joining, concat=False)
        # adapt the number of in_channels for the ExtResNetBlock
        in_channels = out_channels + 3 if self.use_coord else out_channels

        self.basic_module = ExtResNetBlock(in_channels, 
                                           out_channels, 
                                           norm=norm, 
                                           activation=activation, 
                                           dropout_p=dropout_p,
                                           use_scSE=use_scSE)
        

    def forward(self, encoder_features, x, return_input=False):
        x = self.upsampling(encoder_features, x)
        x = self.joining(encoder_features, x)
        if self.use_coord:
            x = self.coord_conv(x)
        if return_input:
            x1 = self.basic_module(x)
            return x1, x
        x = self.basic_module(x)
        return x
    
    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x

class Upsampling(nn.Module):
    def __init__(self, transposed_conv, in_channels=None, out_channels=None, scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)
    
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            xx_channel = x_range.expand([batch_size_shape, 1, -1])

            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            y_range = torch.linspace(-1, 1, dim_y, device=input_tensor.device)
            yy_channel, xx_channel = torch.meshgrid(y_range, x_range)
            yy_channel = yy_channel.expand([batch_size_shape, 1, -1, -1])
            xx_channel = xx_channel.expand([batch_size_shape, 1, -1, -1])

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            y_range = torch.linspace(-1, 1, dim_y, device=input_tensor.device)
            z_range = torch.linspace(-1, 1, dim_z, device=input_tensor.device)
            zz_channel, yy_channel, xx_channel = torch.meshgrid(z_range, y_range, x_range)
            zz_channel = zz_channel.expand([batch_size_shape, 1, -1, -1, -1])
            yy_channel = yy_channel.expand([batch_size_shape, 1, -1, -1, -1])
            xx_channel = xx_channel.expand([batch_size_shape, 1, -1, -1, -1])
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out
    