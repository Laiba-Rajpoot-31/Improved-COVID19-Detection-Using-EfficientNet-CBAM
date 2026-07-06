import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    Focuses on 'what' is meaningful in an input image.
    Uses both Max Pooling and Average Pooling, followed by a shared MLP.
    """
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Global Average Pooling and Global Max Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared Multi-Layer Perceptron (MLP)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Element-wise addition of the two outputs
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Focuses on 'where' is an informative part, which is complementary to Channel Attention.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average and max pooling along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel axis
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially infers attention maps along two separate dimensions,
    channel and spatial, then the attention maps are multiplied to the input feature map
    for adaptive feature refinement.
    """
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply Channel Attention
        out = x * self.ca(x)
        # Apply Spatial Attention
        out = out * self.sa(out)
        return out
