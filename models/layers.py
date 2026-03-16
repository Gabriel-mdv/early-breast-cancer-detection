import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = ConvBNAct(in_channels, hidden_dim, 1)
        self.depthwise = ConvBNAct(hidden_dim, hidden_dim, 3, stride, 1)
        self.project = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        x = self.bn(x)
        return self.act(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
    def forward(self, x):
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        attn = self.sigmoid(self.conv(fusion))
        return fusion * attn
