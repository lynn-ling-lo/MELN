import torch
import torch.nn as nn

from models.transformer import VisionTransformer
from timm.models.layers import to_2tuple




class PureViT(nn.Module):
    def __init__(
        self, 
        input_size=[224, 224],
        backbone_dim=256, 
        ratio=16,
        embed_dim=768,
        num_heads=5
    ):
        super(PureViT, self).__init__()
        self.transformer = VisionTransformer(num_classes=num_heads)
        input_size = to_2tuple(224)
        patch_size = to_2tuple(16)
        self.img_size = input_size
        input_channel = 3

        self.patchify = nn.Conv2d(input_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

        print('PureViT loaded!')

    def forward(self, x):
        x = self.patchify(x)#.flatten(2).transpose(1, 2)
        x = self.transformer(x)

        return x



