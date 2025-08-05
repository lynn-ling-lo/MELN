import torch
import torch.nn as nn

from models.backbone import get_blocks, bottleneck_IR
from models.transformer import VisionTransformer
from models.mermix import MerMix

import copy


class Backbone(nn.Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        self.body = nn.Sequential(*(modules[:-3]))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        return x


class Baseline(nn.Module):
    def __init__(
        self, 
        input_size=[112, 112],
        backbone_dim=256, 
        ratio=16,
        embed_dim=768,
        num_heads=5
    ):
        super(Baseline, self).__init__()

        self.backbone = Backbone(input_size, 50, 'ir')
        self.backbone.load_state_dict(torch.load('/path/to/IR50/backbone_ir50_ms1m_epoch120.pth', map_location='cpu'), strict=False)
        self.embed_dim = embed_dim

        self.transformer = VisionTransformer(num_classes=num_heads)

        self.proj = nn.Conv2d(backbone_dim, self.transformer.embed_dim, kernel_size=1)
        print('Baseline model loaded!')

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        x = self.transformer(x)

        return x


class Mymodel(nn.Module):
    def __init__(
        self, 
        input_size=[112, 112],
        backbone_dim=256, 
        keep_rate = 0.3,
        ratio=16,
        embed_dim=768,
        num_heads=5
    ):
        super(Mymodel, self).__init__()

        self.backbone = Backbone(input_size, 50, 'ir')
        self.backbone.load_state_dict(torch.load('/path/to/IR50/backbone_ir50_ms1m_epoch120.pth', map_location='cpu'), strict=False)
        self.embed_dim = embed_dim

        self.transformer = MerMix(keep_rate=keep_rate)
        

        self.proj = nn.Conv2d(backbone_dim, self.transformer.embed_dim, kernel_size=1)
        print('MERMix model loaded!')

    def forward(self, x, neutral, neutral_B, warm_up=True):
        x = self.backbone(x)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        
        neutral_B = self.proj(self.backbone(neutral_B))
        neutral_B = neutral_B.permute(0, 2, 3, 1).flatten(1, 2)
        
        if self.training:
            x_ori, out_mask, me_aux  = self.transformer(x)
            
            if warm_up == False:
                # Mix for data augmentation
                
                x_aug = self._mix_batch(x, neutral_B, out_mask)
                
                x_aug, out_mask_aug, me_aux_aug = self.transformer(x_aug)
                
                return x_ori,out_mask, me_aux, x_aug, out_mask_aug, me_aux_aug
            else:

                return x_ori, out_mask, me_aux
        else:
            x  = self.transformer(x)
            return x

    def _mix_batch(self, x, neutral, mask):
        
        B,N,C = x.shape
        mask_r = 1-mask
        
        x_masked = x*(mask.unsqueeze(-1).repeat(1,1,C))
        neutral_masked = neutral* (mask_r.unsqueeze(-1).repeat(1,1,C))
        x_mix = x_masked.add_(neutral_masked)
        return x_mix
        
        '''if use_cutmix:
            lam = self.lam_constant
            if self.mask_type == 'block':
                mask, lam = generate_mask(lam, x.device, self.minimum_tokens)
            elif self.mask_type == 'random':
                mask, lam = generate_mask_random(lam, x.device, self.minimum_tokens)
            else:
                raise ValueError(f"unsupported mask type {self.mask_type}")

            mask_224 = torch.nn.functional.interpolate(mask, size=(224, 224), mode='nearest')

            x_flip = x.flip(0).mul_(mask_224)
            x.mul_(1 - mask_224).add_(x_flip)

        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam, mask'''