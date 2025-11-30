# model.py (fixed: no in-place ops, no layers created inside forward)
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class AFNOLikeSpectral(nn.Module):
    """
    AFNO-like spectral mixing but implemented without in-place ops.
    Multiplicative learnable complex weights applied to a low-frequency patch of the FFT.
    """
    def __init__(self, channels=64, num_bands=16):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        # learnable complex weights (real + imag)
        self.real = nn.Parameter(torch.randn(channels, num_bands, num_bands) * 0.01)
        self.imag = nn.Parameter(torch.randn(channels, num_bands, num_bands) * 0.01)
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: B x C x H x W
        B, C, H, W = x.shape
        # compute FFT -> complex tensor (B x C x H x (W//2+1))
        fft = torch.fft.rfft2(x, dim=(-2, -1))  # complex tensor

        # slice sizes for low-frequency patch
        fh = min(self.num_bands, fft.shape[-2])
        fw = min(self.num_bands, fft.shape[-1])

        # build complex weights (C x fh x fw)
        w = torch.complex(self.real[:, :fh, :fw], self.imag[:, :fh, :fw])  # C x fh x fw

        # avoid inplace by creating a copy to modify
        fft_mod = fft.clone()
        # broadcast multiply into the low-frequency region
        # fft_mod[:, :, :fh, :fw] = fft[:, :, :fh, :fw] * w[None, :, :, :]
        # Use an intermediate multiplication (no inplace on original)
        patch = fft[:, :, :fh, :fw] * w[None, :, :, :]
        fft_mod[:, :, :fh, :fw] = patch

        # inverse FFT using modified tensor
        x2 = torch.fft.irfft2(fft_mod, s=(H, W), dim=(-2, -1))
        x2 = self.post_conv(x2)
        x2 = self.norm(x2)
        x2 = self.act(x2)

        # return residual sum (no in-place)
        return x + x2


class ImageBackbone(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', pretrained=True, out_dim=512):
        super().__init__()
        # create timm backbone with features_only=True to get spatial map
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        last_ch = self.backbone.feature_info[-1]['num_chs']
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(last_ch, out_dim)
        self._last_ch = last_ch

    def forward(self, x):
        feats = self.backbone(x)  # list of feature maps
        last = feats[-1]          # B x C x H x W
        pooled = self.pool(last).flatten(1)
        out = self.proj(pooled)
        return out, last

    def out_channels(self):
        return self._last_ch


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, nhead=8, num_layers=2, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = dim * 4
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation="gelu",
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class EnvMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=128, dropout=0.1):
        super().__init__()
        if in_dim <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
                nn.GELU()
            )

    def forward(self, x):
        if self.net is None:
            return None
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, num_classes, env_in_dim=0, image_embed_dim=512, transformer_dim=256, afno_channels=64, pretrained=True):
        super().__init__()
        # image backbone
        self.image_backbone = ImageBackbone(out_dim=image_embed_dim, pretrained=pretrained)
        # AFNO-like spectral module (fixed channel size)
        self.afno = AFNOLikeSpectral(channels=afno_channels, num_bands=16)

        # adapt convolution to match AFNO channels (create once)
        backbone_last_ch = self.image_backbone.out_channels()
        self.adapt_spatial = nn.Conv2d(backbone_last_ch, afno_channels, kernel_size=1)

        # projection from global image embedding to transformer dim
        self.image_proj = nn.Linear(image_embed_dim, transformer_dim)

        # transformer encoder
        self.transformer = TransformerEncoderBlock(dim=transformer_dim, nhead=8, num_layers=2,
                                                   dim_feedforward=transformer_dim * 4)

        # tokens and pos emb
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 4, transformer_dim) * 0.02)

        # env mlp
        self.env_mlp = EnvMLP(env_in_dim, hidden=128, out_dim=transformer_dim)

        # pooled spatial projection (use afno_channels -> transformer_dim)
        self.pooled_proj = nn.Linear(afno_channels, transformer_dim)

        # heads
        self.head_image = nn.Linear(transformer_dim, num_classes)
        self.head_env = nn.Linear(transformer_dim, num_classes)
        self.head_fusion = nn.Linear(transformer_dim * 2, num_classes)

    def forward(self, image, env=None):
        # image: B x 3 x H x W
        img_vec, spatial = self.image_backbone(image)  # img_vec: B x D, spatial: B x C x H x W

        # adapt channels once (registered conv)
        spatial = self.adapt_spatial(spatial)  # B x afno_channels x H x W

        # spectral mixing (AFNO-like)
        spatial = self.afno(spatial)  # residual

        # global tokens
        img_token = self.image_proj(img_vec).unsqueeze(1)  # B x 1 x D

        env_token = None
        if env is not None and self.env_mlp.net is not None:
            env_token = self.env_mlp(env).unsqueeze(1)
        else:
            # use zero token if no env
            env_token = torch.zeros_like(img_token)

        pooled_spatial = nn.AdaptiveAvgPool2d(1)(spatial).flatten(1)  # B x afno_channels
        pooled_spatial_token = self.pooled_proj(pooled_spatial).unsqueeze(1)  # B x 1 x D

        cls_token = self.cls_token.expand(image.shape[0], -1, -1)  # B x 1 x D

        seq = torch.cat([cls_token, img_token, env_token, pooled_spatial_token], dim=1)  # B x 4 x D
        seq = seq + self.pos_emb

        trans_out = self.transformer(seq)  # B x 4 x D

        cls_out = trans_out[:, 0]
        img_out = trans_out[:, 1]
        env_out = trans_out[:, 2]
        pooled_out = trans_out[:, 3]

        logits_image = self.head_image(img_out)
        logits_env = self.head_env(env_out)
        fusion = torch.cat([cls_out, pooled_out], dim=1)
        logits_fusion = self.head_fusion(fusion)

        logits = (logits_image + logits_env + logits_fusion) / 3.0
        return logits
