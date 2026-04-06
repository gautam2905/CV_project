import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            c_in = in_ch if i == 0 else out_ch
            layers.extend([
                nn.Conv3d(c_in, out_ch, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=2):
        super().__init__()
        self.conv_block = ConvBlock3D(in_ch, out_ch, num_convs)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )

    def forward(self, x):
        return self.conv_block(x) + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, channels_list, num_convs=2):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels_list:
            self.stages.append(ResidualBlock3D(prev_ch, ch, num_convs))
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = ch

    def forward(self, x):
        skips = []
        for stage, pool in zip(self.stages, self.pools):
            x = stage(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, num_convs=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ResidualBlock3D(in_ch + skip_ch, out_ch, num_convs)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, bottleneck_ch, channels_list, num_convs=2):
        super().__init__()
        self.stages = nn.ModuleList()
        prev_ch = bottleneck_ch
        for ch in channels_list:
            self.stages.append(DecoderStage(prev_ch, ch, ch, num_convs))
            prev_ch = ch

    def forward(self, x, skips):
        outputs = []
        for stage, skip in zip(self.stages, reversed(skips)):
            x = stage(x, skip)
            outputs.append(x)
        return x, outputs


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_channels=32,
                 channel_multipliers=(1, 2, 4, 8, 10), num_conv_per_stage=2,
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        channels = [base_channels * m for m in channel_multipliers]
        enc_channels = channels[:-1]
        bottleneck_ch = channels[-1]

        self.encoder = Encoder(in_channels, enc_channels, num_conv_per_stage)
        self.bottleneck = ResidualBlock3D(enc_channels[-1], bottleneck_ch, num_conv_per_stage)
        self.decoder = Decoder(bottleneck_ch, list(reversed(enc_channels)), num_conv_per_stage)
        self.final_conv = nn.Conv3d(enc_channels[0], num_classes, kernel_size=1)

        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for ch in reversed(enc_channels[1:]):
                self.ds_heads.append(nn.Conv3d(ch, num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.01)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.01)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_enc, skips = self.encoder(x)
        x_bot = self.bottleneck(x_enc)
        x_dec, dec_outputs = self.decoder(x_bot, skips)
        logits = self.final_conv(x_dec)

        result = {"logits": logits}

        if self.deep_supervision and self.training:
            ds_logits = []
            for head, out in zip(self.ds_heads, dec_outputs[:-1]):
                ds_logits.append(head(out))
            result["deep_supervision"] = ds_logits

        return result
