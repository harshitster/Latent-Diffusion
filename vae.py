import torch

import blocks

class VAEConfig:
    def __init__(self):
        self.down_channels = None
        self.up_channels = None
        self.bottleneck_channels = None
        self.latent_channels = None
        self.down_sample = None
        self.up_sample = None
        self.n_layers = None
        self.n_heads = None
        self.n_groups = None

class VAE(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.config = VAEConfig()

        self.conv_in = torch.nn.Conv3d(in_channels=self.in_channels, out_channels=self.config.down_channels[0], kernel_size=3, padding=1, stride=1)

        self.encoder_blocks = torch.nn.ModuleList()
        for i in range(len(self.config.down_channels) - 1):
            self.encoder_block.append(
                blocks.EncoderBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i + 1],
                    time_emb_dim=None,
                    downsample=self.config.down_sample[i],
                    n_heads=None,
                    n_layers=self.config.n_layers,
                    attention=False,
                    n_groups=self.config.n_groups
                )
            )

        for i in range(len(self.config.bottleneck_channels) - 1):
            self.encoder_block.appen(
                blocks.Bottleneck(
                    in_channels=self.config.bottleneck_channels[i],
                    out_channels=self.config.bottleneck_channels[i + 1],
                    time_emb_dim=None,
                    n_heads=self.config.n_heads,
                    n_layers=self.config.n_layers,
                    attention=True,
                    n_groups=self.config.n_groups
                )
            )

        self.encoder_out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=self.config.n_groups, num_channels=self.config.bottleneck_channels[-1]),
            torch.nn.SiLU(),
            torch.nn.Conv3d(in_channels=self.config.bottleneck_channels[-1], out_channels=self.config.latent_channels, kernel_size=3, padding=1)
        )

        self.mu_activation = torch.nn.Linear(in_features=self.config.latent_channels, out_features=self.config.latent_channels, bias=False)
        self.logvar_activation = torch.nn.Linear(in_features=self.config.latent_channels, out_features=self.config.latent_channels, bias=False)

        self.dec_conv_in = torch.nn.Conv3d(in_channels=self.config.latent_channels, out_channels=self.config.bottleneck_channels[-1], kernel_size=3, padding=1)

        self.decoder_blocks = torch.nn.ModuleList()
        for i in reversed(range(1, len(self.config.bottleneck_channels))):
            self.decoder_blocks.append(
                blocks.Bottleneck(
                    in_channels=self.config.bottleneck_channels[i],
                    out_channels=self.config.bottleneck_channels[i - 1],
                    time_emb_dim=None,
                    n_heads=self.config.n_heads,
                    n_layers=self.config.n_layers,
                    attention=True,
                    n_groups=self.config.n_groups
                )
            )

        for i in range(len(self.config.up_channels) - 1):
            self.decoder_blocks.append(
                blocks.DecoderBlock(
                    in_channels=self.config.up_channels[i],
                    out_channels=self.config.up_channels[i + 1],
                    time_emb_dim=None,
                    upsample=self.config.up_sample[i],
                    n_heads=self.config.n_heads,
                    n_layers=self.config.n_layers,
                    attention=False,
                    n_groups=self.config.n_groups
                )
            )

        self.decoder_out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=self.config.n_groups, num_channels=self.config.up_channels[-1]),
            torch.nn.SiLU(),
            torch.nn.Conv3d(in_channels=self.config.up_channels[-1], out_channels=self.in_channels, kernel_size=3, padding=1)
        )

    def sample(self, x):
        b, c, d, w, h = x.shape
        x = x.reshape(x.size(0), -1)

        mu = self.mu_activation(x)
        logvar = self.logvar_activation(x)
        sigma = torch.sqrt(torch.exp(logvar))
        z = torch.randn_like(sigma)

        x = mu + sigma * z
        x = x.reshape(x.size(0), c, d, w, h)

        return x

    def forward(self, x):
        out = self.conv_in(x)

        for encoder_block in self.encoder_blocks:
            out = encoder_block(out)
        out = self.encoder_out(out)

        out = self.sample(out)

        out = self.dec_conv_in(out)
        for decoder_block in self.decoder_blocks:
            out = decoder_block(out)
        out = self.decoder_out(out)

        return out