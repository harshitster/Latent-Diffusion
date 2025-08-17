import torch

import blocks

class UNetConfig:
    def __init__(self):
        self.down_channels = None
        self.up_channels = None
        self.bottleneck_channels = None
        self.time_emb_dim = None
        self.down_sample = [False if i == 0 else True for i in range(len(self.down_channels) - 1)]
        self.up_sample = self.down_sample   
        self.n_heads = None
        self.n_timesteps = None
        self.n_groups = None
        self.n_layers = None
        self.attention = True

class UNet(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.config = UNetConfig()

        self.timestep_embedding_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.time_emb_dim, self.config.time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.time_emb_dim, self.config.time_emb_dim)
        )

        self.conv_in = torch.nn.Conv3d(in_channels=in_channels, out_channels=self.config.down_channels[0], kernel_size=1)

        self.encoder_blocks = torch.nn.ModuleList([
            blocks.EncoderBlock(
                in_channels=self.config.down_channels[i],
                out_channels=self.config.down_channels[i + 1],
                time_emb_dim=self.config.time_emb_dim,
                downsample=self.config.down_sample[i],
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                attention=self.config.attention,
                n_groups=self.config.n_groups
            )

            for i in range(len(self.config.down_channels) - 1)
        ])

        self.bottleneck_blocks = torch.nn.ModuleList([
            blocks.Bottleneck(
                in_channels=self.config.bottleneck_channels[i],
                out_channels=self.config.bottleneck_channels[i + 1],
                time_emb_dim=self.config.time_emb_dim,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                attention=self.config.attention,
                n_groups=self.config.n_groups
            )

            for i in range(len(self.config.bottleneck_channels) - 1)
        ])

        self.decoder_blocks = torch.nn.ModuleList([
            blocks.DecoderBlock(
                in_channels=self.config.up_channels[i],
                out_channels=self.config.up_channels[i + 1],
                time_emb_dim=self.config.time_emb_dim,
                upsample=self.config.up_sample[i],
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                attention=self.config.attention,
                n_groups=self.config.n_groups
            )

            for i in range(len(self.config.up_channels) - 1)
        ])


        self.out_block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=self.config.n_groups, num_channels=16),
            torch.nn.Conv3d(in_channels=16, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        )

        self.positional_encoding = blocks.PositionalEncoding(time_emb_dim=self.config.time_emb_dim, n_timesteps=self.config.n_timesteps)

    def forward(self, x, t):
        x = self.conv_in(x)

        t_emb = self.positional_encoding[t]
        t_emb = self.timestep_embedding_proj(t_emb)

        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, t_emb)
            skip_connections.append(x)

        for bottleneck_block in self.bottleneck_blocks:
            x = bottleneck_block(x, t_emb)
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, skip_connections.pop(), t_emb)
        
        x = self.out_block(x)

        return x