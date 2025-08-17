import torch
import numpy as np

class PositionalEncoding(torch.nn.Module):
    def __init__(self, time_emb_dim: int, n_timesteps: int):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.n_timesteps = n_timesteps

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(self.time_emb_dim, self.n_timesteps))

    def _get_sinsoid_encoding_table(self, time_emb_dim, n_timesteps):
        def _get_position_angle_vec(timestep):
            return [timestep / np.power(10000, 2 * (d // 2) / time_emb_dim) for d in range(time_emb_dim)]
        
        sinusoid_table = np.array([_get_position_angle_vec(timestep) for timestep in range(n_timesteps)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)
    
    def forward(self, timestep):
        positional_encodings = self.pos_table[timestep,:]

        return positional_encodings
    
class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, downsample: bool, n_heads: int, n_layers: int, attention: bool, n_groups: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.downsample = downsample
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention = attention
        self.n_groups = n_groups

        self.resnet_conv1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.in_channels if i == 0 else self.out_channels),
                torch.nn.SiLU(),
                torch.nn.Conv3d(in_channels=self.in_channels if i == 0 else self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            )

            for i in range(self.n_layers)
        ])

        if self.time_emb_dim is not None:
            self.time_emb_layer = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(in_features=self.time_emb_dim, out_features=self.out_channels)
                )

                for _ in range(self.n_layers)
            ])

        self.resnet_conv2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.out_channels),
                torch.nn.SiLU(),
                torch.nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            )

            for _ in range(self.n_layers)
        ])

        if self.attention:
            self.attention_norm = torch.nn.ModuleList([
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.out_channels)

                for _ in range(self.n_layers)
            ])

            self.attention = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.n_heads, batch_first=True)

                for _ in range(self.n_layers)
            ])

        self.residual_input_conv = torch.nn.ModuleList([
            torch.nn.Conv3d(in_channels=self.in_channels if i == 0 else out_channels, out_channels=self.out_channels, kernel_size=1)

            for i in range(self.n_layers)
        ])

        if self.downsample:
            self.downsample_conv = torch.nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.downsample_conv = torch.nn.Identity()

    def forward(self, x, t_emb=None):
        out = x

        for i in range(self.n_layers):
            resnet_input = out
            out = self.resnet_conv1[i](out)
            if t_emb is not None:
                out += self.time_emb_layer[i](t_emb)[:, :, None, None, None]
            out = self.resnet_conv2[i](out)
            out += self.residual_input_conv[i](resnet_input)

            if self.attention:
                b, c, d, w, h = out.shape
                in_attn = out.reshape(b, c, d * w * h)
                in_attn = self.attention_norm[i](in_attn).transpose(1, 2)
                out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(b, c, d, w, h)
                out += out_attn

        out = self.downsample_conv(out)

        return out
    
class Bottleneck(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, n_heads: int, n_layers: int, attention: bool, n_groups: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention = attention
        self.n_groups = n_groups

        self.resnet_conv1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.in_channels if i == 0 else self.out_channels),
                torch.nn.SiLU(),
                torch.nn.Conv3d(in_channels=self.in_channels if i == 0 else self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            )

            for i in range(self.n_layers + 1)
        ])

        if self.time_emb_dim is not None:
            self.time_emb_layer = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(in_features=self.time_emb_dim, out_features=self.out_channels)
                )

                for _ in range(self.n_layers + 1)
            ])

        self.resnet_conv2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.out_channels),
                torch.nn.SiLU(),
                torch.nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            )

            for _ in range(self.n_layers + 1)
        ])

        if self.attention:
            self.attention_norm = torch.nn.ModuleList([
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.out_channels)

                for _ in range(self.n_layers)
            ])

            self.attention = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.n_heads, batch_first=True)

                for _ in range(self.n_layers)
            ])

        self.residual_input_conv = torch.nn.ModuleList([
            torch.nn.Conv3d(in_channels=self.in_channels if i == 0 else self.out_channels, out_channels=self.out_channels, kernel_size=1)

            for i in range(self.n_layers + 1)
        ])

    def forward(self, x, t_emb=None):
        out = x

        resnet_input = x
        out = self.resnet_conv1[0](out)
        if self.time_emb_dim is not None:
            out += self.time_emb_layer[0](t_emb)[:, :, None, None, None]
        out = self.resnet_conv2[0](out)
        out += self.residual_input_conv[0](resnet_input)


        for i in range(self.n_layers):
            b, c, d, w, h = out.shape
            in_attn = out.reshape(b, c, d * w * h)
            in_attn = self.attention_norm[i](in_attn).transpose(1, 2)
            out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(b, c, d, w, h)
            out += out_attn

            resnet_input = out
            out = self.resnet_conv1[i + 1](out)
            if self.time_emb_dim is not None:
                out += self.time_emb_layer[i + 1](t_emb)[:, :, None, None, None]
            out = self.resnet_conv2[i + 1](out)
            out += self.residual_input_conv[i + 1](resnet_input)

        return out
    
class DecoderBlock(torch.nn.ModuleList):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, upsample: bool, n_heads: int, n_layers: int, attention: bool, n_groups: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.upsample = upsample
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention = attention
        self.n_groups = n_groups

        self.resnet_conv1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.in_channels if i == 0 else self.out_channels),
                torch.nn.SiLU(),
                torch.nn.Conv3d(in_channels=self.in_channels if i == 0 else self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            )

            for i in range(self.n_layers)
        ])

        if self.time_emb_dim is not None:
            self.time_emb_layer = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(in_features=self.time_emb_dim, out_features=self.out_channels)
                )
                
                for _ in range(self.n_layers)
            ])

        self.resnet_conv2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.out_channels),
                torch.nn.SiLU(),
                torch.nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            )

            for _ in range(self.n_layers)
        ])

        if self.attention:
            self.attention_norm = torch.nn.ModuleList([
                torch.nn.GroupNorm(num_groups=self.n_groups, num_channels=self.out_channels)

                for _ in range(self.n_layers)
            ])

            self.attention = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(embed_dim=self.out_channels, num_heads=self.n_heads, batch_first=True)

                for _ in range(self.n_layers)
            ])

        self.residual_input_conv = torch.nn.ModuleList([
            torch.nn.Conv3d(in_channels=self.in_channels if i == 0 else self.out_channels, out_channels=self.out_channels, kernel_size=1)

            for i in range(self.n_layers)
        ])

        if self.upsample:
            self.upsample_conv = torch.nn.ConvTranspose3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample_conv = torch.nn.Identity()

    def forward(self, x, skip_connection=None, t_emb=None):
        x = self.upsample_conv(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        out = x
        
        for i in range(self.n_layers):
            resnet_input = out
            out = self.resnet_conv1[i](out)
            if t_emb is not None:
                out += self.time_emb_layer[i](t_emb)[:, :, None, None, None]
            out = self.resnet_conv2[i](out)
            out += self.residual_input_conv(resnet_input)

            if self.attention:
                b, c, d, w, h = out.shape
                in_attn = out.reshape(b, c, d * w * h)
                in_attn = self.attention_norm[i](in_attn).transpose(1, 2)
                out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(b, c, d, w, h)
                out += out_attn

        return out