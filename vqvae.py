import torch

import blocks

class VQVAEConfig:
    def __init__(self):
        self.down_channels = None
        self.up_channels = None
        self.bottleneck_channels = None
        self.latent_channels = None
        self.codebook_size = None
        self.down_sample = None
        self.up_sample = None
        self.n_heads = None
        self.n_layers = None
        self.n_groups = None

class VQVAE(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.config = VQVAEConfig()

        self.enc_conv_in = torch.nn.Conv3d(in_channels=in_channels, out_channels=self.config.down_channels, kernel_size=3, padding=1)

        self.encoder_blocks = torch.nn.ModuleList()
        for i in range(len(self.config.down_channels) - 1):
            self.encoder_blocks.append(
                blocks.EncoderBlock(
                    in_channels=self.config.down_channels[i],
                    out_channels=self.config.down_channels[i + 1],
                    time_emb_dim=None,
                    downsample=self.config.down_sample[i],
                    n_heads=None,
                    n_layers=self.config.n_layers,
                    attention=False,
                    n_groups=self.config.n_groups
                )
            )

        for i in range(len(self.config.bottleneck_channels) - 1):
            self.encoder_block.append(
                blocks.Bottleneck(
                    in_channels=self.config.bottleneck_channels[i],
                    out_channels=self.config.bottleneck_channels[i + 1],
                    time_emb_dim=None,
                    n_heads=self.config.n_heads,
                    attention=True,
                    n_groups=self.config.n_groups
                )
            )

        self.encoder_out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=self.config.n_groups, num_channels=self.config.bottleneck_channels[-1]),
            torch.nn.SiLU(),
            torch.nn.Conv3d(in_channels=self.config.bottleneck_channels[-1], out_channels=self.config.latent_channels, kernel_size=3, padding=1)
        )


        self.quantization = torch.nn.ModuleDict({
            'quant_in': torch.nn.Conv3d(in_channels=self.config.latent_channels, out_channels=self.config.latent_channels, kernel_size=1),
            'embedding': torch.nn.Embedding(num_embeddings=self.config.codebook_size, embedding_dim=self.config.latent_channels),
            'quant_out': torch.nn.Conv3d(in_channels=self.config.latent_channels, out_channels=self.config.latent_channels, kernel_size=1)
        })

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

    def quantize(self, x):
        b, c, d, w, h = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = x.reshape(x.size(0), -1, x.size(-1))

        dist = torch.cdist(x, self.quantization['embedding'].weight[None, :].repeat((x.size(0), 1, 1)))
        min_enc_indices = torch.argmin(dist, dim=-1)

        quant_out = torch.index_select(self.quantization['embedding'].weight, 0, min_enc_indices.view(-1))

        x = x.reshape((-1, x.size(-1)))

        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantization_loss = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss
        }
        quant_out = x + (quant_out - x).detach()

        quant_out = quant_out.reshape((b, d, w, h, c)).permute(0, 4, 1, 2, 3)
        return quant_out, quantization_loss
    
    def forward(self, x):
        out = self.enc_conv_in(x)
        for encode_layer in self.encoder_blocks:
            out = encode_layer(out)
        
        out = self.encoder_out(out)

        out = self.quantization['quant_in'](out)
        z, quantization_losses = self.quantize(out)
        out = z
        out = self.quantization['quant_out'](out)

        out = self.dec_conv_in(out)
        for decoder_layer in self.decoder_blocks:
            out = decoder_layer(out)

        out = self.decoder_out(out)

        return out, z, quantization_losses