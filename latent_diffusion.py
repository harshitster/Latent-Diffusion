import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

import blocks
from diffusion_kernel import Kernel, DDPMConfig
from vae import VAE
from vqvae import VQVAE


class ConditioningConfig:
    def __init__(self):
        self.conditioning_type = None  # 'text', 'class', 'image', None
        self.num_classes = None
        self.text_embed_dim = 512
        self.class_embed_dim = 512
        self.cross_attention = True
        self.conditioning_dropout = 0.1  # For classifier-free guidance


class LatentUNetConfig:
    def __init__(self):
        self.down_channels = [320, 640, 1280, 1280]
        self.up_channels = [
            1280 + 1280,
            1280 + 640,
            640 + 320,
            320,
        ]  # Account for skip connections
        self.bottleneck_channels = [1280, 1280]
        self.time_emb_dim = 320
        self.down_sample = [True, True, True, False]
        self.up_sample = [False, True, True, True]
        self.n_heads = 8
        self.n_timesteps = 1000
        self.n_groups = 32
        self.n_layers = 2
        self.attention = True


class CrossAttentionBlock(torch.nn.Module):
    def __init__(
        self, query_dim: int, context_dim: int, n_heads: int, head_dim: int = 64
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        inner_dim = head_dim * n_heads

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=query_dim)
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim), torch.nn.Dropout(0.1)
        )

    def forward(self, x, context=None):
        h = self.n_heads

        b, c, d, w, h_spatial = x.shape
        x_reshaped = x.reshape(b, c, -1).transpose(1, 2)  # (b, d*w*h, c)

        x_norm = self.norm(x).reshape(b, c, -1).transpose(1, 2)

        q = self.to_q(x_norm)

        if context is None:
            context = x_norm

        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(b, -1, h, self.head_dim).transpose(1, 2)  # (b, h, seq, head_dim)
        k = k.view(b, -1, h, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, h, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, -1, h * self.head_dim)
        out = self.to_out(out)

        # Add residual and reshape back
        out = out + x_reshaped
        out = out.transpose(1, 2).reshape(b, c, d, w, h_spatial)

        return out


class ConditionedEncoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        context_dim: int,
        downsample: bool,
        n_heads: int,
        n_layers: int,
        attention: bool,
        n_groups: int,
    ):
        super().__init__()

        self.encoder_block = blocks.EncoderBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            downsample=downsample,
            n_heads=n_heads,
            n_layers=n_layers,
            attention=attention,
            n_groups=n_groups,
        )

        if context_dim is not None:
            self.cross_attention = CrossAttentionBlock(
                query_dim=out_channels, context_dim=context_dim, n_heads=n_heads
            )
        else:
            self.cross_attention = None

    def forward(self, x, t_emb=None, context=None):
        x = self.encoder_block(x, t_emb)

        if self.cross_attention is not None and context is not None:
            x = self.cross_attention(x, context)

        return x


class ConditionedDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        context_dim: int,
        upsample: bool,
        n_heads: int,
        n_layers: int,
        attention: bool,
        n_groups: int,
    ):
        super().__init__()

        self.decoder_block = blocks.DecoderBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
            upsample=upsample,
            n_heads=n_heads,
            n_layers=n_layers,
            attention=attention,
            n_groups=n_groups,
        )

        if context_dim is not None:
            self.cross_attention = CrossAttentionBlock(
                query_dim=out_channels, context_dim=context_dim, n_heads=n_heads
            )
        else:
            self.cross_attention = None

    def forward(self, x, skip_connection=None, t_emb=None, context=None):
        x = self.decoder_block(x, skip_connection, t_emb)

        if self.cross_attention is not None and context is not None:
            x = self.cross_attention(x, context)

        return x


class LatentUNet(torch.nn.Module):
    def __init__(
        self, in_channels: int, conditioning_config: ConditioningConfig = None
    ):
        super().__init__()

        self.config = LatentUNetConfig()
        self.conditioning_config = conditioning_config or ConditioningConfig()

        # Conditioning embeddings
        context_dim = None
        if self.conditioning_config.conditioning_type == "text":
            context_dim = self.conditioning_config.text_embed_dim
        elif self.conditioning_config.conditioning_type == "class":
            context_dim = self.conditioning_config.class_embed_dim
            self.class_embedding = torch.nn.Embedding(
                self.conditioning_config.num_classes, context_dim
            )

        # Time embedding
        self.timestep_embedding_proj = torch.nn.Sequential(
            torch.nn.Linear(self.config.time_emb_dim, self.config.time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.config.time_emb_dim, self.config.time_emb_dim),
        )

        # Input convolution
        self.conv_in = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.config.down_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Encoder blocks
        self.encoder_blocks = torch.nn.ModuleList(
            [
                ConditionedEncoderBlock(
                    in_channels=self.config.down_channels[i],
                    out_channels=self.config.down_channels[i + 1],
                    time_emb_dim=self.config.time_emb_dim,
                    context_dim=context_dim,
                    downsample=self.config.down_sample[i],
                    n_heads=self.config.n_heads,
                    n_layers=self.config.n_layers,
                    attention=self.config.attention,
                    n_groups=self.config.n_groups,
                )
                for i in range(len(self.config.down_channels) - 1)
            ]
        )

        # Bottleneck blocks
        self.bottleneck_blocks = torch.nn.ModuleList(
            [
                blocks.Bottleneck(
                    in_channels=self.config.bottleneck_channels[i],
                    out_channels=self.config.bottleneck_channels[i + 1],
                    time_emb_dim=self.config.time_emb_dim,
                    n_heads=self.config.n_heads,
                    n_layers=self.config.n_layers,
                    attention=self.config.attention,
                    n_groups=self.config.n_groups,
                )
                for i in range(len(self.config.bottleneck_channels) - 1)
            ]
        )

        # Decoder blocks
        self.decoder_blocks = torch.nn.ModuleList(
            [
                ConditionedDecoderBlock(
                    in_channels=self.config.up_channels[i],
                    out_channels=self.config.up_channels[i + 1],
                    time_emb_dim=self.config.time_emb_dim,
                    context_dim=context_dim,
                    upsample=self.config.up_sample[i],
                    n_heads=self.config.n_heads,
                    n_layers=self.config.n_layers,
                    attention=self.config.attention,
                    n_groups=self.config.n_groups,
                )
                for i in range(len(self.config.up_channels) - 1)
            ]
        )

        # Output block
        self.out_block = torch.nn.Sequential(
            torch.nn.GroupNorm(
                num_groups=self.config.n_groups,
                num_channels=self.config.up_channels[-1],
            ),
            torch.nn.SiLU(),
            torch.nn.Conv3d(
                in_channels=self.config.up_channels[-1],
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        # Positional encoding for timesteps
        self.positional_encoding = blocks.PositionalEncoding(
            time_emb_dim=self.config.time_emb_dim, n_timesteps=self.config.n_timesteps
        )

    def forward(self, x, t, context=None):
        # Time embedding
        t_emb = self.positional_encoding(t)
        t_emb = self.timestep_embedding_proj(t_emb)

        # Process conditioning
        if context is not None:
            if self.conditioning_config.conditioning_type == "class":
                context = self.class_embedding(context)

            # Classifier-free guidance dropout
            if self.training and self.conditioning_config.conditioning_dropout > 0:
                mask = (
                    torch.rand(context.shape[0])
                    > self.conditioning_config.conditioning_dropout
                )
                context = context * mask.unsqueeze(1).to(context.device)

        # Input convolution
        x = self.conv_in(x)

        # Encoder
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, t_emb, context)
            skip_connections.append(x)

        # Bottleneck
        for bottleneck_block in self.bottleneck_blocks:
            x = bottleneck_block(x, t_emb)

        # Decoder
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, skip_connections.pop(), t_emb, context)

        # Output
        x = self.out_block(x)

        return x


class LatentDiffusionConfig:
    def __init__(self):
        # Autoencoder settings
        self.autoencoder_type = "vae"  # 'vae' or 'vqvae'
        self.autoencoder_path = None
        self.latent_scaling_factor = 0.18215

        # Diffusion settings
        self.n_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.schedule_type = "linear"

        # Training settings
        self.learning_rate = 1e-4
        self.n_epochs = 100
        self.checkpoint_dir = "./checkpoints"
        self.snapshot_freq = 10


class LatentDiffusionModel(torch.nn.Module):
    def __init__(self, autoencoder_config, conditioning_config=None):
        super().__init__()

        self.config = LatentDiffusionConfig()
        self.conditioning_config = conditioning_config

        # Initialize autoencoder
        if self.config.autoencoder_type == "vae":
            self.autoencoder = VAE(in_channels=autoencoder_config["in_channels"])
        elif self.config.autoencoder_type == "vqvae":
            self.autoencoder = VQVAE(in_channels=autoencoder_config["in_channels"])
        else:
            raise ValueError(
                f"Unknown autoencoder type: {self.config.autoencoder_type}"
            )

        # Load pretrained autoencoder if path provided
        if self.config.autoencoder_path:
            self.autoencoder.load_state_dict(torch.load(self.config.autoencoder_path))
            # Freeze autoencoder parameters
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        # Initialize diffusion kernel
        kernel_config = DDPMConfig()
        kernel_config.n_timesteps = self.config.n_timesteps
        kernel_config.beta_start = self.config.beta_start
        kernel_config.beta_end = self.config.beta_end
        kernel_config.schedule_type = self.config.schedule_type

        self.kernel = Kernel()
        self.kernel.config = kernel_config
        self.kernel.__init__()  # Re-initialize with new config

        # Initialize U-Net for latent space
        latent_channels = autoencoder_config.get("latent_channels", 4)
        self.unet = LatentUNet(
            in_channels=latent_channels, conditioning_config=conditioning_config
        )

    def encode(self, x):
        """Encode input to latent space"""
        self.autoencoder.eval()
        with torch.no_grad():
            if self.config.autoencoder_type == "vae":
                # For VAE, we need to get the latent representation
                latents = self.autoencoder.conv_in(x)
                for encoder_block in self.autoencoder.encoder_blocks:
                    latents = encoder_block(latents)
                latents = self.autoencoder.encoder_out(latents)
                # Apply scaling factor
                latents = latents * self.config.latent_scaling_factor
            elif self.config.autoencoder_type == "vqvae":
                # For VQ-VAE, encode and get quantized representation
                out = self.autoencoder.enc_conv_in(x)
                for encoder_block in self.autoencoder.encoder_blocks:
                    out = encoder_block(out)
                out = self.autoencoder.encoder_out(out)
                out = self.autoencoder.quantization["quant_in"](out)
                latents, _ = self.autoencoder.quantize(out)
                latents = latents * self.config.latent_scaling_factor

        return latents

    def decode(self, latents):
        """Decode latents back to original space"""
        self.autoencoder.eval()
        with torch.no_grad():
            # Remove scaling factor
            latents = latents / self.config.latent_scaling_factor

            if self.config.autoencoder_type == "vae":
                # For VAE decoding
                out = self.autoencoder.dec_conv_in(latents)
                for decoder_block in self.autoencoder.decoder_blocks:
                    out = decoder_block(out)
                out = self.autoencoder.decoder_out(out)
            elif self.config.autoencoder_type == "vqvae":
                # For VQ-VAE decoding
                out = self.autoencoder.quantization["quant_out"](latents)
                out = self.autoencoder.dec_conv_in(out)
                for decoder_block in self.autoencoder.decoder_blocks:
                    out = decoder_block(out)
                out = self.autoencoder.decoder_out(out)

        return out

    def forward(self, x, t, context=None):
        """Forward pass for training"""
        # Encode to latent space
        latents = self.encode(x)

        # Add noise for diffusion training
        noise = torch.randn_like(latents)
        noisy_latents = self.kernel.noise(latents, noise, t)

        # Predict noise with U-Net
        noise_pred = self.unet(noisy_latents, t, context)

        return noise_pred, noise


class LatentDiffusionTrainer:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or LatentDiffusionConfig()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )
        self.loss_function = torch.nn.MSELoss()

    def train_step(self, batch_data, context=None):
        """Single training step"""
        self.model.train()

        batch_size = batch_data.shape[0]
        t = torch.randint(
            0, self.config.n_timesteps, (batch_size,), device=batch_data.device
        )

        # Forward pass
        noise_pred, noise = self.model(batch_data, t, context)

        # Compute loss
        loss = self.loss_function(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, data_loader, load_checkpoint=None):
        """Full training loop"""
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.config.n_epochs):
            epoch_losses = []

            for batch_idx, batch_data in enumerate(
                tqdm(data_loader, desc=f"Epoch {epoch+1}")
            ):
                # Handle conditioning if present
                if isinstance(batch_data, tuple):
                    data, context = batch_data
                else:
                    data, context = batch_data, None

                loss = self.train_step(data, context)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{self.config.n_epochs} - Loss: {avg_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.snapshot_freq == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": avg_loss,
                }
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir, f"latent_diffusion_epoch_{epoch+1}.pth"
                )
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)


class LatentDiffusionSampler:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or LatentDiffusionConfig()

    def sample(self, sample_shape, context=None, guidance_scale=7.5, eta=0.0):
        """
        Sample from the latent diffusion model

        Args:
            sample_shape: Shape of latent to sample (B, C, D, H, W)
            context: Conditioning context
            guidance_scale: Classifier-free guidance scale
            eta: DDIM eta parameter (0 = DDIM, 1 = DDPM)
        """
        self.model.eval()

        # Start with random noise in latent space
        latents = torch.randn(sample_shape, device=next(self.model.parameters()).device)

        # Sampling loop
        with torch.no_grad():
            for i in tqdm(reversed(range(self.config.n_timesteps)), desc="Sampling"):
                t = torch.full(
                    (sample_shape[0],), i, device=latents.device, dtype=torch.long
                )

                # Classifier-free guidance
                if context is not None and guidance_scale > 1.0:
                    # Unconditional prediction
                    noise_pred_uncond = self.model.unet(latents, t, None)
                    # Conditional prediction
                    noise_pred_cond = self.model.unet(latents, t, context)
                    # Apply guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = self.model.unet(latents, t, context)

                # Denoise step
                latents, _ = self.model.kernel.denoise(latents, noise_pred, t)

        # Decode to original space
        samples = self.model.decode(latents)

        return samples

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
