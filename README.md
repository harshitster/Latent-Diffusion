# Latent Diffusion Models - 3D Implementation

A PyTorch implementation of Latent Diffusion Models for high-resolution 3D volumetric data generation, based on the paper ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752) by Rombach et al.

## Overview

This repository provides a complete 3D implementation of Latent Diffusion Models (LDMs), featuring:

- **Two-Stage Training**: First train VAE/VQ-VAE, then train diffusion in latent space
- **Cross-Attention Conditioning**: Support for text, class, and image conditioning
- **Classifier-Free Guidance**: Enhanced sample quality through guidance scaling
- **Memory Efficient**: Perform diffusion in compressed latent space rather than raw 3D volumes
- **Flexible Architecture**: Modular design supporting both VAE and VQ-VAE backends
- **3D Volumetric Processing**: Specialized for medical imaging, 3D shapes, and volumetric data

## Key Advantages of Latent Diffusion

### Computational Efficiency
- **Reduced Memory Usage**: Diffusion operates on compressed latents (typically 8x-64x smaller)
- **Faster Training**: Significantly reduced computational requirements
- **High Resolution**: Generate high-quality 3D volumes that would be infeasible with pixel-space diffusion

### Enhanced Conditioning
- **Class Conditioning**: Control generation with categorical labels
- **Classifier-Free Guidance**: Fine-tune conditioning strength

## Architecture

### Two-Stage Pipeline

**Stage 1: Autoencoder Training**
- Train VAE or VQ-VAE on your 3D dataset
- Learn compressed latent representation
- Freeze autoencoder for Stage 2

**Stage 2: Latent Diffusion Training**
- Train U-Net to denoise in latent space
- Add cross-attention for conditioning
- Enable classifier-free guidance

### Core Components

- **`latent_diffusion.py`**: Complete latent diffusion implementation
- **`vae.py`** / **`vqvae.py`**: Autoencoder backends for latent space compression
- **`blocks.py`**: Shared building blocks (ResNet, Attention, etc.)
- **`diffusion_kernel.py`**: DDPM diffusion process implementation

## Usage

### 1. First Train the Autoencoder

```python
from vae import VAE, VAEConfig
from vqvae import VQVAE, VQVAEConfig
import torch

# Configure VAE
vae_config = VAEConfig()
vae_config.down_channels = [64, 128, 256, 512]
vae_config.up_channels = [512, 256, 128, 64]
vae_config.latent_channels = 4
vae_config.n_groups = 32

# Train VAE (your training loop here)
vae = VAE(in_channels=1)  # For single-channel 3D data
# ... training code ...
torch.save(vae.state_dict(), 'trained_vae.pth')
```

### 2. Configure and Train Latent Diffusion

```python
from latent_diffusion import (
    LatentDiffusionModel, 
    LatentDiffusionTrainer,
    LatentDiffusionConfig,
    ConditioningConfig
)

# Configure conditioning (optional)
conditioning_config = ConditioningConfig()
conditioning_config.conditioning_type = 'text'  # or 'class', 'image', None
conditioning_config.text_embed_dim = 512
conditioning_config.conditioning_dropout = 0.1

# Configure latent diffusion
ld_config = LatentDiffusionConfig()
ld_config.autoencoder_type = 'vae'
ld_config.autoencoder_path = 'trained_vae.pth'
ld_config.n_timesteps = 1000
ld_config.learning_rate = 1e-4

# Initialize model
autoencoder_config = {
    'in_channels': 1,
    'latent_channels': 4
}

model = LatentDiffusionModel(
    autoencoder_config=autoencoder_config,
    conditioning_config=conditioning_config
)

# Train
trainer = LatentDiffusionTrainer(model, ld_config)
trainer.train(data_loader)
```

### 3. Generate Samples

```python
from latent_diffusion import LatentDiffusionSampler

# Load trained model
sampler = LatentDiffusionSampler(model, ld_config)
sampler.load_checkpoint('checkpoints/latent_diffusion_epoch_100.pth')

# Generate unconditional samples
sample_shape = (4, 4, 16, 64, 64)  # (batch, latent_channels, depth, height, width)
samples = sampler.sample(sample_shape)

# Generate conditional samples with classifier-free guidance
context = get_text_embeddings("a brain MRI scan")  # Your text encoder
samples = sampler.sample(
    sample_shape, 
    context=context, 
    guidance_scale=7.5
)
```

## Model Architecture Details

### Latent U-Net with Cross-Attention

```python
# Default configuration
LatentUNetConfig:
    down_channels = [320, 640, 1280, 1280]
    up_channels = [2560, 1920, 960, 320]  # Includes skip connections
    time_emb_dim = 320
    n_heads = 8
    cross_attention = True  # For conditioning
```

### Cross-Attention Mechanism

The model uses cross-attention to inject conditioning information:
- **Query**: Features from U-Net layers
- **Key/Value**: Conditioning embeddings (text, class, etc.)
- **Multi-Head**: 8 attention heads by default

### Classifier-Free Guidance

During sampling, the model can interpolate between conditional and unconditional predictions:

```
ε_pred = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
```

## Configuration Options

### Autoencoder Settings
```python
LatentDiffusionConfig:
    autoencoder_type = 'vae'  # or 'vqvae'
    autoencoder_path = 'path/to/pretrained/autoencoder.pth'
    latent_scaling_factor = 0.18215  # Stabilizes training
```

### Conditioning Settings
```python
ConditioningConfig:
    conditioning_type = 'text'  # 'text', 'class', 'image', or None
    text_embed_dim = 512        # Dimension of text embeddings
    num_classes = 1000          # For class conditioning
    conditioning_dropout = 0.1  # For classifier-free guidance
```

### Training Settings
```python
LatentDiffusionConfig:
    n_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    learning_rate = 1e-4
    n_epochs = 100
```

## File Structure

```
├── latent_diffusion.py     # Main latent diffusion implementation
├── blocks.py               # Building blocks (EncoderBlock, DecoderBlock, etc.)
├── diffusion_kernel.py     # DDPM diffusion kernel
├── vae.py                  # Variational Autoencoder
├── vqvae.py               # Vector Quantized VAE
├── unet.py                # Standard U-Net (for comparison)
├── patch_discriminator.py  # Discriminator for adversarial training
└── README.md              # This file
```

## Training Pipeline

### Stage 1: Autoencoder Pretraining

1. **Prepare 3D Dataset**: Volumetric medical images, 3D shapes, etc.
2. **Train VAE/VQ-VAE**: Learn compressed latent representation
3. **Validate Reconstruction**: Ensure high-quality reconstruction
4. **Save Checkpoint**: Freeze autoencoder weights

### Stage 2: Latent Diffusion Training

1. **Load Pretrained Autoencoder**: Freeze encoder/decoder weights
2. **Configure Conditioning**: Set up text encoder or class embeddings
3. **Train U-Net**: Learn to denoise in latent space
4. **Enable Guidance**: Train with conditioning dropout

## Key Differences from Pixel-Space DDPM

### Memory Efficiency
- **Latent Space**: 16x16x16 latents vs 128x128x128 images
- **Reduced VRAM**: Can train on larger volumes
- **Faster Inference**: Fewer denoising steps needed

### Enhanced Quality
- **Perceptual Loss**: Autoencoder trained with perceptual objectives
- **Stable Training**: More stable than high-resolution pixel diffusion
- **Better Conditioning**: Cross-attention enables precise control

### Conditioning Capabilities
- **Text-to-3D**: Natural language descriptions → 3D volumes
- **Multi-Modal**: Combine text, class, and image conditioning
- **Guidance Control**: Adjust conditioning strength at inference

## Advanced Features

### Classifier-Free Guidance

Train the model to handle both conditional and unconditional generation:

```python
# During training, randomly drop conditioning
if torch.rand(1) < conditioning_dropout:
    context = None  # Unconditional training
```

### DDIM Sampling

Fast sampling with fewer denoising steps:

```python
# Sample with 50 steps instead of 1000
samples = sampler.sample(shape, eta=0.0, num_steps=50)
```

### Multiple Conditioning Types

Combine different conditioning modalities:

```python
conditioning_config.conditioning_type = ['text', 'class']
# Model learns to condition on both text and class labels
```

## Memory Considerations for 3D

### Autoencoder Design
- **Spatial Compression**: 8x downsampling per spatial dimension
- **Channel Expansion**: Increase channels while reducing spatial size
- **Attention Layers**: Only in lower resolution layers

### Gradient Checkpointing
```python
# Enable gradient checkpointing for memory efficiency
torch.utils.checkpoint.checkpoint(self.encoder_block, x)
```

## Citation

If you use this implementation, please cite the original Latent Diffusion Models paper:

```bibtex
@misc{rombach2022highresolutionimagesynthesislatent,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2022},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2112.10752}, 
}
```

And the original DDPM paper:

```bibtex
@misc{ho2020denoisingdiffusionprobabilisticmodels,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.11239}, 
}
```

## Examples and Applications

### Medical Imaging
- **Brain MRI Generation**: Condition on anatomical labels
- **CT Scan Synthesis**: Generate pathological variations
- **Microscopy**: Synthesize cellular structures

### 3D Shape Generation
- **CAD Models**: Generate mechanical parts from descriptions
- **Architectural Models**: Create building layouts
- **Character Models**: Generate 3D game assets

### Scientific Visualization
- **Molecular Structures**: Generate protein conformations
- **Fluid Dynamics**: Synthesize flow patterns
- **Climate Data**: Generate weather simulations

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce batch size
- Use gradient checkpointing
- Train autoencoder with smaller latent dimensions

**Poor Sample Quality**
- Increase guidance scale (7.5-15.0)
- Train autoencoder longer
- Use more diffusion timesteps

**Slow Training**
- Reduce U-Net model size
- Use mixed precision training
- Enable gradient accumulation

## Acknowledgments

This implementation builds upon the work of Rombach et al. in Latent Diffusion Models, combined with the foundational DDPM framework by Ho et al. Special thanks to the open-source community for advancing generative modeling research.