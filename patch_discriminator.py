import torch

class DiscriminatorConfig:
    def __init__(self):
        self.in_channels = 3
        self.out_channels = None
        self.kernel_size = None
        self.strides = None
        self.padding = None

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.config = DiscriminatorConfig()

        channels = [self.config.in_channels] + self.config.out_channels  + [1]
        self.d_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=self.config.kernel_size,
                    stride=self.config.strides,
                    padding=self.config.padding
                ),
                torch.nn.BatchNorm3d(
                    num_features=channels[i + 1] 
                ) if i != len(channels) - 2 else torch.nn.Identity(),
                torch.nn.LeakyReLU(
                    negative_slope=0.2
                ) if i != len(channels) - 2 else torch.nn.Identity()
            )

            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        out = x

        for layer in self.d_layers:
            out = layer(out)

        return out