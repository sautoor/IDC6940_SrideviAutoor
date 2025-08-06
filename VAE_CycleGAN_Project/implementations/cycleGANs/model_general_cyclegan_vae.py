import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, factor=2):
        super(Upsample, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * factor**2, kernel_size=kernel_size, stride=stride, padding=padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0, f"out_channels * factor**2 ({out_channels * factor**2}) must be divisible by in_channels ({in_channels})"
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x):
        z = self.conv(x) + x.repeat_interleave(self.repeats, dim=1) # conv w skip
        return F.pixel_shuffle(z, self.factor)
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, factor=2):
        super(Downsample, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels // factor**2, kernel_size=kernel_size, stride=stride, padding=padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels % factor**2 == 0
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = int(in_channels * factor**2 // out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = F.pixel_unshuffle(y, self.factor)

        x = F.pixel_unshuffle(x, self.factor)
        x = x.view(x.shape[0], self.out_channels, self.group_size, *x.shape[2:])
        x = x.mean(dim=2)  # Average over the group dimension

        return y + x  # Add the downsampled skip connection


class GroupedLinear(torch.nn.Module): # nn.Conv2d but block multiplication: each block size = out_channels / groups...
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        groups=1,
        device=None,
        dtype=None,
        skip: bool = True,  # Whether to add a skip connection
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._conv = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     groups=groups,
                                     bias=bias,
                                     device=device,
                                     dtype=dtype)
        
        if in_channels >= out_channels:
            assert in_channels % out_channels == 0, "in_channels must be divisible by out_channels for skip"
            self.group_size = int(in_channels // out_channels)
            self.downsample = True
        else:
            assert out_channels % in_channels == 0, "out_channels must be divisible by in_channels for skip"
            self.group_size = int(out_channels // in_channels)
            self.downsample = False

        self.skip = skip
        # initialize convolution to 0 
        torch.nn.init.normal_(self._conv.weight.data, 0.0, 0.0000002)
        torch.nn.init.constant_(self._conv.bias.data, 0.0) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._conv(x)
        
        if self.skip:
            if self.downsample:
                z = z + x.view(x.shape[0], self.out_channels, self.group_size, *x.shape[2:]).mean(dim=2)  # Average over the group dimension
            else:
                z = z + x.repeat_interleave(self.group_size, dim=1)  # Repeat each channel group_size times

        return z


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks, n_layers, latent_dim, lambda_kl=1e-4):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        self.lambda_kl = lambda_kl

        # Initial convolution block
        out_features = 64
        encoder = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features


        # Downsampling
        for _ in range(n_layers):
            out_features *= 2
            encoder += [
                # nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                Downsample(in_features, out_features, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            encoder += [ResidualBlock(out_features)]

        self.encoder = nn.Sequential(*encoder)

        # normal distribution
        groups = 1
        self.fc_mu = GroupedLinear(out_features, latent_dim, groups=groups, skip=True) # * self.latent_img_size**2
        self.fc_logvar = GroupedLinear(out_features, latent_dim, groups=groups, skip=False)
        self.fc_decode = GroupedLinear(latent_dim, out_features, groups=groups, skip=True)


        # Decoder
        decoder = []
        for _ in range(num_residual_blocks):
            decoder += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(n_layers):
            out_features //= 2
            decoder += [
                # nn.Upsample(scale_factor=2),
                # nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                Upsample(in_features, out_features, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        decoder += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.decoder = nn.Sequential(*decoder)

    def kld_loss(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / mu.size(0)

    def forward(self, x):
        # return self.model(x)\

        e = self.encoder(x)
        mu = self.fc_mu(e)
        logvar = self.fc_logvar(e) - 10
        logvar = torch.clamp(logvar, -10, 0)  # Clamp logvar to avoid numerical issues
        
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        if self.training:  # Only calculate KL divergence loss during training
            kl_loss = self.kld_loss(mu, logvar)
            wandb.log({"kl_loss": kl_loss.item()})  # Log KL divergence loss to Weights & Biases
            (kl_loss * self.lambda_kl).backward(retain_graph=True)  # Retain graph for backpropagation

        z = self.fc_decode(z)
        
        return self.decoder(z)

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
