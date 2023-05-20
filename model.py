import torch
import torch.nn as nn
import torch.nn.functional as F


# Taken from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    def __init__(self, rad):
        super(PhaseShuffle, self).__init__()
        self.rad = rad

    def forward(self, x):
        if self.rad == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.rad + 1) - self.rad
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                           x.shape)
        return x_shuffle


class Generator(nn.Module):
    """
    BiWaveGAN Generator.

    Input shape:  (batch_size, 1, latent_dim)
    Output shape: (batch_size, 1, slice_len)
    """

    def __init__(self,
                 slice_len=32768,
                 latent_dim=100,
                 model_size=32):
        super(Generator, self).__init__()
        assert slice_len in [16384, 32768, 65536]
        self.slice_len = slice_len
        self.latent_dim = latent_dim
        self.model_size = model_size
        self.dim_mul = 16 if slice_len == 16384 else 32

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * model_size * self.dim_mul),
            nn.ReLU(inplace=True)
        )
        dm_ms = self.dim_mul * model_size
        convtrans_list = [
            nn.ConvTranspose1d(dm_ms, dm_ms // 2, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(dm_ms // 2, dm_ms // 4, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(dm_ms // 4, dm_ms // 8, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(dm_ms // 8, dm_ms // 16, kernel_size=25, stride=4, padding=11, output_padding=1),
            nn.ReLU(inplace=True)
        ]

        if slice_len == 16384:
            convtrans_list.extend([
                nn.ConvTranspose1d(model_size, 1, kernel_size=25, stride=4, padding=11, output_padding=1),
                nn.Tanh()
            ])

        elif slice_len == 32768:
            convtrans_list.extend([
                nn.ConvTranspose1d(dm_ms // 16, model_size, kernel_size=25, stride=4, padding=11, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(model_size, 1, kernel_size=25, stride=2, padding=12, output_padding=1),
                nn.Tanh()
            ])

        else:
            convtrans_list.extend([
                nn.ConvTranspose1d(dm_ms // 16, model_size, kernel_size=25, stride=4, padding=11, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(model_size, 1, kernel_size=25, stride=4, padding=11, output_padding=1),
                nn.Tanh()
            ])
        self.convtrans_layers = nn.Sequential(*convtrans_list)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.dim_mul * self.model_size, 16)
        x = self.convtrans_layers(x)
        return x


class Encoder(nn.Module):
    """
    BiWaveGAN Encoder.

    Input shape:  (batch_size, 1, slice_len)
    Output shape: (batch_size, 1, latent_dim)
    """

    def __init__(self,
                 slice_len=37268,
                 latent_dim=100,
                 model_size=32):
        assert slice_len in [16384, 32768, 65536]
        super(Encoder, self).__init__()
        self.slice_len = slice_len
        self.latent_dim = latent_dim
        self.model_size = model_size
        self.dim_mul = 16 if slice_len == 16384 else 32
        dm_ms = self.dim_mul * model_size

        if slice_len == 16384:
            conv_list = [
                nn.Conv1d(1, model_size, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]

        elif slice_len == 32768:
            conv_list = [
                nn.Conv1d(1, model_size, kernel_size=25, stride=2, padding=12),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(model_size, dm_ms // 16, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]

        else:
            conv_list = [
                nn.Conv1d(1, model_size, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(model_size, dm_ms // 16, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]

        conv_list.extend([
            nn.Conv1d(dm_ms // 16, dm_ms // 8, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(dm_ms // 8, dm_ms // 4, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(dm_ms // 4, dm_ms // 2, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(dm_ms // 2, dm_ms, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ])
        self.conv_layers = nn.Sequential(*conv_list)
        self.fc = nn.Linear(4 * 4 * model_size * self.dim_mul, latent_dim)
        # activation in final layer?

    def forward(self, x):
        z = self.conv_layers(x)
        z = z.view(-1, 1, 4 * 4 * self.dim_mul * self.model_size)
        z = self.fc(z)
        return z


class Discriminator(nn.Module):
    def __init__(self,
                 slice_len=32768,
                 latent_dim=100,
                 model_size=32,
                 discrim_filters=512,
                 z_discrim_depth=4,
                 joint_discrim_depth=3,
                 phaseshuffle_rad=2):
        assert slice_len in [16384, 32768, 65536]
        super(Discriminator, self).__init__()
        self.slice_len = slice_len
        self.latent_dim = latent_dim
        self.model_size = model_size
        self.discrim_filters = discrim_filters
        self.z_discrim_depth = z_discrim_depth
        self.joint_discrim_depth = joint_discrim_depth
        self.phaseshuffle_rad = phaseshuffle_rad

        # construct x discriminator
        x_discrim_conv_list = [
            nn.Conv1d(1, model_size, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PhaseShuffle(rad=phaseshuffle_rad),
            nn.Conv1d(model_size, 2 * model_size, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PhaseShuffle(rad=phaseshuffle_rad),
            nn.Conv1d(2 * model_size, 4 * model_size, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PhaseShuffle(rad=phaseshuffle_rad),
            nn.Conv1d(4 * model_size, 8 * model_size, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PhaseShuffle(rad=phaseshuffle_rad),
            nn.Conv1d(8 * model_size, 16 * model_size, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]

        if slice_len == 16384:
            x_discrim_conv_list.extend([
                nn.Conv1d(16 * model_size, 16 * model_size, kernel_size=16, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        elif slice_len == 32768:
            x_discrim_conv_list.extend([
                nn.Conv1d(16 * model_size, 32 * model_size, kernel_size=25, stride=2, padding=12),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(32 * model_size, 32 * model_size, kernel_size=16, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        else:
            x_discrim_conv_list.extend([
                nn.Conv1d(16 * model_size, 32 * model_size, kernel_size=25, stride=4, padding=11),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv1d(32 * model_size, 32 * model_size, kernel_size=16, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        self.x_discrim = nn.Sequential(*x_discrim_conv_list)

        # construct z discriminator
        z_discrim_conv_list = [
            nn.Conv1d(self.latent_dim, discrim_filters, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        for i in range(z_discrim_depth - 1):
            z_discrim_conv_list.append(nn.Conv1d(discrim_filters, discrim_filters, kernel_size=1, stride=1, padding=0))
            z_discrim_conv_list.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.z_discrim = nn.Sequential(*z_discrim_conv_list)

        # construct joint discriminator
        joint_input_filters = 16 * model_size + discrim_filters if slice_len == 16384 else 32 * model_size + discrim_filters
        joint_discrim_modules = [
            nn.Conv1d(joint_input_filters, discrim_filters, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        for i in range(joint_discrim_depth - 2):
            joint_discrim_modules.append(nn.Conv1d(discrim_filters, discrim_filters, kernel_size=1, stride=1, padding=0))
            joint_discrim_modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        joint_discrim_modules.append(nn.Conv1d(discrim_filters, 1, kernel_size=1, stride=1, padding=0))
        self.joint_discrim = nn.Sequential(*joint_discrim_modules)

    def forward(self, x, z):
        x = self.x_discrim(x)
        z = z.reshape(-1, self.latent_dim, 1)
        z = self.z_discrim(z)
        concat = torch.cat((x, z), dim=1)
        output = self.joint_discrim(concat).reshape(-1)
        return output

    def gradient_penalty(self, real, z_hat, fake, z, device):
        batch_size, channels, audio_len = real.shape
        latent_dim = z.shape[1]
        eps = torch.rand((batch_size, 1, 1)).repeat(1, channels, audio_len).to(device)
        interpolated_data = real * eps + fake * (1 - eps)
        eps = torch.rand((batch_size, 1, 1)).repeat(1, 1, latent_dim).to(device)
        interpolated_z = z_hat * eps + z * (1 - eps)
        mixed_scores = self(interpolated_data, interpolated_z)

        gradients = torch.autograd.grad(
            outputs=mixed_scores,
            inputs=(interpolated_data, interpolated_z),
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )
        grad_x = gradients[0].view(gradients[0].shape[0], -1)
        grad_z = gradients[1].view(gradients[1].shape[0], -1)
        grad_cat = torch.cat((grad_x, grad_z), dim=1)
        grad_norm = grad_cat.norm(2, dim=1)
        grad_penalty = torch.mean((grad_norm - 1) ** 2)

        return grad_penalty


class BiWaveGAN:
    def __init__(self,
                 slice_len,
                 latent_dim,
                 model_size,
                 discrim_filters,
                 z_discrim_depth,
                 joint_discrim_depth,
                 phaseshuffle_rad,
                 device):
        self.slice_len = slice_len
        self.latent_dim = latent_dim
        self.model_size = model_size
        self.G = Generator(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size).to(device)
        self.E = Encoder(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size).to(device)
        self.D = Discriminator(slice_len=slice_len, latent_dim=latent_dim, model_size=model_size,
                               discrim_filters=discrim_filters, z_discrim_depth=z_discrim_depth,
                               joint_discrim_depth=joint_discrim_depth, phaseshuffle_rad=phaseshuffle_rad).to(device)

    def generate(self, z):
        return self.G(z)

    def encode(self, x):
        return self.E(x)

    def discriminate(self, x, z):
        return self.D(x, z)

    def reconstruct(self, x):
        return self.G(self.E(x))

    def train(self):
        for submodel in [self.G, self.E, self.D]:
            submodel.train()

    def eval(self):
        for submodel in [self.G, self.E, self.D]:
            submodel.eval()
