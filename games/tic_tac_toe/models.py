import torch
import torch.nn.functional as F
from torch import nn


class Conv2DBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, bn=False, relu=False):
        super().__init__()
        self.conv = nn.Conv2d(filters_in, filters_out, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        self.relu=None
        if bn:
            self.bn = nn.BatchNorm2d(filters_out)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        if self.bn:
            h = self.bn(h)
        if self.relu:
            h = self.relu(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv2DBlock(filters_in=filters, filters_out=filters, kernel_size=3, bn=True, relu=False)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


class Representation(nn.Module):
    def __init__(self, num_blocks, channels_in, latent_dim=4):
        super().__init__()
        self.conv0 = Conv2DBlock(channels_in, latent_dim, kernel_size=3, bn=True, relu=True)
        # self.residual_blocks = nn.ModuleList([ResidualBlock(latent_dim)] * num_blocks)

    def forward(self, x):
        x = self.conv0(x)

        # for block in self.residual_blocks:
        #     x = block(x)

        return x


class Prediction(nn.Module):
    def __init__(self, num_blocks, channels_in, size_x, size_y, policy_output_size, latent_dim=4):
        super().__init__()
        self.conv0 = Conv2DBlock(channels_in, latent_dim, 3, bn=True, relu=True)
        # self.residual_blocks = nn.ModuleList([ResidualBlock(latent_dim)] * num_blocks)
        # self.shared_conv = Conv2DBlock(latent_dim, latent_dim, 1, bn=True, relu=True)

        latent_size = latent_dim * size_x * size_y

        self.fc_output_policy_block = nn.Sequential(
            nn.Linear(latent_size, policy_output_size),
            nn.Softmax(dim=-1)
        )

        self.fc_output_value_block = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv0(x)

        # for block in self.residual_blocks:
        #     x = block(x)

        # x = self.shared_conv(x)

        x = torch.flatten(x)

        policy = self.fc_output_policy_block(x)
        value = self.fc_output_value_block(x)

        return policy, value


class Dynamics(nn.Module):
    def __init__(self, num_blocks, size_x, size_y, state_channels_in, action_channels_in, latent_dim=4):
        super().__init__()

        self.conv0 = Conv2DBlock(state_channels_in + action_channels_in, latent_dim, 3, bn=True, relu=True)
        # self.residual_blocks = nn.ModuleList([ResidualBlock(latent_dim)] * num_blocks)

        latent_size = size_x * size_y * latent_dim

        self.fc_output_reward_block = nn.Sequential(
            nn.Linear(latent_size, 1),
            nn.Tanh()
        )

    def forward(self, x_tuple):
        x = torch.cat(x_tuple, dim=1)

        x = self.conv0(x)
        # for block in self.residual_blocks:
        #     x = block(x)

        state_output = x

        x = x.flatten()
        reward_output = self.fc_output_reward_block(x)

        return state_output, reward_output


