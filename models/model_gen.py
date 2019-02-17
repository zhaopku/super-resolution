import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class SRGANGenerator(nn.Module):
	def __init__(self, args):
		# upsample_block_num: log of scale factor, base at 2
		# each upscale factor upscale the base image by a factor of 2

		self.scale_factor = args.upscale_factor
		self.src_in_channels = args.in_channels

		upsample_block_num = int(math.log(self.scale_factor, 2))

		super(SRGANGenerator, self).__init__()
		self.block1 = nn.Sequential(
			# see https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
			# for H_out and W_out
			# 3 in_channels and 64 out_channels, stride is default to 1
			nn.Conv2d(in_channels=self.src_in_channels, out_channels=64, kernel_size=9, padding=4),
			nn.PReLU() # class torch.nn.PReLU(num_parameters=1, init=0.25)
		)
		# residual block with 64 in_channels and 64 out_channels
		self.block2 = ResidualBlock(64)
		self.block3 = ResidualBlock(64)
		self.block4 = ResidualBlock(64)
		self.block5 = ResidualBlock(64)
		self.block6 = ResidualBlock(64)
		self.block7 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.PReLU()
		)

		block8 = [UpsampleBLock(in_channels=64, up_scale=2) for _ in range(upsample_block_num)]
		block8.append(nn.Conv2d(64, self.src_in_channels, kernel_size=9, padding=4))
		self.block8 = nn.Sequential(*block8)

	def forward(self, x):
		block1 = self.block1(x)
		block2 = self.block2(block1)
		block3 = self.block3(block2)
		block4 = self.block4(block3)
		block5 = self.block5(block4)
		block6 = self.block6(block5)
		block7 = self.block7(block6)
		block8 = self.block8(block1 + block7)
		#TODO: figure out what happened here
		return (torch.tanh(block8) + 1) / 2

class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(channels)
		self.prelu = nn.PReLU()
		self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		residual = self.conv1(x)
		residual = self.bn1(residual)
		residual = self.prelu(residual)
		residual = self.conv2(residual)
		residual = self.bn2(residual)

		# this is element-wise sum
		return x + residual


class UpsampleBLock(nn.Module):
	def __init__(self, in_channels, up_scale):
		super(UpsampleBLock, self).__init__()
		self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
		# https://pytorch.org/docs/stable/nn.html?highlight=pixelshuffle#torch.nn.PixelShuffle
		self.pixel_shuffle = nn.PixelShuffle(up_scale)
		self.prelu = nn.PReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.pixel_shuffle(x)
		x = self.prelu(x)
		return x
