import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Discriminator(nn.Module):
	def __init__(self, args):
		super(Discriminator, self).__init__()
		self.in_channels = args.in_channels

		self.net = nn.Sequential(
			nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(0.2),

			nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),

			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),

			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, kernel_size=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(1024, 1, kernel_size=1)
		)

		self.sigmoid = torch.nn.Sigmoid()
		self.log_sigmoid = torch.nn.LogSigmoid()

	def forward(self, x):
		batch_size = x.size(0)

		logits = self.net(x).view(batch_size, -1)

		probs = self.sigmoid(logits)
		log_probs = self.log_sigmoid(logits)

		# return probs and log_probs
		return probs, log_probs
