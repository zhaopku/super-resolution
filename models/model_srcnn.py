import torch
from torch import nn

class ModelSRCNN(nn.Module):
	def __init__(self, args):
		super(ModelSRCNN, self).__init__()

		self.args = args
		self.conv_0 = nn.Conv2d(1, 64, kernel_size=9, padding=0)
		self.relu_0 = nn.ReLU()
		self.conv_1 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
		self.relu_1 = nn.ReLU()
		self.conv_2 = nn.Conv2d(32, 1, kernel_size=5, padding=0)

	def forward(self, x):
		x_conv_0 = self.conv_0(x)
		x_relu_0 = self.relu_0(x_conv_0)
		x_conv_1 = self.conv_1(x_relu_0)
		x_relu_1 = self.relu_1(x_conv_1)
		x_conv_2 = self.conv_2(x_relu_1)

		return x_conv_2
