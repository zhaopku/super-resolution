from os import listdir
from os.path import join
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch

MAX_PIXEL = 1536
MIN_PIXEL = 0


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
	return crop_size - (crop_size % upscale_factor)

# ToTensor(): to convert the numpy images to torch images
# swap color axis because
# numpy image: H x W x C
# torch image: C X H X W

def train_hr_transform(crop_size):
	return Compose([
		# Crop the given PIL Image at a random location.
		RandomCrop(crop_size),
		ToTensor(),
	])


def train_lr_transform(crop_size, upscale_factor):
	return Compose([
		ToPILImage(),
		# Resize() takes as input a PIL image, first argument is desired output size
		# https://pytorch.org/docs/stable/torchvision/transforms.html
		Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
		ToTensor()
	])


def display_transform():
	return Compose([
		ToPILImage(),
		# make the output image larger
		# Resize(200, interpolation=Image.BICUBIC),
		# CenterCrop(200),
		ToTensor()
	])

def create_new_lr_image(lr_images, hr_images):
	lr_images_new = torch.zeros_like(hr_images)

	_, _, old_height, old_width = lr_images.size()
	batch_size, channel, height, width = hr_images.size()

	for idx in range(batch_size):
		for c in range(channel):
			for h in range(height):
				for w in range(width):
					if h < old_height and w < old_width:
						lr_images_new[idx][c][h][w] = lr_images[idx][c][h][w]

	return lr_images_new

class TrainDatasetFromFolder(Dataset):
	def __init__(self, dataset_dir, crop_size, upscale_factor, ratio=1.0):
		super(TrainDatasetFromFolder, self).__init__()
		all_image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

		train_size = int(len(all_image_filenames) * ratio)
		self.image_filenames = all_image_filenames[:train_size]

		crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

		self.hr_transform = train_hr_transform(crop_size)
		self.lr_transform = train_lr_transform(crop_size, upscale_factor)

	def __getitem__(self, index):
		# for high resolution images, first random crop to augment data, then convert to Torch tensor
		# xx = Image.open(self.image_filenames[index])
		# yy = RandomCrop(50)(xx)
		# zz = ToTensor()(yy)
		#print(index)
		hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
		# to obtain low resolution image, use resize at the high resolution image
		lr_image = self.lr_transform(hr_image)
		return lr_image, hr_image

	def __len__(self):
		return len(self.image_filenames)

class ValDatasetFromFolder(Dataset):
	def __init__(self, dataset_dir, upscale_factor):
		super(ValDatasetFromFolder, self).__init__()
		self.upscale_factor = upscale_factor
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)][:150]

	def __getitem__(self, index):
		# this is the original image, without random crop
		hr_image = Image.open(self.image_filenames[index])
		# size of the original image
		w, h = hr_image.size
		crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
		lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
		hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
		# original high resolution image
		hr_image = CenterCrop(crop_size)(hr_image)
		# downscaled low resolution image
		lr_image = lr_scale(hr_image)
		# x, y = lr_image.size
		# super resolution image with naive method. Same shape as the original high resolution image
		hr_restore_img = hr_scale(lr_image)
		ConvertToTensor = ToTensor()

		return ConvertToTensor(lr_image), ConvertToTensor(hr_restore_img), ConvertToTensor(hr_image)

	def __len__(self):
		return len(self.image_filenames)


class TestRawDatasetFromFolder(Dataset):
	"""
	only used to convert raw low resolution images to high resolution images
	does not need high resolution images
	"""
	def __init__(self, dataset_dir, upscale_factor):
		super(TestRawDatasetFromFolder, self).__init__()
		self.upscale_factor = upscale_factor
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

	def __getitem__(self, index):
		lr_image = Image.open(self.image_filenames[index])
		w, h = lr_image.size

		hr_scale = Resize((self.upscale_factor * w, self.upscale_factor * h), interpolation=Image.BICUBIC)

		hr_restore_img = hr_scale(lr_image)
		return ToTensor()(lr_image), ToTensor()(hr_restore_img)

	def __len__(self):
		return len(self.image_filenames)

class TestDatasetFromFolder(Dataset):
	def __init__(self, dataset_dir, upscale_factor):
		super(TestDatasetFromFolder, self).__init__()
		self.upscale_factor = upscale_factor
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)][:150]

	def __getitem__(self, index):
		# this is the original image, without random crop
		hr_image = Image.open(self.image_filenames[index])
		# size of the original image
		w, h = hr_image.size
		lr_scale = Resize((h // self.upscale_factor, w // self.upscale_factor), interpolation=Image.BICUBIC)
		hr_scale = Resize((h, w), interpolation=Image.BICUBIC)
		# downscaled low resolution image
		lr_image = lr_scale(hr_image)
		# x, y = lr_image.size
		# super resolution image with naive method. Same shape as the original high resolution image
		hr_restore_img = hr_scale(lr_image)
		ConvertToTensor = ToTensor()

		return ConvertToTensor(lr_image), ConvertToTensor(hr_restore_img), ConvertToTensor(hr_image)

	def __len__(self):
		return len(self.image_filenames)
