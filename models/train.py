import torch
from tqdm import tqdm
import argparse
import os
import torch.optim as optimizer
from models.data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder, display_transform, create_new_lr_image
from torch.utils.data import DataLoader
from models.model_srcnn import ModelSRCNN
from models import utils
from models.model_gen import SRGANGenerator
import pytorch_ssim
import math
import torchvision.utils
from tensorboardX import SummaryWriter
from models.model_discriminator import Discriminator
from models.gan_loss import GeneratorLoss

class Train:
	def __init__(self):
		self.args = None
		self.training_set = None
		self.val_set = None
		self.train_loader = None
		self.val_loader = None
		self.model = None

		self.result_dir = None
		self.out_image_dir = None

		self.naive_results = None
		self.naive_results_computed = False
		self.writer = None
		self.generator_loss = None

	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		data_args = parser.add_argument_group('Dataset options')
		data_args.add_argument('--data_dir', type=str, default='/Users/mengzhao/light_field')
		data_args.add_argument('--category', type=str, default='light_field')
		data_args.add_argument('--train_dir', type=str, default='train')
		data_args.add_argument('--val_dir', type=str, default='val')
		data_args.add_argument('--test_dir', type=str, default='test')
		data_args.add_argument('--summary_dir', type=str, default='summaries')

		data_args.add_argument('--result_dir', default='./result')
		data_args.add_argument('--ratio', type=float, default=1.0, help='ratio of training data used')

		# neural network options
		nn_args = parser.add_argument_group('Network options')
		nn_args.add_argument('--crop_size', default=128, type=int, help='training images crop size')
		nn_args.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
		                    help='super resolution upscale factor')
		nn_args.add_argument('--model', type=str, default='SRGAN', help='string to specify model')
		nn_args.add_argument('--in_channels', type=int, default=3)

		# training options
		training_args = parser.add_argument_group('Training options')
		training_args.add_argument('--batch_size', type=int, default=3)
		training_args.add_argument('--n_save', type=int, default=50, help='number of test images to save on disk')
		training_args.add_argument('--epochs', type=int, default=100, help='number of training epochs')
		training_args.add_argument('--lr', type=float, default=0.001, help='learning rate')

		training_args.add_argument('--gamma', type=float, default=0.001, help='coefficient for adversarial loss')
		training_args.add_argument('--sigma', type=float, default=1.0, help='coefficient for vgg loss')

		training_args.add_argument('--theta', type=float, default=0.01, help='coefficient for discriminator learning rate')

		return parser.parse_args(args)

	def construct_data(self):
		self.training_set = TrainDatasetFromFolder(os.path.join(self.args.data_dir, self.args.category, self.args.train_dir),
		                                           crop_size=self.args.crop_size,
		                                   upscale_factor=self.args.upscale_factor, ratio=self.args.ratio)

		self.val_set = ValDatasetFromFolder(os.path.join(self.args.data_dir, self.args.category, self.args.val_dir),
		                                   upscale_factor=self.args.upscale_factor)

		self.train_loader = DataLoader(dataset=self.training_set, num_workers=1, batch_size=self.args.batch_size, shuffle=True)

		# images are of different size, hence test batch size must be 1
		self.val_loader = DataLoader(dataset=self.val_set, num_workers=1, batch_size=1, shuffle=False)

	def construct_model(self):
		if self.args.model == 'SRCNN':
			self.model = ModelSRCNN(args=self.args)
		elif self.args.model == 'SRGAN_GEN':
			self.model = SRGANGenerator(args=self.args)
			self.optimizer = optimizer.Adam(self.model.parameters(), lr=self.args.lr)
			print('{}, #param = {}'.format(self.args.model, sum(param.numel() for param in self.model.parameters())))
		elif self.args.model == 'SRGAN':
			self.generator = SRGANGenerator(args=self.args)
			self.discriminator = Discriminator(args=self.args)

			self.optimizerG = optimizer.Adam(self.generator.parameters(), lr=self.args.lr)
			self.optimizerD = optimizer.Adam(self.discriminator.parameters(), lr=self.args.lr*self.args.theta)
			print('{}, #generator param = {}, discriminator param {}'.
			      format(self.args.model, sum(param.numel() for param in self.generator.parameters()),
			             sum(param.numel() for param in self.discriminator.parameters())))
		else:
			print('Invalid model string: {}'.format(self.args.model))

		# average over all the pixels in the batch
		self.mse_loss = torch.nn.MSELoss(reduction='mean')
		# average over all samples
		self.bce_loss = torch.nn.BCELoss(reduction='mean')

		self.generator_loss = GeneratorLoss(args=self.args)



	def construct_out_dir(self):
		self.result_dir = utils.construct_dir(prefix=self.args.result_dir, args=self.args)
		self.out_image_dir = os.path.join(self.result_dir, 'images')
		self.model_dir = os.path.join(self.result_dir, 'models')
		self.out_path = os.path.join(self.result_dir, 'result.txt')

		self.summary_dir = utils.construct_dir(prefix=self.args.summary_dir, args=self.args)

		if not os.path.exists(self.summary_dir):
			os.makedirs(self.summary_dir)
		self.writer = SummaryWriter(log_dir=self.summary_dir)

		if not os.path.exists(self.out_image_dir):
			os.makedirs(self.out_image_dir)

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

	def main(self, args=None):
		print('PyTorch Version {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))
		self.args = self.parse_args(args=args)

		self.construct_data()

		self.construct_model()

		self.construct_out_dir()

		# self.model_path = './saved_models/up_2.pth'
		# if torch.cuda.is_available():
		# 	(generate_state_dict, discriminator_state_dict) = torch.load(self.model_path, map_location='gpu')
		# else:
		# 	(generate_state_dict, discriminator_state_dict) = torch.load(self.model_path, map_location='cpu')
		#
		# self.generator.load_state_dict(generate_state_dict)
		# self.discriminator.load_state_dict(discriminator_state_dict)

		with open(self.out_path, 'w') as self.out:
			if self.args.model == 'SRGAN':
				self.gan_train_loop()
			else:
				self.train_loop()

	def gan_train_loop(self):
		# put to GPU
		if torch.cuda.is_available():
			self.discriminator.cuda()
			self.generator.cuda()
			self.mse_loss = self.mse_loss.cuda()
			self.generator_loss.cuda()

		for e in range(self.args.epochs):
			# switch to training mode
			self.discriminator.train()
			self.generator.train()

			# errG: generator loss, loss for generator not failing to fool discriminator
			# D_G_z1: probability of discriminator being fooled before update
			# D_G_z2: probability of discriminator being fooled after update
			# D_x: probability of a real image recognized by discriminator
			# errD_real: loss for discriminator not recognizing real images
			# errD_fake: loss for discriminator not recognizing fake images

			train_results = {'mse_loss': 0.0, 'adversarial_loss': 0.0, 'D_G_z1': 0.0,
							 'D_G_z2': 0.0, 'D_x': 0.0, 'errD_real': 0.0,
							 'errD_fake': 0.0, 'perception_loss': 0.0,
							 'n_samples':0}

			for idx, (lr_image, hr_image) in enumerate(tqdm(self.train_loader)):

				cur_batch_size = lr_image.size(0)
				train_results['n_samples'] += cur_batch_size
				# put data to GPU
				if torch.cuda.is_available():
					lr_image = lr_image.cuda()
					hr_image = hr_image.cuda()

				# update

				#################
				# discriminator #
				#################
				# maximize log(D(x)) + log(1 - D(G(z)))
				# D(x): probability of real being original
				# 1 - D(G(z)): probability of fake not being original

				## with all real batch
				self.discriminator.zero_grad()
				real_label = torch.full((cur_batch_size,), 1)
				if torch.cuda.is_available():
					real_label = real_label.cuda()
				hr_probs, log_hr_probs = self.discriminator(hr_image)

				errD_real = self.bce_loss(hr_probs.view(-1), real_label)
				train_results['errD_real'] += errD_real.data.cpu() * cur_batch_size
				# errD_real.backward()
				D_x = hr_probs.data.cpu()
				train_results['D_x'] += D_x.data.cpu().sum()

				## with all fake batch
				sr_image = self.generator(lr_image)
				if torch.cuda.is_available():
					sr_image = sr_image.cuda()

				fake_label = torch.full((cur_batch_size,), 0)
				if torch.cuda.is_available():
					fake_label = fake_label.cuda()
				sr_probs, log_sr_probs = self.discriminator(sr_image)

				errD_fake = self.bce_loss(sr_probs.view(-1), fake_label)
				train_results['errD_fake'] += errD_fake.data.cpu() * cur_batch_size
				# errD_fake.backward()

				D_G_z1 = sr_probs.data.cpu()
				train_results['D_G_z1'] += D_G_z1.data.cpu().sum()

				D_loss = errD_real + errD_fake
				D_loss.backward(retain_graph=True)
				self.optimizerD.step()

				#################
				#   generator   #
				#################

				# maximize log(D(G(z)))

				self.generator.zero_grad()

				sr_probs, log_sr_probs = self.discriminator(sr_image)
				D_G_z2 = sr_probs.data.cpu()

				G_loss, adversarial_loss, mse_loss, perception_loss \
					= self.generator_loss(sr_probs=sr_probs, sr_images=sr_image, hr_images=hr_image)

				train_results['adversarial_loss'] += adversarial_loss.data.cpu() * cur_batch_size
				train_results['D_G_z2'] += D_G_z2.data.cpu().sum()
				# plus we want to minimize the mse loss
				train_results['mse_loss'] += mse_loss.data.cpu() * cur_batch_size
				train_results['perception_loss'] += perception_loss.data.cpu() * cur_batch_size

				G_loss.backward()
				self.optimizerG.step()

				# if idx == 2:
				# 	break

			# write to out file
			result_line = 'Epoch %d, ' % e
			for k, v in train_results.items():
				if k == 'n_samples':
					continue
				# if k.startswith('D'):
				# 	self.writer.add_scalar('train/{}'.format(k), v, e)
				# 	result_line += '{} = {}, '.format(k, v)
				else:
					result_line += '{} = {}, '.format(k, v/train_results['n_samples'])
					self.writer.add_scalar('train/{}'.format(k), v/train_results['n_samples'], e)

			print(result_line)
			self.out.write(result_line+'\n')

			self.out.flush()
			self.gan_validate(epoch=e)

	def gan_validate(self, epoch):
		with torch.no_grad():
			self.generator.eval()
			self.discriminator.eval()
			val_results = {'mse_loss': 0, 'D_G_z':0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'n_samples': 0}

			if not self.naive_results_computed:
				self.naive_results = {'mse_loss': 0, 'D_G_z':0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'n_samples': 0}

			# TODO: to finish
			val_images = []
			for idx, (lr_image, naive_hr_image, hr_image) in enumerate(tqdm(self.val_loader)):
				# put data to GPU

				cur_batch_size = lr_image.size(0)
				val_results['n_samples'] += cur_batch_size

				if torch.cuda.is_available():
					lr_image = lr_image.cuda()
					naive_hr_image = naive_hr_image.cuda()
					hr_image = hr_image.cuda()

				sr_image = self.generator(lr_image)
				sr_probs, log_sr_probs = self.discriminator(sr_image)
				val_results['D_G_z'] += sr_probs.data.cpu().sum()

				mse_loss = self.mse_loss(input=sr_image, target=hr_image)
				val_results['mse_loss'] += mse_loss.data.cpu() * cur_batch_size

				batch_ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
				val_results['ssims'] += batch_ssim * cur_batch_size
				val_results['psnr'] = 10 * math.log10(1 / (val_results['mse_loss'] / val_results['n_samples']))
				val_results['ssim'] = val_results['ssims'] / val_results['n_samples']

				# to save memory
				naive_sr_probs, naive_log_sr_probs = self.discriminator(naive_hr_image)

				self.naive_results['D_G_z'] += naive_sr_probs.data.cpu().sum()

				if not self.naive_results_computed:
					naive_mse_loss = self.mse_loss(input=naive_hr_image, target=hr_image).data.cpu()
					self.naive_results['mse_loss'] += naive_mse_loss * cur_batch_size
					naive_batch_ssim = pytorch_ssim.ssim(naive_hr_image, hr_image).item()
					self.naive_results['ssims'] += naive_batch_ssim * cur_batch_size
					self.naive_results['psnr'] = 10 * math.log10(1 / (self.naive_results['mse_loss'] / val_results['n_samples']))
					self.naive_results['ssim'] = self.naive_results['ssims'] / val_results['n_samples']

				# only save certain number of images

				# transform does not support batch processing
				lr_image = create_new_lr_image(lr_image, hr_image)
				if idx < self.args.n_save:
					for image_idx in range(cur_batch_size):
						val_images.extend(
							[display_transform()(lr_image[image_idx].data.cpu()),
							 display_transform()(naive_hr_image[image_idx].data.cpu()),
							 display_transform()(hr_image[image_idx].data.cpu()),
							 display_transform()(sr_image[image_idx].data.cpu())])

				# if idx == 5:
				# 	break

			val_results['D_G_z'] = val_results['D_G_z'] / val_results['n_samples']
			val_results['mse_loss'] = val_results['mse_loss'] / val_results['n_samples']
			# write to out file
			result_line = '\tVal\t'
			for k, v in val_results.items():
				result_line += '{} = {}, '.format(k, v)
				self.writer.add_scalar('val/{}'.format(k), v, epoch)

			if not self.naive_results_computed:
				result_line += '\n'
				self.naive_results['D_G_z'] = self.naive_results['D_G_z'] / val_results['n_samples']
				for k, v in self.naive_results.items():
					result_line += 'naive_{} = {} '.format(k, v)
				self.naive_results_computed = True
			else:
				result_line += '\n\t'
				self.naive_results['D_G_z'] = self.naive_results['D_G_z'] / val_results['n_samples']
				result_line += 'naive D_G_z = {}'.format(self.naive_results['D_G_z']/val_results['n_samples'])

			print(result_line)
			self.out.write(result_line+'\n')
			self.out.flush()

			self.out.flush()
			# save model
			torch.save((self.generator.state_dict(), self.discriminator.state_dict()),
			           os.path.join(self.model_dir, str(epoch)+'.pth'))

			self.save_image(val_images, epoch)

	def train_loop(self):

		# put model to GPU if available
		if torch.cuda.is_available():
			self.model.cuda()
			self.mse_loss = self.mse_loss.cuda()

		for e in range(self.args.epochs):
			# switch the model to training mode
			self.model.train()
			train_loss = 0
			for idx, (lr_image, hr_image) in enumerate(tqdm(self.train_loader)):
				# put data to GPU
				if torch.cuda.is_available():
					lr_image = lr_image.cuda()
					hr_image = hr_image.cuda()

				# sr_image, should have the same shape as hr_image
				sr_image = self.model(lr_image)

				loss = self.mse_loss(input=sr_image, target=hr_image)

				train_loss += loss.data.cpu()

				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()

			print('Epoch = {}, Train loss = {}'.format(e, train_loss))
			self.out.write('Epoch = {}, Train loss = {}\n'.format(e, train_loss))
			self.out.flush()

			self.validate(epoch=e)

	def validate(self, epoch):
		with torch.no_grad():
			self.model.eval()
			val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'val_size': 0}

			if not self.naive_results_computed:
				self.naive_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'val_size': 0}

			val_images = []
			for idx, (lr_image, naive_hr_image, hr_image) in enumerate(tqdm(self.val_loader)):
				# put data to GPU

				cur_batch_size = lr_image.size(0)
				val_results['val_size'] += cur_batch_size

				if torch.cuda.is_available():
					lr_image = lr_image.cuda()
					naive_hr_image = naive_hr_image.cuda()
					hr_image = hr_image.cuda()

				sr_image = self.model(lr_image)

				batch_mse = ((sr_image - hr_image) ** 2).data.mean()
				val_results['mse'] += batch_mse * cur_batch_size
				batch_ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
				val_results['ssims'] += batch_ssim * cur_batch_size
				val_results['psnr'] = 10 * math.log10(1 / (val_results['mse'] / val_results['val_size']))
				val_results['ssim'] = val_results['ssims'] / val_results['val_size']

				if not self.naive_results_computed:
					naive_batch_mse = ((naive_hr_image - hr_image) ** 2).data.mean()
					self.naive_results['mse'] += naive_batch_mse * cur_batch_size
					naive_batch_ssim = pytorch_ssim.ssim(naive_hr_image, hr_image).item()
					self.naive_results['ssims'] += naive_batch_ssim * cur_batch_size
					self.naive_results['psnr'] = 10 * math.log10(1 / (self.naive_results['mse'] / val_results['val_size']))
					self.naive_results['ssim'] = self.naive_results['ssims'] / val_results['val_size']

				# only save certain number of images

				# transform does not support batch processing
				lr_image = create_new_lr_image(lr_image, hr_image)
				if idx < self.args.n_save:
					for image_idx in range(cur_batch_size):


						val_images.extend(
							[display_transform()(lr_image[image_idx].data.cpu()),
							 display_transform()(naive_hr_image[image_idx].data.cpu()),
							 display_transform()(hr_image[image_idx].data.cpu()),
							 display_transform()(sr_image[image_idx].data.cpu())])

			# write to out file
			result_line = '\tVal\t'
			for k, v in val_results.items():
				result_line += '{} = {} '.format(k, v)

			if not self.naive_results_computed:
				result_line += '\n'
				for k, v in self.naive_results.items():
					result_line += 'naive_{} = {} '.format(k, v)
				self.naive_results_computed = True

			print(result_line)
			self.out.write(result_line+'\n')
			self.out.flush()
			# save model
			torch.save(self.model.state_dict(), os.path.join(self.model_dir, str(epoch)+'.pth'))

			self.save_image(val_images, epoch)

	def save_image(self, images, epoch):
		# number of out images, 4*5 is number of sub-images in an output image
		n_out_images = int(len(images) / 4)

		cur_out_image_dir = os.path.join(self.out_image_dir, 'epoch_%d' % epoch)

		if not os.path.exists(cur_out_image_dir):
			os.makedirs(cur_out_image_dir)

		for idx in tqdm(range(n_out_images), desc='saving validating image'):
			image = torch.stack(images[idx*(4):(idx+1)*(4)])
			if image.size()[0] < 1:
				break
			image = torchvision.utils.make_grid(image, nrow=4, padding=10)
			save_path = os.path.join(cur_out_image_dir, 'index_%d.jpg' % idx)
			torchvision.utils.save_image(image, save_path, padding=5)


"""
result/SRGAN_light_field_lr_0.001_bt_50_cr_128_up_8_gamma_0.1_theta_0.1/models/15.pth
"""