import h5py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from_dir = '/Users/mengzhao/light_field/arxiv'
to_dir = '/Users/mengzhao/light_field/images'

folders = ['blender', 'gantry']

for folder in folders:
	sub_folders = os.listdir(os.path.join(from_dir, folder))
	for sub_folder in tqdm(sub_folders):
		if sub_folder.startswith('.'):
			continue
		print('bsub -n 3 -R  "select[gpu_model1==GeForceGTX1080Ti] rusage[mem=4096,ngpus_excl_p=1]" '
		      'python convert_images.py --item {} --upscale_factor 4 &&'.format(sub_folder))
		continue
		file_name = os.path.join(from_dir, folder, sub_folder, 'lf.h5')

		with h5py.File(file_name, 'r') as file:
			images = np.asarray(file['LF'])

			(v, h, x, y, c) = images.shape

			for i in range(v):
				for j in range(h):
					single_image = images[i,j]
					result = Image.fromarray(single_image)
					path = os.path.join(to_dir, sub_folder, sub_folder+'_{}_{}.jpg'.format(i, j))
					if not os.path.exists(os.path.join(to_dir, sub_folder)):
						os.makedirs(os.path.join(to_dir, sub_folder))
					result.save(path)
					break
				break
