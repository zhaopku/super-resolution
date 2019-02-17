import os
from shutil import copyfile
import random
from tqdm import tqdm
from PIL import Image


from_dir = '/Users/mengzhao/VOCdevkit/VOC2012/JPEGImages'
to_dir = '/Users/mengzhao/light_field/light_field'

all_files = os.listdir(from_dir)
random.shuffle(all_files)


def create(file_names, tag):
	min_size = 100000
	for file_name in tqdm(file_names, desc=tag):
		if not file_name.endswith('.jpg'):
			continue
		from_path = os.path.join(from_dir, file_name)

		img = Image.open(from_path)

		cur_min_size = min(img.size)

		if cur_min_size < 128:
			continue

		to_path = os.path.join(to_dir, tag, file_name)

		copyfile(from_path, to_path)
	return min_size

n_train = int(len(all_files)*0.9)

m = create(all_files[:n_train], tag='train')
print(m)
m2 = create(all_files[n_train:], tag='val')
print(m2)