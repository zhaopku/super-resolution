import os
from PIL import Image

folder = './data/light_field'

sub_folders = ['train', 'val', 'test']

cnt = 0
for sub_folder in sub_folders:
	#cur_folder = os.path.join(folder, sub_folder)
	all_files = os.listdir(os.path.join(folder, sub_folder))

	for file_name in all_files:
		img = Image.open(os.path.join(folder, sub_folder, file_name))

		if min(img.size) < 256:
			os.remove(os.path.join(folder, sub_folder, file_name))
			cnt += 1

print(cnt)