import os
import numpy as np
from PIL import Image
from tqdm import tqdm

path = 'test_out_images/stillLife/up_4'

all_file_names = os.listdir(path)

for file_name in all_file_names:
	if not file_name.endswith('jpg') or file_name.endswith('lr.jpg'):
		continue

	img = Image.open(os.path.join(path, file_name))

	# for papillon
	#img2 = img.crop((316, 301, 695, 682))
	# for statue
	#img2 = img.crop((336, 102, 501, 231))

	img2 = img.crop((114, 128, 296, 319))

	img2.save(os.path.join(path, 'cr_'+file_name))