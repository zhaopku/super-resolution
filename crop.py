import os
import numpy as np
from PIL import Image
from tqdm import tqdm

path = 'test_out_images/papillon/up_2'

all_file_names = os.listdir(path)

for file_name in all_file_names:
	if not file_name.endswith('jpg') or file_name.endswith('lr.jpg'):
		continue

	img = Image.open(os.path.join(path, file_name))

