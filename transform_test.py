import torch
from PIL import Image
from models import data_utils
import numpy as np


crop_size = 71

hr_transform = data_utils.train_hr_transform(crop_size)

im = Image.open('./read/sample.jpg')

x = hr_transform(im)

print()