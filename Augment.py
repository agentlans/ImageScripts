import argparse
import numpy as np

from PIL import Image
import albumentations as A
import cv2

from pathlib import Path
import os

# My packages
from Helpers import *

parser = argparse.ArgumentParser(description='Generates variations of the given images')

parser.add_argument('-i', metavar="INPUT_DIR", required=True, help='input directory containing the images')
parser.add_argument('-o', metavar="OUTPUT_DIR", required=True, 
  help='output directory that will contain the augmented images')
parser.add_argument('--flip', default=0.5, type=float, help='probability of flipping an image (default 0.5)')
parser.add_argument('--bc', default=0.5, type=float, help='probability of random brightness and contrast (default 0.5)')
parser.add_argument('--rot', default=0.5, type=float, help='probability of rotating an image (default 0.5)')
parser.add_argument('--n', default=1, type=int, help='number of replicates for each file')

# Read the command line arguments
args = parser.parse_args()
input_dir = args.i
output_dir = args.o
flip_p = args.flip
bc_p = args.bc
rot_p = args.rot
n = args.n

# Initialize output directory
make_dir(Path(output_dir))

# Set up the transforms
transform = A.Compose([
    A.ShiftScaleRotate(p=rot_p, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=flip_p),
    A.RandomBrightnessContrast(p=bc_p)
])

# Go through the files one by one
files = list_files(input_dir)
for f in tqdm(files):
  fname = Path(f).stem
  ext = Path(f).suffix
  with Image.open(f) as img:
    for i in range(n):
      img2 = transform(image=np.asarray(img))['image']
      # Save to output directory
      filename2 = os.path.join(output_dir, fname + '_' + str(i) + ext)
      Image.fromarray(img2).save(filename2)

