import argparse
import numpy as np
import pandas as pd
from PIL import Image

# My packages
from Helpers import *
from CLIP import CLIP

parser = argparse.ArgumentParser(
                    #prog='Zero-shot image classifier',
                    description='Puts images into directories based on text descriptions')

parser.add_argument('-i', metavar="INPUT_DIR", required=True, help='input directory containing the images')
parser.add_argument('-o', metavar="OUTPUT_DIR", required=True, 
  help='output directory that will contain the classified images')
parser.add_argument('-c', '--categories', required=True, nargs="*", help='categories that each image can be classified')
parser.add_argument("--chunk_size", default=3, type=int, help="number of images to process at a time (default 3)")
parser.add_argument('-m', '--model', default='openai/clip-vit-large-patch14', 
  help='the CLIP model used for classification (default "openai/clip-vit-large-patch14")')

# Read the command line arguments
args = parser.parse_args()
input_dir = args.i
output_dir = args.o
chunk_size = args.chunk_size
labels = args.categories
if not labels:
  raise Exception('There must be at least one category for classification.')
model = args.model

# Load a CLIP model with those categories
clip = CLIP(labels, model_name=model)
# Initialize output directory
od = OutputDirectory(output_dir, labels)

def process(filenames):
  "Processes the filenames."
  clip.process(list(map(Image.open, filenames)))
  classes = clip.classes()
  for filename, c in zip(filenames, classes):
    od.move_to(filename, c)

# Go through the directory in chunks
files = list_files(input_dir)
_ = map_tqdm(process, chunked(files, chunk_size))

