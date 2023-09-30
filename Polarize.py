import argparse
import numpy as np
import pandas as pd
from PIL import Image

# My packages
from Helpers import *
from CLIP import CLIP

parser = argparse.ArgumentParser(
                    #prog='Zero-shot image classifier',
                    description='Copies the most extreme images based on some quality to an output directory')

parser.add_argument('-i', metavar="INPUT_DIR", required=True, help='input directory containing the images')
parser.add_argument('-o', metavar="OUTPUT_DIR", required=True, 
  help='output directory that will contain the polarized images')
parser.add_argument('--pos', required=True, help='description of images to be assigned positive scores')
parser.add_argument('--neg', required=True, help='description of images to be assigned negative scores')
parser.add_argument('-n', required=True, type=int, help='number of images with high and low scores to copy to output')
parser.add_argument("--chunk_size", default=3, type=int, help="number of images to process at a time (default 3)")
parser.add_argument('-m', '--model', default='openai/clip-vit-large-patch14', 
  help='the CLIP model used for classification (default "openai/clip-vit-large-patch14")')

# Read the command line arguments
args = parser.parse_args()
input_dir = args.i
output_dir = args.o
chunk_size = args.chunk_size
n_samples = args.n
labels = [args.pos, args.neg]
model = args.model

# Load a CLIP model with those categories
clip = CLIP(labels, model_name=model)
# Initialize output directories
od = OutputDirectory(output_dir, ['high_score', 'low_score'])

def process(filenames):
  "Processes the filenames and computes score."
  clip.process(list(map(Image.open, filenames)))
  p = clip.probs()
  score = np.log(p[:,0]) - np.log(p[:,1])
  return pd.DataFrame({'Filename': filenames, 'Score': score})

# Go through the directory in chunks
files = list_files(input_dir)
results = pd.concat(map_tqdm(process, chunked(files, chunk_size)))

# Sort by score
results = results.sort_values(by=['Score'])

# How many samples to select?
n = results.shape[0]
nsel = min(n_samples, n/2)
low_score = results.head(nsel)['Filename']
high_score = results.tail(nsel)['Filename']

# Copy the files to their output directories
for filename in low_score:
  od.copy_to(filename, 'low_score')

for filename in high_score:
  od.copy_to(filename, 'high_score')

