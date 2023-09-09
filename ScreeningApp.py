# ScreeningApp.py
# Copyright 2023 Alan Tseng
# 
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with 
# this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import gradio as gr
from pathlib import Path
from random import shuffle
import shutil
from os import listdir
from os.path import isfile, join
import sys

parser = argparse.ArgumentParser(description='Webapp for quickly screening a directory of images')
parser.add_argument('input_dir', type=str, help='input directory containing the images')

args = parser.parse_args()

# Load the list of images (assume every file is an image)
input_dir = args.input_dir
images = []
for f in listdir(input_dir):
  full_path = join(input_dir, f)
  if isfile(full_path):
    images.append(full_path)
if images == []:
  raise RuntimeError('No files in the input directory')

shuffle(images)

accept_dir = (Path(input_dir) / "Accept")
reject_dir = (Path(input_dir) / "Reject")

accept_dir.mkdir(exist_ok=True)
reject_dir.mkdir(exist_ok=True)

image = None

def next_picture():
  global image
  if len(images) > 0:
    image = images[-1]
    images.pop()
    # Update image
    return image

accepted = []
rejected = []

def accept():
  if image:
    accepted.append(image)
    shutil.move(image, accept_dir)
  return next_picture()

def reject():
  if image:
    rejected.append(image)
    shutil.move(image, reject_dir)
  return next_picture()

with gr.Blocks() as demo:
  img = gr.Image(label="Image")
  with gr.Row():
    cmd_no = gr.Button('No')
    cmd_yes = gr.Button('Yes')
  cmd_yes.click(fn=accept, outputs=img)
  cmd_no.click(fn=reject, outputs=img)

demo.launch()
