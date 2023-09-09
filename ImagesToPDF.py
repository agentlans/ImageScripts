# ImagesToPDF.py
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

import cairo
import PIL.Image as Image
import glob
import itertools

import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Create a PDF file with the given image files')
parser.add_argument('input_dir', type=str, help='input directory containing the images')
parser.add_argument('-o', metavar='output_file', type=str,
                    help='output PDF file')
parser.add_argument('--rows', type=int, default=1,
                    help='number of rows of images per page')
parser.add_argument('--cols', type=int, default=1,
                    help='number of columns of images per page')
parser.add_argument('--margin', type=int, default=0,
                    help='size of margins around each image in mm')
parser.add_argument('--res', type=int, default=300,
                    help='resolution of the images in pixels per inch')

args = parser.parse_args()

# Load the list of images (assume every file is an image)
input_dir = args.input_dir
imgs = []
for f in listdir(input_dir):
  full_path = join(input_dir, f)
  if isfile(full_path):
    imgs.append(full_path)
if imgs == []:
  raise RuntimeError('No files in the input directory')

# Read the other parameters
rows = args.rows
cols = args.cols
margin_mm = args.margin
out_file = args.o
resolution_px_per_in = args.res

assert(rows > 0 and cols > 0 and margin_mm >= 0)

# Constants and functions for layout

PTS_PER_MM = 2.8346456693
MM_PER_IN = 25.4 # mm/inch

def to_points(mm):
  return int(mm * PTS_PER_MM)

def pt_to_mm(pt):
  return pt / PTS_PER_MM

class Converter:
  def __init__(self, resolution_px_per_in=300):
    self.res = resolution_px_per_in
  def px_to_mm(self, px):
    return MM_PER_IN / self.res * px
  def mm_to_px(self, mm):
    return self.res / MM_PER_IN * mm
  def px_to_pt(self, px):
    return to_points(self.px_to_mm(px))
  def pt_to_px(self, pt):
    return self.mm_to_px(pt_to_mm(pt))

class LinearLayout:
  def __init__(self, total_space, num_items, item_margin):
    "Given total space, distribute num_items so that each item is spaced by item_margin."
    self.total_space = total_space
    self.num_items = num_items
    self.item_margin = item_margin
  def item_size(self):
    "Returns the maximum size for each item"
    item_space = self.total_space - (self.num_items + 1) * self.item_margin
    return item_space / self.num_items
  def location(self, i):
    "Returns the smallest coordinate of item i."
    return i * self.item_size() + (i+1) * self.item_margin

def closer(a, b, x):
  "True if a is closer to x than b is to x."
  return abs(a-x) < abs(b-x)

class ItemBox:
  def __init__(self, width, height, box_width, box_height, can_rotate=True):
    """Need to resize image (width, height) to fit inside a box (box_width, box_height)
    and the image must be centred with respect to the box.
    """
    # Determine whether it's better to rotate the image or not
    box_aspect = box_width / box_height
    img_aspect = width / height
    if closer(1/img_aspect, img_aspect, box_aspect) and can_rotate:
      self.rotate = True
      width, height = height, width
    else:
      self.rotate = False
    # Find the maximum scaling
    w_scale = box_width / width
    h_scale = box_height / height
    scale = min(h_scale, w_scale)
    # Find the dimensions of the resized image
    self.image_width = scale * width
    self.image_height = scale * height

class Placement:
  pass

class Page:
  def __init__(self, width_mm, height_mm, rows=1, cols=1, margin_mm=5, resolution_px_per_in=300):
    """Initializes page of size width_mm x height_mm. The page has rows and columns of images.
    Each image is separated by margin_mm. Horizontal and vertical resolution set as provided."""
    self.w = width_mm
    self.h = height_mm
    self.rows, self.cols = rows, cols
    self.row_layout = LinearLayout(to_points(height_mm), rows, to_points(margin_mm))
    self.col_layout = LinearLayout(to_points(width_mm),  cols, to_points(margin_mm))
    self.res = Converter(resolution_px_per_in)
    # Maximum sizes of the boxes
    self.box_height = self.row_layout.item_size() # You can access these members
    self.box_width  = self.col_layout.item_size()
  # Width and height of the page in points
  def width(self):
    return to_points(self.w)
  def height(self):
    return to_points(self.h)
  # (top, left) of the box at (i, j) in points
  def box_coords(self, i, j):
    top  = self.row_layout.location(i)
    left = self.col_layout.location(j)
    return top, left
  def img_coords(self, i, j, img_width_px, img_height_px, can_rotate=True):
    "Calculates how an image (img_width_px, img_height_px) can be resized and rotated to fit at location (i,j) on the page."
    box_top, box_left = self.box_coords(i,j) # in points
    # Convert pixels to points first
    conv = self.res
    # Calculate the placement in points
    box = ItemBox(conv.px_to_pt(img_width_px), conv.px_to_pt(img_height_px), self.box_width, self.box_height, can_rotate)
    p = Placement()
    p.image_width = int(conv.pt_to_px(box.image_width)) # in pixels
    p.image_height = int(conv.pt_to_px(box.image_height))
    p.left = box_left # in points
    p.top = box_top
    p.rotate = box.rotate
    return p

def prepare_image(im, new_width, new_height, rotate):
  if rotate:
    im2 = im.resize((new_height, new_width))
    im2 = im2.rotate(90, expand=True)
  else:
    im2 = im.resize((new_width, new_height))
  return im2

# From https://pycairo.readthedocs.io/en/latest/tutorial/pillow.html
# From a PIL Image object, returns a PyCairo ImageSurface
def from_pil(im: Image, alpha: float=1.0, format: cairo.Format=cairo.FORMAT_ARGB32) -> cairo.ImageSurface:
    """
    :param im: Pillow Image
    :param alpha: 0..1 alpha to add to non-alpha images
    :param format: Pixel format for output surface
    """
    assert format in (
        cairo.FORMAT_RGB24,
        cairo.FORMAT_ARGB32,
    ), f"Unsupported pixel format: {format}"
    if 'A' not in im.getbands():
        im.putalpha(int(alpha * 256.))
    arr = bytearray(im.tobytes('raw', 'BGRa'))
    surface = cairo.ImageSurface.create_for_data(arr, format, im.width, im.height)
    return surface

"""
imgs = list(reversed(glob.glob("/home/user/Documents/*.png")))
rows = 3
cols = 2
margin_mm = 5
"""

p = Page(210, 297, rows, cols, margin_mm, resolution_px_per_in)
coords = itertools.cycle(itertools.product(range(rows), range(cols)))

first_page = True
with cairo.PDFSurface(out_file, p.width(), p.height()) as surface:
    ctx = cairo.Context(surface)
    for (i,j) in coords:
      if imgs == []:
        break # Finished
      if (i,j) == (0,0) and not first_page:
        # New page
        ctx.show_page()
      if first_page:
          first_page = False
      # Open picture
      filename = imgs.pop()
      print(filename)
      im = Image.open(filename)
      # Compute where to put this picture
      pl = p.img_coords(i, j, im.size[0], im.size[1], True)
      # Get the picture onto a surface
      ctx.save()
      img_surface = from_pil(prepare_image(im, pl.image_width, pl.image_height, pl.rotate))
      m = max(pl.image_width, pl.image_height)
      s = p.res.px_to_pt(m) / m
      ctx.scale(s, s)
      # Centre the picture within each box
      x_box = p.box_width
      y_box = p.box_height
      delta_x = (x_box - p.res.px_to_pt(pl.image_width))/2
      delta_y = (y_box - p.res.px_to_pt(pl.image_height))/2
      # Paint
      ctx.set_source_surface(img_surface, (pl.left + delta_x)/s, (pl.top + delta_y)/s)
      ctx.paint()
      ctx.restore()

