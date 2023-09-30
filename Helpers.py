from pathlib import Path
import shutil
from more_itertools import chunked
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from slugify import slugify

def list_files(mypath):
  "Lists files inside the directory. Not recursive"
  return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

def map_tqdm(f, lst):
  "Applies f to every element in lst. Comes with a progress meter."
  out = []
  for x in tqdm(list(lst)):
    #try:
      out.append(f(x))
    #except:
    #  pass
  return out

def map_chunkwise(f, lst, chunk_size):
  """Breaks lst into chunks and applies f to every element in each chunk.
  Returns a list containing the results from processing each chunk.
  """
  return map_tqdm(f, chunked(lst, chunk_size))

def make_dir(dir_path):
  "Creates a directory and its parents if necessary."
  dir_path.mkdir(parents=True, exist_ok=True)

def dir_slugify(x):
  "Given arbitrary string, returns a string that can be a valid directory name."
  return slugify(x, regex_pattern=r'\W')

class OutputDirectory:
  def __init__(self, dir_path, category_names):
    """Creates an output directory and its subdirectories where
    each subdirectory is a different category."""
    self.path = Path(dir_path)
    self.dict = dict()
    # Create subdirectories
    for cat in category_names:
      subdir = dir_slugify(cat)
      make_dir(self.path / subdir)
      # Keep track of which subdirectory is each category
      self.dict[cat] = subdir
  def get_subdir(self, category):
    "Returns the subdirectory corresponding to a given category."
    return self.path / self.dict[category]
  def move_to(self, src, category):
    "Moves the file at src to the right subdirectory."
    shutil.move(src, self.get_subdir(category))
  def copy_to(self, src, category):
    "Copies the file at src to the right subdirectory."
    shutil.copy(src, self.get_subdir(category))

def make_output_dirs(main_dir, subdir_names):
  "Creates main_dir/subdir for the given subdirectory names."
  for subdir in subdir_names:
    (Path(main_dir) / subdir).mkdir(parents=True, exist_ok=True)

def copy_file_to(input_file, main_output_dir, subdir):
  "Copies a file to a subdirectory of the output directory."
  shutil.copy(input_file, Path(main_output_dir) / subdir)

def move_file_to(input_file, main_output_dir, subdir):
  "Moves a file to a subdirectory of the output directory."
  shutil.move(input_file, Path(main_output_dir) / subdir)

def delete_file(filename):
  Path(filename).unlink()

