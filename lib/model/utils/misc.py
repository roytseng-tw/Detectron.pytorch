import os
import socket
from datetime import datetime


def get_run_name():
  return datetime.now().strftime('%b%d-%H-%M-%S') + '_' + socket.gethostname()


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
  """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
  """
  filename_lower = filename.lower()
  return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def get_imagelist_from_dir(dirpath):
  images = []
  for f in os.listdir(dirpath):
    if is_image_file(f):
      images.append(f)
  return images
