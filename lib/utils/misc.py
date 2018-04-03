import os
import socket
from datetime import datetime


def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()


def get_output_dir(args, run_name):
    """ Get root output directory for each run """
    cfg_filename, _ = os.path.splitext(os.path.split(args.cfg_file)[1])
    return os.path.join(args.output_base_dir, cfg_filename, run_name)


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
