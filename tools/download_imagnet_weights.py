"""Script to downlaod ImageNet pretrained weights from Google Drive

Extra packages required to run the script:
    colorama, argparse_color_formatter
"""

import argparse
import os
import requests
from argparse_color_formatter import ColorHelpFormatter
from colorama import init, Fore

import _init_paths  # pylint: disable=unused-import
from core.config import cfg


def parse_args():
    """Parser command line argumnets"""
    parser = argparse.ArgumentParser(formatter_class=ColorHelpFormatter)
    parser.add_argument('--output_dir', help='Directory to save downloaded weight files',
                        default=os.path.join(cfg.DATA_DIR, 'pretrained_model'))
    parser.add_argument('-t', '--targets', nargs='+', metavar='file_name',
                        help='Files to download. Allowed values are: ' +
                        ', '.join(map(lambda s: Fore.YELLOW + s + Fore.RESET,
                                      list(PRETRAINED_WEIGHTS.keys()))),
                        choices=list(PRETRAINED_WEIGHTS.keys()),
                        default=list(PRETRAINED_WEIGHTS.keys()))
    return parser.parse_args()


# ---------------------------------------------------------------------------- #
# Mapping from filename to google drive file_id
# ---------------------------------------------------------------------------- #
PRETRAINED_WEIGHTS = {
    'resnet50_caffe.pth': '1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1',
    'resnet101_caffe.pth': '1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l',
    'resnet152_caffe.pth': '1NSCycOb7pU0KzluH326zmyMFUU55JslF',
    'vgg16_caffe.pth': '19UphT53C0Ua9JAtICnw84PPTa3sZZ_9k',
}


# ---------------------------------------------------------------------------- #
# Helper fucntions for download file from google drive
# ---------------------------------------------------------------------------- #

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
    init()  # colorama init. Only has effect on Windows
    args = parse_args()
    for filename in args.targets:
        file_id = PRETRAINED_WEIGHTS[filename]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        destination = os.path.join(args.output_dir, filename)
        download_file_from_google_drive(file_id, destination)
        print('Download {} to {}'.format(filename, destination))


if __name__ == "__main__":
    main()
