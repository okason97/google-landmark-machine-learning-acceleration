# -*- coding: utf-8 -*-

# !/usr/bin/python

# Note: requires the tqdm package (pip install tqdm)

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv, tqdm
from urllib import request, error
from PIL import Image
from io import BytesIO
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def parse_data(data_file):
    csvreader = pd.read_csv(data_file)
    csvreader = csvreader[csvreader['landmark_id'] != 'None']
    csvreader = csvreader[csvreader['landmark_id'].isin(csvreader['landmark_id'].value_counts().head(10).index.values)]
    key_url_list = csvreader[['id','url']]
    return key_url_list.values


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=8)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    loader()
