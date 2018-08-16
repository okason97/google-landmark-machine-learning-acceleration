
# coding: utf-8

# In[1]:


import os
from multiprocessing import Pool, Value
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fractions import math

from skimage import color, exposure, io
from skimage.transform import resize

from sklearn import preprocessing
from sklearn.externals import joblib


# In[2]:


in_dir = "../input/images/"
out_dir = "../input/preprocessed-images/"
normalized_dir = "../input/normalized-images/"
shape = 256
scaler_filename = "../models/images_StandardScaler.save"


# In[3]:


def process_image(image_dir):
    image = io.imread(in_dir+image_dir)
    
    r = math.gcd(image.shape[0], image.shape[1])
    widht_ratio = int(image.shape[1] / r)
    height_ratio = int(image.shape[0] / r)

    # crop
    if widht_ratio > height_ratio*1.5:
        image = image[:,int((image.shape[1]-shape)/2):int((image.shape[1]-shape)/2+shape)]       
    elif height_ratio > widht_ratio*1.5:
        image = image[int((image.shape[0]-shape)/2):int((image.shape[0]-shape)/2+shape),:]

    image = resize(image, (shape, shape), mode='reflect', anti_aliasing=True)
    image = color.rgb2gray(image)
    image = exposure.equalize_hist(image)
    print("preprocessed: "+image_dir)
    print("saved in: "+out_dir+image_dir)
    io.imsave(out_dir+image_dir,image)    
    return image

def normalize_image(image, image_dir):
    image = np.array(io.imread(out_dir+image_dir))
        
    # Load scaler
    scaler = joblib.load(scaler_filename)

    # standarization or normalization
    image = image.reshape(1,-1)

    image = scaler.transform(image)
    image = image.reshape(shape,shape)

    print("normalized image saved in: "+normalized_dir+image_dir)
    io.imsave(normalized_dir+image_dir,image)    


# In[4]:


if not os.path.exists(out_dir):
    os.mkdir(out_dir)

processes = 6

split_n = 8000

# scaler = preprocessing.MinMaxScaler(feature_range=(0,255))
scaler = preprocessing.StandardScaler()

dir_list = np.array(os.listdir(in_dir))

pool = Pool(processes=processes)  # Num of CPUs

for sub_dir_list in np.array_split(dir_list, split_n):
    # crop, resize, rgb to grey and hist equalization.
    images = np.array(pool.map(process_image, sub_dir_list, chunksize = 8))

    # standarization or normalization
    images = np.reshape(images,(len(images),-1))
    scaler.partial_fit(images)

joblib.dump(scaler, scaler_filename)

pool.imap_unordered(normalize_image, images_dirs, chunksize = 8)

pool.close()
pool.terminate()

