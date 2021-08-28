#!/usr/bin/env python
# coding: utf-8


import dlib
import glob
import os
from tqdm import tqdm
from utils.alignment import align_face


# In[10]:


images_path = 'raw'
SHAPE_PREDICTOR_PATH = 'pretrained_models/shape_predictor_68_face_landmarks.dat'
IMAGE_SIZE = 1024


# In[11]:


predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


# In[12]:


os.chdir(images_path)


# In[13]:


images_names = glob.glob(f'*')


# In[14]:


print(images_names)


# In[15]:


aligned_images = []
for image_name in tqdm(images_names):
    try:
        print("aligned")
        aligned_image = align_face(filepath=f'../{images_path}/{image_name}',
                                       predictor=predictor, output_size=IMAGE_SIZE)
        aligned_images.append(aligned_image)
    except Exception as e:
        print(e)


os.makedirs(f'{images_path}/aligned', exist_ok=True)
os.makedirs(f'{images_path}/aligned/0', exist_ok=True)


print(aligned_images)
print(images_names)
for image, name in zip(aligned_images,images_names):
    real_name = name.split('.')[0]
    try:
        name = f'{images_path}/aligned/0/{real_name}.jpeg'
        print("save:",name)
        image.save(name)
    except Exception as e:
        print(e)


