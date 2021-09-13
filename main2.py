import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
import matplotlib.pyplot as plt


save_path = 'pretrained_models'
image_dir_name = 'image'

image_name = '336944_15'
use_multi_id_training = False

global_config.device = 'cuda'
paths_config.input_data_id = image_dir_name
paths_config.input_data_path = f'{image_dir_name}_original'
paths_config.stylegan2_ada_ffhq = 'pretrained_models/webtoon001693.pkl'
paths_config.checkpoints_dir = 'checkpoints'
paths_config.style_clip_pretrained_mappers = 'pretrained_models'
hyperparameters.use_locality_regularization = False

os.makedirs(f'./{image_dir_name}_original', exist_ok=True)
os.makedirs(f'./{image_dir_name}_processed', exist_ok=True)
os.makedirs(save_path, exist_ok=True)

original_image = Image.open(os.path.join(f'{image_dir_name}_original', f'{image_name}.jpg'))

with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    old_G = pickle.load(f)['G_ema'].cuda()

with open('checkpoints/model_RKJBGNOQHLMZ_336944_15.pkl', 'rb') as f:
    new_G = pickle.load(f).cuda()


def plot_syn_images(syn_images, name): 
    for i, img in enumerate(syn_images): 
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
        plt.axis('off') 
        resized_image = Image.fromarray(img,mode='RGB').resize((256,256)) 
        resized_image.save(f"results/img_{i}_{name}.jpg")
        print("image saved")
        del img 
        del resized_image 
        torch.cuda.empty_cache()

w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
w_pivot = torch.load(f'{embedding_dir}/0.pt')

eigvec = torch.load('factor2.pt')["eigvec"].to(global_config.device)

degree = [-6, -4, -2, 0, 2, 4, 6]

for j in degree:
    direction = j * eigvec[:, 0].unsqueeze(0)
    old_image = old_G.synthesis(w_pivot + direction, noise_mode='const', force_fp32 = True)
    new_image = new_G.synthesis(w_pivot + direction, noise_mode='const', force_fp32 = True)
    plot_syn_images([old_image, new_image], f"fix_{j}")

