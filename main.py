import os
import pickle
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
# from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt


save_path = 'pretrained_models'
image_dir_name = 'image'

image_name = '336944_15'
use_multi_id_training = False

global_config.device = 'cuda'
paths_config.input_data_id = image_dir_name
paths_config.input_data_path = f'{image_dir_name}_original'
paths_config.stylegan2_ada_ffhq = '../stylegan2-ada-pytorch/results/00005-webtoon_3channel-webtoon256-bgcfnc-resumeffhq256/models/network-snapshot-007338.pkl'
# paths_config.stylegan2_ada_ffhq = f'{save_path}/network-snapshot-007338.pkl'
paths_config.checkpoints_dir = 'checkpoints'
paths_config.style_clip_pretrained_mappers = 'pretrained_models'

hyperparameters.use_last_w_pivots = False
hyperparameters.max_pti_steps = 350
hyperparameters.pti_learning_rate = 3e-4
# hyperparameters.use_locality_regularization = False
hyperparameters.pt_lpips_lambda = 0

os.makedirs(f'./{image_dir_name}_original', exist_ok=True)
os.makedirs(f'./{image_dir_name}_processed', exist_ok=True)
os.makedirs(save_path, exist_ok=True)

original_image = Image.open(os.path.join(f'{image_dir_name}_original', f'{image_name}.jpg'))

model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)


def load_generators(model_id, image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    return old_G, new_G

generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
old_G, new_G = load_generators(model_id, generator_type)

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

eigvec = torch.load('pretrained_models/factor.pt')["eigvec"].to(global_config.device)

ix = [0]

degree = [-6, -4, -2, 0, 2, 4, 6]

for i in ix:
    for n, j in enumerate(degree):
        direction = j * eigvec[:, i].unsqueeze(0)
        old_image = old_G.synthesis(w_pivot + direction, noise_mode='const', force_fp32 = True)
        new_image = new_G.synthesis(w_pivot + direction, noise_mode='const', force_fp32 = True)
        plot_syn_images([old_image, new_image], f"fix_{i}_{n}")


# for direction, factor_and_edit in latents_after_edit.items():
#   print(f'Showing {direction} change')
#   for latent in factor_and_edit.values():
#     exit()


# def display_alongside_source_image(images): 
#     res = np.concatenate([np.array(image) for image in images], axis=1) 
#     res = Image.fromarray(res)
#     res.save("img.jpg")
#     print("image saved")


# original_image = Image.open(os.path.join(f'{image_dir_name}_original', f'{image_name}.jpg')).resize((256, 256), Image.ANTIALIAS)
# original_image.save('a.jpg')
