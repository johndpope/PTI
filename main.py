import os
import torch
import pickle
from PIL import Image
from configs import paths_config, hyperparameters, global_config
from scripts.run_pti import run_PTI

ix = [0, 1, 2, 3, 4]
# degrees = [-6, -4, -2, 0, 2, 4, 6]
degrees = [-3, -2, -1, 0, 1, 2, 3]

save_path = 'pretrained_models'
image_dir_name = 'image'
out_dir = 'results'
image_name = '336944_15'
use_multi_id_training = False

global_config.device = 'cuda'
paths_config.input_data_id = image_dir_name
paths_config.input_data_path = f'{image_dir_name}_original'
paths_config.checkpoints_dir = 'checkpoints'

# paths_config.stylegan2_ada_ffhq = os.path.join(save_path, 'network-snapshot-008467.pkl')
# factor_path = os.path.join(save_path, 'default-all.pt')
paths_config.stylegan2_ada_ffhq = os.path.join(save_path, 'network-snapshot-xflip-017095.pkl')
factor_path = os.path.join(save_path, 'xflip-all.pt')
# paths_config.stylegan2_ada_ffhq = os.path.join(save_path, 'network-snapshot-mixing-010080.pkl')
# factor_path = os.path.join(save_path, 'mixing-all.pt')

hyperparameters.use_last_w_pivots = False
hyperparameters.max_pti_steps = 350
hyperparameters.pti_learning_rate = 3e-4
hyperparameters.pt_lpips_lambda = 1
hyperparameters.pt_lpips_layers = [0, 1, 2, 3]


def save_image(img, name):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
    resized_image = Image.fromarray(img,mode='RGB').resize((256,256)) 
    resized_image.save(os.path.join(out_dir, f'{name}.jpg'))
    del img 
    del resized_image 
    torch.cuda.empty_cache()


if __name__ == '__main__':

    os.makedirs(f'{image_dir_name}_original', exist_ok=True)
    os.makedirs(f'{image_dir_name}_processed', exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    original_image = Image.open(os.path.join(f'{image_dir_name}_original', f'{image_name}.jpg'))
    model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)
    generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name

    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()

    with open(os.path.join(paths_config.checkpoints_dir, f'model_{model_id}_{generator_type}.pt'), 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    w_path_dir = os.path.join(paths_config.embedding_base_dir, paths_config.input_data_id)
    embedding_dir = os.path.join(w_path_dir, paths_config.pti_results_keyword, image_name)
    w_pivot = torch.load(os.path.join(embedding_dir, '0.pt'))
    eigvec = torch.load(factor_path)["eigvec"].to(global_config.device)

    for i in ix:
        for j, degree in enumerate(degrees):
            direction = degree * eigvec[:, i].unsqueeze(0)
            old_image = old_G.synthesis(w_pivot + direction, noise_mode='const', force_fp32=True)
            new_image = new_G.synthesis(w_pivot + direction, noise_mode='const', force_fp32=True)
            save_image(old_image, f"old_{i}_{j}")
            save_image(new_image, f"new_{i}_{j}")
            print(f"(degree: {degree}): {j}-th image saved")