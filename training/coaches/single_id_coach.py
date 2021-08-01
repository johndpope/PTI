import os
import torch
# import pickle
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        print(w_path_dir)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            print(fname)
            print(image.shape)

            # self.restart_training()  # HYUNG-KWON KO CHANGED

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            print("[INFO] generated embedding_dir: ", embedding_dir)

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                print("[INFO] load w_pivot")
                w_pivot = self.load_inversions(w_path_dir, image_name)
                print("[INFO] load w_pivot complete")

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                print("[INFO] calc w_pivot")
                w_pivot = self.calc_inversions(image, image_name)
                print("[INFO] calc w_pivot complete")

            # w_pivot = w_pivot.detach().clone().to(global_config.device)
            w_pivot = w_pivot.to(global_config.device)

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            print("[INFO] save w_pivot")

            log_images_counter = 0
            real_images_batch = image.to(global_config.device)


            for i in tqdm(range(hyperparameters.max_pti_steps)):

                generated_images = self.forward(w_pivot)

                # print(generated_images.min())
                # print(generated_images.max())
                # print(real_images_batch.min())
                # print(real_images_batch.max())

                loss, l2_loss_val, loss_lpips = self.calc_loss(i, generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1

            print("loss: ", loss)

            torch.save(self.G, f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')

            # snapshot_pkl = f'{paths_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pkl'
            # with open(snapshot_pkl, 'wb') as f:
            #     pickle.dump(self.G, f)
