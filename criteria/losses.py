import torch.nn as nn

l1_criterion = nn.L1Loss(reduction='mean')
l2_criterion = nn.MSELoss(reduction='mean')


def l1_loss(real_images, generated_images):
    loss = l1_criterion(real_images, generated_images)
    return loss


def l2_loss(real_images, generated_images):
    loss = l2_criterion(real_images, generated_images)
    return loss
