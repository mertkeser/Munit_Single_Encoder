import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt # For Visualization
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt # For Visualization
import torch.optim as optimizer

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--save_dir", type=str, default="/output/results/ours_with_munit_64/", help="directory of saved Images")
parser.add_argument("--first_data_dir", type=str, default="/output/DataSets/cityscapes2975_512x256/", help="directory of first domain")
parser.add_argument("--second_data_dir", type=str, default="/output/DataSets/gta2975_512x256/", help="directory of second domain")
parser.add_argument("--dataset_name", type=str, default="ours_city_to_gta_munit_64_0_res", help="Name of the Dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=201, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving generator samples (Number of Epoch)")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Create sample and checkpoint directories
if not os.path.exists("/output/results/architecture_ours/%s/images" % opt.dataset_name):
    os.makedirs("/output/results/architecture_ours/%s/images" % opt.dataset_name)
if not os.path.exists("/output/results/architecture_ours/%s/saved_models" % opt.dataset_name):
    os.makedirs("/output/results/architecture_ours/%s/saved_models" % opt.dataset_name)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
com_e = DAugNet_Encoder()
dec_a = DAugNet_Decoder()
shared_Disc = Disc_Shared()
disc_a = Discriminator(shared_Disc)
disc_b = Discriminator(shared_Disc)

# Optimizers
ae_params = list(com_e.parameters()) + list(dec_a.parameters())

OptimizeGenerator = optimizer.Adam(ae_params, lr=opt.lr, betas=(opt.b1, opt.b2))

disc_params_a = disc_a.parameters()
OptimizeDiscriminator_A = optimizer.Adam(disc_params_a, lr=opt.lr, betas=(opt.b1, opt.b2))

disc_params_b = disc_b.parameters()
OptimizeDiscriminator_B = optimizer.Adam(disc_params_b, lr=opt.lr, betas=(opt.b1, opt.b2))

if cuda:
    com_e = com_e.cuda()
    dec_a = dec_a.cuda()
    shared_Disc = shared_Disc.cuda()
    disc_a = disc_a.cuda()
    disc_b = disc_b.cuda()
    criterion_GAN.cuda()
    criterion_pixel.cuda()

if opt.epoch != 0:
    # Load pretrained models
    com_e.load_state_dict(torch.load("/output/results/architecture_ours/%s/saved_models/com_e_%d.pth" % (opt.dataset_name, opt.epoch)))
    dec_a.load_state_dict(torch.load("/output/results/architecture_ours/%s/saved_models/dec_a_%d.pth" % (opt.dataset_name, opt.epoch)))
    disc_a.load_state_dict(torch.load("/output/results/architecture_ours/%s/saved_models/disc_a_%d.pth" % (opt.dataset_name, opt.epoch)))
    disc_b.load_state_dict(torch.load("/output/results/architecture_ours/%s/saved_models/disc_b_%d.pth" % (opt.dataset_name, opt.epoch)))
    style_a = (torch.load("/output/results/architecture_ours/%s/saved_models/style_a_%d.pth" % (opt.dataset_name, opt.epoch)))
    style_b = (torch.load("/output/results/architecture_ours/%s/saved_models/style_b_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    com_e.apply(weights_init)
    dec_a.apply(weights_init)
    disc_a.apply(weights_init)
    disc_b.apply(weights_init)

# Learning rate update schedulers
#lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
#)
#lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
#    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
#)
#lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
#    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
#)

#Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = transforms.Compose([#transforms.Resize(( np.int(opt.img_height),np.int(opt.img_width) ), Image.BICUBIC),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Training data loader
dataloader = DataLoader(
    ImageDataset(path_1=opt.first_data_dir, path_2=opt.second_data_dir, transforms=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
dataloader_visualization = DataLoader(
    ImageDataset(path_1=opt.first_data_dir, path_2=opt.second_data_dir, transforms=transforms_),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------
Pixel_loss_AA_epoch = []
Pixel_loss_AB_epoch = []
Pixel_loss_BA_epoch = []
Pixel_loss_BB_epoch = []
Pixel_loss_BAB_epoch = []
Pixel_loss_ABA_epoch = []
GAN_loss_epoch = []
Disc_train_epoch = []
Total_loss_epoch = []

prev_time = time.time()

#style_a = Variable(torch.randn(np.random.normal(0, 1, [1, 256*2])).cuda())
#style_b = Variable(torch.randn(np.random.normal(0, 1, [1, 256*2])).cuda())
style_a = Variable(torch.from_numpy(np.random.normal(0, 1, [1, 256*2])).cuda()).to(dtype=torch.float)
style_b = Variable(torch.from_numpy(np.random.normal(0, 1, [1, 256*2])).cuda()).to(dtype=torch.float)

for epoch in range(opt.epoch, opt.n_epochs):
    #Initialization of Loss Keepers in each Iterization
    #epoch = epoch + epoch_resume
    t0 = time.time()  # Start of the Time
    # A is the cityscapes dataset
    # B is the gta or textureless dataset

    Pixel_loss_AA = []
    Pixel_loss_AB = []
    Pixel_loss_BA = []
    Pixel_loss_BB = []
    Pixel_loss_BAB = []
    Pixel_loss_ABA = []
    GAN_loss = []
    Disc_train = []
    Total_loss = []

    for i, (real_A, real_B) in enumerate(dataloader):

        # Set model input
        real_A = Variable(real_A.cuda()).to(dtype=torch.float) # Float Tensor
        real_B = Variable(real_B.cuda()).to(dtype=torch.float)
        #real_A = real_A.cuda().detach()
        #real_B = real_B.cuda().detach()

        # Adversarial ground truths
        #valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
        #fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        OptimizeGenerator.zero_grad()
        h_a_c = com_e(real_A) #Float Tensor
        h_b_c = com_e(real_B)

        style_a_mapped = style_a
        style_b_mapped = style_b
        style_train_a = torch.cat([style_a_mapped] * (h_a_c.size(0)))
        style_train_b = torch.cat([style_b_mapped] * (h_b_c.size(0)))
        shape = [-1, 2, h_a_c.size(1)] + (h_a_c.dim() - 2) * [1]  # [-1, 2, channel, 1, 1]
        style_train_a = style_train_a.view(shape)  # [batch_size, 2, n_channels, ...]
        style_train_b = style_train_b.view(shape)

        # Adversarial ground truths
        reconAA = dec_a(h_a_c, style_train_a)
        reconBA = dec_a(h_b_c, style_train_a)
        reconAB = dec_a(h_a_c, style_train_b)
        reconBB = dec_a(h_b_c, style_train_b)

        h_ba_c = com_e(reconBA)
        h_ab_c = com_e(reconAB)

        reconABA = dec_a(h_ab_c, style_train_a)
        reconBAB = dec_a(h_ba_c, style_train_b)

        loss_Pixel_A_A = criterion_pixel(reconAA, real_A)
        loss_Pixel_B_B = criterion_pixel(reconBB, real_B)
        loss_Pixel_A_B_A = criterion_pixel(reconABA, real_A)
        loss_Pixel_B_A_B = criterion_pixel(reconBAB, real_B)

        real_A_patch = patchify_image(real_A, 8)
        real_B_patch = patchify_image(real_B, 8)
        recon_BA_patch = patchify_image(reconBA, 8)
        recon_AB_patch = patchify_image(reconAB, 8)

        pred_fake_a = disc_a(recon_BA_patch)
        loss_GAN_A2B = criterion_GAN(pred_fake_a,
                                     torch.empty([len(pred_fake_a), 1], dtype=torch.float).fill_(1.0).cuda())

        pred_fake_b = disc_b(recon_AB_patch)
        loss_GAN_B2A = criterion_GAN(pred_fake_b,
                                          torch.empty([len(pred_fake_b), 1], dtype=torch.float).fill_(1.0).cuda())

        Pixel_loss_AA.append(loss_Pixel_A_A.item())
        Pixel_loss_BB.append(loss_Pixel_B_B.item())
        Pixel_loss_BAB.append(loss_Pixel_B_A_B.item())
        Pixel_loss_ABA.append(loss_Pixel_A_B_A.item())

        loss_recon = 10 * (loss_Pixel_A_A + loss_Pixel_B_B + loss_Pixel_B_A_B + loss_Pixel_A_B_A)
        loss_GAN = 1 * (loss_GAN_A2B + loss_GAN_B2A)

        loss_Total = loss_recon + loss_GAN
        loss_Total.backward(retain_graph=True)
        OptimizeGenerator.step()

        Total_loss.append(loss_Total.item())
        GAN_loss.append(loss_GAN.item())

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        OptimizeDiscriminator_A.zero_grad()

        pred_real = disc_a(real_A_patch)  # First one is output of encoder, second one is plus noise
        loss_A_real = criterion_GAN(pred_real,
                                    torch.empty([len(pred_real), 1], dtype=torch.float).fill_(1.0).cuda())

        # Calculation of Fake Loss for Discriminator A2B
        #         style_a_mapped = style_map_a(style_a)
        style_a_mapped = style_a
        style_train_a = torch.cat([style_a_mapped] * (h_a_c.size(0)))
        shape = [-1, 2, h_a_c.size(1)] + (h_a_c.dim() - 2) * [1]  # [-1, 2, channel, 1, 1]
        style_train_a = style_train_a.view(shape)  # [batch_size, 2, n_channels, ...]
        h_b_c = com_e(real_B)
        reconBA = dec_a(h_b_c, style_train_a)

        pred_fake = disc_a(recon_BA_patch)
        loss_A_fake = criterion_GAN(pred_fake,
                                    torch.empty([len(pred_fake), 1], dtype=torch.float).fill_(0.0).cuda())

        loss_D_A = 1 * (loss_A_real + loss_A_fake)  # * 0.5

        loss_D_A.backward()
        OptimizeDiscriminator_A.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        OptimizeDiscriminator_B.zero_grad()

        pred_real = disc_b(real_B_patch)  # First one is output of encoder, second one is plus noise
        loss_B_real = criterion_GAN(pred_real,
                                    torch.empty([len(pred_real), 1], dtype=torch.float).fill_(1.0).cuda())

        #         style_b_mapped = style_map_b(style_b)
        style_b_mapped = style_b
        style_train_b = torch.cat([style_b_mapped] * (h_b_c.size(0)))
        shape = [-1, 2, h_a_c.size(1)] + (h_a_c.dim() - 2) * [1]  # [-1, 2, channel, 1, 1]
        style_train_b = style_train_b.view(shape)
        h_a_c = com_e(real_A)
        reconAB = dec_a(h_a_c, style_train_b)
        # Calculation of Fake Loss for Discriminator A2B
        pred_fake = disc_b(recon_AB_patch)
        loss_B_fake = criterion_GAN(pred_fake,
                                    torch.empty([len(pred_fake), 1], dtype=torch.float).fill_(0.0).cuda())

        loss_D_B = 1 * (loss_B_real + loss_B_fake)  # * 0.5

        loss_D_B.backward()
        OptimizeDiscriminator_B.step()

        Disc_train.append(loss_D_B.item() + loss_D_A.item())

    # --------------
    #  Log Progress
    # --------------
    # Print log
    Pixel_loss_AA_epoch.append(np.mean(Pixel_loss_AA))
    Pixel_loss_BB_epoch.append(np.mean(Pixel_loss_BB))
    Pixel_loss_BAB_epoch.append(np.mean(Pixel_loss_BAB))
    Pixel_loss_ABA_epoch.append(np.mean(Pixel_loss_ABA))
    GAN_loss_epoch.append(np.mean(GAN_loss))
    Disc_train_epoch.append(np.mean(Disc_train))
    Total_loss_epoch.append(np.mean(Total_loss))

    print('For %d Epoch, Pixel Loss AA %f,Pixel Loss BB %f,Time One Epoch: %d Sec' % (epoch, Pixel_loss_AA_epoch[-1], Pixel_loss_BB_epoch[-1], time.time() - t0))
    print('Loss From Disc %f, GAN Training Loss %f, Total Loss %f' % (Disc_train_epoch[-1], GAN_loss_epoch[-1], Total_loss_epoch[-1]))

    if epoch % opt.sample_interval == 0:
        # Save model checkpoints
        torch.save(com_e.state_dict(), "/output/results/architecture_ours/%s/saved_models/com_e_%d.pth" % (opt.dataset_name, epoch))
        torch.save(dec_a.state_dict(), "/output/results/architecture_ours/%s/saved_models/dec_a_%d.pth" % (opt.dataset_name, epoch))
        torch.save(disc_a.state_dict(), "/output/results/architecture_ours/%s/saved_models/disc_a_%d.pth" % (opt.dataset_name, epoch))
        torch.save(disc_b.state_dict(), "/output/results/architecture_ours/%s/saved_models/disc_b_%d.pth" % (opt.dataset_name, epoch))
        torch.save(style_a, "/output/results/architecture_ours/%s/saved_models/style_a_%d.pth" % (opt.dataset_name, epoch))
        torch.save(style_b, "/output/results/architecture_ours/%s/saved_models/style_b_%d.pth" % (opt.dataset_name, epoch))

    # Save Images in Real Format
    if epoch % opt.sample_interval == 0:
        no_im = 3
        for i in range(no_im):
            real_A, real_B = next(iter(dataloader_visualization))
            real_A = Variable(real_A.cuda()).to(dtype=torch.float)
            real_B = Variable(real_B.cuda()).to(dtype=torch.float)

            save_image(real_A, "/output/results/architecture_ours/%s/images/Dom_A_epoch_%d_%s.png" % (opt.dataset_name, epoch, i), normalize=True)
            save_image(real_B, "/output/results/architecture_ours/%s/images/Dom_B_epoch_%d_%s.png" % (opt.dataset_name, epoch, i), normalize=True)

            h_a_c = com_e(real_A)  # First one is output of encoder, second one is plus noise
            h_b_c = com_e(real_B)

            #             style_a_mapped = style_map_a(style_a)
            #             style_b_mapped = style_map_b(style_b)

            style_a_mapped = style_a
            style_b_mapped = style_b
            style_train_a = torch.cat([style_a_mapped] * (h_a_c.size(0)))
            style_train_b = torch.cat([style_b_mapped] * (h_b_c.size(0)))
            shape = [-1, 2, h_a_c.size(1)] + (h_a_c.dim() - 2) * [1]  # [-1, 2, channel, 1, 1]
            style_train_a = style_train_a.view(shape)  # [batch_size, 2, n_channels, ...]
            style_train_b = style_train_b.view(shape)

            reconAA = dec_a(h_a_c, style_train_a)
            reconBA = dec_a(h_b_c, style_train_a)
            reconAB = dec_a(h_a_c, style_train_b)
            reconBB = dec_a(h_b_c, style_train_b)

            save_image(reconAA, "/output/results/architecture_ours/%s/images/Dom_A_recon_epoch_%d_%s.png" % (opt.dataset_name, epoch, i), normalize=True)
            save_image(reconBB, "/output/results/architecture_ours/%s/images/Dom_B_recon_epoch_%d_%s.png" % (opt.dataset_name, epoch, i), normalize=True)
            save_image(reconBA, "/output/results/architecture_ours/%s/images/Dom_A_fake_epoch_%d_%s.png" % (opt.dataset_name, epoch, i), normalize=True)
            save_image(reconAB, "/output/results/architecture_ours/%s/images/Dom_B_fake_epoch_%d_%s.png" % (opt.dataset_name, epoch, i), normalize=True)