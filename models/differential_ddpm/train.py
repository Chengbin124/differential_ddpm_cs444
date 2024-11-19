# accelerate launch train.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '/data3/jerryhuang/differential_ddpm_cs444/external_libraries/denoising_diffusion_pytorch'))

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

print(Unet)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False,
    differential_transformer=True,
)
# Define the diffusion process
diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)
# Define the trainer
trainer = Trainer(
    diffusion,
    folder="/data3/jerryhuang/denoising-diffusion-pytorch/cifar10_images",
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    calculate_fid=True,  # whether to calculate fid during training
)
# Start training
trainer.train()
