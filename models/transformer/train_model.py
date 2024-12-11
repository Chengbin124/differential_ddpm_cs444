from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps = 1000,           
    sampling_timesteps = 250    
)

trainer = Trainer(
    diffusion,
    folder="/data3/jerryhuang/denoising-diffusion-pytorch/cifar10_images",
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         
    gradient_accumulate_every = 2,    
    ema_decay = 0.995,                
    amp = True,                       
    calculate_fid = True              
)

trainer.train()