import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse
import logging
import numpy as np
from scipy.stats import entropy
import torchvision.utils as vutils
import random
from modules import UNet_conditional
from ddpm_conditional import Diffusion

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDPM Model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/DDPM_conditional/ckpt.pt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--ema_ckpt",
        type=str,
        default="models/DDPM_conditional/ema_ckpt.pt",
        help="Path to the EMA model checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the evaluation on"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50_000,
        help="Number of samples to generate for evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    return parser.parse_args()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path, ema=False, device="cuda", num_classes=10):
    model = UNet_conditional(num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    if ema:
        ema_model = copy.deepcopy(model)
        return ema_model
    return model


def generate_images(model, diffusion, device, num_samples=50000, batch_size=64):
    model.eval()
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating Images"):
            current_batch_size = min(
                batch_size, num_samples - len(all_images) * batch_size
            )
            if current_batch_size <= 0:
                break
            sampled = diffusion.sample(
                model, n=current_batch_size, labels=None, cfg_scale=0
            )
            all_images.append(sampled)
    all_images = torch.cat(all_images, dim=0)[:num_samples]

    logging.info(
        f"Fake Images - dtype: {all_images.dtype}, min: {all_images.min().item()}, max: {all_images.max().item()}"
    )
    images_to_save = all_images.float() / 255.0
    vutils.save_image(
        images_to_save[:64], "generated_images.png", normalize=False, nrow=8
    )
    logging.info("Generated images saved to 'generated_images.png'")
    return all_images


def prepare_real_images(device, num_samples=50000, batch_size=64, image_size=32):
    """
    Uses similar scaling/preprocessing as seen in the original paper https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/tpu_utils/tpu_utils.py
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    real_images = []
    with torch.no_grad():
        total_loaded = 0
        for images, _ in tqdm(dataloader, desc="Loading Real Images"):
            images = images.to(device)
            images = (images + 1) / 2
            real_images.append(images)
            total_loaded += images.size(0)
            if total_loaded >= num_samples:
                break

    real_images = torch.cat(real_images, dim=0)[:num_samples]
    logging.info(
        f"Real Images - dtype: {real_images.dtype}, min: {real_images.min().item()}, max: {real_images.max().item()}"
    )
    return real_images


def calculate_inception_score(images, device="cuda", batch_size=64, splits=10):
    # We reference https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
    # We use the statistics from ImageNet for the normalization step

    from torchvision.models import inception_v3
    import torch.nn.functional as F

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    up = torch.nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    images = images.to(device).float() / 255.0

    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Calculating IS"):
            batch = images[i : i + batch_size]
            batch = up(batch)
            batch = normalize(batch)
            preds_batch = inception_model(batch)
            preds.append(F.softmax(preds_batch, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits) : (k + 1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores = [entropy(pyx, py) for pyx in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def compute_fid_in_chunks(fid_metric, images, batch_size=512, real=True):
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        fid_metric.update(batch, real=real)


def compute_fid(real_images, fake_images, device="cuda", fid_batch_size=512):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    real_images = (real_images * 255).clamp(0, 255).type(torch.uint8)
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    compute_fid_in_chunks(fid, real_images, batch_size=fid_batch_size, real=True)
    compute_fid_in_chunks(fid, fake_images, batch_size=fid_batch_size, real=False)

    fid_score = fid.compute().item()
    logging.info(f"Overall FID Score: {fid_score:.4f}")
    return fid_score


def main():
    args = parse_args()
    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    diffusion = Diffusion(
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=args.image_size,
        device=device,
    )

    logging.info("Loading models...")
    ema_model = load_model(args.ema_ckpt, ema=True, device=device, num_classes=10)
    logging.info("Models loaded successfully.")

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        ema_model = nn.DataParallel(ema_model, device_ids=[0, 1, 2, 3])

    logging.info(f"Generating {args.num_samples} images...")
    fake_images = generate_images(
        model=ema_model,
        diffusion=diffusion,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    logging.info("Image generation completed.")

    logging.info("Loading real images...")
    real_images = prepare_real_images(
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    fid_overall = compute_fid(
        real_images, fake_images, device=device, fid_batch_size=256
    )
    is_score, is_std = calculate_inception_score(
        fake_images, device=device, batch_size=args.batch_size
    )
    logging.info(f"Inception Score: {is_score:.4f} ± {is_std:.4f}")

    print(f"Overall FID Score: {fid_overall:.4f}")
    print(f"Inception Score: {is_score:.4f} ± {is_std:.4f}")


if __name__ == "__main__":
    main()
