import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os

from modules import SelfAttention, UNet_conditional

device = "cuda" if torch.cuda.is_available() else "cpu"

attention_maps = []


# Referenced https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging for adding hook functions for visualizing the attention maps
# https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
def visualize_heatmaps_grid(
    images, attention_maps, save_dir="visualizations", save_name="heatmaps_grid.png"
):
    os.makedirs(save_dir, exist_ok=True)
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for idx, (img, attn) in enumerate(zip(images, attention_maps)):
        img = img.permute(1, 2, 0).cpu().numpy()
        attn = attn.mean(dim=1)[0].cpu().numpy()

        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-12)

        axes[idx, 0].imshow((img + 1) / 2)
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"Image {idx+1}")

        axes[idx, 1].imshow((img + 1) / 2, alpha=0.7)
        axes[idx, 1].imshow(attn, cmap="jet", alpha=0.3)
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"Attention Heatmap {idx+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close(fig)


def visualize_gradients_grid(
    images, gradients, save_dir="visualizations", save_name="gradients_grid.png"
):
    os.makedirs(save_dir, exist_ok=True)
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for idx, (img, grad) in enumerate(zip(images, gradients)):
        img = img.detach().permute(1, 2, 0).cpu().numpy()
        grad = grad.detach().permute(1, 2, 0).abs().mean(dim=-1).cpu().numpy()

        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-12)

        axes[idx, 0].imshow(img)
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"Image {idx+1}")

        axes[idx, 1].imshow(grad, cmap="viridis")
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"Gradient Map {idx+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close(fig)


def main():
    model = UNet_conditional(num_classes=10).to(device)
    model.load_state_dict(
        torch.load(
            "/data3/jerryhuang/differential_ddpm_cs444/models/DDPM_differential/ema_ckpt.pt",
            map_location=device,
        )
    )
    model.eval()

    def hook_fn_attention(module, input, output):
        attention_maps.append(output)

    hooks = []
    for _, module in model.named_modules():
        if isinstance(module, SelfAttention):
            hooks.append(module.register_forward_hook(hook_fn_attention))

    transform = T.Compose(
        [T.Resize(32), T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform, download=True
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    images, _ = next(iter(dataloader))
    images = images.to(device)

    # activate hooks
    with torch.no_grad():
        model(images, t=torch.randint(0, 1000, (images.size(0),), device=device))

    visualize_heatmaps_grid(images, attention_maps)
    images.requires_grad = True
    gradients = []
    with torch.enable_grad():
        outputs = model(
            images, t=torch.randint(0, 1000, (images.size(0),), device=device)
        )
        loss = outputs.mean()
        loss.backward()
        gradients = images.grad

    visualize_gradients_grid(images, gradients)


if __name__ == "__main__":
    main()
