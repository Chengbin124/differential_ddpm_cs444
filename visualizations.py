import os
from PIL import Image
import matplotlib.pyplot as plt

image_folder = "/data3/jerryhuang/differential_ddpm_cs444/results/DDPM_differential/"
output_file = "ema_images_grid.png"

def numeric_sort(file):
    return int(file.split('_')[0])

ema_images = sorted([img for img in os.listdir(image_folder) if '_ema' in img and img.endswith('.jpg')], key=numeric_sort)
selected_images = [img for img in ema_images if int(img.split('_')[0]) % 50 == 0]


fig, axes = plt.subplots(1, len(selected_images), figsize=(20, 5))
for i, img_file in enumerate(selected_images):
    img_path = os.path.join(image_folder, img_file)
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].axis('off')
    epoch = img_file.split('_')[0]
    axes[i].set_title(f"Epoch {epoch}", fontsize=10)

plt.tight_layout()
fig.savefig(output_file, dpi=300)
