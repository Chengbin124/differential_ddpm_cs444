from diffusers import DiffusionPipeline
import torch


def load_model(model_type="pretrained", model_path=None):
    if model_type == "pretrained":
        print("Loading pre-trained model...")
        # https://huggingface.co/google/ddpm-celebahq-256
        pipe = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32")
    elif model_type == "custom" and model_path:
        print(f"Loading custom model from {model_path}...")
        pipe = DiffusionPipeline.from_pretrained(model_path)
    else:
        raise ValueError("Invalid model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe


def generate_image(pipe):
    print("Generating image...")

    image = pipe().images[0]
    return image


if __name__ == "__main__":
    model_type = "pretrained"
    model_path = ""

    pipe = load_model(model_type=model_type, model_path=model_path)
    image = generate_image(pipe)
    image.save("generated_image.png")  # Save the image as a PNG file
