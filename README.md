from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU for faster processing

# Function to generate an image from a text prompt
def generate_image(prompt, save_path="output_image.png"):
    # Generate the image
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    # Save the image
    image.save(save_path)
    print(f"Image saved to {save_path}")

# Example usage
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use GPU for faster processing

# Function to generate an image from a text prompt
def generate_image(prompt, save_path="output_image.png"):
    # Generate the image
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    # Save the image
    image.save(save_path)
    print(f"Image saved to {save_path}")

# Example usage
if __name__ == "__main__":
    prompt = input("Enter your text prompt: ")
    generate_image(prompt)
