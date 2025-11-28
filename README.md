Text-to-Image Generator:

This project is a simple text-to-image generation tool built using the **Stable Diffusion** model. It takes a text prompt as input and generates a corresponding image. I created this as part of my Microsoft AI-NSI internship to explore how generative AI models work in practice.

About the Project:

The main idea behind this project was to understand how modern diffusion models convert text descriptions into images. I used the "Hugging Face Diffusers" library because it offers a clean and easy-to-use interface for Stable Diffusion.

The project runs on Google Colab, which provides GPU support for faster image generation.

Features:

* Generate images from any natural language prompt
* Uses Stable Diffusion (v1.5)
* GPU-accelerated for faster output
* Clean and simple codebase
* Easy to modify or extend

Tech Used:

* Python
* PyTorch
* Hugging Face Diffusers
* Google Colab
* Stable Diffusion v1.5

How to Run:

1. Clone the repository:

   ```bash
   git clone https://github.com/Girishp12113/Text-to-Image-generator
   ```

2. Open the notebook on Google Colab.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Enter any text prompt and run the generation cell.

Sample Code:

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "a futuristic city with neon lights"
image = pipe(prompt).images[0]
image.save("output.png")

Project Structure:

├── notebook.ipynb
├── outputs/
├── requirements.txt
└── README.md

Outputs:

You can add your generated images in the `outputs` folder and update them here whenever needed.

What I Learned:

Working on this project helped me understand:

* How Stable Diffusion works
* How generative models process text and images
* How to use GPUs for faster inference
* Integrating Hugging Face models into real code
* Basics of prompt engineering

Future Improvements:

* Adding a small UI for user input
* Improving image resolution
* Allowing multiple image outputs per prompt
* Exploring fine-tuning
