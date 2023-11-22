'''
Fetches and caches the Kandinsky models.
'''

import torch
from diffusers import KandinskyV3Pipeline

def get_kandinsky_pipelines():
    # Kandinsky 3 pipelines
    pipe_prior_3 = KandinskyV3Img2ImgPipeline.from_pretrained(
        "kandinsky-community/kandinsky-3", torch_dtype=torch.float16)

    return (pipe_prior_3)

if __name__ == "__main__":
    get_kandinsky_pipelines()
