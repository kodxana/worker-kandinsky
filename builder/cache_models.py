'''
Fetches and caches the Kandinsky models.
'''

import torch
from huggingface_hub import hf_hub_download
from kandinsky3 import get_T2I_unet, get_T5encoder, get_movq, Kandinsky3T2IPipeline

def get_kandinsky_pipelines():
    # Kandinsky 3 pipelines
    unet_path = hf_hub_download(
        repo_id="ai-forever/Kandinsky3.0", filename='weights/kandinsky3.pt')
    movq_path = hf_hub_download(
          repo_id="ai-forever/Kandinsky3.0", filename='weights/movq.pt')

    return (unet_pat, movq_path)

if __name__ == "__main__":
    get_kandinsky_pipelines()
