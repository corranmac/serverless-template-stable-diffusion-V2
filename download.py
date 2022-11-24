# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from huggingface_hub import hf_hub_download
import os

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    os.mkdir("checkpoints")
    os.mkdir("midas")
    os.chdir("checkpoints")
    hf_hub_download(repo_id="stabilityai/stable-diffusion-2-base", filename="512-base-ema.ckpt",token=str(HF_AUTH_TOKEN))
    hf_hub_download(repo_id="stabilityai/stable-diffusion-2", filename="768-v-ema.ckpt",token=str(HF_AUTH_TOKEN))
    os.chdir(".")
    os.chdir("midas")
    hf_hub_download(repo_id="stabilityai/stable-diffusion-2-depth", filename="512-depth-ema.ckpt",token=str(HF_AUTH_TOKEN))
    os.chdir(".")
if __name__ == "__main__":
    download_model()
