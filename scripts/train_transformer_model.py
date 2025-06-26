import os

from jsonargparse import CLI
# import wandb
import torch

from protac_splitter.llms.training import train_model
# # Get the API key from the environment
# api_key = os.environ.get('WANDB_API_KEY')
# wandb.login(key=api_key)
# print('Logged in to wandb')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU with index 0
    print(f"GPU available: {torch.cuda.is_available()}")
    CLI(train_model)