import os

from jsonargparse import CLI
# import wandb

from protac_splitter.llms.training import train_model
from protac_splitter.llms.training_rl_models import (
    train_ppo_model,
    train_dpo_model,
)

# # Get the API key from the environment
# api_key = os.environ.get('WANDB_API_KEY')
# wandb.login(key=api_key)
# print('Logged in to wandb')

if __name__ == '__main__':
    CLI([train_model, train_ppo_model])