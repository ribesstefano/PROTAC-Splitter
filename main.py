from train_model import train_model, train_mlm_model, train_ppo_model
from jsonargparse import CLI

if __name__ == '__main__':
    CLI([train_model, train_mlm_model, train_ppo_model])