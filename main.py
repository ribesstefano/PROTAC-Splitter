from train_model import train_model
from jsonargparse import CLI
from transformers import AutoTokenizer
from typing import Optional
import os

if __name__ == '__main__':
    CLI(train_model)