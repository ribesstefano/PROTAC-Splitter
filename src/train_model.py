from jsonargparse import CLI
from protac_splitter.llms.training import train_model

if __name__ == '__main__':
    CLI(train_model)