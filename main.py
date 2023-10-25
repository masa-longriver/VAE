import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data import Dataset
from model import VAE
from run import run
from utils import EarlyStopping, save_loss, save_img

parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    help="Select the dataset from ['food-101', 'food-101-small']")
args = parser.parse_args()


if __name__ == '__main__':
    config_dir = os.path.join(os.getcwd(),
                              'configs/main.json')
    with open(config_dir, 'r') as f:
        config = json.load(f)
    
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    Dataset = Dataset(args.dataset)
    train_dl, valid_dl, test_dl = Dataset.get_dataloader()
    config_data = Dataset.get_config()
    config['data'] = config_data
    
    model = VAE(config).to(config['device'])
    optimizer = optim.Adam(model.parameters(),
                           lr=config['optim']['lr'])
    es = EarlyStopping(config, args.dataset)

    train_losses = []
    valid_losses = []
    for epoch in range(config['epochs']):
        train_loss = run(config, model, train_dl, optimizer, state='train')
        valid_loss = run(config, model, valid_dl, optimizer, state='eval')
        print(f"Epoch: {epoch+1}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}",
              flush=True)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if es.check(valid_loss, model, epoch):
            break
    
    test_loss = run(config, model, test_dl, optimizer, state='eval')
    print(f"test_loss: {test_loss:.4f}", flush=True)

    save_loss(train_losses, valid_losses, args.dataset)
    save_img(config, model, test_dl, args.dataset)