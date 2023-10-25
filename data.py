import json
import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class Dataset():
    def __init__(self, dataset):
        self._dataset = dataset
        self.check_dataset()
        self._dataset_dir = os.path.join(os.getcwd(),
                                         'data/datasets',
                                         f'{self._dataset.lower()}.pkl')
        config_nm = os.path.join(os.getcwd(),
                                 'configs',
                                 f'{self._dataset.lower()}.json')
        with open(config_nm, 'r') as f:
            self._config = json.load(f)
    
    def check_dataset(self):
        supported_dataset = ['food-101', 'food-101-small']
        if self._dataset.lower() not in supported_dataset:
            raise NotImplementedError("Dataset is not supported.")
    
    def save_dataset(self, train_ds, valid_ds, test_ds):
        with open(self._dataset_dir, 'wb') as f:
            pickle.dump([train_ds, valid_ds, test_ds], f)
    
    def create_dataset(self):
        data_dir = self._config['path']
        transform = transforms.Compose([
            transforms.Resize((self._config['height'], self._config['width'])),
            transforms.ToTensor()
        ])
        ds = datasets.ImageFolder(root=data_dir, transform=transform)
        train_size = int(len(ds) * self._config['train_size'])
        valid_size = int(len(ds) * self._config['valid_size'])
        test_size = len(ds) - train_size - valid_size
        train_ds, valid_ds, test_ds = random_split(
            ds,[train_size, valid_size, test_size])
        self.save_dataset(train_ds, valid_ds, test_ds)

        return train_ds, valid_ds, test_ds
    
    def get_dataloader(self):
        if os.path.exists(self._dataset_dir):
            with open(self._dataset_dir, 'rb') as f:
                train_ds, valid_ds, test_ds = pickle.load(f)
        else:
            train_ds, valid_ds, test_ds = self.create_dataset()
        
        train_dl = DataLoader(train_ds,
                              batch_size=self._config['batch_size'],
                              shuffle=True)
        valid_dl = DataLoader(valid_ds,
                              batch_size=self._config['batch_size'],
                              shuffle=True)
        test_dl  = DataLoader(test_ds,
                              batch_size=self._config['batch_size'],
                              shuffle=False)
        
        return train_dl, valid_dl, test_dl
    
    def get_config(self):
        
        return self._config