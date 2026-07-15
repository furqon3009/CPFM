import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import numpy as np
from .augmentations import DataTransform

class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Check samples dimensions.
        # The dimension of the data is expected to be (N, C, L)
        # where N is the #samples, C: #channels, and L is the sequence length
        if len(x_data.shape) == 2:
            x_data = x_data.unsqueeze(1)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:
            x_data = x_data.transpose(0, 2, 1)

        x_data_weak, x_data_strong = DataTransform(x_data, dataset_configs)

        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        if isinstance(x_data_weak, np.ndarray):
            x_data_weak = torch.from_numpy(x_data_weak)

        if isinstance(x_data_strong, np.ndarray):
            x_data_strong = torch.from_numpy(x_data_strong)

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        # Normalize data
        if dataset_configs.normalize:
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
            data_mean2 = torch.mean(x_data_weak, dim=(0, 2))
            data_std2 = torch.std(x_data_weak, dim=(0, 2))
            self.transform2 = transforms.Normalize(mean=data_mean2, std=data_std2)
            data_mean3 = torch.mean(x_data_strong, dim=(0, 2))
            data_std3 = torch.std(x_data_strong, dim=(0, 2))
            self.transform3 = transforms.Normalize(mean=data_mean3, std=data_std3)

        self.x_data = x_data.float()
        self.x_data_weak = x_data_weak.float()
        self.x_data_strong = x_data_strong.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        x = self.x_data[index]
        x_weak = self.x_data_weak[index]
        x_strong = self.x_data_strong[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        if self.transform2:    
            x_weak = self.transform2(self.x_data_weak[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data_weak[index].shape)
        if self.transform3:    
            x_strong = self.transform3(self.x_data_strong[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data_strong[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y, index, x_weak, x_strong

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader

def split_data(data_path, domain_id, dataset_configs, hparams, dtype, run_id):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))

    train_data, val_data, train_label, val_label = train_test_split(
        dataset_file["samples"],
        dataset_file["labels"],
        stratify=dataset_file["labels"],
        test_size=0.1,
        random_state=run_id
    )

    dataset_train = {}
    dataset_val = {}

    dataset_train["samples"] = train_data
    dataset_train["labels"] = train_label
    dataset_val["samples"] = val_data
    dataset_val["labels"] = val_label

    dataset_train = Load_Dataset(dataset_train, dataset_configs)
    dataset_val = Load_Dataset(dataset_val, dataset_configs)

    if dtype == "test":  # you don't need to shuffle or drop last batch while testing
        shuffle = False
        drop_last = False
    else:
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last
    

    # Dataloaders
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)

    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                              batch_size=hparams["batch_size"],
                                              shuffle=False,
                                              drop_last=drop_last,
                                              num_workers=0)

    return data_loader_train, data_loader_val
