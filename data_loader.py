from paddle.vision import datasets, transforms
from paddle.io import DataLoader
import os

def load_training(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.DatasetFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    # print(os.path.join(root_path, dir))
    return train_loader

def load_testing(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.DatasetFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader