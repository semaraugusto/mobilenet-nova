import torchvision.transforms.v2 as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Generator

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        # print(x)
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_dataloaders(batch_size: int=512, num_workers: int=24, transform_train=None, transform_test=None):
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.ColorJitter(),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    testset = CIFAR10(
        root='./CIFAR10', train=False, download=True, transform=transform_test)

    trainset = CIFAR10(
        root='./CIFAR10', train=True, download=True)

    # split the train set into train/validation
    train_set_size = int(len(trainset) * 0.8)
    valid_set_size = len(trainset) - train_set_size

    seed = Generator().manual_seed(42)
    trainset, validset = random_split(trainset, [train_set_size, valid_set_size], generator=seed)

    trainset = DatasetWrapper(trainset, transform_train)
    validset = DatasetWrapper(validset, transform_test)

    # Create train dataloader
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create validation dataloader
    validloader = DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create test dataloader
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader
