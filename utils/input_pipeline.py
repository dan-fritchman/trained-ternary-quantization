import numpy as np
from PIL import Image, ImageEnhance
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
import torch


TRAIN_DIR = '~/data/tiny-imagenet-200/training'
VAL_DIR = '~/data/tiny-imagenet-200/validation'


"""It assumes that training image data is in the following form:
TRAIN_DIR/class4/image44.jpg
TRAIN_DIR/class4/image12.jpg
...
TRAIN_DIR/class55/image33.jpg
TRAIN_DIR/class55/image543.jpg
...
TRAIN_DIR/class1/image6.jpg
TRAIN_DIR/class1/image99.jpg
...

And the same for validation data.
"""


def get_cifar10():
    batch_size = 128

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_iterator = trainloader
    val_iterator = testloader
    train_size = len(trainset)
    val_size = len(testset)
    num_classes = len(classes)
    return train_iterator, val_iterator, train_size, val_size, num_classes


def get_image_folders():
    """
    Build an input pipeline for training and evaluation.
    For training data it does data augmentation.
    """

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    # intensities of enhancers
    factors = {
        0: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
        1: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        2: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        3: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
    }

    # randomly change color of an image
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        # random enhancers in random order
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image

    def rotate(image):
        degree = np.clip(np.random.normal(0.0, 15.0), -40.0, 40.0)
        return image.rotate(degree, Image.BICUBIC)

    # training data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.Lambda(rotate),
        transforms.RandomCrop(56),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # for validation data
    val_transform = transforms.Compose([
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # mean and std are taken from here:
    # http://pytorch.org/docs/master/torchvision/models.html

    train_folder = ImageFolder(TRAIN_DIR, train_transform)
    val_folder = ImageFolder(VAL_DIR, val_transform)
    return train_folder, val_folder