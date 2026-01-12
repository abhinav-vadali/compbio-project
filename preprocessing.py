from s3torchconnector import S3IterableDataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_datasets(path, trainTransform=None, testTransform = None):
    train_dataset = ImageFolder(root = path + "/train", transform = trainTransform)
    dev_dataset = ImageFolder(root = path + "/val", transform = testTransform)
    test_dataset = ImageFolder(root = path + "/test", transform = testTransform)
    return train_dataset, dev_dataset, test_dataset

def create_train_transforms():
    return transforms.Compose([
    
    # we must resize the image to 224x224 as expected by VGG16
    transforms.Resize((224,224)),
    
    # randomized flips and rotations are useful for data augmentation and to prevent overfitting
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    
    # convert the image to a tensor and normalize it using the mean and stddev of the dataset
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def create_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),

        # we do not apply the random transformations to the test set
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
