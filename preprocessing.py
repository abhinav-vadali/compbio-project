from s3torchconnector import S3IterableDataset
import torch
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_datasets(DATASET_URI, REGION):
    train_dataset = S3IterableDataset.from_prefix(DATASET_URI + "train/", region=REGION)
    dev_dataset = S3IterableDataset.from_prefix(DATASET_URI + "val/", region=REGION)
    test_dataset = S3IterableDataset.from_prefix(DATASET_URI + "test/", region=REGION)
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
