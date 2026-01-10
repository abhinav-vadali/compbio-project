from s3torchconnector import S3IterableDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from io import BytesIO
from torchvision import datasets, transforms

DATASET_URI = "s3://cancer-classification-data-bucket/BreaKHis_Total_dataset/"
REGION = "us-east-1"

dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# transformations to apply to preprocess training images
train_transforms = transforms.Compose([
    
    # we must resize the image to 224x224 as expected by VGG16
    transforms.Resize((224,224)),
    
    # randomized flips and rotations are useful for data augmentation and to prevent overfitting
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    
    # convert the image to a tensor and normalize it using the mean and stddev of the dataset
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # we do not apply the random transformations to the test set
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# create a custom dataset class to apply the transformations
class CancerDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
    def __len__(self):
        length = 0
        for item in self.data:
            length += 1
        return length
    def __get__item__(self, idx):
        i = 0
        for item in self.data:
            if i == idx:
                img = Image.open(BytesIO(item.read()))
                return img
            i += 1


train_dataset = CancerDataset(dataset, train_transforms)

test_dataset = CancerDataset(dataset, test_transforms)

train_data_loader = DataLoader(dataset, batch_size = 32, shuffle = True)

for data in train_data_loader:



##for data in dataloader:
    ##print(data.size())
#prints the image from bytes data
    def print_image(data: bytes) -> None:
        img = Image.open(BytesIO(data.read()))
        plt.imshow(img) 
        plt.axis('off')
        plt.show()

i = 0
for data in dataset:
    if (i == 2392):
        print_image(data)
    i+=1