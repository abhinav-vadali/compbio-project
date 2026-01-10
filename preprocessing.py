from s3torchconnector import S3IterableDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, IterableDataset, random_split
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO
from torchvision import transforms

DATASET_URI = "s3://cancer-classification-data-bucket/BreaKHis_Total_dataset/"
REGION = "us-east-1"

train_dataset = S3IterableDataset.from_prefix(DATASET_URI + "train/", region=REGION)
dev_dataset = S3IterableDataset.from_prefix(DATASET_URI + "val/", region=REGION)
test_dataset = S3IterableDataset.from_prefix(DATASET_URI + "test/", region=REGION)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# create a custom dataset class to apply the transformations
class CancerDataset(IterableDataset):
    def __init__(self, data, transforms = None):
        self.data = data
        self.transforms = transforms
    def __iter__(self):
        for item in self.data:
            img = Image.open(BytesIO(item.read())).convert('RGB')
            label = 0 if 'benign' in item.key else 1
            if self.transforms:
                img = self.transforms(img)
            yield img, label

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

training_data = CancerDataset(train_dataset, transforms = train_transforms)
validation_data = CancerDataset(dev_dataset, transforms = test_transforms)
testing_data = CancerDataset(test_dataset, transforms = test_transforms)

train_data_loader = DataLoader(training_data, batch_size=32)
val_data_loader = DataLoader(validation_data, batch_size=32)
test_data_loader = DataLoader(testing_data, batch_size=32)``


#prints the image from bytes data
def print_image(data: bytes) -> None:
    img = Image.open(BytesIO(data.read()))
    plt.imshow(img) 
    plt.axis('off')
    plt.show()