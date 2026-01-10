from preprocessing import create_test_transforms, create_train_transforms, load_datasets
from datasets import CancerDataset
from torch.utils.data import DataLoader

DATASET_URI = "s3://cancer-classification-data-bucket/BreaKHis_Total_dataset/"
REGION = "us-east-1"

train_dataset, dev_dataset, test_dataset = load_datasets(DATASET_URI, REGION)

train_transforms = create_train_transforms()
test_transforms = create_test_transforms()

training_data = list(CancerDataset(train_dataset, transforms = train_transforms))
validation_data = CancerDataset(dev_dataset, transforms = test_transforms)
testing_data = CancerDataset(test_dataset, transforms = test_transforms)

train_data_loader = DataLoader(training_data, batch_size=32, shuffle = True)
val_data_loader = DataLoader(validation_data, batch_size=32, shuffle = False)
test_data_loader = DataLoader(testing_data, batch_size=32, shuffle = False)