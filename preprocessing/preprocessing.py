from s3torchconnector import S3IterableDataset
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

DATASET_URI = "s3://cancer-classification-data-bucket/BreaKHis_Total_dataset/"
REGION = "us-east-1"

dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

def prepare_image(data):
    img = Image.open(BytesIO(data.read()))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

i = 0
for data in dataset:
    if (i == 0):
        prepare_image(data)
    i+=1