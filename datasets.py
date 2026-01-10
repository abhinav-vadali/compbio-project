from PIL import Image
from io import BytesIO
from torch.utils.data import IterableDataset

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