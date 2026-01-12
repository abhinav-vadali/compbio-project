from preprocessing import create_test_transforms, create_train_transforms, load_datasets
from torch.nn import BCEWithLogitsLoss
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from vgg16 import vgg16model

train_transforms = create_train_transforms()
test_transforms = create_test_transforms()

train_data, dev_data, test_data = load_datasets("/Users/abhinavvadali/data", trainTransform = train_transforms, testTransform = test_transforms)

##validation_data = CancerDataset(dev_dataset, transforms = test_transforms)
##testing_data = CancerDataset(test_dataset, transforms = test_transforms)

train_data_loader = DataLoader(train_data, batch_size=32, shuffle = True)
##val_data_loader = DataLoader(validation_data, batch_size=32, shuffle = False)
##test_data_loader = DataLoader(testing_data, batch_size=32, shuffle = False)


device = "mps" if torch.backends.mps.is_available() else "cpu"

model = vgg16model().to(device)

optimizer = Adam(params = model.parameters(), lr = 0.001)

# calculate weight for the benign class since it is underrepresented in the sample using the inverse class frequency method
minority_weight = torch.tensor([7784/(2*2479)], device = device)

loss_function = BCEWithLogitsLoss(weight = minority_weight)

num_epochs = 2

if __name__ == "__main__":
    for epoch in range(num_epochs):
        for batch_idx, (imgs, labels) in enumerate(train_data_loader):
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            scores = model(imgs)
            # Zero parameter gradients
            optimizer.zero_grad()
            loss = loss_function(scores, labels)
            # Backward pass and parameter updates
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")
