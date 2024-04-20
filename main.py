import h5py
import torch
from uNet import UNet
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = []
masks = []

for i in range(1, 700):
    print(f"translating image .mat to array {i}")
    first_path = r".\brainTumorDataPublic_1-766"
    second_path = r".mat"
    path = first_path + "\\" + str(i) + second_path
    with h5py.File(path, 'r') as file:
        # Here you access the images and masks datasets
        image = np.array(file['cjdata/image'])
        mask = np.array(file['cjdata/tumorMask'])
        images.append(image)
        masks.append(mask)

images = np.stack(images, axis=0)
images = images.astype(np.float32) / 255.0  # Normalize to range 0-1 and ensure type is float32
images = np.expand_dims(images, axis=1)
masks = np.stack(masks, axis=0)
masks = masks.astype(np.float32)
images_tensor = torch.tensor(images)
masks_tensor = torch.tensor(masks).float()


X_train, X_val, y_train, y_val = train_test_split(images_tensor, masks_tensor, test_size=0.1, random_state=42)

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Shape of the tensor:", images_tensor.shape)
for epoch in range(50):  # Number of epochs
    print(f"beginning epoch: {epoch}")
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

for image in X_val:
    image=image.squeeze(0)
    plt.imshow(image, cmap='gray')
    plt.title('MRI Image')
    plt.axis('off')
    plt.show()