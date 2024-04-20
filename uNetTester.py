import torch
from uNet import UNet  # Ensure this is the same UNet structure as used during training
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Initialize the model and load the trained weights
model = UNet()
model.load_state_dict(torch.load('./unet_model.pth'))
model.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

all_images = []
all_masks = []

for i in range(1,100):
    base_dir = r".\brainTumorDataPublic_1-766"
    file_name = str(i) + ".mat"
    path = os.path.join(base_dir, file_name)
    with h5py.File(path, 'r') as file:
        image = np.array(file['cjdata/image'])
        mask = np.array(file['cjdata/tumorMask'])
        all_images.append(image)
        all_masks.append(mask)

# Normalize and reshape images and masks
test_images = np.stack(all_images, axis=0).astype(np.float32) / 255.0
test_images = np.expand_dims(test_images, axis=1)
test_masks = np.stack(all_masks, axis=0)

test_images_tensor = torch.tensor(test_images)
test_masks_tensor = torch.tensor(test_masks, dtype=torch.float32)

# Create dataset and DataLoader
test_dataset = TensorDataset(test_images_tensor, test_masks_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []
with torch.no_grad():
    for data, true_mask in test_loader:
        data = data.to(device)
        output = model(data)
        predicted_mask = torch.sigmoid(output)
        predicted_mask = (predicted_mask > 0.5).float()  # Threshold probabilities to create binary mask
        predictions.append(predicted_mask.cpu().numpy())

def overlay_masks(image, true_mask, predicted_mask):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image.squeeze(), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(image.squeeze(), cmap='gray')
    ax[1].imshow(true_mask.squeeze(), cmap='jet', alpha=0.5)
    ax[1].set_title('True Mask Overlay')
    ax[1].axis('off')

    ax[2].imshow(image.squeeze(), cmap='gray')
    ax[2].imshow(predicted_mask.squeeze(), cmap='jet', alpha=0.5)
    ax[2].set_title('Predicted Mask Overlay')
    ax[2].axis('off')

    plt.show()

# Visualize the results
for (img, true_mask), pred_mask in zip(test_dataset, predictions):
    overlay_masks(img.numpy(), true_mask.numpy(), pred_mask)
