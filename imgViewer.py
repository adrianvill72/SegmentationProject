import h5py
import numpy as np
import matplotlib.pyplot as plt

for i in range(1,10):
    first_path = r".\brainTumorDataPublic_1-766"
    second_path=r".mat"
    path=first_path+"\\"+str(i)+second_path
    print(path)
    with h5py.File(path, 'r') as file:
        # List all groups and datasets in the file
        def print_structure(name, obj):
            print(name, type(obj))

        file.visititems(print_structure)

    with h5py.File(path, 'r') as file:
        # Access the image dataset
        images = np.array(file['cjdata/image'])

        # For visualization, ensure the image data is correctly oriented
        # You might need to adjust this depending on how the images are stored
        plt.imshow(images, cmap='gray')
        plt.title('MRI Image')
        plt.axis('off')  # Remove axis ticks and labels
        plt.show()