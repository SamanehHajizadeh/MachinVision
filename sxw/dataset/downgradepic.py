import json
from os.path import join

import cv2
import matplotlib.pyplot as plt

class ImageDataset():
    def __init__(self, dataset_directory='clean'):
        self.dataset_directory = dataset_directory
        with open(join(self.dataset_directory, 'metadata.json')) as f:
            self.metadata = json.load(f)
        self.names = list(self.metadata.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            response = []
            for i in range(index[0], index[1]):
                response.append((self.metadata[self.names[i]], cv2.imread(join(self.dataset_directory, self.names[i]))))
            return response
        elif isinstance(index, str):
            data = self.metadata[index]
            img = cv2.imread(join(self.dataset_directory, index))
            return img, data
        data = self.metadata[self.names[index]]
        img = cv2.imread(join(self.dataset_directory, self.names[index]))
        return img, data

def get_device(img, meta):
    device = meta['device']
    return img[device['y']:device['y'] + device['height'], device['x']:device['x'] + device['width']]

def resize_img(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimensions = (width, height)
    return cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)


imgAndDataLoading = ImageDataset()

img, data1 = imgAndDataLoading[0]

# Crop the picture and specify the device.
device = get_device(img, data1)
#plt.imshow(device)
#plt.show()

#Save actual size of image
shape = device.shape
print(shape)

# Resize the image and decide on percentage of rescaling
img_resized = resize_img(device, 0.10)
#plt.imshow(img_resized)
#plt.show()


#After downgrading the quality, You can back to actual size of image after downgrading the quality
resize2 = cv2.resize(img_resized, (shape[1], shape[0]))
#plt.imshow(resize2)
#plt.show()


#Run the script for whole dataset

for img, data1 in imgAndDataLoading:
    # Crop the picture and specify the device.
    device = get_device(img, data1)
    #Save actual size of image
    shape = device.shape
    # Resize the image and decide on percentage of rescaling
    img_resized = resize_img(device, 0.10)
    #After downgrading the quality, You can back to actual size of image after downgrading the quality
    resize2 = cv2.resize(img_resized, (shape[1], shape[0]))
    #Show the output in plt
    #plt.imshow(resize2)
    #plt.show()
    #generate downside pic
    cv2.imwrite('output/' + data1['filename'] + '.png', resize2)
    print(data1['filename']  + "Finished!")