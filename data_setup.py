from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rioxarray as rio

MAX_PIXEL_VAL = 65535.0 #because satellite imagery is stored as uint16's

#pre-calculated mean and std dev for each of the six channels, using the find_mean_std function below
mean = np.array([0.01196699, 0.01541834, 0.01535424, 0.03466259, 0.02900563, 0.02068319])
std = np.array([0.0114159, 0.01277009, 0.01595061, 0.02045764, 0.0223029, 0.01961419])


root = Path('data/dset-s2')

train_imgs = list((root/'tra_scene').glob('*tif'))
train_masks = list((root/'tra_truth').glob('*tif'))

val_imgs = list((root/'val_scene').glob('*.tif'))
val_masks = list((root/'val_truth').glob('*.tif'))

# As the images and corresponding masks are matched by name, we will sort both lists to keep them synchronized.
train_imgs.sort(); train_masks.sort()
val_imgs.sort(); val_masks.sort()

def find_mean_std(img_paths):
    """Find the (approximate) mean and std dev of the pixel values across each image,
    (after making sure to scale the image pixel values between 0 and 1.)"""

    num_imgs = len(img_paths)

    #6 channels --> 6 means and std dev's
    means = np.zeros(6)
    tot_sums = np.zeros(6)
    num_pixels = 0
    tot_sum_squares = np.zeros(6)

    #find the means
    for img_path in img_paths:
        img = rio.open_rasterio(img_path)
        img.data = img.data.astype(np.float32)/MAX_PIXEL_VAL
        #find the means for each channel across the pixels
        tot_sums += np.sum(img.data, axis = (1,2))
        num_pixels += img.data.shape[-2] * img.data.shape[-1]


    means = tot_sums / num_pixels

    #second pass for the standard deviations
    for img_path in img_paths:
        img = rio.open_rasterio(img_path)
        img.data = img.data.astype(np.float32)/MAX_PIXEL_VAL
        #find the means for each channel across the pixels
        tot_sum_squares += np.sum((img.data-means[:,None,None])**2, axis = (1,2))

    std_devs = np.sqrt(tot_sum_squares/num_pixels)

    return means, std_devs

#mean, std = find_mean_std(train_imgs)

def unnormalize_img(img_arr):
    """Inverts the image normalization to obtain the unprocessed data.

    Args
    ----
    img_arr: ndarray of shape (6,width, height).

    Returns
    -------
    unnormalized image of shape (6, width, height).
    """
    img_arr = img_arr*std[:,None,None] + mean[:,None,None]
    return MAX_PIXEL_VAL * img_arr

class WaterDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        assert len(img_paths) == len(mask_paths)
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_filepath = self.img_paths[idx]
        mask_filepath = self.mask_paths[idx]

        #change to channels-last convention
        img = rio.open_rasterio(img_filepath).data
        mask = rio.open_rasterio(mask_filepath).data

        if self.transform is not None:
            img = img.transpose((1,2,0))
            mask = mask.transpose((1,2,0))
            transformed = self.transform(image=img, mask = mask)
            img = transformed['image']
            mask = transformed['mask']
        return {'image' : img, 'mask' : mask}

def create_dataloader(img_paths, mask_paths, img_size = (256,256), batch_size = 16, train = True):
    width, height = img_size
    if train:
        transform = A.Compose([
        A.ToFloat(max_value=MAX_PIXEL_VAL),
        A.RandomCrop(width=width, height=height),
        A.RandomRotate90(),
        A.Flip(),
        A.Normalize(mean = mean, std = std, max_pixel_value = 1),
        ToTensorV2(transpose_mask = True)
        ])
    else:
        transform = A.Compose([
        A.ToFloat(max_value=MAX_PIXEL_VAL),
        A.RandomCrop(width=width, height=height),
        A.Normalize(mean = mean, std = std, max_pixel_value = 1),
        ToTensorV2(transpose_mask = True)
        ])

    water_data = WaterDataset(img_paths = img_paths,
                              mask_paths = mask_paths,
                              transform = transform)

    data_loader = DataLoader(water_data,
                             batch_size = batch_size,
                             shuffle = True,
                             num_workers = 0)

    return data_loader
