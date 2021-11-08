import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS2018_Cross_Validation_Dataset(Dataset):
    def __init__(self, data_path, cross_validation_index, mode='train', img_modality='all', mask_type='all'):
        self.data_path = data_path
        self.img_modality = img_modality
        self.mask_type = mask_type
        self.name_list = [file for file in os.listdir(data_path)]
        self.name_list = sorted(self.name_list)

        # The dataset split to train set and valid set for cross validation
        start_index = (len(self.name_list) // cross_validation_index[1]) * (cross_validation_index[0] - 1)
        end_index = (len(self.name_list) // cross_validation_index[1]) * cross_validation_index[0]
        if mode == 'train':
            del self.name_list[start_index: end_index]
        elif mode == 'valid':
            self.name_list = self.name_list[start_index: end_index]

    def __getitem__(self, index):
        name = self.name_list[index]
        if self.img_modality == 'flair':
            img_array = np.load(os.path.join(self.data_path, name, name + '_flair.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't1':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t1.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't1ce':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t1ce.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't2':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t2.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 'all':
            flair_array = np.load(os.path.join(self.data_path, name, name + '_flair.npy'))
            t1_array = np.load(os.path.join(self.data_path, name, name + '_t1.npy'))
            t1ce_array = np.load(os.path.join(self.data_path, name, name + '_t1ce.npy'))
            t2_array = np.load(os.path.join(self.data_path, name, name + '_t2.npy'))
            img_array = np.stack((flair_array, t1_array, t1ce_array, t2_array), axis=0)

        if self.mask_type == 'et':
            mask_array = np.load(os.path.join(self.data_path, name, name + '_et.npy'))
            mask_array = mask_array[np.newaxis, :].copy()
        elif self.mask_type == 'tc':
            mask_array = np.load(os.path.join(self.data_path, name, name + '_tc.npy'))
            mask_array = mask_array[np.newaxis, :].copy()
        elif self.mask_type == 'wt':
            mask_array = np.load(os.path.join(self.data_path, name, name + '_wt.npy'))
            mask_array = mask_array[np.newaxis, :].copy()
        elif self.mask_type == 'all':
            et_array = np.load(os.path.join(self.data_path, name, name + '_et.npy'))
            tc_array = np.load(os.path.join(self.data_path, name, name + '_tc.npy'))
            wt_array = np.load(os.path.join(self.data_path, name, name + '_wt.npy'))
            mask_array = np.stack((wt_array, tc_array, et_array), axis=0)

        img_tensor = torch.from_numpy(img_array)
        mask_tensor = torch.from_numpy(mask_array)
        data = {'img': img_tensor, 'mask': mask_tensor, 'name': name}
        return data

    def __len__(self):
        return len(self.name_list)


class BraTS2018_ImgMask_Dataset(Dataset):
    def __init__(self, data_path, img_modality='all', mask_type='wt', transforms=None):
        self.data_path = data_path
        self.img_modality = img_modality
        self.mask_type = mask_type
        self.transforms = transforms
        self.name_list = [file for file in os.listdir(data_path)]

    def __getitem__(self, index):
        name = self.name_list[index]
        if self.img_modality == 'flair':
            img_array = np.load(os.path.join(self.data_path, name, name + '_flair.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't1':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t1.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't1ce':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t1ce.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't2':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t2.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 'all':
            flair_array = np.load(os.path.join(self.data_path, name, name + '_flair.npy'))
            t1_array = np.load(os.path.join(self.data_path, name, name + '_t1.npy'))
            t1ce_array = np.load(os.path.join(self.data_path, name, name + '_t1ce.npy'))
            t2_array = np.load(os.path.join(self.data_path, name, name + '_t2.npy'))
            img_array = np.stack((flair_array, t1_array, t1ce_array, t2_array), axis=0)

        if self.mask_type == 'et':
            mask_array = np.load(os.path.join(self.data_path, name, name + '_et.npy'))
            mask_array = mask_array[np.newaxis, :].copy()
        elif self.mask_type == 'tc':
            mask_array = np.load(os.path.join(self.data_path, name, name + '_tc.npy'))
            mask_array = mask_array[np.newaxis, :].copy()
        elif self.mask_type == 'wt':
            mask_array = np.load(os.path.join(self.data_path, name, name + '_wt.npy'))
            mask_array = mask_array[np.newaxis, :].copy()
        elif self.mask_type == 'all':
            et_array = np.load(os.path.join(self.data_path, name, name + '_et.npy'))
            tc_array = np.load(os.path.join(self.data_path, name, name + '_tc.npy'))
            wt_array = np.load(os.path.join(self.data_path, name, name + '_wt.npy'))
            mask_array = np.stack((wt_array, tc_array, et_array), axis=0)

        if self.transforms is not None:
            img_array, mask_array = self.transforms(img_array, mask_array)

        img_tensor = torch.from_numpy(img_array.copy())
        mask_tensor = torch.from_numpy(mask_array.copy())
        data = {'img': img_tensor, 'mask': mask_tensor, 'name': name}
        return data

    def __len__(self):
        return len(self.name_list)


class BraTS2018_Img_Dataset(Dataset):
    def __init__(self, data_path, img_modality='all'):
        self.data_path = data_path
        self.img_modality = img_modality
        self.name_list = [file for file in os.listdir(data_path)]

    def __getitem__(self, index):
        name = self.name_list[index]
        if self.img_modality == 'flair':
            img_array = np.load(os.path.join(self.data_path, name, name + '_flair.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't1':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t1.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't1ce':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t1ce.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 't2':
            img_array = np.load(os.path.join(self.data_path, name, name + '_t2.npy'))
            img_array = img_array[np.newaxis, :].copy()
        elif self.img_modality == 'all':
            flair_array = np.load(os.path.join(self.data_path, name, name + '_flair.npy'))
            t1_array = np.load(os.path.join(self.data_path, name, name + '_t1.npy'))
            t1ce_array = np.load(os.path.join(self.data_path, name, name + '_t1ce.npy'))
            t2_array = np.load(os.path.join(self.data_path, name, name + '_t2.npy'))
            img_array = np.stack((flair_array, t1_array, t1ce_array, t2_array), axis=0)

        img_tensor = torch.from_numpy(img_array)
        data = {'img': img_tensor, 'name': name}
        return data

    def __len__(self):
        return len(self.name_list)
