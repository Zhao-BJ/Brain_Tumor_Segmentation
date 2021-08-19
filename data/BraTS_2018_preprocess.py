"""
This script file aim to process original MICCAI BraTS 2018 dataset to npy file.
original dataset structure:
MICCAI_BraTS_2018_Data_Train:
    original:
        HGG:
            cases:
                *_t1.nii.gz
                *_t1ce.nii.gz
                *_t2.nii.gz
                *_flair.nii.gz
                *_seg.nii.gz
            ***
        LGG:
            cases:
                *_t1.nii.gz
                *_t1ce.nii.gz
                *_t2.nii.gz
                *_flair.nii.gz
                *_seg.nii.gz
            ***
        survival_data.csv

process dataset structure:
MICCAI_BraTS_2018_Data_Train:
    process:
        HGG:
            cases:
                *_t1.npy
                *_t1ce.npy
                *_t2.npy
                *_flair.npy
                *_seg.npy
            ***
        LGG:
            cases:
                *_t1.npy
                *_t1ce.npy
                *_t2.npy
                *_flair.npy
                *_seg.npy
            ***
"""
import os
import argparse
import numpy as np
import nibabel as nib


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/MICCAI_BraTS_2018_Data_Training/original')
    parser.add_argument('--npy_data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/MICCAI_BraTS_2018_Data_Training/process/norm_npy')

    args = parser.parse_args()
    return args


def normalize_intensity(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def main():
    args = get_argument()

    # tumor level list (include HGG and LGG)
    levels_list = [file for file in os.listdir(args.org_data_path) if file.endswith('GG')]
    print('The tumor level number is: ', len(levels_list))

    # level process
    for level in levels_list:
        # cases list
        cases_list = [file for file in os.listdir(os.path.join(args.org_data_path, level))]
        print('The case number is: ', len(cases_list))

        # case level process
        for case in cases_list:

            # modality list
            modalities_list = [file for file in os.listdir(os.path.join(args.org_data_path, level, case)) if file.endswith('.nii.gz')]
            print('The modality number is: ', len(modalities_list))

            # modality process
            for modality in modalities_list:
                if modality[-10:] == 'seg.nii.gz':                                                           # the label process
                    nii = nib.load(os.path.join(args.org_data_path, level, case, modality))
                    arr = nii.get_fdata()
                    new_arr = np.zeros((3, arr.shape[0], arr.shape[1], arr.shape[2]), dtype=np.float32)
                    new_arr[0, arr == 4] = 1                                                                # channel 0 is Enhancing Tumor (ET)
                    new_arr[1, arr == 1] = 1
                    new_arr[1, arr == 4] = 1                                                                # channel 1 is Tumor Core (TC), entails NCR_NET and ET
                    new_arr[2, arr > 0] = 1                                                                  # channel 2 is Whole Tumor (WT), entals NCR_NET, ED and ET
                else:                                                                                                  # other modality process
                    nii = nib.load(os.path.join(args.org_data_path, level, case, modality))
                    arr = nii.get_fdata()                                                                        # get original data array

                    # remove a small number of outliers with large values
                    low = np.percentile(arr, 0.01)
                    high = np.percentile(arr, 99.9)
                    arr[arr < low] = low
                    arr[arr > high] = high

                    arr = normalize_intensity(arr)                                                          # normalized to [0, 1]

                    new_arr = np.float32(arr)

                # the processed data save as npy data to accelerate subsequent model training
                npy_path = os.path.join(args.npy_data_path, level, case)
                if not os.path.exists(npy_path):
                    os.makedirs(npy_path)
                np.save(os.path.join(npy_path, modality[:-7] + '.npy'), new_arr)


if __name__ == '__main__':
    main()
