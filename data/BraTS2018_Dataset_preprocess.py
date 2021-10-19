"""
这里的目的是对BraTS2018数据集进行预处理，并将最后结果保存为npy文件，包括如下步骤：
重采样：统一原始图像的真实大小
填充或裁剪：对重采样后的图像填充或裁剪为指定的大小，便于神经网络模型输入
窗宽窗位调整：在MR图像选择合适的值范围，从而增强目标的对比度，血管增强
标准化：采用Z-score标准化
保存为npy：便于模型训练时数据读取

原始数据保存路径为
*/BraTS2018/original:
    train:
        HGG:
            Brats18_*:
                Brats18_*_flair.nii.gz
                Brats18_*_seg.nii.gz
                Brats18_*_t1.nii.gz
                Brats18_*_t1ce.nii.gz
                Brats18_*_t2.nii.gz
            *** （共210个）
        LGG:
            Brats18_*:
                Brats18_*_flair.nii.gz
                Brats18_*_seg.nii.gz
                Brats18_*_t1.nii.gz
                Brats18_*_t1ce.nii.gz
                Brats18_*_t2.nii.gz
            ***（共75个）
    valid:
        Brats18_*:
            Brats18_*_flair.nii.gz
            Brats18_*_t1.nii.gz
            Brats18_*_t1ce.nii.gz
            Brats18_*_t2.nii.gz
        *** （共66个）
处理后的数据保存路径为: 其中norm_npy是经过上述五个步骤处理的用于模型训练的数据，resampled_nii是处理到第二步之后的重采样图像
*/BraTS2018/process_spacing*: （这里会指定重采样后的层厚，如spacing1表示重采样后的层厚为1，spacing05表示重采样后的层厚为0.5）
    norm_npy:
        train:
            Brats18_*:
                Brats18_*_flair.npy
                Brats18_*_seg.npy
                Brats18_*_t1.npy
                Brats18_*_t1ce.npy
                Brats18_*_t2.npy
            *** （共285个）
        valid:
            Brats18_*:
                Brats18_*_flair.npy
                Brats18_*_t1.npy
                Brats18_*_t1ce.npy
                Brats18_*_t2.npy
            *** （共66个）
    resampled_nii:
        train:
            Brats18_*:
                Brats18_*_flair.nii.gz
                Brats18_*_seg.nii.gz
                Brats18_*_t1.nii.gz
                Brats18_*_t1ce.nii.gz
                Brats18_*_t2.nii.gz
            *** （共285个）
        valid:
            Brats18_*:
                Brats18_*_flair.nii.gz
                Brats18_*_t1.nii.gz
                Brats18_*_t1ce.nii.gz
                Brats18_*_t2.nii.gz
            *** （共66个）
"""
import os
import math
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom


# 对医疗图像进行重采样，仅仅需要将out_spacing替换成自己想要的输出即可
def resample(image, new_spacing=[0.5, 0.5, 0.5], interpolation_order=0):
    '''
    :param image: 3D SimpleITK volume
    :param new_spacing: 指定的统一体素大小（层厚）
    :return: 重采样之后的numpy阵列
    '''
    spacing = image.GetSpacing()                                                                         # [x, y, z]或[W, H, D]
    spacing = spacing[::-1]                                                                                    # [D, H, W]  反转列表，符合sitk导出阵列的形状

    image = sitk.GetArrayFromImage(image)                                                         # shape=(D, H, W)
    real_resize_factor = tuple([spacing[i] / new_spacing[i] for i in range(3)])
    image = zoom(image, real_resize_factor, order=interpolation_order)
    return image


#  将3D的numpy阵列填充或裁剪为指定大小的阵列
def padding_or_cropping_3D(array, new_shape=(160, 256, 256)):
    pad_index = [[0,0],[0,0],[0,0]]
    crop_index = [0, 0, 0]
    for i, i_size in enumerate(array.shape):
        if i_size < new_shape[i]:
            if (new_shape[i] - i_size) % 2 != 0:
                pad_index[i] = [math.ceil((new_shape[i] - i_size) / 2), int((new_shape[i] - i_size) / 2)]
            else:
                pad_index[i] = [int((new_shape[i] - i_size) / 2), int((new_shape[i] - i_size) / 2)]
        elif i_size > new_shape[i]:
            crop_index[i] = int((i_size - new_shape[i]) / 2)
    array = np.pad(array, pad_index, mode='constant', constant_values=(array.min(), array.min()))
    array = array[int(crop_index[0]): int(crop_index[0] + new_shape[0]), int(crop_index[1]): int(crop_index[1] + new_shape[1]), int(crop_index[2]): int(crop_index[2] + new_shape[2])]
    return array


# 调整CT图像的窗宽窗位并使用Z-score标准化，但输入数据是numpy阵列
def threshold_for_center_and_width_and_Zscore(center, width, array):
    # 阈值处理，窗宽窗位
    min = (2 * center - width) / 2.0
    array = array - min
    array[array < 0] = 0
    array[array > width] = width

    # 标准化处理，Z-score
    std = np.std(array)
    mean = np.average(array)
    array = (array - mean) / std
    return array


def convert_array_to_nii_format(array, org_nii_img, new_spacing, save_path):
    arr_sitk = sitk.GetImageFromArray(array)
    arr_sitk.SetSpacing(new_spacing)
    arr_sitk.SetDirection(org_nii_img.GetDirection())
    arr_sitk.SetOrigin(org_nii_img.GetOrigin())
    sitk.WriteImage(arr_sitk, save_path)


def get_argument():
    parser = argparse.ArgumentParser()

    # 数据集路径设置
    parser.add_argument('--org_data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/BraTS2018/original')
    parser.add_argument('--resampled_data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/BraTS2018/process_spacing1/resampled_nii')
    parser.add_argument('--npy_data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/BraTS2018/process_spacing1/norm_npy')

    # 预处理超参数
    parser.add_argument('--new_spacing', default=[1., 1., 1.])                                # 由于BraTS2018数据的原始图像层厚就是[1, 1, 1]，因此重采样被跳过
    parser.add_argument('--new_shape', default=(152, 200, 176))                           # 数据处理后的最终大小，(D, H, W)

    # 窗宽和窗位设置，MR的四个模态在不同的窗宽窗位下可能会更好
    parser.add_argument('--flair_center', default=175)
    parser.add_argument('--flair_width', default=350)
    parser.add_argument('--t1_center', default=200)
    parser.add_argument('--t1_width', default=400)
    parser.add_argument('--t1ce_center', default=50)
    parser.add_argument('--t1ce_width', default=100)
    parser.add_argument('--t2_center', default=550)
    parser.add_argument('--t2_width', default=1100)

    args = parser.parse_args()
    return args


def main():
    args = get_argument()

    # 子集（即训练集和验证集）列表
    subset_lists = [file for file in os.listdir(args.org_data_path)]
    print('子集数量为：', len(subset_lists))

    # 逐个子集处理
    for subset in subset_lists:
        print('正在处理的子集是：', subset)

        # 对训练集和验证集的处理，路径有所不同，因此分开处理
        if subset == 'train':
            # 神经胶质瘤级别（即HGG和LGG）列表
            level_lists = [file for file in os.listdir(os.path.join(args.org_data_path, subset)) if file.endswith('GG')]
            print('神经胶质瘤的级别数量为：', len(level_lists))

            # 逐个胶质瘤级别处理
            for level in level_lists:
                print('正在处理的胶质瘤级别是：', level)

                # 样本列表
                sample_lists = [file for file in os.listdir(os.path.join(args.org_data_path, subset, level))]
                print('样本数量为：', len(sample_lists))

                # 逐个样本处理
                for sample in sample_lists:
                    print('正在处理的样本是：', sample)

                    # 每个样本的模态列表
                    modality_lists = [file for file in os.listdir(os.path.join(args.org_data_path, subset, level, sample)) if file.endswith('.nii.gz')]
                    print('每个样本的模态数量为：', len(modality_lists))

                    # 创建每个样本的新nii保存路径
                    if not os.path.exists(os.path.join(args.resampled_data_path, subset, sample)):
                        os.makedirs(os.path.join(args.resampled_data_path, subset, sample))
                    # 创建每个样本的新npy保存路径
                    if not os.path.exists(os.path.join(args.npy_data_path, subset, sample)):
                        os.makedirs(os.path.join(args.npy_data_path, subset, sample))

                    # 逐个模态处理
                    for modality in modality_lists:
                        print('正在处理的模态是：', modality)

                        org_img = sitk.ReadImage(os.path.join(args.org_data_path, subset, level, sample, modality))

                        # 预处理步骤1：重采样（由于原始图像的层厚就是[1, 1, 1]，因此重采样跳过）
                        org_arr = sitk.GetArrayFromImage(org_img)

                        # 预处理步骤2：填充或裁剪
                        poc_arr = padding_or_cropping_3D(array=org_arr, new_shape=args.new_shape)
                        # 对填充或裁剪后的图像保存为新的nii.gz文件
                        convert_array_to_nii_format(array=poc_arr, org_nii_img=org_img, new_spacing=args.new_spacing, save_path=os.path.join(args.resampled_data_path, subset, sample, modality))

                        # 预处理步骤3和4：窗宽窗位调整和标准化
                        if modality[-13:] == '_flair.nii.gz':
                            cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.flair_center, width=args.flair_width, array=poc_arr)
                        elif modality[-10:] == '_t1.nii.gz':
                            cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.t1_center, width=args.t1_width, array=poc_arr)
                        elif modality[-12:] == '_t1ce.nii.gz':
                            cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.t1ce_center, width=args.t1ce_width, array=poc_arr)
                        elif modality[-10:] == '_t2.nii.gz':
                            cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.t2_center, width=args.t2_width, array=poc_arr)
                        elif modality[-11:] == '_seg.nii.gz':
                            cwZ_arr = np.zeros((3, poc_arr.shape[0], poc_arr.shape[1], poc_arr.shape[2]), dtype=np.uint8)
                            cwZ_arr[0, poc_arr == 4] = 1                                                 # 通道0是增强肿瘤（ET）
                            cwZ_arr[1, poc_arr == 1] = 1
                            cwZ_arr[1, poc_arr == 4] = 1                                                 # 通道1是肿瘤核心（TC），包括增强肿瘤、坏死和非增强肿瘤（NCR_NET）
                            cwZ_arr[2, poc_arr > 0] = 1                                                   # 通道2是整个肿瘤（WT），包括增强肿瘤、坏死和非增强肿瘤、瘤周水肿（ED）
                        cwZ_arr = np.float32(cwZ_arr)

                        # 预处理步骤5：保存为npy
                        np.save(os.path.join(args.npy_data_path, subset, sample, modality[:-7] + '.npy'), arr=cwZ_arr)

        elif subset == 'valid':
            # 样本列表
            sample_lists = [file for file in os.listdir(os.path.join(args.org_data_path, subset)) if file.endswith('_1')]
            print('样本数量为：', len(sample_lists))

            # 逐个样本处理
            for sample in sample_lists:
                print('正在处理的样本是：', sample)

                # 每个样本的模态列表
                modality_lists = [file for file in os.listdir(os.path.join(args.org_data_path, subset, sample)) if file.endswith('.nii.gz')]
                print('每个样本的模态数量为：', len(modality_lists))

                # 创建每个样本的新nii保存路径
                if not os.path.exists(os.path.join(args.resampled_data_path, subset, sample)):
                    os.makedirs(os.path.join(args.resampled_data_path, subset, sample))
                # 创建每个样本的新npy保存路径
                if not os.path.exists(os.path.join(args.npy_data_path, subset, sample)):
                    os.makedirs(os.path.join(args.npy_data_path, subset, sample))

                # 逐个模态处理
                for modality in modality_lists:
                    print('正在处理的模态是：', modality)

                    org_img = sitk.ReadImage(os.path.join(args.org_data_path, subset, sample, modality))

                    # 预处理步骤1：重采样（由于原始图像的层厚就是[1, 1, 1]，因此重采样跳过）
                    org_arr = sitk.GetArrayFromImage(org_img)

                    # 预处理步骤2：填充或裁剪
                    poc_arr = padding_or_cropping_3D(array=org_arr, new_shape=args.new_shape)
                    # 对填充或裁剪后的图像保存为新的nii.gz文件
                    convert_array_to_nii_format(array=poc_arr, org_nii_img=org_img, new_spacing=args.new_spacing, save_path=os.path.join(args.resampled_data_path, subset, sample, modality))

                    # 预处理步骤3和4：窗宽窗位调整和标准化
                    if modality[-13:] == '_flair.nii.gz':
                        cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.flair_center, width=args.flair_width, array=poc_arr)
                    elif modality[-10:] == '_t1.nii.gz':
                        cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.t1_center, width=args.t1_width, array=poc_arr)
                    elif modality[-12:] == '_t1ce.nii.gz':
                        cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.t1ce_center, width=args.t1ce_width, array=poc_arr)
                    elif modality[-10:] == '_t2.nii.gz':
                        cwZ_arr = threshold_for_center_and_width_and_Zscore(center=args.t2_center, width=args.t2_width, array=poc_arr)
                    cwZ_arr = np.float32(cwZ_arr)

                    # 预处理步骤5：保存为npy
                    np.save(os.path.join(args.npy_data_path, subset, sample, modality[:-7] + '.npy'), arr=cwZ_arr)


if __name__ == '__main__':
    main()
