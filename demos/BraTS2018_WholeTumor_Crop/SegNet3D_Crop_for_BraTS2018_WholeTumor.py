import os
import argparse
import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops
from data.BraTS2018_Dataset import BraTS2018_Img_Dataset
from models.SegNet3D import SegNet3D
from utils.image_utils import object_crop


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_arguments():
    parser = argparse.ArgumentParser()

    # GPU settings
    parser.add_argument('--cuda', default=True, help='Whether to use GPU')

    # Hyper parameter settings
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--roi_size', default=(112, 160, 112))

    # Data settings
    parser.add_argument('--data_path', default='~/Datasets/Brain_Tumor/BraTS2018/process_spacing1/norm_npy/valid')
    parser.add_argument('--org_npy_path', default='~/Datasets/Brain_Tumor/BraTS2018/process_spacing1/norm_npy/valid')
    parser.add_argument('--org_nii_path', default='~/Datasets/Brain_Tumor/BraTS2018/process_spacing1/resampled_nii/valid')
    parser.add_argument('--save_npy_path', default='~/Datasets/Brain_Tumor/BraTS2018/process_spacing1_crop_112_160_112/norm_npy/valid')
    parser.add_argument('--save_nii_path', default='~/Datasets/Brain_Tumor/BraTS2018/process_spacing1_crop_112_160_112/resampled_nii/valid')

    # Model settings
    parser.add_argument('--inplanes', default=4, help='The number of modality data input')
    parser.add_argument('--planes', default=32)
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--resume', default='~/Projects/Brain_Tumor_Segmentation/demos/BraTS2018_WholeTumor_Crop/results/SegNet3D/weights_500.pth')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Data settings
    data_set = BraTS2018_Img_Dataset(data_path=args.data_path, img_modality='all')
    data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)

    # Model settings
    model = SegNet3D(inplanes=args.inplanes, planes=args.planes, num_classes=args.num_classes)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=torch.device(device)))
    if args.cuda:
        model = model.cuda()

    # Seg Data
    with torch.no_grad():
        model.eval()

        for idx, data in enumerate(data_loader):
            if args.cuda:
                imgs = data['img'].to(torch.float32).cuda()
            else:
                imgs = data['img'].to(torch.float32)

            if args.cuda:
                with autocast():
                    prob = model(imgs, True)
                    pred = torch.sigmoid(prob)
                    mid = (pred.max() + pred.min()) / 2
                    pred = pred > mid
                pred = pred.cpu().data.numpy()
            else:
                prob = model(imgs, True)
                pred = torch.sigmoid(prob)
                mid = (pred.max() + pred.min()) / 2
                pred = pred > mid
                pred = pred.data.numpy()

            # 计算二值图像的连通图属性
            for i in range(pred.shape[0]):
                binary = pred[i, 0, :, :, :]
                label_image = label(binary)
                regions = regionprops(label_image)
                area_list = [region.area for region in regions]
                if area_list:
                    idx_max = np.argmax(area_list)
                    binary[label_image != idx_max + 1] = 0

                regions = regionprops(label(binary))
                D_coo = (regions[0].bbox[0] + regions[0].bbox[3]) // 2
                H_coo = (regions[0].bbox[1] + regions[0].bbox[4]) // 2
                W_coo = (regions[0].bbox[2] + regions[0].bbox[5]) // 2

                # 切割npy文件
                # 创建样本模态保存路径
                save_npy_path = os.path.join(args.save_npy_path, data['name'][i])
                if not os.path.exists(save_npy_path):
                    os.makedirs(save_npy_path)

                # 读取npy样本下的所有模态
                modality_lists = [file for file in os.listdir(os.path.join(args.org_npy_path, data['name'][i])) if file.lower().endswith('.npy')]

                # 逐个模态的切割
                for modality in modality_lists:
                    if modality[-8:] != '_seg.npy':
                        org_arr = np.load(os.path.join(args.org_npy_path, data['name'][i], modality))
                        obj_arr, err_coo, crop_coo = object_crop(org_arr=org_arr, roi_size=args.roi_size, d_coo=D_coo, h_coo=H_coo, w_coo=W_coo)

                        # 保存切割后的目标和坐标
                        np.save(os.path.join(save_npy_path, modality), arr=obj_arr)
                        coord = [err_coo[0], err_coo[1], err_coo[2], err_coo[3], err_coo[4], err_coo[5], crop_coo[0], crop_coo[1], crop_coo[2], crop_coo[3], crop_coo[4], crop_coo[5]]
                print(coord)
                np.savetxt(os.path.join(save_npy_path, data['name'][i] + '.txt'), coord, fmt='%d')

                # 切割nii文件
                # 创建样本模态保存路径
                save_nii_path = os.path.join(args.save_nii_path, data['name'][i])
                if not os.path.exists(save_nii_path):
                    os.makedirs(save_nii_path)

                # 读取nii样本下的所有模态
                modality_lists = [file for file in os.listdir(os.path.join(args.org_nii_path, data['name'][i])) if file.lower().endswith('.nii.gz')]

                # 逐个模态的切割
                for modality in modality_lists:
                    if modality[-11:] != '_seg.nii.gz':
                        org_img = sitk.ReadImage(os.path.join(args.org_nii_path, data['name'][i], modality))
                        org_arr = sitk.GetArrayFromImage(org_img)
                        obj_arr, err_coo, crop_coo = object_crop(org_arr=org_arr, roi_size=args.roi_size, d_coo=D_coo, h_coo=H_coo, w_coo=W_coo)

                        # 保存切割后的目标和坐标
                        obj_img = sitk.GetImageFromArray(obj_arr)
                        obj_img.SetSpacing(org_img.GetSpacing())
                        obj_img.SetDirection(org_img.GetDirection())
                        obj_img.SetOrigin(org_img.GetOrigin())
                        sitk.WriteImage(image=obj_img, fileName=os.path.join(save_nii_path, modality))
                        coord = [err_coo[0], err_coo[1], err_coo[2], err_coo[3], err_coo[4], err_coo[5], crop_coo[0], crop_coo[1], crop_coo[2], crop_coo[3], crop_coo[4], crop_coo[5]]
                print(coord)
                np.savetxt(os.path.join(save_nii_path, data['name'][i] + '.txt'), coord, fmt='%d')


if __name__ == '__main__':
    main()
