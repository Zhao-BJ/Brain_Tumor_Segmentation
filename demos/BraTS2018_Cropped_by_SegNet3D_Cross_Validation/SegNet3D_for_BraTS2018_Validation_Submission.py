import os
import argparse
import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from data.BraTS2018_Dataset import BraTS2018_Img_Dataset
from models.SegNet3D import SegNet3D


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_arguments():
    parser = argparse.ArgumentParser()

    # GPU settings
    parser.add_argument('--cuda', default=True, help='Whether to use GPU')

    # Hyper parameter settings
    parser.add_argument('--batch_size', default=1)

    # Data settings
    parser.add_argument('--data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/BraTS2018/process_spacing1_crop_112_160_112/norm_npy/train')
    parser.add_argument('--org_nii_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/BraTS2018/process_spacing1_crop_112_160_112/resampled_nii/train')
    parser.add_argument('--save_path', default='/data2/aiteam_cta/ZhaoBJ/Projects/Brain_Tumor_Segmentation/demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation/results/SegNet3D/weights_1000/BraTS2018_Train')

    # Model settings
    parser.add_argument('--inplanes', default=4, help='Four modality data input stack')
    parser.add_argument('--planes', default=64)
    parser.add_argument('--num_classes', default=3)
    parser.add_argument('--resume', default='/data2/aiteam_cta/ZhaoBJ/Projects/Brain_Tumor_Segmentation/demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation/results/SegNet3D/weights_1000.pth')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Data settings
    valid_set = BraTS2018_Img_Dataset(data_path=args.data_path, img_modality='all')
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Model settings
    model = SegNet3D(inplanes=args.inplanes, planes=args.planes, num_classes=args.num_classes)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=torch.device(device)))
    if args.cuda:
        model = model.cuda()

    ##############################################################
    # valid model
    ##############################################################
    with torch.no_grad():
        model.eval()

        for idx, data in enumerate(valid_loader):
            if args.cuda:
                imgs = data['img'].to(torch.float32).cuda()
            else:
                imgs = data['img'].to(torch.float32)

            if args.cuda:
                with autocast():
                    prob = model(imgs, True)
                    pred = torch.sigmoid(prob)
                    mid = (pred.max() + pred.min()) / 2
                    pred[pred > mid] = 1.0
                    pred[pred <= mid] = 0.0
                    pred = pred.cpu().data.numpy()
            else:
                prob = model(imgs, True)
                pred = torch.sigmoid(prob)
                mid = (pred.max() + pred.min()) / 2
                pred[pred > mid] = 1.0
                pred[pred <= mid] = 0.0
                pred = pred.data.numpy()

            for i in range(pred.shape[0]):
                new_arr = np.zeros((pred.shape[2], pred.shape[3], pred.shape[4]), dtype=np.uint8)
                new_arr[pred[i, 0, :, :, :] == 1] = 2
                new_arr[pred[i, 1, :, :, :] == 1] = 1
                new_arr[pred[i, 2, :, :, :] == 1] = 4

                fin_arr = np.zeros((155, 240, 240), dtype=np.uint8)
                coord = np.loadtxt(os.path.join(args.data_path, data['name'][i], data['name'][i] + '.txt'), dtype=np.int64)
                fin_arr[coord[6] + 5: coord[7] + 5, coord[8] + 24: coord[9] + 24, coord[10] + 40: coord[11] + 40] = new_arr[coord[0]: coord[1], coord[2]: coord[3], coord[4]: coord[5]]

                org_img = sitk.ReadImage(os.path.join(args.org_nii_path, data['name'][i], data['name'][i] + '_flair.nii.gz'))
                fin_img = sitk.GetImageFromArray(fin_arr)
                fin_img.SetSpacing(org_img.GetSpacing())
                fin_img.SetOrigin(org_img.GetOrigin())
                fin_img.SetDirection(org_img.GetDirection())
                sitk.WriteImage(image=fin_img, fileName=os.path.join(args.save_path, data['name'][i] + '.nii.gz'))


if __name__ == '__main__':
    main()
