import os
import logging
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from data.BraTS2018_Dataset import BraTS2018_ImgMask_Dataset
from evaluators.seg_evaluators import calculate_per_channel_dice_score
from losses.seg_losses import SoftDiceLoss
from models.SegNet3D import SegNet3D
from utils.train_utils import trainlog


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_arguments():
    parser = argparse.ArgumentParser()

    # GPU settings
    parser.add_argument('--cuda', default=True, help='Whether to use GPU')

    # Hyper parameter settings
    parser.add_argument('--batch_size', default=1)

    # Data settings
    parser.add_argument('--data_path', default='/data2/aiteam_cta/ZhaoBJ/Datasets/Brain_Tumor/BraTS2018/process_spacing1/norm_npy/train')
    parser.add_argument('--save_path', default='/data2/aiteam_cta/ZhaoBJ/Projects/Brain_Tumor_Segmentation/demos/BraTS2018_WholeTumor_Crop/results/SegNet3D')

    # Model settings
    parser.add_argument('--inplanes', default=4, help='The number of modality data input')
    parser.add_argument('--planes', default=32)
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--resume', default='/data2/aiteam_cta/ZhaoBJ/Projects/Brain_Tumor_Segmentation/demos/BraTS2018_WholeTumor_Crop/results/SegNet3D/weights_500.pth')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # Data settings
    valid_set = BraTS2018_ImgMask_Dataset(data_path=args.data_path, img_modality='all', mask_type='wt')
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logfile = "%s/weightes_500.log" % args.save_path
    trainlog(logfile)

    # Model settings
    model = SegNet3D(inplanes=args.inplanes, planes=args.planes, num_classes=args.num_classes)
    if args.resume is not None:
        logging.info('resuming finetune from %s' % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=torch.device(device)))
    dice_criterion = SoftDiceLoss()
    entropy_criterion = nn.BCEWithLogitsLoss()
    if args.cuda:
        model = model.cuda()
        dice_criterion = dice_criterion.cuda()
        entropy_criterion = entropy_criterion.cuda()

    ##############################################################
    # valid model
    ##############################################################
    with torch.no_grad():
        model.eval()
        dice_loss_total = 0.0
        entropy_loss_total = 0.0
        wt_dice_total = 0.0
        count_total = 0.0

        for idx, data in enumerate(valid_loader):
            if args.cuda:
                imgs = data['img'].to(torch.float32).cuda()
                mask = data['mask'].to(torch.float32).cuda()
            else:
                imgs = data['img'].to(torch.float32)
                mask = data['mask'].to(torch.float32)
            count_total += imgs.size(0)

            if args.cuda:
                with autocast():
                    prob = model(imgs, True)
                    dice_loss = dice_criterion(torch.sigmoid(prob), mask)
                    entropy_loss = entropy_criterion(prob, mask)
            else:
                prob = model(imgs, True)
                dice_loss = dice_criterion(torch.sigmoid(prob), mask)
                entropy_loss = entropy_criterion(prob, mask)
            dice_loss_total += dice_loss.item()
            entropy_loss_total += entropy_loss.item()

            for i in range(prob.size()[0]):
                wt_dice = calculate_per_channel_dice_score(torch.sigmoid(prob[i, 0, :, :, :]), mask[i, 0, :, :, :], cuda=args.cuda)
                if wt_dice.item() < 0.7:
                    print('Dice低于0.4的图像是：', data['name'][i], wt_dice.item())
                wt_dice_total += wt_dice.item()

        dice_loss = dice_loss_total / count_total
        entropy_loss = entropy_loss_total / count_total
        wt_dice = wt_dice_total / count_total
        logging.info('WT Dice:%.4f |Dice Loss:%.4f |Entropy Loss:%.4f' % (wt_dice, dice_loss, entropy_loss))


if __name__ == '__main__':
    main()
