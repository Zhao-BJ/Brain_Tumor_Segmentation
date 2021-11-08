import os
import time
import logging
import visdom
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.BraTS2018_Dataset import BraTS2018_Cross_Validation_Dataset
from evaluators.seg_evaluators import calculate_per_channel_dice_score
from losses.seg_losses import SoftDiceLoss
from models.SegNet3D import SegNet3D
from utils.train_utils import reproducibility, trainlog, dt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 202111


def get_arguments():
    parser = argparse.ArgumentParser()

    # GPU settings
    parser.add_argument('--cuda', default=True, help='Whether to use GPU')

    # Hyper parameter settings
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--learning_rate', default=0.0002)
    parser.add_argument('--epoch_num', default=500)

    # Data settings
    parser.add_argument('--data_path', default='~/Datasets/Brain_Tumor/BraTS2018/process_spacing1_crop_112_160_112/norm_npy/train')
    parser.add_argument('--cross_validation_index', default=[1, 5], help='Split original dataset to train set and valid set')
    parser.add_argument('--save_path', default='~/Projects/Brain_Tumor_Segmentation/demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation/results/SegNet3D_1_5')

    # Model settings
    parser.add_argument('--inplanes', default=4, help='Four modality data input stack')
    parser.add_argument('--planes', default=32)
    parser.add_argument('--num_classes', default=3)
    parser.add_argument('--resume', default=None)

    # Visualization settings
    parser.add_argument('--visdom_env', default='BraTS2018_Cropped_by_SegNet3D_Cross_Validation/SegNet3D_1_5')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    reproducibility(cuda=args.cuda, seed=seed)

    # Visualization settings
    vis = visdom.Visdom('localhost', env=args.visdom_env)

    # Data settings
    train_set = BraTS2018_Cross_Validation_Dataset(data_path=args.data_path, cross_validation_index=args.cross_validation_index, mode='train', img_modality='all', mask_type='all')
    valid_set = BraTS2018_Cross_Validation_Dataset(data_path=args.data_path, cross_validation_index=args.cross_validation_index, mode='valid', img_modality='all', mask_type='all')
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logfile = "%s/trainlog.log" % args.save_path
    trainlog(logfile)

    # Model settings
    model = SegNet3D(inplanes=args.inplanes, planes=args.planes, num_classes=args.num_classes)
    if args.resume is not None:
        logging.info('resuming finetune from %s' % args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=torch.device(device)))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    dice_criterion = SoftDiceLoss()
    entropy_criterion = nn.BCEWithLogitsLoss()
    if args.cuda:
        model = model.cuda()
        dice_criterion = dice_criterion.cuda()
        entropy_criterion = entropy_criterion.cuda()

    # Train
    logging.info("==" * 30)
    logging.info(dt())
    start_time = time.time()

    for epoch in range(1, args.epoch_num + 1):
        ##############################################################
        # train model
        ##############################################################
        model.train()
        for idx, data in enumerate(train_loader):
            if args.cuda:
                imgs = data['img'].to(torch.float32).cuda()
                mask = data['mask'].to(torch.float32).cuda()
            else:
                imgs = data['img'].to(torch.float32)
                mask = data['mask'].to(torch.float32)

            optimizer.zero_grad()
            if args.cuda:
                with autocast():
                    prob = model(imgs)
                    dice_loss = dice_criterion(torch.sigmoid(prob), mask)
                    entropy_loss = entropy_criterion(prob, mask)
                    loss = dice_loss + entropy_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                prob = model(imgs)
                dice_loss = dice_criterion(torch.sigmoid(prob), mask)
                entropy_loss = entropy_criterion(prob, mask)
                loss = dice_loss + entropy_loss
                loss.backward()
                optimizer.step()

            if (idx + 1) % 10 == 0:
                print('Epoch:%3d |idx:%3d |Dice Loss:%.3f |Entropy Loss:%.3f' % (epoch, idx + 1, dice_loss.item(), entropy_loss.item()))

        ##############################################################
        # valid model
        ##############################################################
        with torch.no_grad():
            model.eval()
            dice_loss_total = 0.0
            entropy_loss_total = 0.0
            et_dice_total = 0.0
            tc_dice_total = 0.0
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
                    if wt_dice.item() < 0.4:
                        print('WT Dice低于0.4的图像是：', data['name'][i], wt_dice.item())
                    wt_dice_total += wt_dice.item()
                    tc_dice = calculate_per_channel_dice_score(torch.sigmoid(prob[i, 1, :, :, :]), mask[i, 1, :, :, :], cuda=args.cuda)
                    if tc_dice.item() < 0.4:
                        print('TC Dice低于0.4的图像是：', data['name'][i], tc_dice.item())
                    tc_dice_total += tc_dice.item()
                    et_dice = calculate_per_channel_dice_score(torch.sigmoid(prob[i, 2, :, :, :]), mask[i, 2, :, :, :], cuda=args.cuda)
                    if et_dice.item() < 0.4:
                        print('ET Dice低于0.4的图像是：', data['name'][i], et_dice.item())
                    et_dice_total += et_dice.item()

            dice_loss = dice_loss_total / count_total
            entropy_loss = entropy_loss_total / count_total
            et_dice = et_dice_total / count_total
            tc_dice = tc_dice_total / count_total
            wt_dice = wt_dice_total / count_total
            logging.info('Epoch:%3d |WT Dice:%.4f |TC Dice:%.4f |ET Dice:%.4f |Dice Loss:%.4f |Entropy Loss:%.4f' % (epoch, wt_dice, tc_dice, et_dice, dice_loss, entropy_loss))
            vis.line(Y=[[wt_dice, tc_dice, et_dice]], X=[epoch], win='dice', update=None if epoch == 1 else 'append', opts=dict(title='dice', legend=['wt_dice', 'tc_dice', 'et_dice']))
            vis.line(Y=[[dice_loss, entropy_loss]], X=[epoch], win='loss', update=None if epoch == 1 else 'append', opts=dict(title='loss', legend=['dice_loss', 'entropy_loss']))


            ############################################################
            # save model
            ############################################################
            if et_dice > 0.6:
                save_model_path = os.path.join(args.save_path, 'weights_%d.pth' % epoch)
                torch.save(model.state_dict(), save_model_path)
                logging.info('saved model to %s' % save_model_path)

            logging.info("Time elapsed: %4.2f" % ((time.time() - start_time) / 60))
            logging.info('--' * 30)


if __name__ == '__main__':
    main()
