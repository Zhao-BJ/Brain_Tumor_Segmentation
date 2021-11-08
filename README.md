## Brain Tumor Segmentation

本项目的目标是实现大脑肿瘤的分割，所使用的模型基于PyTorch实现。

中文 | English

## 数据集

**BraTS2018**：MICCAI BraTS 2018 Challenge 数据集，任务是分割不同的神经胶质瘤子区域。详细信息及预处理见：[data/BraTS2018_Dataset_preprocess.md](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/data/BraTS2018_Dataset_preprocess.md)

## 实验

### BraTS2018的SegNet3D两阶段分割

在这个实验中，我们使用SegNet3D网络对BraTS2018数据集进行两阶段的分割。具体来说，由于脑肿瘤的目标只有一个，因此，第一阶段中，我们对整个肿瘤进行定位，然后计算出整个肿瘤的中心坐标，再根据经验设置一个合适的裁剪大小，以整个肿瘤的中心坐标为中心进行裁剪。然后第二阶段中，以裁剪后的图像块为输入训练肿瘤分割。SegNet3D网络的详细信息见：[model/SegNet3D.md]()。

对BraTS2018的预处理中，会将图像裁剪为$ (D, H, W) = (144, 192, 160) $的图像块，以这个大小输入SegNet3D网络进行整个肿瘤的定位，并在第一阶段中会根据定位结果裁剪图像为$ (D, H, W) = (112, 160, 112) $，以此来增加肿瘤目标在整个图像中的占比。

第一阶段的整个肿瘤定位是通过训练网络对整个肿瘤分割实现的，我们可以设置裁剪的大小来确保裁剪到整个肿瘤的区域，因此，肿瘤定位不需要太精确，只要能找到肿瘤的位置即可。第一阶段的代码位于[demos/BraTS2018_WholeTumor_Crop](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/tree/main/demos/BraTS2018_WholeTumor_Crop)包内，定位网络训练代码于：[SegNet3D.py](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/demos/BraTS2018_WholeTumor_Crop/SegNet3D.py)。然后对训练好的模型，使用定位裁剪脚本代码获得新的的数据：[SegNet3D_Crop_for_BraTS2018_WholeTumor.py](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/demos/BraTS2018_WholeTumor_Crop/SegNet3D_Crop_for_BraTS2018_WholeTumor.py)。注意，在裁剪的过程中会保存裁剪的坐标，这个坐标用来还原原始图像的大小。此外，对有标签的数据集，可以使用脚本[SegNet3D_Valid_BraTS2018_WholeTumor_Crop.py](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/demos/BraTS2018_WholeTumor_Crop/SegNet3D_Valid_BraTS2018_WholeTumor_Crop.py)验证整个肿瘤的分割性能。

第二阶段的肿瘤子区域分割代码位于[demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation)包内，我们使用整个训练集或5-折交叉验证进行训练，然后在竞赛系统计算训练集和验证集的评价指标。第二阶段的训练代码为[SegNet3D.py](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation/SegNet3D.py)。计算并生成最终结果的脚本为[SegNet3D_for_BraTS2018_Validation_Submission.py](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/demos/BraTS2018_Cropped_by_SegNet3D_Cross_Validation/SegNet3D_for_BraTS2018_Validation_Submission.py)。

竞赛提交系统地址为：https://ipp.cbica.upenn.edu/categories/brats2018。其中，本实验最终获得的训练集结果为：ET Dice = 0.92949, WT Dice = 0.99216, TC Dice = 0.99038。验证集结果为：ET Dice = 0.57008, WT Dice = 0.78826, TC Dice = 0.66277。可以看出目前严重过拟合了。

