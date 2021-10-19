## BraTS2018 数据集

### MICCAI BraTS 2018 Challenge

#### 数据描述

竞赛任务是分割不同神经胶质瘤子区域，包括：`1)增强肿瘤(Enhancing Tumor, ET)`、`2)肿瘤核心(Tumor Core, TC)`、`3)整个肿瘤(Whole Tumor, WT)`。
![Glioma sub-regions](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/pictures/Glioma%20sub-regions.png)
神经胶质瘤子区域。A.整个肿瘤（黄色），可见于T2-FLAIR模态。B.肿瘤核心（红色），可见于T2模态。C.增强的肿瘤结构（浅蓝色），可见于T1Gd模态，以及核心的囊性/坏死成分（绿色）。D.组合的最终标签，包括：水肿（黄）、非增强肿瘤核心（红）、核心的囊性/坏死成分（绿）、增强核心（蓝）。图片来自[BraTS IEEE TMI论文](https://ieeexplore.ieee.org/document/6975210)。

训练数据集提供的标签包括：
`1`代表坏死和非增强肿瘤核心(necrotic and non-enhancing tumor, NCR_NET)
`2`代表瘤周水肿(edema, ED)
`4`代表增强肿瘤(enhancing tumor, ET)
`0`代表其他

TC描述了肿瘤的大部分，包括ET以及NCR_NET。WT描述了整个肿瘤，包括所有标签。

#### 数据预处理

原始标签中，NCR_NET, ED, ET是分开标注的，彼此不重叠。然而为了对三个子区域进行分割，需要对三个子区域分成3个通道表示，其中第0通道代表ET，即原标签中的`4`。第1通道代表TC，即原标签中的`1 + 4`。第2通道代表WT，即原标签中的`1 + 2 + 4`。

对BraTS2018数据集的图像预处理，包括如下步骤：

* 重采样：统一原始图像的真实大小
* 填充或裁剪：对重采样后的图像填充或裁剪为指定的大小，便于神经网络模型输入
* 窗宽窗位调整：在MR图像中选择合适的值范围，从而增强目标的对比度
* 标准化：采用Z-score标准化
* 保存为npy：便于模型训练时数据读取

`重采样`：BraTS2018数据集的原始图像的训练集和验证集的层厚就是[1, 1, 1]，因此我们不再重采样。

`填充或裁剪`：原始图像的像素尺寸为（155，240，240），我们根据经验，将图像中大脑无关的边缘部分裁剪掉，并指定裁剪后的图像大小为（152，200，176）。

`窗宽窗位调整`：我们根据经验，选择不同模态合适的值范围，即$ flair \in [0, 350], t1 \in [0, 400], t1ce \in [0, 100], t2 \in [0, 1100] $。依据这个范围设置窗宽窗位。

数据集处理前后需按固定结构存储，其存储的路径结构如下：

```python
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
                Brats18_*_seg.nii.gz
                Brats18_*_t1.nii.gz
                Brats18_*_t1ce.nii.gz
                Brats18_*_t2.nii.gz
            *** （共66个）
```

