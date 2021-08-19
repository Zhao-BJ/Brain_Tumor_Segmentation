# Brain Tumor Segmentation
本项目的目标是实现大脑肿瘤的分割，所使用的模型基于PyTorch实现。

## 数据集
### MICCAI BraTS 2018 Challenge
#### 数据描述
竞赛任务是分割不同神经胶质瘤子区域，包括：`1)增强肿瘤(Enhancing Tumor, ET)`、`2)肿瘤核心(Tumor Core, TC)`、`3)整个肿瘤(Whole Tumor, WT)`。  
![](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/pictures/Glioma%20sub-regions.png "Glioma sub-regions")  
神经胶质瘤子区域。A.整个肿瘤（黄色），可见于T2-FLAIR模态。B.肿瘤核心（红色），可见于T2模态。C.增强的肿瘤结构（浅蓝色），可见于T1Gd模态，以及核心的囊性/坏死成分（绿色）。D.组合的最终标签，包括：水肿（黄）、非增强肿瘤核心（红）、核心的囊性/坏死成分（绿）、增强核心（蓝）。图片来自[BraTS IEEE TMI论文](https://ieeexplore.ieee.org/document/6975210)。

训练数据集提供的标签包括：  
`1`代表坏死和非增强肿瘤核心(necrotic and non-enhancing tumor, NCR_NET)  
`2`代表瘤周水肿(edema, ED)  
`4`代表增强肿瘤(enhancing tumor, ET)  
`0`代表其他  

TC描述了肿瘤的大部分，包括ET以及NCR_NET。WT描述了整个肿瘤，包括所有标签。

#### 数据预处理
原始标签中，NCR_NET, ED, ET是分开标注的，彼此不重叠。然而为了对三个子区域进行分割，需要对三个子区域分成3个通道表示，其中第0通道代表ET，即原标签中的`4`。第1通道代表TC，即原标签中的`1 + 4`。第2通道代表WT，即原标签中的`1 + 2 + 4`。

此外，对所有模态的数据归一化到[0, 1]，并最终保存为npy格式的文件，便于模型训练时读取。详细处理见于[data/BraTS_2018_preprocess.py](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/data/BraTS_2018_preprocess.py)