# Brain Tumor Segmentation
本项目的目标是实现大脑肿瘤的分割，所使用的模型基于PyTorch实现。

## 数据集
### MICCAI BraTS 2018 Challenge
竞赛任务是分割不同神经胶质瘤子区域，包括：`1)增强肿瘤(Enhancing Tumor, ET)`、`2)肿瘤核心(Tumor Core, TC)`、`3)整个肿瘤(Whole Tumor, WT)`。
![](https://github.com/Zhao-BJ/Brain_Tumor_Segmentation/blob/main/pictures/Glioma%20sub-regions.png "Glioma sub-regions")
神经胶质瘤子区域。A.整个肿瘤（黄色），可见于T2-FLAIR模态。B.肿瘤核心（红色），可见于T2模态。C.增强的肿瘤结构（浅蓝色），可见于T1Gd模态，以及核心的囊性/坏死成分（绿色）。D.组合的最终标签，包括：水肿（黄）、非增强肿瘤核心（红）、核心的囊性/坏死成分（绿）、增强核心（蓝）。图片来自BraTS IEEE TMI论文。