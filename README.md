# Sub-Image Anomaly Detection with Deep Pyramid Correspondences in PaddlePaddle

基于PaddlePaddle复现 [Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://arxiv.org/abs/2005.02357).

***SPatially-Adaptive(SPADE)*** presents an anomaly segmentation approach which does not require a training stage.  
It is fast, robust and achieves SOTA on `MVTec AD` dataset.  

代码由PyTorch翻译至PaddlePaddle，参考[@byungjae89_SPADE-pytorch](https://github.com/byungjae89/SPADE-pytorch).感谢！

* I used K=5 nearest neighbors, which differs from the original paper K=50.


## 实验环境
硬件

* Intel(R) Xeon® CPU E5-2630v4×2
* NVIDIA Tesla V100 32G
* 256 GB RAM

系统环境

* CentOS 7.3

* Python 3.6+
* PaddlePaddle 2.0.1

若你已经下载了 [`MVTec AD`](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 数据集，请把它移动至 `data/mvtec_anomaly_detection.tar.xz`，若没有，训练启动前，脚本将自动下载该数据集。


## Bug
* Pickle 不能序列化Paddle Tensor，特征学习时无法保存feature map。
