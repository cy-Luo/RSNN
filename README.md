# RSNN
基于Relu-Softmax Neural Network的MNIST手写数字识别
在本实验中，我们使用了MNIST数据集来评估我们的两层神经网络模型在手写数字识别任务上的性能。
## 1.1. MNIST数据集
MNIST（Modified National Institute of Standards and Technology）数据集是一个手写数字识别数据集。它包含了从0到9的手写数字的灰度图像，其中训练集包含60,000个样本，测试集包含10,000个样本。这些图像的尺寸为28x28像素，像素值范围在0到255之间。
我们将图像数据展平成一个具有784个特征（28x28）的一维向量，以便输入到我们的神经网络模型。
## 1.2. RSNN
在本实验中，我们使用了一个简单的两层神经网络Relu-Softmax Neural Network（RSNN）来对MNIST数据集进行分类。这个神经网络包括一个输入层，一个隐藏层和一个输出层。具体结构如下：
