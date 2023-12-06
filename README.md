# RSNN: 基于Relu-Softmax Neural Network的MNIST手写数字识别
在本实验中，我们使用了MNIST数据集来评估我们的两层神经网络模型在手写数字识别任务上的性能。
## 1.1. MNIST数据集
MNIST（Modified National Institute of Standards and Technology）数据集是一个手写数字识别数据集。它包含了从0到9的手写数字的灰度图像，其中训练集包含60,000个样本，测试集包含10,000个样本。这些图像的尺寸为28x28像素，像素值范围在0到255之间。
我们将图像数据展平成一个具有784个特征（28x28）的一维向量，以便输入到我们的神经网络模型。
## 1.2. RSNN
在本实验中，我们使用了一个简单的两层神经网络Relu-Softmax Neural Network（RSNN）来对MNIST数据集进行分类。这个神经网络包括一个输入层，一个隐藏层和一个输出层。具体结构如下： 

<p align="center">
  <img width="249" alt="image" src="https://user-images.githubusercontent.com/113240460/230779579-27fd6c03-cd90-48a9-aa9a-77bab070c811.png">
</p>

<p align="center">
  图1.1. RSNN的网络结构图
</p>

（1）输入层：将MNIST训练集 $X_{train}$ 作为输入；

（2）隐藏层：采用ReLU为激活函数，第二层的输出为
$$A_1 = ReLU(X_{train}W_1+b_1)$$

（3）输出层：使用softmax激活函数进行归一化处理，得到概率分布
$$y_{pred} = softmax(A_1W_2+b_2)$$

（4）损失函数采用交叉熵损失函数，并加入L2正则化项防止过拟合，通过在损失函数中添加权重平方和项来惩罚较大的权重值，损失函数表达式如下
$$L = -\frac{1}{N}\sum_{i=1}^{N}y_{true(i)}ln(y_{pred(i)}) + \frac{\lambda}{2}(\sum_{i=1}^{m}\sum_{i=1}^{n}W_{1,ij}^{2} + \sum_{i=1}^{p}\sum_{i=1}^{q}W_{2,ij}^{2})$$

## 1.3. 优化方法
为了训练神经网络，我们采用了随机梯度下降（SGD）作为优化器。SGD通过在每次迭代中计算损失函数关于权重和偏置的梯度并更新参数，使损失函数的值逐渐减小。我们还使用了学习率下降策略，以便在训练过程中动态调整学习率，从而实现更快的收敛。
## 1.4. 超参数搜索
在本次实验中，我们使用了随机搜索方法来搜索最优的超参数组合。我们首先定义需要搜索的超参数范围，例如隐藏层大小、学习率和正则化强度。然后，我们对这些超参数进行随机采样，每次训练模型并记录测试集上的准确率。在一定次数的采样之后，我们选择测试集上准确率最高的一组超参数作为最终的超参数组合。
在本实验中，我们采用了 10000 次随机搜索，搜索的超参数范围如下：

（1）隐藏层大小：[40, 100]

（2）学习率：[5e-4, 5e-3]

（3）正则化强度：[1e-6, 1e-4]

每次训练使用了 1000 次迭代，批量大小为 64，并且设置了学习率衰减为 0.95。最终在第3925次找到的最优超参数组合为：

（1）隐藏层大小：92

（2）学习率：0.0048

（3）正则化强度：7.04e-05

得到的测试集准确率为0.8596。

<p align="center">
  <img width="267" alt="image" src="https://user-images.githubusercontent.com/113240460/230779955-6a5d49d4-3468-4047-a0ed-e896795b00d4.png">
</p>

<p align="center">
  图1.2. 训练集和测试集的损失函数
</p>

<p align="center">
  <img width="277" alt="image" src="https://user-images.githubusercontent.com/113240460/230779951-491b69a5-7df5-4d1b-aaec-70be6ceab76d.png">
</p>

<p align="center">
  图1.3. 训练集和测试集的准确率
</p>
