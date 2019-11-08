# Dynamic-Image
代码用于行为识别任务，将动态图作为行为的表达送入神经网络中进行分类。
本代码根据Two-stream Network的代码思路进行模仿（https://github.com/jeffreyyihuang/two-stream-action-recognition）
## Experiments information
dataset： UCF101
split_file: '01'
GPU：1080
## Code description
- dataloader.py 用于将UCF101数据集转化成dataset，并返回train_loader和test_loader
- di_cnn.py 用于定义网络结构，此处采用Resnet101结构作为基础结构
- network.py 用于定义不同层的Resnet的网络结构并加载预训练权重
- utils.py 用于计算top1，top5精度，以及对实验中的数据进行记录
- generate_dimage.py 用于生成动态图
## Comments
本代码的精度尚未达到原论文（Dynamic Image for Action Recognition）的精度，参数仍需要调整，后续会继续完善代码，有问题请联系469642692@qq.com

