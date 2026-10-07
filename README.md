FedTIDE：面向有限通信轮次的参考状态纠正联邦学习方法
本仓库为论文《面向有限通信轮次的参考状态纠正联邦学习方法》的代码实现。
FedTIDE 面向统计初始化后的有限通信优化场景，在冻结预训练特征提取器且无公开代理数据的条件下，通过固定参考模型、状态条件梯度、曲率预处理、受约束状态纠正和分布式候选评估，对联邦分类器进行追加优化。

运行环境
pip install torch torchvision numpy scipy

支持数据集
CIFAR-10
CIFAR-100
SVHN

默认随机种子：
2025, 2026, 2027

支持模式
full：完整 FedTIDE
global_only：仅使用全局方向
no_projection：移除保护投影
no_curvature：移除曲率预处理
init_fedavg：相同初始化后的分类头 FedAvg

输出文件
report.txt：实验汇总
rounds.csv：逐轮指标
mechanisms.csv：机制诊断信息

主要实验设置
Backbone：ResNet-18
Clients：30
Rounds：10
Dirichlet alpha：0.1
Seeds：2025, 2026, 2027
Feature extractor：frozen
Public proxy data：none

本实现不使用公开代理数据，客户端逐样本状态和逐样本特征不会上传至服务器。

