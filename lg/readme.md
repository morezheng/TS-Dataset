# LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script

## 简单说明

该数据集由麦克马斯特大学的Philip Kollmeyer等人提供，包含了一个全新的3Ah LG HG2电池在特定测试条件下的电压、电流、温度等数据，这些数据被用于设计基于深度前馈神经网络（FNN）方法的电池荷电状态（SOC）估计器。数据集还包括了数据采集、数据准备以及FNN示例脚本的开发。

## 目录及文件结构

下载并解压数据集后，文件夹结构如下：

- `FNN_xEV_Li_ion_SOC_EstimatorScript_March_2020.mlx`：FNN示例脚本文件，用于训练和测试SOC估计器。
- `data`：包含测试数据的文件夹，具体文件结构如下：
  - `training_data.mat`：训练数据文件。
  - `testing_data.mat`：测试数据文件。
  - `validation_data.mat`：验证数据文件。

## 数据集说明

### 数据集样本数

- 训练数据样本数：[具体数值]
- 测试数据样本数：[具体数值]
- 验证数据样本数：[具体数值]

### 数据集频率

数据采集的频率为[具体频率]，即每[时间间隔]记录一次数据。

### 变量说明

| 变量名称                         | 描述                                           |
| -------------------------------- | ---------------------------------------------- |
| 电压（Voltage）                  | 电池在测试过程中的电压值，单位为伏特（V）。    |
| 电流（Current）                  | 电池在测试过程中的电流值，单位为安培（A）。    |
| 温度（Temperature）              | 电池在测试过程中的温度值，单位为摄氏度（°C）。 |
| 荷电状态（State of Charge, SOC） | 电池的剩余电量百分比，范围为0%到100%。         |
| 容量（Capacity）                 | 电池在测试过程中的容量值，单位为安时（Ah）。   |

## 附录

### 数据集来源

数据集来源于Mendeley Data，具体链接为：[LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script](https://data.mendeley.com/datasets/cp3473x7xv/3)

### 引用信息

如果使用该数据集，请引用以下文献：

- C. Vidal, P. Kollmeyer, M. Naguib, P. Malysz, O. Gross, and A. Emadi, “Robust xEV Battery State-of-Charge Estimator Design using Deep Neural Networks,” in Proc WCX SAE World Congress Experience, Detroit, MI, Apr 2020.
- C. Vidal, P. Kollmeyer, E. Chemali and A. Emadi, "Li-ion Battery State of Charge Estimation Using Long Short-Term Memory Recurrent Neural Network with Transfer Learning," 2019 IEEE Transportation Electrification Conference and Expo (ITEC), Detroit, MI, USA, 2019, pp. 1-6.

请根据实际情况填写数据集样本数和数据集频率的具体数值。