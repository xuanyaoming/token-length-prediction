# token-length-prediction

## 介绍
这是论文【Predicting Token Length for Large Language Models with Multi-Branch Headers】的官方代码仓库

## 一、安装
```
conda create -n token-length-prediction python=3.11
conda activate token-length-prediction
pip install transformers==4.46.3
pip install trl==0.12.2
pip install deepspeed
pip install accelerate
pip install torchtune
pip install tensorboard
```
如果安装失败，可尝试：
```
pip install -r requirements.txt
```

## 二、实验

### 2.1 训练模型
论文中实现了两种训练策略，其中ToDS训练慢但表现好，TeDS训练快但是效果非常差。在实验前，
需要确保accelerate被正确配置。论文中的实验大多实在4张80GB A100的环境下进行，如果尝试
在其他硬件环境下进行实验，请在代码中更改对应的配置。
```
# 配置accelerate
accelerate config

# deepspeed 配置路径：configs\zero2_config.json
```
#### 2.1.1 ToDS 训练
STEP 1: 更改`train.py`最后一行为`train_in_tods(batch_size=80, set="gripper")`，其中 set 取值可以是 gripper, IW1, IW2, IW3, IW4 

STEP 2: 执行`accelerate launch train.py` 每个epoch都会保存一次检查点和主进程的header权重，保存路径在`model/{set name}/random/headers`

#### 2.1.2 TeDS 训练
STEP 1: 更改`train.py`最后一行为`train_in_teds(batch_size=60, set="gripper")`，其中 set 取值可以是 gripper, IW1, IW2, IW3, IW4 

STEP 2: 执行`accelerate launch train.py` 每个epoch都会保存一次检查点和主进程的header权重，保存路径在`model/{set name}/fast/headers`

### 2.2 测试模型
直接执行`python test.py`即可，可以在test_acc函数中编辑列表的数据来更改测试所用的模型和数据集。对于每个模型和数据组合，将会测试1000组数据，执行时间在13分钟左右。

### 2.3 推理效率测试
执行`python overhead_comparison.py`，注意由于没有应用LLM领域前沿的推理加速技术，推理速度随序列长度增加而明显下降，但是实验数据足够说明EST header不会对模型推理产生明显影响。

## 三、其他说明
1. 数据方面，我们已经完成对Instruction in Wild数据集和我们自己收集的gripper数据集进行预处理，相关代码主要在`trainingset.py`文件中
2. `data\gripper`目录下两个文件存储的数据是相同的
3. 目前不支持通过参数更改head_dim的功能，请在`model_arch.py`中通过直接通过代码修改。默认head_dim是64，仅对应论文中gripper数据集相关实验
4. transformers的版本必须是4.46.3，最新版transformers的KV缓存机制会导致显存溢出，速度也更慢