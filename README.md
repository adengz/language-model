# language-model 语言模型

## **用Log Loss训练LSTM模型**

[full_vocab.ipynb](https://nbviewer.jupyter.org/github/adengz/language-model/blob/main/full_vocab.ipynb) [5] ~ [8]

句子分词后，由字典下标编码的输入向量经过一层200维的word embedding，被传至一层200维的单向LSTM。每一步LSTM的隐状态被一个线性变换层映射到整个字典的维度。用Log Loss来训练在整个字典上的分类问题，预测的目标为每一个当前输入词的下一个词。

### 模型/训练参数

* EMBEDDING_DIM = 200
* HIDDEN_SIZE = 200
* DROPOUT = 0.2
* BATCH_SIZE = 64
* OPTIM_CLS = Adam
* LR = 1e-3
* EPOCHS = 15

### 最佳模型

DEV最佳准确率：32.40%

TEST准确率：31.98%

### 模型最常见的35个预测错误

|prediction|ground truth|
|:----:|:----:|
|Bob|He|
|Bob|Sue|
|Bob|She|
|the|his|
|.|and|
|was|had|
|he|she|
|.|to|
|to|.|
|was|decided|
|the|her|
|the|a|
|was|went|
|was|didn|
|was|'s|
|Bob|The|
|a|the|
|was|got|
|.|for|
|Bob|They|
|.|in|
|to|and|
|.|at|
|was|wanted|
|.|!|
|Bob|His|
|Bob|When|
|was|could|
|.|the|
|Bob|But|
|to|for|
|.|,|
|and|.|
|her|his|
|was|fel|

### 分析

最常见的错误中，预测的词比较单一，只有Bob, the, was, to和.。极有可能是由于这些词在训练集中词频过高。也有可能是因为本身预测算法使用的是greedy search/argmax，会使预测的输出集中于这些高频词。好消息是，这些错误，标点符号以外，Bob（名词）、the（冠词）、was（动词）和to（介词），与正确的词的词性是基本一致的。可以考虑用beam search来做预测，避免出现太多高频词。

## 使用更大的context

[full_vocab.ipynb](https://nbviewer.jupyter.org/github/adengz/language-model/blob/main/full_vocab.ipynb) [9] ~ [12]

与前一任务类似，唯一的区别是考虑输入句子的前文。将前文embed后传入LSTM，得到的最后一步隐状态 + cell state连同embedded输入句子一起送给LSTM。

### 模型/训练参数

* EMBEDDING_DIM = 200
* HIDDEN_SIZE = 200
* DROPOUT = 0.2
* BATCH_SIZE = 64
* OPTIM_CLS = Adam
* LR = 1e-3
* EPOCHS = 15

### 最佳模型

DEV最佳准确率：35.19%

TEST准确率：34.79%

### 模型最常见的35个预测错误

|prediction|ground truth|
|:----:|:----:|
|the|his|
|Bob|Sue|
|was|had|
|.|and|
|He|Bob|
|.|to|
|the|a|
|Sue|Bob|
|was|decided|
|the|her|
|he|she|
|was|didn|
|was|went|
|was|'s|
|was|got|
|to|.|
|and|.|
|.|for|
|She|Sue|
|a|the|
|and|to|
|was|wanted|
|.|at|
|.|!|
|.|in|
|.|,|
|was|felt|
|was|could|
|to|for|
|.|on|
|to|a|
|a|his|
|to|home|
|was|and|
|.|the|

### 分析

提供上文后，模型整体性能提升了。原先最多的错误预测Bob -> He不见了，大概率因为真正需要预测的部分以He开头的很多，而且在上文存在的情况下，使用代词也是正常的现象。错误was由于所对应的正确选项太多，所以本质上没有太大改观。

## **Binary Log Loss实验**

[neg_sample.ipynb](https://nbviewer.jupyter.org/github/adengz/language-model/blob/main/neg_sample.ipynb)

模型分别有三个输入，三个输出。

句子用字典下标编码输入后，经过一层200维的input embedding后，传入一层200维单向LSTM，输出每一个单词对应的隐状态。

同样，预测的目标，也就是一个句子后移一个词后的新句子，经过字典下标编码，输入一层200维的output embedding后输出。注意这里output embedding与LSTM的维度必须一致。

从字典中抽取的非目标词，同样经过字典下标编码，输入同样的output embedding后输出。

损失函数按以下公式计算

![图片](https://uploader.shimo.im/f/W9ZsNXUsbbhfz45H.png!thumbnail?fileGuid=CJ3GrG3xTwHxYCXX)

其中**h**_<sub>t</sub>_是输入词经过input embedding和LSTM输出的隐状态，_y<sub>t</sub>_是output embedded目标词，_y_'是output embedded非目标词。score(_y_,**h**)便是两个输入向量的点积，由此可见output embedding与LSTM的维度必须一致。

负例采样的频率正比于负样本在训练集中出现的频率的_f_次幂

* UNIF: _f_ = 0，所有概率相同，均匀抽样
* UNIG: _f_ = 1，抽样概率正比于训练集中的词频

### 模型/训练参数

* EMBEDDING_DIM = 200
* DROPOUT = 0.2
* BATCH_SIZE = 64
* OPTIM_CLS = Adam
* LR = 1e-3
* EPOCHS = 30

### UNIF抽样 _f_ = 0.0

|_r_|DEV最佳准确率|TEST准确率|
|:----:|:----:|:----:|
|20|26.94%|26.59%|
|100|27.23%|26.80%|
|500|26.69%|26.22%|

### UNIG-f抽样 负样本个数 _r_ = 20

|_f_|DEV最佳准确率|TEST准确率|
|:----:|:----:|:----:|
|0.25|27.59%|27.88%|
|0.50|28.98%|28.64%|
|0.75|27.87%|27.92%|
|1.00|23.01%|22.61%|

除UNIG (_f_ = 1.0) 外，均好于UNIF _r_ = 20的结果。高频词的确应被赋予更高的采样权重，但完全正比于词频也会降低模型的预测能力。

### 分析

采用二分类 + 负采样的方法本意是为了节约训练时间，但对于所提供的小数据集，结果适得其反。原因可能在于训练及预测过程中额外的score计算，以及loss计算和准确率计算需要不同的计算图。以准确率来衡量，效果也略逊于直接用全字典来进行分类。

