# Pretrained_Model_Alignment
将视觉预训练模型与文字预训练模型进行对齐，可用于零样本学习。Aligning visual pre-trained models with text pre-trained models. Can be applied to zero-shot learning.

## 项目介绍 (Project Introduction)
> both in Chinese and English. The translation is assisted by gpt.

在许多情况下，我们所拥有的资源仅仅是分别在文字或图片数据集上预训练好的单模态预训练模型。鉴于此，本项目的目标是通过少量的训练，使得两个不同模态的预训练模型（文字与图片）能够实现对齐。值得注意的是，本项目所用的对齐数据集并非类似于CLIP所需的海量“图像-描述”数据集，而是一个普通的图像多分类数据集。

本项目的核心思路是将图像多分类数据集中的类别转化为相应的文字描述，并运用对比学习的思想，为每张图片分配一个正确描述和一个错误描述（错误描述的内容不固定）。随后，分别计算这些描述的嵌入，并将其与图片的嵌入进行相似度比较。该项目的模型结构如图1所示。

In many cases, the resources we possess are merely unimodal pre-trained models that have been trained separately on either text or image datasets. Therefore, the objective of this project is to achieve alignment between two different modalities of pre-trained models (text and images) with minimal additional training. It is worth noting that the alignment dataset used in this project is not a large-scale "image-description" dataset like that required by CLIP, but rather a standard image classification dataset.

The core idea of this project is to convert the categories in the image classification dataset into corresponding textual descriptions and apply contrastive learning principles. For each image, a correct description and an incorrect description (with varying content) are assigned. Subsequently, the embeddings of these descriptions are computed and compared with the embeddings of the images for similarity. The structure of the model is illustrated in Figure 1.

![model](images/model_description.png)

**Fig 1. Model Structure**

除了常用的余弦相似度，本项目还设计了一个孪生神经网络相似度，使其在两个预训练模型嵌入维度不同的情况下也可用。其中两个孪生分支的参数可以相同也可以不相同（根据具体实验需求），模型结构如图2所示。

In addition to the commonly used cosine similarity, this project also designed a Siamese neural network-based similarity measure, which can be used even when the two pre-trained models have different embedding dimensions. The parameters of the two branches in the Siamese network can be either identical or different, depending on the specific experimental requirements. The model structure is illustrated in Figure 2.

![similarity](images/similarity_description.png)

**Fig 2. Similarity Model Structure**

## 如何使用 (How to use it)

1. clone这个仓库。
2. 下载AwA2数据集，下载链接：[AwA2](https://cvml.ista.ac.at/AwA2/AwA2-data.zip)。数据集包含50个类别，尽管数据集提供了每个类别的具体属性，但在实验中并没有直接使用。我们在训练的40个类别中进行对齐，并使用10个测试类别进行测试以验证其对齐后的零样本能力。注意将本项目提供的`Animals_with_Attributes2/class.txt`文件放到数据集文件夹中。这个文件是作为对每个类别的描述使用的，不过也可以使用同文件夹下的其他三个描述文件进行实验（非常建议实验`Animals_with_Attributes2/class_sentence.txt`文件）
3. 下载预训练文本嵌入模型参数到weights文件夹中。本项目使用的是sentence_transformer库中提供的“all-mpnet-base-v2”模型。中国大陆下载时可能存在代理问题，注意修改代码。如无法下载可以去官方github下载zip解压。
4. 下载并解压提供的model-9.zip为model-9.pth，这是预训练图像嵌入模型参数。
5. 运行train.py
6. 经过实验，我们发现微调vit模型+使用孪生神经网络相似度可以取得较好的效果。训练时注意学习率需要较小，且相似度网络的学习率应该低于微调vit模型的学习率（相似度网络会更快地拟合）。
