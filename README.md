# Pretrained_Model_Alignment
将视觉预训练模型与文字预训练模型进行对齐，可用于零样本学习。Aligning visual pre-trained models with text pre-trained models, enabling zero-shot learning.

在许多情况下，我们所拥有的资源仅仅是分别在文字或图片数据集上预训练好的单模态预训练模型。鉴于此，本项目的目标是通过少量的训练，使得两个不同模态的预训练模型（文字与图片）能够实现对齐。值得注意的是，本项目所用的对齐数据集并非类似于CLIP所需的海量“图像-描述”数据集，而是一个普通的图像多分类数据集。

本项目的核心思路是将图像多分类数据集中的类别转化为相应的文字描述，并运用对比学习的思想，为每张图片分配一个正确描述和一个错误描述（错误描述的内容不固定）。随后，分别计算这些描述的嵌入，并将其与图片的嵌入进行相似度比较。该项目的模型结构如图1所示。

In many cases, the resources we possess are merely unimodal pre-trained models that have been trained separately on either text or image datasets. Therefore, the objective of this project is to achieve alignment between two different modalities of pre-trained models (text and images) with minimal additional training. It is worth noting that the alignment dataset used in this project is not a large-scale "image-description" dataset like that required by CLIP, but rather a standard image classification dataset.

The core idea of this project is to convert the categories in the image classification dataset into corresponding textual descriptions and apply contrastive learning principles. For each image, a correct description and an incorrect description (with varying content) are assigned. Subsequently, the embeddings of these descriptions are computed and compared with the embeddings of the images for similarity. The structure of the model is illustrated in Figure 1.

![model](images/model_description.png)

**Fig 1. Model structure**

除了常用的余弦相似度，本项目还设计了一个孪生神经网络相似度，使其在两个预训练模型嵌入维度不同的情况下也可用。其中两个孪生分支的参数可以相同也可以不相同（根据具体实验需求）
