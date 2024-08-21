import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import tqdm

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, attributes='class_sentence.txt', train=None, pre_load=True, image_folder='JPEGImages'):
        """
        :param root_dir: 数据集根目录
        :param transform: 数据预处理
        :param attributes: 所需的属性
        :param train: None表示全部数据，True表示训练集，False表示测试集
        :param pre_load: 是否预加载图片
        :param image_folder: 图片文件夹名称
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pre_load = pre_load

        self.imgs = []
        self.labels = []
        self.img_paths = []

        self.classes = None  # id to class
        self.attributes = None  # id to attributes
        with open(os.path.join(root_dir, 'classes.txt'), 'r') as file:
            self.classes = file.readlines()
            for i in range(len(self.classes)):
                self.classes[i] = self.classes[i].split('\t')[1].strip()
        with open(os.path.join(root_dir, attributes), 'r') as file:
            self.attributes = file.readlines()

        self.used_classes = self.classes
        if train is not None:
            if train:
                with open(os.path.join(root_dir, 'trainclasses.txt'), 'r') as file:
                    self.used_classes = [line.strip() for line in file]
            else:
                with open(os.path.join(root_dir, 'testclasses.txt'), 'r') as file:
                    self.used_classes = [line.strip() for line in file]

        self.class_to_id = {self.classes[i]: i for i in range(len(self.classes))}

        for label in self.used_classes:
            class_dir = os.path.join(root_dir, image_folder, label)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.labels.append(self.class_to_id[label])
                    self.img_paths.append(img_path)

        if self.pre_load:
            for img_path in tqdm.tqdm(self.img_paths):
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                self.imgs.append(img)

        # print(self.classes)
        # print(self.attributes)
        # print(self.class_to_id)
        # print(self.used_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.pre_load:
            image = self.imgs[idx]
        else:
            img_path = self.img_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        class_index = self.labels[idx]
        return image, class_index

    def get_class_attributes(self, class_index):
        return self.attributes[class_index]

    def get_class_name(self, class_index):
        return self.classes[class_index]


if __name__ == "__main__":
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageDataset(root_dir="E:\大学本科课程\大三4\项目\Animals_with_Attributes2", pre_load=False, train=False, transform=data_transform)
    print(dataset[0])