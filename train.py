import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageDataset
from image_encoder import Image_encoder
from similarity import nn_similarity, cosine_similarity
from text_encoder import Text_encoder
from torch.utils.tensorboard import SummaryWriter

def main():
    seed = 42
    random.seed(seed)        # 设置Python内置随机生成器的种子
    np.random.seed(seed)     # 设置NumPy随机生成器的种子
    torch.manual_seed(seed)  # 设置PyTorch随机生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    batch_size = 32
    epoch = 1000
    lr = 1e-5
    record = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = ImageDataset(root_dir=".\Animals_with_Attributes2", transform=data_transform, train=True, attributes='class_sentence.txt', pre_load=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = ImageDataset(root_dir=".\Animals_with_Attributes2", transform=data_transform, train=False, attributes='class_sentence.txt', pre_load=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    text_encoder = Text_encoder(device, 'weights/all-mpnet-base-v2')
    image_encoder = Image_encoder(device, model_weight_path='weights/model-9.pth')

    attributes = train_dataset.attributes
    attributes_encoded = torch.from_numpy(text_encoder.encode(attributes)).to(device)

    similarity = nn_similarity().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(image_encoder.parameters(), lr=lr)

    optimizer2 = torch.optim.Adam(similarity.parameters(), lr=0.1 * lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=1, patience=3)

    if record:
        writer = SummaryWriter('runs/image_similarity')

    # for param in image_encoder.parameters():
    #     param.requires_grad = False

    # TPR = 1
    # FPR = 1

    total_step = 0
    for e in range(epoch):
        image_encoder.train()
        similarity.train()
        # pos_portion = max(0.01, min(0.99, FPR / (TPR + FPR))) if TPR + FPR > 0 else 0.5
        # neg_portion = 1 - pos_portion
        # print(f"Epoch {e + 1}/{epoch}, pos_portion: {pos_portion}, neg_portion: {neg_portion}")
        for batch_images, batch_labels in train_dataloader:
            image_encoder.zero_grad()
            similarity.zero_grad()
            batch_images = batch_images.to(device)
            encoded_image = image_encoder(batch_images)
            encoded_attr = attributes_encoded[batch_labels.to(device)].to(device)
            similarity_true = similarity(encoded_image, encoded_attr)

            batch_fake_labels = torch.randint(low=0, high=49, size=batch_labels.shape)
            batch_fake_labels = batch_fake_labels + (batch_fake_labels >= batch_labels)  # 确保生成的假数据与源数据完全不同
            batch_fake_attr = attributes_encoded[batch_fake_labels.to(device)].to(device)
            similarity_fake = similarity(encoded_image, batch_fake_attr)

            real_label = torch.ones_like(batch_labels, dtype=torch.float).to(device)
            # loss = pos_portion * criterion(similarity_true, real_label) + neg_portion * criterion(similarity_fake, -real_label)
            loss = criterion(similarity_true, real_label) + criterion(similarity_fake, -real_label)
            loss.backward()
            optimizer.step()
            optimizer2.step()

            total_step += 1
            if record:
                writer.add_scalar('Train Loss', loss.item(), total_step)
            if total_step % 100 == 0:
                print(f"Step {total_step}, Loss: {loss.item()}")

        image_encoder.eval()
        similarity.eval()
        with torch.no_grad():
            total_true = 0
            correct_true = 0
            total_false = 0
            correct_false = 0
            for batch_images, batch_labels in test_dataloader:
                batch_images = batch_images.to(device)
                encoded_image = image_encoder(batch_images)
                encoded_attr = attributes_encoded[batch_labels.to(device)].to(device)
                similarity_true = similarity(encoded_image, encoded_attr)
                pred = (similarity_true > 0).long()
                correct_true += (pred == 1).sum().item()
                total_true += pred.shape[0]

                batch_fake_labels = torch.randint(low=0, high=49, size=batch_labels.shape)
                batch_fake_labels = batch_fake_labels + (batch_fake_labels >= batch_labels)
                batch_fake_attr = attributes_encoded[batch_fake_labels.to(device)].to(device)
                similarity_fake = similarity(encoded_image, batch_fake_attr)
                pred = (similarity_fake > 0).long()
                correct_false += (pred == 0).sum().item()
                total_false += pred.shape[0]

            print(f"Epoch {e + 1}/{epoch}, TPR: {correct_true / total_true}, FPR: {correct_false / total_false}, ACC: {(correct_true + correct_false) / (total_true + total_false)}")

            if record:
                writer.add_scalar('Test TPR', correct_true / total_true, e)
                writer.add_scalar('Test FPR', correct_false / total_false, e)
                writer.add_scalar('Test ACC', (correct_true + correct_false) / (total_true + total_false), e)

            TPR = correct_true / total_true
            FPR = correct_false / total_false

            torch.save(image_encoder.state_dict(), f'./weights/image_similarity/image-{e+1}.pth')
            torch.save(similarity.state_dict(), f'./weights/image_similarity/similarity-{e+1}.pth')
            # scheduler.step(correct_true / total_true)


if __name__ == '__main__':
    main()