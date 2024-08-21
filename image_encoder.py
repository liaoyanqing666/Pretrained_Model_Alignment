import torch
from torch.utils.data import DataLoader, Dataset
from dataset import *
from vit_model import vit_base_patch16_224_in21k as create_model


def Image_encoder(device, model_weight_path=None):
    model = create_model(num_classes=50, has_logits=False).to(device)
    if model_weight_path is not None:
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # Create dataset and dataloader
    dataset = ImageDataset(root_dir=".\Animals_with_Attributes2\JPEGImages", transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # create model
    model = create_model(num_classes=50, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for batch_images, batch_labels, batch_paths in dataloader:
            print(batch_images)
            print(batch_labels)
            print(batch_paths)
            batch_images = batch_images.to(device)
            # predict class
            output = model(batch_images).cpu()
            print(f"Output tensor shape: {output.shape}")

if __name__ == '__main__':
    main()
