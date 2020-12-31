import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class ICVGIPDataset(Dataset):
    def __init__(
        self,
        image_dir="data/leftImg8bit/train",
        labels_dir="data/gtFine/train",
        print_dataset=False,
        input_img_size=(388, 388),
        output_img_size=(388, 388),
    ):
        X = []
        y = []
        for root, directories, files in os.walk(image_dir, topdown=False):
            for name in files:
                X.append(os.path.join(root, name))

        for root, directories, files in os.walk(labels_dir, topdown=False):
            for name in files:
                if "_gtFine_labellevel3Ids.png" in name:
                    y.append(os.path.join(root, name))
        assert len(X) == len(y)
        X.sort()
        y.sort()
        self.samples = list(zip(X, y))
        del X, y
        if print_dataset:
            self.print_dataset()

        self.input_img_size = input_img_size
        self.output_img_size = output_img_size
        print(input_img_size)

    def __len__(self):
        return len(self.samples)

    def print_dataset(self):
        for X, y in self.samples:
            print(X, y)

    def __getitem__(self, index):
        print(self.samples[index])
        image_path, label_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.input_img_size)
        image = torch.Tensor(
            np.asarray(image).reshape(
                (3, self.input_img_size[0], self.input_img_size[1])
            )
        )

        labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        labels = cv2.resize(labels, self.output_img_size, cv2.INTER_NEAREST)
        labels[np.where(labels > 26)] = 26

        labels = torch.Tensor(np.asarray(labels)).long()
        # image = self.transform(image)
        # image
        return image, labels

    def transform(self, image):
        transform_ops = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return transform_ops(image)


def get_dataloader(
    image_dir="data/leftImg8bit/train",
    labels_dir="data/gtFine/train",
    print_dataset=False,
    batch_size=8,
    input_img_size=(388, 388),
    output_img_size=(388, 388),
):
    dataset = ICVGIPDataset(
        image_dir=image_dir,
        labels_dir=labels_dir,
        print_dataset=print_dataset,
        input_img_size=input_img_size,
        output_img_size=output_img_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


if __name__ == "__main__":
    dataset = get_dataloader()
