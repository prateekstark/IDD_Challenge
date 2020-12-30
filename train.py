import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from arch.unet import UNet20
import torch.nn.functional as F
import matplotlib.pyplot as plt
from arch.hrnet import get_seg_model
from dataloaders import get_dataloader


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    parser = argparse.ArgumentParser(description="IDD Challenge 2020")
    parser.add_argument("--model", help="Default=hrnet", type=str, default="hrnet")
    parser.add_argument("--num_classes", help="Default=27", type=int, default=27)
    parser.add_argument("--train_batch_size", help="Default=8", type=int, default=8)
    parser.add_argument("--lr", help="Default=0.001", type=float, default=0.001)
    parser.add_argument("--epochs", help="Default=60", type=int, default=60)
    parser.add_argument(
        "--pretrained_model", help="Default=None", type=str, default=None
    )
    parser.add_argument(
        "--train_img_dir",
        help="Default=data/leftImg8bit/train",
        type=str,
        default="data/leftImg8bit/train",
    )
    parser.add_argument(
        "--train_label_dir",
        help="Default=data/gtFine/train",
        type=str,
        default="data/gtFine/train",
    )
    args = parser.parse_args()

    train_img_dir = args.train_img_dir
    train_label_dir = args.train_label_dir
    num_classes = args.num_classes
    train_batch_size = args.train_batch_size
    epochs = args.epochs
    lr = args.lr

    if "unet" in args.model:
        train_dataloader = get_dataloader(
            image_dir=train_img_dir,
            labels_dir=train_label_dir,
            batch_size=train_batch_size,
            print_dataset=True,
            input_img_size=(572, 572),
            output_img_size=(388, 388),
        )
        model = UNet20(output_channels=num_classes).to(device)
    elif "hrnet" in args.model:
        train_dataloader = get_dataloader(
            image_dir=train_img_dir,
            labels_dir=train_label_dir,
            batch_size=train_batch_size,
            print_dataset=True,
            input_img_size=(572, 572),
            output_img_size=(572, 572),
        )
        config = {}
        config["NUM_CLASSES"] = num_classes
        config["PRETRAINED"] = None
        config["MODEL"] = {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "STAGE1": {
                    "BLOCK": "BOTTLENECK",
                    "FUSE_METHOD": "SUM",
                    "NUM_BLOCKS": [1],
                    "NUM_CHANNELS": [32],
                    "NUM_MODULES": 1,
                    "NUM_RANCHES": 1,
                },
                "STAGE2": {
                    "BLOCK": "BASIC",
                    "FUSE_METHOD": "SUM",
                    "NUM_BLOCKS": [2, 2],
                    "NUM_BRANCHES": 2,
                    "NUM_CHANNELS": [16, 32],
                    "NUM_MODULES": 1,
                },
                "STAGE3": {
                    "BLOCK": "BASIC",
                    "FUSE_METHOD": "SUM",
                    "NUM_BLOCKS": [2, 2, 2],
                    "NUM_BRANCHES": 3,
                    "NUM_CHANNELS": [16, 32, 64],
                    "NUM_MODULES": 1,
                },
                "STAGE4": {
                    "BLOCK": "BASIC",
                    "FUSE_METHOD": "SUM",
                    "NUM_BLOCKS": [2, 2, 2, 2],
                    "NUM_BRANCHES": 4,
                    "NUM_CHANNELS": [16, 32, 64, 128],
                    "NUM_MODULES": 1,
                },
            }
        }
        model = get_seg_model(config)
        model = model.to(device)
    else:
        raise ValueError("Model architecture does not exist!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step_losses = []
    epoch_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        batch = 0
        for X, y in tqdm(train_dataloader):
            print(X.shape, y.shape)
            optimizer.zero_grad()
            output = model(X.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())

            if batch % 50 == 0:
                print("Batch %d, Epoch %d" % (batch, epoch))
            batch += 1
        epoch_loss = epoch_loss / len(train_dataloader)
        print("Average Loss: {}".format(epoch_loss))
        epoch_losses.append(epoch_loss)
        torch.save(model, "checkpoint_{}_{}.pth".format(args.model, epoch))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)
    plt.savefig("{}_train_analysis.png".format(args.model))
