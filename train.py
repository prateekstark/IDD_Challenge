import torch
import logging
import argparse
from tqdm import tqdm
from arch.unet import UNet20
import matplotlib.pyplot as plt
from arch.hrnet import get_seg_model
from dataloaders import get_dataloader


if __name__ == "__main__":
    logging.basicConfig(
        filename="logfile.log",
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Device used: {}".format(dev))

    parser = argparse.ArgumentParser(description="IDD Challenge 2020")
    parser.add_argument("--model", help="Default=hrnet", type=str, default="hrnet")
    parser.add_argument("--num_classes", help="Default=27", type=int, default=27)
    parser.add_argument("--train_batch_size", help="Default=8", type=int, default=8)
    parser.add_argument("--lr", help="Default=0.01", type=float, default=0.01)
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

    logger.info("train_img_dir={}".format(train_img_dir))
    logger.info("train_label_dir={}".format(train_label_dir))
    logger.info("num_classes={}".format(num_classes))
    logger.info("train_batch_size={}".format(train_batch_size))
    logger.info("epochs={}".format(epochs))
    logger.info("learning_rate={}".format(lr))


    logger.info("Model Used: {}".format(args.model))
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

    if(args.pretrained_model):
        model = torch.load(args.pretrained_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-16, last_epoch=-1, verbose=True)

    step_losses = []
    epoch_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            output = model(X.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        scheduler.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        logger.info("Average Loss: {}".format(epoch_loss))
        epoch_losses.append(epoch_loss)
        torch.save(model, "checkpoint_{}_{}.pth".format(args.model, epoch))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)
    plt.savefig("{}_train_analysis.png".format(args.model))
