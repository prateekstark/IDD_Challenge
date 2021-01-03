import torch
import random
import logging
import argparse
from tqdm import tqdm
from arch.unet import UNet20, UNet256
import matplotlib.pyplot as plt
from arch.hrnet import get_seg_model
from dataloaders import get_dataloader


def one_hot(targets, C=27):    
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        targets = one_hot(targets, 27)
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

if __name__ == "__main__":
    identifier = random.randint(0, 1000000)
    logging.basicConfig(
        filename="logfile_{}.log".format(identifier),
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Device used: {}".format(device))

    parser = argparse.ArgumentParser(description="IDD Challenge 2020")
    parser.add_argument("--model", help="Default=hrnet", type=str, default="hrnet")
    parser.add_argument("--optim", help="Default=adadelta", type=str, default="adadelta")
    parser.add_argument("--criterion", help="Default=crossentropy", type=str, default="crossentropy")
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

    logger.info("identifier={}".format(identifier))
    logger.info("train_img_dir={}".format(train_img_dir))
    logger.info("train_label_dir={}".format(train_label_dir))
    logger.info("num_classes={}".format(num_classes))
    logger.info("train_batch_size={}".format(train_batch_size))
    logger.info("epochs={}".format(epochs))
    logger.info("learning_rate={}".format(lr))
    logger.info("Model Used: {}".format(args.model))
    logger.info("optim Used: {}".format(args.optim))


    if "unet_orig" in args.model:
        train_dataloader = get_dataloader(
            image_dir=train_img_dir,
            labels_dir=train_label_dir,
            batch_size=train_batch_size,
            print_dataset=True,
            input_img_size=(572, 572),
            output_img_size=(388, 388),
        )
        model = UNet20(output_channels=num_classes).to(device)
    elif "unet_256" in args.model:
        train_dataloader = get_dataloader(
            image_dir=train_img_dir,
            labels_dir=train_label_dir,
            batch_size=train_batch_size,
            print_dataset=True,
            input_img_size=(256, 256),
            output_img_size=(256, 256),
        )
        model = UNet256(num_classes=num_classes).to(device)
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

    scheduler_bool = False
    if(args.pretrained_model):
        model = torch.load(args.pretrained_model)

    if(args.optim == 'adadelta'):
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6, last_epoch=-1, verbose=True)
        scheduler_bool = True

    if(args.criterion == 'crossentropy'):
        criterion = torch.nn.CrossEntropyLoss()
    elif(args.criterion == 'focalTversky'):
        criterion = FocalTverskyLoss()
    else:
        raise ValueError("Criterion not recognized")

    step_losses = []
    epoch_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for X, y in tqdm(train_dataloader):
            X = X.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            output = model(X.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        if(scheduler_bool):
            scheduler.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        logger.info("Average Loss: {}".format(epoch_loss))
        epoch_losses.append(epoch_loss)
        torch.save(model, "checkpoint_{}_{}_{}.pth".format(args.model, epoch, identifier))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)
    plt.savefig("{}_{}_train_analysis.png".format(args.model, identifier))
