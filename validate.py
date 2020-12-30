import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from dataloaders import get_dataloader


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    parser = argparse.ArgumentParser(description="IDD Challenge 2020 validation")
    parser.add_argument(
        "--val_img_dir",
        help="Default=data/leftImg8bit/val",
        type=str,
        default="data/leftImg8bit/val",
    )
    parser.add_argument(
        "--model_path",
        help="Default=checkpoint_7.pth",
        type=str,
        default="checkpoint_7.pth",
    )
    parser.add_argument("--output_dir", help="Default=pred", type=str, default="pred")
    parser.add_argument("--input_img_size", help="Default=572", type=int, default=572)
    args = parser.parse_args()

    val_img_dir = args.val_img_dir
    MODEL_PATH = args.model_path

    model = torch.load(MODEL_PATH)
    model.eval()
    for root, directories, files in os.walk(val_img_dir, topdown=False):
        for name in files:
            img_path = os.path.join(root, name)
            image = Image.open(img_path).convert("RGB")
            image = image.resize((args.input_img_size, args.input_img_size))
            image = np.asarray(image).reshape(
                (1, 3, args.input_img_size, args.input_img_size)
            )
            image = torch.from_numpy(image).float().to(device)

            with torch.no_grad():
                output = model(image)
            output = torch.argmax(output, dim=1)
            output = output[0].cpu().detach().numpy().astype("uint8")
            output = cv2.resize(output, (1920, 1080), cv2.INTER_NEAREST)

            dir_present = False
            for dir_ in os.listdir():
                if args.output_dir == dir_:
                    dir_present = True
                    break

            if not dir_present:
                os.mkdir(args.output_dir)

            print(
                "Written Image: {}".format(
                    "pred/" + name[:9] + "_gtFine_labellevel3Ids.png"
                )
            )
            cv2.imwrite("pred/" + name[:9] + "_gtFine_labellevel3Ids.png", output)
