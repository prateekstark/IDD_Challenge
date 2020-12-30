import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet20(nn.Module):
    def __init__(self, dropout_rate=0.2, output_channels=27):
        super(UNet20, self).__init__()
        # Input 572 * 572 * 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        # 570 * 570 * 64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        # 568 * 568 *64

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout5 = nn.Dropout2d(p=dropout_rate)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.bn6 = nn.BatchNorm2d(256)
        self.dropout6 = nn.Dropout2d(p=dropout_rate)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.bn7 = nn.BatchNorm2d(512)
        self.dropout7 = nn.Dropout2d(p=dropout_rate)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.bn8 = nn.BatchNorm2d(512)
        self.dropout8 = nn.Dropout2d(p=dropout_rate)

        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3))
        self.bn9 = nn.BatchNorm2d(1024)
        self.dropout9 = nn.Dropout2d(p=dropout_rate)

        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3))
        self.bn10 = nn.BatchNorm2d(1024)
        self.dropout10 = nn.Dropout2d(p=dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2
        )

        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3))
        self.bn11 = nn.BatchNorm2d(512)
        self.dropout11 = nn.Dropout2d(p=dropout_rate)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.bn12 = nn.BatchNorm2d(512)
        self.dropout12 = nn.Dropout2d(p=dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2
        )

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3))
        self.bn13 = nn.BatchNorm2d(256)
        self.dropout13 = nn.Dropout2d(p=dropout_rate)

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.bn14 = nn.BatchNorm2d(256)
        self.dropout14 = nn.Dropout2d(p=dropout_rate)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2
        )

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3))
        self.bn15 = nn.BatchNorm2d(128)
        self.dropout15 = nn.Dropout2d(p=dropout_rate)

        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.bn16 = nn.BatchNorm2d(128)
        self.dropout16 = nn.Dropout2d(p=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2
        )

        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3))
        self.bn17 = nn.BatchNorm2d(64)
        self.dropout17 = nn.Dropout2d(p=dropout_rate)

        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.bn18 = nn.BatchNorm2d(64)
        self.dropout18 = nn.Dropout2d(p=dropout_rate)

        self.conv19 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=(1, 1))
        self.bn19 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        y1 = x[:, :, 88:-88, 88:-88]
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        y2 = x[:, :, 40:-40, 40:-40]
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        y3 = x[:, :, 16:-16, 16:-16]
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.dropout8(x)

        y4 = x[:, :, 4:-4, 4:-4]
        x = self.maxpool4(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.dropout10(x)

        x = self.upconv1(x)
        x = torch.cat((y4, x), dim=1)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.dropout11(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.dropout12(x)

        x = self.upconv2(x)
        x = torch.cat((y3, x), dim=1)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = self.dropout13(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = F.relu(x)
        x = self.dropout14(x)

        x = self.upconv3(x)
        x = torch.cat((y2, x), dim=1)

        x = self.conv15(x)
        x = self.bn15(x)
        x = F.relu(x)
        x = self.dropout15(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = F.relu(x)
        x = self.dropout16(x)

        x = self.upconv4(x)
        x = torch.cat((y1, x), dim=1)

        x = self.conv17(x)
        x = self.bn17(x)
        x = F.relu(x)
        x = self.dropout17(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = F.relu(x)
        x = self.dropout18(x)

        x = self.conv19(x)
        x = self.bn19(x)
        return x