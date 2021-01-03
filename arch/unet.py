import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet20(nn.Module):
    def __init__(self, dropout_rate=0.05, output_channels=27):
        super(UNet20, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        # 570 * 570 * 64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        # 568 * 568 *64

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.bn6 = nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3))
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.bn8 = nn.BatchNorm2d(512)

        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3))
        self.bn9 = nn.BatchNorm2d(1024)

        self.conv10 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3))
        self.bn10 = nn.BatchNorm2d(1024)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2
        )
        self.dropout5 = nn.Dropout2d(p=dropout_rate)

        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3))
        self.bn11 = nn.BatchNorm2d(512)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3))
        self.bn12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2
        )
        self.dropout6 = nn.Dropout2d(p=dropout_rate)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3))
        self.bn13 = nn.BatchNorm2d(256)

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.bn14 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2
        )
        self.dropout7 = nn.Dropout2d(p=dropout_rate)

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3))
        self.bn15 = nn.BatchNorm2d(128)
        self.dropout15 = nn.Dropout2d(p=dropout_rate)

        self.conv16 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.bn16 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2
        )
        self.dropout8 = nn.Dropout2d(p=dropout_rate)

        self.conv17 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3))
        self.bn17 = nn.BatchNorm2d(64)

        self.conv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.bn18 = nn.BatchNorm2d(64)

        self.conv19 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=(1, 1))
        self.bn19 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        y1 = x[:, :, 88:-88, 88:-88]
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        y2 = x[:, :, 40:-40, 40:-40]
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        y3 = x[:, :, 16:-16, 16:-16]
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)

        y4 = x[:, :, 4:-4, 4:-4]
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)

        x = self.upconv1(x)
        x = torch.cat((y4, x), dim=1)
        x = self.dropout5(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)

        x = self.upconv2(x)
        x = torch.cat((y3, x), dim=1)
        x = self.dropout6(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)

        x = self.conv14(x)
        x = self.bn14(x)
        x = F.relu(x)

        x = self.upconv3(x)
        x = torch.cat((y2, x), dim=1)
        x = self.dropout7(x)

        x = self.conv15(x)
        x = self.bn15(x)
        x = F.relu(x)

        x = self.conv16(x)
        x = self.bn16(x)
        x = F.relu(x)

        x = self.upconv4(x)
        x = torch.cat((y1, x), dim=1)
        x = self.dropout8(x)

        x = self.conv17(x)
        x = self.bn17(x)
        x = F.relu(x)

        x = self.conv18(x)
        x = self.bn18(x)
        x = F.relu(x)

        x = self.conv19(x)
        return x


class UNet256(nn.Module):
    def __init__(self, num_classes):
        super(UNet256, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        # output_out = torch.softmax(output_out, dim=1)
        return output_out