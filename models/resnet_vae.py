import torch
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.nn as nn


def convrelu(in_channels, out_channels, kernel, padding, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.Conv2d(out_channels, out_channels, kernel, padding=padding))
    layers = nn.Sequential(*layers)
    return layers


class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.base_model = resnet18(pretrained=True)

        self.base_layers = list(self.base_model.children())
        # self.base_layers[0] = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0)
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 20, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = convrelu(20, 512, 3, 1)
        self.conv_up2 = convrelu(512, 256, 3, 1)
        self.conv_up1 = convrelu(256, 256, 3, 1)
        self.conv_up0 = convrelu(256, 128, 3, 1)
        # self.pconv = nn.Sequential(
        #                     convrelu(3, 64, 3, 1, pool=True),
        #                     convrelu(64, 128, 3, 1, pool=True),
        #                     convrelu(128, 128, 3, 1, pool=True),
        #                     convrelu(128, 128, 3, 1, pool=True),
        #                     convrelu(128, 128, 3, 1, pool=True),
        #                     convrelu(128, 128, 3, 1, pool=False),
        #                     convrelu(128, 20, 1, 0, pool=False)
        #                     )

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, 64, 3, 1)
        # self.up1 = nn.ConvTranspose2d(in_channels=20, out_channels=512, kernel_size=2, stride=2)
        # self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        # self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        # self.up4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.conv_last = nn.Conv2d(64, n_class, 1)
        # self.mean = nn.Linear(980, 980)
        # self.var = nn.Linear(980, 980)

    def forward(self, input, decode=False):
        if not decode:
            #x_original = self.conv_original_size0(input)
            #x_original = self.conv_original_size1(x_original)
            # p_layers = self.pconv(input)
            layer0 = self.layer0(input)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            layer4 = self.layer4_1x1(layer4)
            enc = layer4
            #mean = self.mean(layer4.view(layer4.size(0), -1))
            #var = self.var(layer4.view(layer4.size(0), -1))
            #enc = self.reparameterize(mean, var)
        else:
            enc = input
        x = self.upsample(enc)
        # layer3 = self.layer3_1x1(layer3)
        # x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        # layer2 = self.layer2_1x1(layer2)
        # x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        # layer1 = self.layer1_1x1(layer1)
        # x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        # layer0 = self.layer0_1x1(layer0)
        # x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

#
# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, in_channels, kernel, padding=padding),
#         nn.BatchNorm2d(in_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#     )


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = self.dconv(x)
        x = self.conv(x)
        return x


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,1,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 1024)
        self.conv_up6 = convrelu(256, 512, 3, 1)
        self.up6 = ResizeConv2d(512, 512, 3, 2)
        self.conv_up5= convrelu(512, 256, 3, 1)
        self.up5 = ResizeConv2d(256, 256, 3, 2)
        self.conv_up4= convrelu(256, 256, 3, 1)
        self.up4 = ResizeConv2d(256, 256, 3, 2)
        self.conv_up3 = convrelu(256, 256, 3, 1)
        self.up3 = ResizeConv2d(256, 256, 3, 2)
        self.conv = nn.Conv2d(256,256,3,1,0)
        self.conv_up2 = convrelu(256, 128, 3, 1)
        self.up2 = ResizeConv2d(128, 128, 3, 2)
        self.conv_up1= convrelu(128, 64, 3, 1)
        self.up1 = ResizeConv2d(64, 64, 3, 2)
        self.conv_up0 = convrelu(64, 64, 3, 1)
        self.up0 = ResizeConv2d(64, 64, 3, 2)
        self.conv_f1 = nn.Conv2d(64,64,3,1,1)
        self.bn = nn.BatchNorm2d(64)
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 256, 2, 2)
        x = self.conv_up6(x)
        x = self.up6(x)
        x = self.conv_up5(x)
        x = self.up5(x)
        x = self.conv_up4(x)
        x = self.up4(x)
        x = self.conv(x)
        x = self.conv_up3(x)
        x = self.up3(x)
        x = self.conv_up2(x)
        x = self.up2(x)
        x = self.conv_up1(x)
        x = self.up1(x)
        x = self.conv_up0(x)
        x = self.up0(x)
        x = F.relu(self.bn(self.conv_f1(x)))
        x = self.outconv(x)

        return x

class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        # mean, logvar = self.encoder(x)
        # z = self.reparameterize(mean, logvar)
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean



if __name__ == "__main__":
    vae = VAE(500)
    model = ResNetUNet(1)
    im = torch.zeros((1, 3, 224, 224))
    out, enc = model(im)
    print (out.shape, enc.shape)
    print(model(enc, decode=True)[0].shape)
