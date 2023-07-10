# coding=utf-8
"""
实现stack hourglass network:  https://arxiv.org/pdf/1603.06937.pdf
"""

import torch.nn as nn
from collections import OrderedDict


class Residual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Residual, self).__init__()
        half_out_dim = out_dim // 2

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, half_out_dim, 1)
        self.bn2 = nn.BatchNorm2d(half_out_dim)
        self.conv2 = nn.Conv2d(half_out_dim, half_out_dim, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(half_out_dim)
        self.conv3 = nn.Conv2d(half_out_dim, out_dim, 1)
        self.skip_layer = nn.Conv2d(in_dim, out_dim, 1)
        if in_dim != out_dim:
            self.need_skip = True
        else:
            self.need_skip = False

    def forward(self, x):
        residual = x
        if self.need_skip:
            residual = self.skip_layer(residual)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n=4, f_size=256):
        """
        Args:
            n: hourglass的层级
            f_size: 输入特征维度
        """
        super(Hourglass, self).__init__()
        self.n = n
        self.f_size = f_size

        self.up1 = Residual(f_size, f_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f_size, f_size)

        if self.n > 1:
            self.low2 = Hourglass(self.n - 1, f_size)
        else:
            self.low2 = Residual(f_size, f_size)

        self.low3 = Residual(f_size, f_size)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(up1)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class CBR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StackHourglass(nn.Module):
    def __init__(self, n_stack=8, in_dim=256, n_kp=3, n_hg_layer=4):
        """
        Args:
            n_stack: hourglass网络堆叠个数
            in_dim:  输入特征维度
            n_kp: 关键点个数
            n_hg_layer: hourglass的层数
        """
        super(StackHourglass, self).__init__()
        self.n_stack = n_stack
        self.in_dim = in_dim
        self.n_kp = n_kp
        self.n_hg_layer = n_hg_layer
        # 输入图片预处理
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),    # (n, 3, 256, 256)==>(n, 64, 128, 128)
            nn.BatchNorm2d(64),           # (n, 64, 128, 128)
            nn.ReLU(inplace=True),        # (n, 64, 128, 128)
            Residual(64, 128),            # (n, 128, 128, 128)
            nn.MaxPool2d(2, 2),           # (n, 128, 64, 64)
            Residual(128, 128),           # (n, 128, 64, 64)
            Residual(128, in_dim)         # (n, 256, 64, 64)
        )
        self.hgs = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ('hg_' + str(i), Hourglass(self.n_hg_layer, self.in_dim))
                ])
            )
            for i in range(self.n_stack)
        ])
        self.features = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ("residual_" + str(i), Residual(self.in_dim, self.in_dim)),
                    ("conv_bn_relu_" + str(i), CBR(self.in_dim, self.in_dim))
                ])
            )
            for i in range(self.n_stack)
        ])
        self.preds = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ("out_" + str(i), nn.Conv2d(self.in_dim, self.n_kp, 1))
                ])
            )
            for i in range(self.n_stack)
        ])
        self.merge_features = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ("merge_feature_" + str(i), nn.Conv2d(self.in_dim, self.in_dim, 1))
                ])
            )
            for i in range(self.n_stack - 1)
        ])
        self.merge_preds = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ("merge_pred_" + str(i), nn.Conv2d(self.n_kp, self.in_dim, 1))
                ])
            )
            for i in range(self.n_stack - 1)
        ])

    def forward(self, imgs):
        """
        Args:
            imgs: 输入图像，  （N, 3, 256, 256）
        Returns:

        """
        x = self.pre(imgs)
        outs = []
        for i in range(self.n_stack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            pred = self.preds[i](feature)
            outs.append(pred)
            if i < self.n_stack - 1:
                x = x + self.merge_features[i](feature) + self.merge_preds[i](pred)

        return torch.stack(outs, 1)


if __name__ == "__main__":
    import torch
    print("_____________")

    # # Test Residual
    # inp = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    # res_model = Residual(3, 6)
    # print(inp.shape, input.dtype)
    # o = res_model(input)
    # print(o.shape)

    # # Test Hourglass
    # inp = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    # hourglass_model = Hourglass(4, 3)
    # o = hourglass_model(inp)
    # print("out_shape: ", o.shape)

    # Test Stack hourglass
    inp = torch.randn((1, 3, 256, 256), dtype=torch.float32)
    stack_hourglass_model = StackHourglass(4, 256, 3, 2)
    # print(stack_hourglass_model)
    o = stack_hourglass_model(inp)

    # torch.save(stack_hourglass_model, "D:/model.pth")
    # script_model = torch.jit.trace(stack_hourglass_model, (inp, ))
    # torch.jit.save(script_model, "D:/script_model.pt")
    print("out_shape: ", o.shape)
