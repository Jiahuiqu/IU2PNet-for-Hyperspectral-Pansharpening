import torch
import torch.nn.functional as F
import torch.nn as nn

class ProxNet1(torch.nn.Module):
    def __init__(self, C):
        super(ProxNet1, self).__init__()

        self.conv11 = nn.Conv2d(C, 128, stride=1, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1)
        self.A1B1 = nn.Conv2d(128, 128, stride=2, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(128 * 2, 128, stride=1, kernel_size=3, padding=1)
        self.B1A3 = nn.ConvTranspose2d(128, 128, stride=2, kernel_size=4, padding=1)
        self.conv22 = nn.Conv2d(128 * 2, 128, stride=1, kernel_size=3, padding=1)
        self.A2B2 = nn.Conv2d(128, 128, stride=2, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(128, 128, stride=1, kernel_size=3, padding=1)
        self.B1C1 = nn.Conv2d(128, 128, stride=2, kernel_size=3, padding=1)
        self.A2C1 = nn.Conv2d(128, 128, stride=4, kernel_size=3, padding=1)
        self.C1A4 = nn.ConvTranspose2d(128 * 2, 128, stride=4, kernel_size=4)
        self.B2A4 = nn.ConvTranspose2d(128 * 2, 128, stride=2, kernel_size=4, padding=1)
        self.C1B3 = nn.ConvTranspose2d(128 * 2, 128, stride=2, kernel_size=4, padding=1)
        self.A3B3 = nn.Conv2d(128 * 2, 128, stride=2, kernel_size = 3, padding = 1)
        self.B3A5 = nn.ConvTranspose2d(128 * 3, 128, stride=2, kernel_size=4, padding=1)
        self.conv15 = nn.Conv2d(128 * 3, 128, stride=1, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(128 * 2, C, stride=1, kernel_size=3, padding=1)
        self.relu = nn.PReLU()


    def forward(self, x):

        A1 = self.relu(self.conv11(x))
        A2 = self.relu(self.conv12(A1))
        B1 = self.relu(self.A1B1(A1))

        A3 = torch.cat((self.relu(self.conv13(A2)),self.relu(self.B1A3(B1))),1)
        B2 = torch.cat((self.relu(self.A2B2(A2)), self.relu(self.conv21(B1))),1)
        C1 = torch.cat((self.relu(self.B1C1(B1)),self.relu(self.A2C1(A2))),1)
        A4 = torch.cat([self.relu(self.conv14(A3)), self.relu(self.C1A4(C1)),self.relu(self.B2A4(B2))],1)

        B3 = torch.cat([self.relu(self.conv22(B2)), self.relu(self.C1B3(C1)),self.relu(self.A3B3(A3))],1)

        A5 = torch.cat([self.relu(self.conv15(A4)), self.relu(self.B3A5(B3))],1)
        A6 = self.relu(self.conv16(A5))

        return A6


class Up_sample(torch.nn.Module):
    def __init__(self,C):

        super(Up_sample, self).__init__()
        self.denet1 = nn.ConvTranspose2d(C, C, stride=2, kernel_size=4, padding=1)
        self.denet2 = nn.ConvTranspose2d(C, C, stride=2, kernel_size=4, padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):

        out = self.relu(self.denet1(x))
        out = self.denet2(out)

        return out


class Down_sample(torch.nn.Module):
    def __init__(self,C):
        super(Down_sample, self).__init__()

        self.ennet1 = nn.Conv2d(C, C, stride=2, kernel_size=3, padding=1)
        self.ennet2 = nn.Conv2d(C, C, stride=2, kernel_size=3, padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):

        out = self.relu(self.ennet1(x))
        out = self.ennet2(out)

        return out


class IU2Pnet(torch.nn.Module):
    def __init__(self, C, N, ratio):
        super(IU2Pnet, self).__init__()

        self.u = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamba = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.R = nn.Conv2d(C, 1, 1)
        self.RT = nn.Conv2d(1, C, 1)
        self.RTRE = nn.Conv2d(C, C, 1)
        self.HTHE = nn.Conv2d(C, C, 1)
        self.HT = Up_sample(C)
        self.H = Down_sample(C)
        self.M = ProxNet1(C)
        self.L = ProxNet1(C)
        self.N = N
        self.ratio = ratio

    def forward(self, X, Y):

        X_hat = F.interpolate(X, scale_factor=self.ratio, mode='bicubic', align_corners=False)
        A = self.RT(self.R(X_hat)-Y)
        B = self.HT(self.H(X_hat)-X)
        M = self.M(X_hat)
        X_hat= X_hat - self.u * (A+B - self.lamba*(X_hat-M))
        for i in range (self.N):
            A = self.RT(self.R(X_hat) - Y)
            B = self.HT(self.H(X_hat) - X)
            M = self.M(X_hat)
            X_hat = X_hat - self.u * (A + B - self.lamba * (X_hat - M))
        # out1 = self.R(X_hat)
        # out2 = self.H(X_hat)

        return X_hat

# if __name__ == "__main__":
#     a = torch.randn(16,31,40,40)
#     b = torch.randn(16,1,160,160)
#     net = NBNet(31, 6, 4)
#     c, d, e = net(a, b)
#     print(c.shape, d.shape, e.shape)
