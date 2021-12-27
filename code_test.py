import torch
import torch.nn as nn
import torch.nn.functional as F

def get_attention(preds, temp):
    """ preds: Bs*C*W*H """
    N, C, H, W = preds.shape

    value = torch.abs(preds)
    # Bs*W*H
    fea_map = value.mean(axis=1, keepdim=True)
    print("fea_map = ", fea_map.shape)
    S_attention = (
        H * W * F.softmax((fea_map/temp).view(N, -1), dim=1)).view(N, H, W)

    # Bs*C
    channel_map = value.mean(axis=2, keepdim=False).mean(
        axis=2, keepdim=False)
    print("channel_map = ", channel_map.shape)
    C_attention = C * F.softmax(channel_map/temp, dim=1)

    return S_attention, C_attention

class CSAM(nn.Module):
    def __init__(self):
        super(CSAM, self).__init__()
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

x = torch.randn([1,2,3,4])
t = 1
# sa, ca = get_attention(x, t)
# print(sa.shape)
# print(ca.shape)

net = CSAM()
cs_a = net(x)
print(cs_a.shape)