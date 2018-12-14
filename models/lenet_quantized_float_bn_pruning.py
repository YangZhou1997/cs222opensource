from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
from .modules.quantize import quantize, quantize_grad, RangeBN
from .modules.quantize import QLinear_P as  QLinear
from .modules.quantize import QConv2d_P as  QConv2d
__all__ = ['lenet_quantized_float_bn_pruning']

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 16


def quan_conv(in_planes, out_planes, kernel_size, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=kernel_size, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)


def init_model(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = quan_conv(1, 10, kernel_size=5)
        self.conv2 = quan_conv(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = QLinear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def lenet_quantized_float_bn_pruning(**kwargs):
	return Net()
