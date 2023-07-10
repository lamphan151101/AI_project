import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

# Đoạn mã mô tả một mô hình DenseNet cho phân loại video và bao gồm một hàm `VioNet` để tạo ra một phiên bản của mô hình này. Dưới đây là mô tả chi tiết của đoạn mã:

# 1. Đoạn mã nhập các module cần thiết từ thư viện PyTorch.

# 2. Lớp `_DenseLayer` đại diện cho một lớp đơn trong DenseBlock của kiến trúc DenseNet. Nó bao gồm hai lớp tích chập với chức năng chuẩn hóa batch và kích hoạt ReLU.

# 3. Lớp `_DenseBlock` đại diện cho một khối gồm nhiều lớp `_DenseLayer`. Nó xếp chồng nhiều lớp lại với mỗi lớp nhận đầu vào là sự kết hợp của đầu ra của các lớp trước và đầu vào.

# 4. Lớp `_Transition` đại diện cho một khối chuyển tiếp trong kiến trúc DenseNet. Nó thực hiện chuẩn hóa batch, kích hoạt ReLU, tích chập 1x1 và pooling trung bình để giảm kích thước không gian.

# 5. Lớp `DenseNet` đại diện cho mô hình DenseNet tổng thể. Nó bao gồm nhiều khối `_DenseBlock` và `_Transition`, cùng với lớp tích chập ban đầu, chuẩn hóa batch và lớp phân loại tuyến tính.

# 6. Hàm `VioNet` tạo ra một phiên bản của mô hình `DenseNet` được thiết kế đặc biệt cho phân loại video. Nó thiết lập các tham số cần thiết như số lớp, thời lượng mẫu, kích thước mẫu, số lượng tính năng ban đầu, tốc độ tăng trưởng và cấu hình khối.

# 7. Nếu `pretrained` được đặt là `True`, hàm sẽ tải các trọng số đã được huấn luyện trước cho mô hình từ một tệp được chỉ định bởi đường dẫn `'F:\AVSS2019\src\Demo\weights.pth'` (giả sử tệp tồn tại). Trọng số được tải lên CPU sử dụng `map_location=torch.device('cpu')`.

# 8. Cuối cùng, hàm trả về mô hình `DenseNet` đã được tạo.

# Xin lưu ý rằng đoạn mã cung cấp có thể chưa hoàn chỉnh, vì nó đề cập đến các phần khác của mã mà không được bao gồm.
class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 sample_size,
                 sample_duration,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super(DenseNet, self).__init__()

        self.sample_size = sample_size
        self.sample_duration = sample_duration

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(3,
                           num_init_features,
                           kernel_size=7,
                           stride=(1, 2, 2),
                           padding=(3, 3, 3),
                           bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)

        out = self.classifier(out)
        return out


def VioNet(pretrained=True):
    model = DenseNet(
        num_classes=2,
        sample_duration=16,
        sample_size=112,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24),
    )

    if pretrained:
        state_dict = torch.load('F:\AVSS2019\src\Demo\weights.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    return model
