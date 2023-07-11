import torch
import torch.nn as nn

import models.densenet as dn
from models.c3d import C3D
from models.densenet import densenet88, densenet121
from models.convlstm import ConvLSTM

# Hàm `VioNet_C3D` khởi tạo một mô hình C3D cho nhận dạng hành động trong video. Dưới đây là phân tích chi tiết của mã:

# 1. Hàm nhận một đối tượng `config` chứa thông tin về thiết bị được sử dụng cho tính toán.

# 2. Hàm tạo một phiên bản của mô hình C3D (`C3D(num_classes=2)`) và gán cho biến `model`.

# 3. Hàm tải trọng số được huấn luyện trước cho mô hình C3D từ tệp `'F:\AVSS2019\src\Demo\weights\C3D_Kinetics.pth'` bằng cách sử dụng `torch.load`. Bộ trạng thái được tải được gán cho mô hình bằng `model.load_state_dict(state_dict)`.

# 4. Hàm lấy các tham số của mô hình bằng `model.parameters()` và gán cho biến `params`.

# 5. Cuối cùng, hàm trả về mô hình C3D đã được khởi tạo (`model`) và các tham số cần được tối ưu hóa trong quá trình huấn luyện (`params`).

# Xin lưu ý rằng đường dẫn `'F:\AVSS2019\src\Demo\weights\C3D_Kinetics.pth'` đã được đặt cứng trong mã nguồn, và bạn có thể cần chỉnh sửa nó để phù hợp với vị trí thực tế của trọng số được huấn luyện trước trên hệ thống của bạn.
def VioNet_C3D(config):
    device = config.device
    model = C3D(num_classes=2).to(device)

    state_dict = torch.load('F:\AVSS2019\src\Demo\weights\C3D_Kinetics.pth')
    model.load_state_dict(state_dict)
    params = model.parameters()

    return model, params

# Hàm `VioNet_ConvLSTM` khởi tạo một mô hình ConvLSTM để nhận dạng hành động trong video. Dưới đây là phân tích chi tiết của mã:

# 1. Hàm nhận một đối tượng `config` chứa thông tin về thiết bị được sử dụng cho tính toán.

# 2. Hàm tạo một phiên bản của mô hình ConvLSTM (`ConvLSTM(256, device)`) và gán cho biến `model`. Đối số `256` là số kênh đầu vào và `device` là thiết bị tính toán (GPU hoặc CPU).

# 3. Vòng lặp `for` được sử dụng để lặp qua tất cả các tên và tham số của mô hình. Trong mô hình ConvLSTM, các tham số của phần mạng tích chập (`conv_net`) được đóng băng bằng cách đặt `param.requires_grad = False`. Điều này có nghĩa là các tham số này sẽ không được cập nhật trong quá trình huấn luyện.

# 4. Hàm lấy các tham số của mô hình bằng `model.parameters()` và gán cho biến `params`.

# 5. Cuối cùng, hàm trả về mô hình ConvLSTM đã được khởi tạo (`model`) và các tham số cần được tối ưu hóa trong quá trình huấn luyện (`params`).

# Xin lưu ý rằng phần mạng tích chập của mô hình ConvLSTM được đóng băng để giữ nguyên các trọng số đã được huấn luyện trước (giả sử là từ mô hình AlexNet). Điều này đảm bảo rằng chỉ có các phần ConvLSTM mới được huấn luyện trong quá trình huấn luyện tiếp theo.
def VioNet_ConvLSTM(config):
    device = config.device
    model = ConvLSTM(256, device).to(device)
    # freeze pretrained alexnet params
    for name, param in model.named_parameters():
        if 'conv_net' in name:
            param.requires_grad = False
    params = model.parameters()

    return model, params

# Hàm `VioNet_densenet` khởi tạo một mô hình DenseNet để nhận dạng hành động trong video. Dưới đây là phân tích chi tiết của mã:

# 1. Hàm nhận một đối tượng `config` chứa thông tin về thiết bị được sử dụng cho tính toán, chỉ số bắt đầu của phần được fine-tuning, kích thước mẫu và thời lượng mẫu.

# 2. Hàm khởi tạo một phiên bản mô hình DenseNet (`densenet121`) với các đối số sau:
#    - `num_classes=2`: Số lượng lớp đầu ra, trong trường hợp này là 2 (hành động và không phải hành động).
#    - `sample_size=sample_size`: Kích thước mẫu đầu vào, được lấy từ `config`.
#    - `sample_duration=sample_duration`: Thời lượng mẫu đầu vào, được lấy từ `config`.
#    Mô hình được chuyển đến thiết bị tính toán (GPU hoặc CPU) được chỉ định trong `config`.

# 3. State dict (trạng thái của mô hình được lưu trữ) được tải từ file 'F:\AVSS2019\src\Demo\weights\DenseNet_Kinetics.pth'. State dict chứa trọng số đã được huấn luyện trước cho mô hình DenseNet.

# 4. Trạng thái đã tải được gán cho mô hình bằng `model.load_state_dict(state_dict)` để khôi phục các trọng số đã được huấn luyện trước.

# 5. Hàm `dn.get_fine_tuning_params(model, ft_begin_idx)` được sử dụng để trả về danh sách các tham số cần được tối ưu hóa trong quá trình huấn luyện. Các tham số này thuộc về phần được fine-tuning của mô hình, bắt đầu từ chỉ số `ft_begin_idx`.

# 6. Cuối cùng, hàm trả về mô hình DenseNet đã được khởi tạo (`model`) và danh sách các tham số cần được tối ưu hóa trong quá trình huấn luyện (`params`).
def VioNet_densenet(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet121(num_classes=2,
                        sample_size=sample_size,
                        sample_duration=sample_duration).to(device)

    state_dict = torch.load('F:\AVSS2019\src\Demo\weights\DenseNet_Kinetics.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params


# the model we finally adopted in DenseNet

# Hàm `VioNet_densenet_lean` tạo một mô hình DenseNet nhỏ hơn (DenseNet-88) để nhận dạng hành động trong video. Dưới đây là phân tích chi tiết của mã:

# 1. Hàm nhận một đối tượng `config` chứa thông tin về thiết bị được sử dụng cho tính toán, chỉ số bắt đầu của phần được fine-tuning, kích thước mẫu và thời lượng mẫu.

# 2. Hàm khởi tạo một phiên bản mô hình DenseNet nhỏ hơn (`densenet88`) với các đối số sau:
#    - `num_classes=2`: Số lượng lớp đầu ra, trong trường hợp này là 2 (hành động và không phải hành động).
#    - `sample_size=sample_size`: Kích thước mẫu đầu vào, được lấy từ `config`.
#    - `sample_duration=sample_duration`: Thời lượng mẫu đầu vào, được lấy từ `config`.
#    Mô hình được chuyển đến thiết bị tính toán (GPU hoặc CPU) được chỉ định trong `config`.

# 3. State dict (trạng thái của mô hình được lưu trữ) được tải từ file 'F:\AVSS2019\src\Demo\weights\DenseNetLean_Kinetics.pth'. State dict chứa trọng số đã được huấn luyện trước cho mô hình DenseNet nhỏ hơn.

# 4. Trạng thái đã tải được gán cho mô hình bằng `model.load_state_dict(state_dict)` để khôi phục các trọng số đã được huấn luyện trước.

# 5. Hàm `dn.get_fine_tuning_params(model, ft_begin_idx)` được sử dụng để trả về danh sách các tham số cần được tối ưu hóa trong quá trình huấn luyện. Các tham số này thuộc về phần được fine-tuning của mô hình, bắt đầu từ chỉ số `ft_begin_idx`.

# 6. Cuối cùng, hàm trả về mô hình DenseNet nhỏ hơn đã được khởi tạo (`model`) và danh sách các tham số cần được tối ưu hóa trong quá trình huấn luyện (`params`).
def VioNet_densenet_lean(config):
    device = config.device
    ft_begin_idx = config.ft_begin_idx
    sample_size = config.sample_size[0]
    sample_duration = config.sample_duration

    model = densenet88(num_classes=2,
                       sample_size=sample_size,
                       sample_duration=sample_duration).to(device)

    state_dict = torch.load('F:\AVSS2019\src\Demo\weights\DenseNetLean_Kinetics.pth')
    model.load_state_dict(state_dict)

    params = dn.get_fine_tuning_params(model, ft_begin_idx)

    return model, params
