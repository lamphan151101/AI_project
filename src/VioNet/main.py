import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from epoch import train, val, test
from model import VioNet_C3D, VioNet_ConvLSTM, VioNet_densenet, VioNet_densenet_lean
from dataset import VioDB
from config import Config

from spatial_transforms import Compose, ToTensor, Normalize
from spatial_transforms import GroupRandomHorizontalFlip, GroupRandomScaleCenterCrop, GroupScaleCenterCrop
from temporal_transforms import CenterCrop, RandomCrop
from target_transforms import Label, Video

from utils import Log


# Hàm `main` là hàm chính để huấn luyện và đánh giá mô hình. Dưới đây là phân tích chi tiết của mã:

# 1. Hàm nhận một đối tượng `config` chứa các thông số cấu hình cho quá trình huấn luyện và đánh giá mô hình.

# 2. Dựa vào giá trị của `config.model`, hàm tải mô hình và lấy danh sách các tham số cần tối ưu tương ứng. Các mô hình có thể được chọn là C3D, ConvLSTM, DenseNet hoặc DenseNet nhỏ hơn (DenseNet-88).

# 3. Hàm tiếp tục bằng việc thiết lập các thông số cho tập dữ liệu huấn luyện và đánh giá:
#    - Tạo các biến để chứa thông tin về tập dữ liệu, kích thước mẫu, bước nhảy, và thời lượng mẫu từ `config`.
#    - Cấu hình quá trình cross validation bằng cách lấy giá trị `config.num_cv`.

# 4. Tạo tập dữ liệu huấn luyện:
#    - Thiết lập các phép biến đổi không gian và thời gian cho dữ liệu huấn luyện.
#    - Tạo đối tượng `VioDB` để đọc dữ liệu từ thư mục và tệp JSON tương ứng với tập dữ liệu huấn luyện.
#    - Sử dụng `DataLoader` để tạo iterator cho tập dữ liệu huấn luyện.

# 5. Tạo tập dữ liệu đánh giá:
#    - Thiết lập các phép biến đổi không gian và thời gian cho dữ liệu đánh giá.
#    - Tạo đối tượng `VioDB` tương tự như bước trước cho tập dữ liệu đánh giá.
#    - Sử dụng `DataLoader` để tạo iterator cho tập dữ liệu đánh giá.

# 6. Tạo các thư mục lưu trữ kết quả nếu chưa tồn tại.

# 7. Khởi tạo hàm mất mát (`nn.CrossEntropyLoss`) và tối ưu hóa (`torch.optim.SGD`).

# 8. Khởi tạo giảm giá trị learning rate (`torch.optim.lr_scheduler.ReduceLROnPlateau`) để điều chỉnh tỷ lệ học tập trong quá trình huấn luyện.

# 9. Thiết lập giá trị mục tiêu (`acc_baseline` và `loss_baseline`) để so sánh và lưu trữ mô hình tốt nhất.

# 10. Vòng lặp qua số epoch được xác định trong `config.num_epoch`:
#     - Gọi hàm `train` để huấn luyện mô hình trên tập dữ liệu huấn luy

# ện.
#     - Gọi hàm `val` để đánh giá mô hình trên tập dữ liệu đánh giá.
#     - Cập nhật giá trị learning rate dựa trên giá trị mất mát trên tập dữ liệu đánh giá.
#     - Nếu đạt được độ chính xác hoặc mất mát tốt hơn so với giá trị mục tiêu, lưu trữ mô hình.

# 11. Cuối cùng, trong hàm `__main__`:
#     - Xác định thiết bị tính toán (GPU hoặc CPU) dựa trên khả năng có sẵn.
#     - Tạo một đối tượng `Config` để cấu hình huấn luyện và đánh giá.
#     - Cấu hình thông số huấn luyện cho các tập dữ liệu khác nhau (`hockey`, `movie`, `vif`).
#     - Với mỗi tập dữ liệu, thực hiện 5 lần cross validation bằng cách thiết lập `config.num_cv` và gọi hàm `main` để huấn luyện và đánh giá mô hình.
def main(config):
    # load model
    if config.model == 'c3d':
        model, params = VioNet_C3D(config)
    elif config.model == 'convlstm':
        model, params = VioNet_ConvLSTM(config)
    elif config.model == 'densenet':
        model, params = VioNet_densenet(config)
    elif config.model == 'densenet_lean':
        model, params = VioNet_densenet_lean(config)
    # default densenet
    else:
        model, params = VioNet_densenet_lean(config)

    # dataset
    dataset = config.dataset
    sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration

    # cross validation phase
    cv = config.num_cv

    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [crop_method,
         GroupRandomHorizontalFlip(),
         ToTensor(), norm])
    temporal_transform = RandomCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    train_batch = config.train_batch
    train_data = VioDB('F:\AVSS2019\src\VioDB\{}_jpg'.format(dataset),
                       'F:\AVSS2019\src\VioDB\{}_jpg{}.json'.format(dataset, cv), 'training',
                       spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    temporal_transform = CenterCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    val_batch = config.val_batch

    val_data = VioDB('F:\AVSS2019\src\VioDB\{}_jpg'.format(dataset),
                     'F:\AVSS2019\src\VioDB\{}_jpg{}.json'.format(dataset, cv), 'validation',
                     spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # make dir
    if not os.path.exists('./pth'):
        os.mkdir('./pth')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # log
    batch_log = Log(
        './log/{}_fps{}_{}_batch{}.log'.format(
            config.model,
            sample_duration,
            dataset,
            cv,
        ), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    epoch_log = Log(
        './log/{}_fps{}_{}_epoch{}.log'.format(config.model, sample_duration,
                                               dataset, cv),
        ['epoch', 'loss', 'acc', 'lr'])
    val_log = Log(
        './log/{}_fps{}_{}_val{}.log'.format(config.model, sample_duration,
                                             dataset, cv),
        ['epoch', 'loss', 'acc'])

    # prepare
    criterion = nn.CrossEntropyLoss().to(device)

    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay

    optimizer = torch.optim.SGD(params=params,
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           factor=config.factor,
                                                           min_lr=config.min_lr)

    acc_baseline = config.acc_baseline
    loss_baseline = 1

    for i in range(config.num_epoch):
        train(i, train_loader, model, criterion, optimizer, device, batch_log,
              epoch_log)
        val_loss, val_acc = val(i, val_loader, model, criterion, device,
                                val_log)
        scheduler.step(val_loss)
        if val_acc > acc_baseline or (val_acc >= acc_baseline and
                                      val_loss < loss_baseline):
            torch.save(
                model.state_dict(),
                './pth/{}_fps{}_{}{}_{}_{:.4f}_{:.6f}.pth'.format(
                    config.model, sample_duration, dataset, cv, i, val_acc,
                    val_loss))
            acc_baseline = val_acc
            loss_baseline = val_loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = Config(
        'densenet_lean',  # c3d, convlstm, densenet, densenet_lean
        'hockey',
        device=device,
        num_epoch=20,
        acc_baseline=0.92,
        ft_begin_idx=0,
    )

    # train params for different datasets
    configs = {
        'hockey': {
            'lr': 1e-2,
            'batch_size': 32
        },
        'movie': {
            'lr': 1e-3,
            'batch_size': 16
        },
        'vif': {
            'lr': 1e-3,
            'batch_size': 16
        }
    }

    for dataset in ['hockey', 'movie', 'vif']:
        config.dataset = dataset
        config.train_batch = configs[dataset]['batch_size']
        config.val_batch = configs[dataset]['batch_size']
        config.learning_rate = configs[dataset]['lr']
        # 5 fold cross validation

        config.num_cv = 1
        main(config)
