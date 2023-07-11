import time
import torch
from utils import AverageMeter

# Hàm `train` đã được triển khai và có chức năng huấn luyện mô hình trên dữ liệu trong mỗi epoch. Dưới đây là mô tả chi tiết về hàm này:

# Hàm `train` nhận các đối số như `epoch` (số thứ tự của epoch), `data_loader` (dataloader cho dữ liệu huấn luyện), `model` (mô hình), `criterion` (hàm mất mát), `optimizer` (bộ tối ưu hóa), `device` (thiết bị tính toán), `batch_log` (đối tượng ghi log batch) và `epoch_log` (đối tượng ghi log epoch).

# Trong hàm này, các đối tượng AverageMeter được tạo ra để tính giá trị trung bình của các thông số như thời gian xử lý batch, thời gian xử lý dữ liệu, mất mát và độ chính xác trong quá trình huấn luyện.

# Mô hình được đặt ở chế độ huấn luyện bằng cách gọi `model.train()`. Sau đó, một vòng lặp for được thực hiện trên `data_loader` để lấy các đầu vào (`inputs`) và nhãn (`targets`) của các batch dữ liệu.

# Các đầu vào và nhãn được chuyển đến thiết bị tính toán (`device`). Thời gian xử lý dữ liệu được tính và cập nhật bằng cách tính thời gian giữa các lần lấy dữ liệu (`data_time`).

# Trước khi tiến hành lan truyền tiến, gradient của các tham số trong mô hình được đặt về 0 bằng cách gọi `optimizer.zero_grad()`.

# Tiến trình lan truyền tiến được thực hiện bằng cách gọi `outputs = model(inputs)` để lấy đầu ra dự đoán từ mô hình. Mất mát (`loss`) được tính bằng cách sử dụng hàm mất mát (`criterion`) trên đầu ra dự đoán và nhãn thực tế. Độ chính xác (`acc`) được tính bằng cách sử dụng hàm `calculate_accuracy` trên đầu ra dự đoán và nhãn thực tế.

# Các thông số như mất mát và độ chính xác được cập nhật bằng cách gọi các phương thức của đối tượng AverageMeter (`losses.update` và `accuracies.update`).

# Sau đó, quá trình lan truyền ngược và cập nhật các tham số trong mô hình được thực hiện bằng cách gọi `loss.backward()` và `optimizer.step()`.

# Thông tin về quá trình huấn luyện, bao gồm thời gian, mất mát và độ chính xác,

#  được in ra màn hình sử dụng hàm `print`.

# Thông tin của batch cũng được ghi log bằng cách gọi `batch_log.log`.

# Cuối cùng, thông tin của epoch, bao gồm mất mát và độ chính xác trung bình của epoch, cũng như learning rate, được ghi log bằng cách gọi `epoch_log.log`.

# Đây là một hàm quan trọng trong quá trình huấn luyện mô hình và nó sẽ thực hiện các bước cần thiết để cập nhật trọng số của mô hình dựa trên gradient tính toán từ mất mát và lan truyền ngược.
def train(
    epoch, data_loader, model, criterion, optimizer, device, batch_log,
    epoch_log
):
    print('training at epoch: {}'.format(epoch))

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # set model to training mode
    model.train()

    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        data_time.update(time.time() - end_time)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        # meter
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        # backward + optimize
        loss.backward()
        optimizer.step()

        # meter
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies
            )
        )

        batch_log.log(
            {
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            }
        )

    epoch_log.log(
        {
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': optimizer.param_groups[0]['lr']
        }
    )

# Hàm `val` được sử dụng để thực hiện quá trình đánh giá (validation) của mô hình trên tập dữ liệu validation trong mỗi epoch. Dưới đây là mô tả chi tiết về hàm này:

# Hàm `val` nhận các đối số như `epoch` (số thứ tự của epoch), `data_loader` (dataloader cho dữ liệu validation), `model` (mô hình), `criterion` (hàm mất mát), `device` (thiết bị tính toán) và `val_log` (đối tượng ghi log validation).

# Trong hàm này, mô hình được đặt ở chế độ đánh giá bằng cách gọi `model.eval()`. Điều này đảm bảo rằng các lớp như Dropout hoặc BatchNorm trong mô hình sẽ hoạt động ở chế độ đánh giá.

# Các đối tượng AverageMeter được tạo ra để tính giá trị trung bình của các thông số như mất mát và độ chính xác trong quá trình đánh giá.

# Sau đó, một vòng lặp for được thực hiện trên `data_loader` để lấy các đầu vào (`inputs`) và nhãn (`targets`) của các batch dữ liệu trong quá trình đánh giá.

# Các đầu vào và nhãn được chuyển đến thiết bị tính toán (`device`). Trong chế độ đánh giá (`with torch.no_grad()`), không cần theo dõi gradient.

# Đầu ra dự đoán (`outputs`) được tính bằng cách gọi `model(inputs)`. Mất mát (`loss`) được tính bằng cách sử dụng hàm mất mát (`criterion`) trên đầu ra dự đoán và nhãn thực tế. Độ chính xác (`acc`) được tính bằng cách sử dụng hàm `calculate_accuracy` trên đầu ra dự đoán và nhãn thực tế.

# Các thông số như mất mát và độ chính xác được cập nhật bằng cách gọi các phương thức của đối tượng AverageMeter (`losses.update` và `accuracies.update`).

# Thông tin về quá trình đánh giá, bao gồm mất mát và độ chính xác trung bình, được in ra màn hình sử dụng hàm `print`.

# Thông tin của epoch, bao gồm mất mát và độ chính xác trung bình của epoch, cũng được ghi log bằng cách gọi `val_log.log`.

# Cuối cùng, hàm trả về giá trị trung bình của mất mát và độ chính xác trên tập dữ liệu validation.
def val(epoch, data_loader, model, criterion, device, val_log):
    print('validation at epoch: {}'.format(epoch))

    # set model to evaluate mode
    model.eval()

    # meters
    losses = AverageMeter()
    accuracies = AverageMeter()

    for _, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # no need to track grad in eval mode
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    print(
        'Epoch: [{}]\t'
        'Loss(val): {loss.avg:.4f}\t'
        'Acc(val): {acc.avg:.3f}'.format(epoch, loss=losses, acc=accuracies)
    )

    val_log.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg, accuracies.avg


def test():
    pass

# Hàm `calculate_accuracy` được sử dụng để tính toán độ chính xác của mô hình dự đoán trên tập dữ liệu validation hoặc test. Dưới đây là mô tả chi tiết về hàm này:

# Hàm `calculate_accuracy` nhận hai đối số là `outputs` (đầu ra dự đoán của mô hình) và `targets` (nhãn thực tế).

# Trước tiên, hàm lấy kích thước của batch (`batch_size`) từ kích thước của nhãn (`targets`). Điều này cho phép tính toán số lượng phần tử trong batch.

# Tiếp theo, hàm sử dụng phương thức `topk` trên đầu ra dự đoán (`outputs`) để lấy chỉ mục của lớp có xác suất dự đoán cao nhất. Điều này được thực hiện bằng cách gọi `outputs.topk(1, 1, True)`. Kết quả là hai giá trị, giá trị thứ nhất (`_`) không được sử dụng trong hàm này và giá trị thứ hai (`pred`) chứa chỉ mục của lớp dự đoán cao nhất cho mỗi mẫu trong batch.

# Tiếp theo, giá trị `pred` được chuyển vị (`pred.t()`) để có cùng chiều với nhãn thực tế. Điều này cho phép so sánh trực tiếp giữa các lớp dự đoán và nhãn thực tế.

# Hàm sử dụng phép so sánh `eq` để so sánh lớp dự đoán và nhãn thực tế. Kết quả là một tensor boolean có cùng kích thước như `pred` và `targets`, trong đó giá trị `True` tương ứng với dự đoán chính xác và giá trị `False` tương ứng với dự đoán sai.

# Sau đó, hàm sử dụng phương thức `float` và `sum` trên tensor boolean `correct` để tính toán số lượng dự đoán chính xác trong batch (`n_correct_elems`).

# Cuối cùng, hàm trả về tỷ lệ độ chính xác bằng cách chia `n_correct_elems` cho `batch_size`.
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size
