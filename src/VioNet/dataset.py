import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

# Tất nhiên! Đây là một ví dụ về hàm `imread` được sử dụng để đọc và chuyển đổi định dạng ảnh sang RGB bằng thư viện PIL. Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận đối số `path`, đó là đường dẫn tới tệp ảnh cần đọc.

# 2. Hàm sử dụng phương thức `Image.open` từ thư viện PIL để mở tệp ảnh tại đường dẫn `path` đã chỉ định.

# 3. Ảnh sau đó được chuyển đổi sang chế độ màu RGB bằng cách sử dụng phương thức `convert` với đối số `'RGB'`.

# 4. Cuối cùng, ảnh đã được chuyển đổi được trả về.

# Hàm này có thể được sử dụng để đọc và chuyển đổi tệp ảnh sang định dạng RGB, đây là định dạng phổ biến trong xử lý ảnh và các tác vụ thị giác máy tính.
def imread(path):
    with Image.open(path) as img:
        return img.convert('RGB')

# Đây là một ví dụ về hàm `video_loader` được sử dụng để tải các khung hình từ một thư mục chứa video. Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận hai đối số: `video_dir_path` là đường dẫn đến thư mục chứa video và `frame_indices` là một danh sách các chỉ số khung hình cần tải.

# 2. Hàm tạo một danh sách rỗng có tên là `video` để lưu trữ các khung hình được tải.

# 3. Với mỗi chỉ số `i` trong `frame_indices`, hàm xây dựng đường dẫn tới tệp ảnh sử dụng phương thức `os.path.join` và chuỗi định dạng `'image_{:05d}.jpg'`. Đây giả định rằng các tệp ảnh được đặt tên theo một mẫu nhất định, ví dụ: `image_00001.jpg`, `image_00002.jpg`,...

# 4. Hàm kiểm tra xem tệp ảnh tại `image_path` có tồn tại bằng cách sử dụng `os.path.exists`. Nếu tệp tồn tại, hàm sẽ đọc ảnh sử dụng hàm `imread` và thêm ảnh vào danh sách `video`.

# 5. Nếu tệp ảnh không tồn tại cho một chỉ số nào đó, hàm sẽ trả về danh sách `video` hiện tại.

# 6. Sau khi duyệt qua tất cả các chỉ số trong `frame_indices` và tải tất cả các khung hình, danh sách `video` chứa các khung hình được trả về.

# Hàm này có thể được sử dụng để tải các khung hình từ một thư mục chứa video, dựa trên các chỉ số khung hình được cung cấp.
def video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(imread(image_path))
        else:
            return video

    return video

# Hàm `n_frames_loader` được sử dụng để tải số khung hình của một video từ tệp tin. Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận một đối số là `file_path`, đó là đường dẫn đến tệp tin chứa số khung hình của video.

# 2. Hàm mở tệp tin sử dụng `open(file_path, 'r')`. Trường hợp 'r' chỉ ra rằng tệp tin sẽ được mở để đọc.

# 3. Hàm đọc nội dung của tệp tin sử dụng `input_file.read()`. Điều này sẽ trả về một chuỗi chứa nội dung của tệp tin.

# 4. Hàm sử dụng `rstrip('\n\r')` để loại bỏ các ký tự xuống dòng (`\n`) và ký tự return (`\r`) cuối chuỗi nếu có.

# 5. Kết quả là một chuỗi số được trả về từ tệp tin. Hàm chuyển đổi chuỗi số này thành dạng số thực (`float`) bằng cách sử dụng `float()`.

# 6. Cuối cùng, hàm trả về số khung hình dưới dạng số thực.

# Hàm `n_frames_loader` này có thể được sử dụng để đọc số khung hình của video từ một tệp tin và trả về giá trị đó.
def n_frames_loader(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))

# Hàm `load_annotation_data` được sử dụng để tải dữ liệu chú thích từ một tệp tin JSON. Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận một đối số là `data_file_path`, đó là đường dẫn đến tệp tin chứa dữ liệu chú thích.

# 2. Hàm sử dụng `open(data_file_path, 'r')` để mở tệp tin dữ liệu. Trường hợp 'r' chỉ ra rằng tệp tin sẽ được mở để đọc.

# 3. Hàm sử dụng `json.load(data_file)` để tải dữ liệu chú thích từ tệp tin. `json.load()` sẽ đọc nội dung của tệp tin và chuyển đổi nó thành đối tượng Python tương ứng (trong trường hợp này là một từ điển hoặc danh sách, tùy thuộc vào cấu trúc của tệp tin JSON).

# 4. Kết quả là một đối tượng Python được trả về, chứa dữ liệu chú thích từ tệp tin JSON.

# Hàm `load_annotation_data` này có thể được sử dụng để tải dữ liệu chú thích từ một tệp tin JSON và trả về dữ liệu chú thích dưới dạng đối tượng Python để sử dụng trong việc xử lý dữ liệu.
def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

# Hàm `get_labels` được sử dụng để lấy danh sách các nhãn lớp từ dữ liệu chú thích. Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận một đối số là `data`, đó là đối tượng chứa dữ liệu chú thích.

# 2. Hàm khởi tạo một từ điển trống có tên là `class_labels_map`. Đây là nơi chúng ta sẽ lưu trữ ánh xạ từ nhãn lớp sang chỉ số.

# 3. Hàm sử dụng một biến `index` để theo dõi chỉ số của từng nhãn lớp.

# 4. Hàm lặp qua danh sách `data['labels']`, đây là danh sách các nhãn lớp có trong dữ liệu chú thích.

# 5. Trong mỗi vòng lặp, hàm thêm một cặp khóa-giá trị vào `class_labels_map`. Khóa là nhãn lớp và giá trị là chỉ số hiện tại.

# 6. Sau mỗi vòng lặp, hàm tăng giá trị của `index` để chuẩn bị cho nhãn lớp tiếp theo.

# 7. Kết quả là `class_labels_map`, một từ điển ánh xạ từ nhãn lớp sang chỉ số của nó.

# Hàm `get_labels` này có thể được sử dụng để lấy danh sách các nhãn lớp từ dữ liệu chú thích và tạo một ánh xạ từ nhãn lớp sang chỉ số của nó. Ánh xạ này có thể hữu ích khi làm việc với mô hình học máy yêu cầu các nhãn lớp được biểu diễn dưới dạng các chỉ số.
def get_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

# Hàm `get_video_names_and_labels` được sử dụng để lấy danh sách tên video và nhãn tương ứng từ dữ liệu chú thích dựa trên một tập con xác định. Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận hai đối số: `data` là đối tượng chứa dữ liệu chú thích và `subset` là tập con được chỉ định (ví dụ: "training", "validation", "test").

# 2. Hàm khởi tạo hai danh sách rỗng: `video_names` và `video_labels`. Đây là nơi chúng ta sẽ lưu trữ tên video và nhãn tương ứng.

# 3. Hàm lặp qua mọi cặp khóa-giá trị trong `data['database']`, đây là cơ sở dữ liệu video.

# 4. Trong mỗi vòng lặp, hàm kiểm tra nếu `val['subset']` (tập con của video hiện tại) trùng khớp với `subset` được chỉ định. Điều này đảm bảo chúng ta chỉ lấy thông tin video từ tập con cần thiết.

# 5. Nếu tập con trùng khớp, hàm lấy nhãn của video từ `val['annotations']['label']` và thêm tên video và nhãn tương ứng vào `video_names` và `video_labels`.

# 6. Sau khi lặp qua tất cả các video trong tập con, hàm trả về `video_names` và `video_labels`, danh sách tên video và nhãn tương ứng.

# Hàm `get_video_names_and_labels` này có thể được sử dụng để lấy danh sách tên video và nhãn tương ứng từ dữ liệu chú thích dựa trên một tập con xác định, chẳng hạn như "training", "validation", hoặc "test".
def get_video_names_and_labels(data, subset):
    video_names = []
    video_labels = []

    for key, val in data['database'].items():
        if val['subset'] == subset:
            label = val['annotations']['label']
            video_names.append(key)
            video_labels.append(label)

    return video_names, video_labels

# Hàm `make_dataset` được sử dụng để tạo tập dữ liệu từ đường dẫn gốc (`root_path`), tệp chú thích (`annotation_path`), và tập con xác định (`subset`). Dưới đây là cách hoạt động của hàm:

# 1. Hàm nhận ba đối số: `root_path` là đường dẫn gốc chứa các thư mục video, `annotation_path` là đường dẫn đến tệp chú thích dữ liệu (định dạng JSON), và `subset` là tập con được chỉ định (ví dụ: "training", "validation", "test").

# 2. Hàm sử dụng hàm `load_annotation_data(annotation_path)` để tải dữ liệu chú thích từ tệp JSON và lưu vào biến `data`.

# 3. Hàm sử dụng hàm `get_video_names_and_labels(data, subset)` để lấy danh sách tên video và nhãn tương ứng từ dữ liệu chú thích và tập con xác định. Kết quả được lưu vào hai danh sách `video_names` và `video_labels`.

# 4. Hàm sử dụng hàm `get_labels(data)` để tạo bản đồ từ nhãn lớp sang chỉ mục và lưu vào biến `class_to_index`.

# 5. Hàm tạo một từ điển `index_to_class` để ánh xạ từ chỉ mục về nhãn lớp. Để làm điều này, hàm lặp qua các cặp tên và nhãn trong `class_to_index` và thêm các mục tương ứng vào `index_to_class`.

# 6. Hàm khởi tạo một danh sách trống `dataset` để lưu trữ thông tin về từng video.

# 7. Hàm lặp qua các cặp tên video và nhãn tương ứng trong `video_names` và `video_labels`.

# 8. Trong mỗi vòng lặp, hàm tạo đường dẫn đến thư mục video bằng cách sử dụng `root_path`, nhãn lớp và tên video. Nếu thư mục video không tồn tại, hàm tiếp tục với vòng lặp tiếp theo.

# 9. Hàm sử dụng `n_frames_loader` để tải số khung hình của video từ tệp `n_frames` và chuyển đổi giá trị thành số nguyên.

# 10. Hàm tạo một từ điển `video` chứa thông tin về tên, đường dẫn, nhãn và số khung hình của video.

# 11. Từ điển `video` được thêm vào danh sách `dataset`.

# 12. Sau khi lặp qua tất cả các video, hàm trả về `dataset` (danh sách các video) và `index_to_class` (từ điển ánh xạ từ chỉ mục về nhãn lớ

# p).

# Hàm `make_dataset` này sẽ tạo ra một tập dữ liệu từ đường dẫn gốc, tệp chú thích và tập con xác định. Tập dữ liệu sẽ chứa thông tin về các video, bao gồm tên, đường dẫn, nhãn và số khung hình của từng video.
def make_dataset(root_path, annotation_path, subset):
    """
    :param root_path: xxx
    :param annotation_path: xxx.json
    :param subset: 'train', 'validation', 'test'
    :return: list_of_videos, index_to_class_decode
    """

    data = load_annotation_data(annotation_path)

    video_names, video_labels = get_video_names_and_labels(data, subset)

    class_to_index = get_labels(data)
    index_to_class = {}
    for name, label in class_to_index.items():
        index_to_class[label] = name

    dataset = []

    for video_name, video_label in zip(video_names, video_labels):
        video_path = os.path.join(
            root_path, video_label, video_name
        )  # $1/$2/$3

        if not os.path.exists(video_path):
            continue

        n_frames = int(n_frames_loader(os.path.join(video_path, 'n_frames')))

        video = {
            'name': video_name,
            'path': video_path,
            'label': class_to_index[video_label],
            'n_frames': n_frames
        }

        dataset.append(video)

    return dataset, index_to_class

# Lớp `VioDB` là một lớp dữ liệu tùy chỉnh được xây dựng trên lớp `Dataset` của PyTorch. Nó được sử dụng để tạo một dataset cho việc huấn luyện và đánh giá mô hình. Dưới đây là mô tả của từng phương thức trong lớp `VioDB`:

# 1. Phương thức `__init__`: Phương thức này được gọi khi khởi tạo một đối tượng `VioDB`. Nó nhận các đối số như `root_path` (đường dẫn gốc), `annotation_path` (đường dẫn tệp chú thích), `subset` (tập con), `spatial_transform` (biến đổi không gian), `temporal_transform` (biến đổi thời gian), và `target_transform` (biến đổi mục tiêu). Trong phương thức này, tập dữ liệu `videos` và từ điển `classes` được tạo bằng cách gọi hàm `make_dataset` với các tham số tương ứng. Các biến `spatial_transform`, `temporal_transform` và `target_transform` được lưu trữ để sử dụng trong các phương thức khác.

# 2. Phương thức `__getitem__`: Phương thức này được sử dụng để truy xuất một mục từ tập dữ liệu. Nó nhận một chỉ mục `index` và trả về một cặp `(clip, target)`. Đầu tiên, nó lấy đường dẫn `path` và số khung hình `n_frames` của video tại chỉ mục `index` từ tập dữ liệu `videos`. Tiếp theo, nó tạo danh sách các khung hình `frames` từ 1 đến `n_frames`. Nếu `temporal_transform` tồn tại, `frames` sẽ được biến đổi bằng cách gọi `temporal_transform` với `frames` làm đối số. Tiếp theo, nó sử dụng phương thức `loader` để tải video từ đường dẫn `path` và `frames`. Nếu `spatial_transform` tồn tại, video được biến đổi bằng cách gọi `spatial_transform` với `clip` làm đối số. Cuối cùng, video được chuyển thành tensor và trả về cùng với `target`.

# 3. Phương thức `__len__`: Phương thức này trả về số lượng video trong tập dữ liệu.

# Lớp `VioDB` giúp quản lý tập dữ liệu và cung cấp phương thức truy xuất mẫu và độ dài của tập dữ liệu để sử dụng trong quá trình huấn luyện và đánh giá mô hình.
class VioDB(Dataset):
    def __init__(
        self,
        root_path,
        annotation_path,
        subset,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None
    ):

        self.videos, self.classes = make_dataset(
            root_path, annotation_path, subset
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = video_loader

    def __getitem__(self, index):

        path = self.videos[index]['path']
        n_frames = self.videos[index]['n_frames']
        frames = list(range(1, 1 + n_frames))

        if self.temporal_transform:
            frames = self.temporal_transform(frames)

        clip = self.loader(path, frames)

        # clip list of images (H, W, C)
        if self.spatial_transform:
            clip = self.spatial_transform(clip)

        # clip: lists of tensors(C, H, W)
        clip = torch.stack(clip).permute(1, 0, 2, 3)

        target = self.videos[index]
        if self.target_transform:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.videos)
