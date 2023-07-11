import os
import sys
import json

from sklearn.model_selection import train_test_split

# Hàm `load_labels(database_path)` trong đoạn code trên có chức năng tải danh sách nhãn từ thư mục chứa dữ liệu. Dưới đây là mô tả tóm tắt của hàm này:

# 1. Tham số đầu vào:
#    - `database_path`: Đường dẫn đến thư mục chứa dữ liệu.

# 2. Tạo một danh sách rỗng để lưu trữ các nhãn.

# 3. Kiểm tra xem `database_path` có phải là một thư mục tồn tại hay không bằng cách sử dụng `os.path.isdir(database_path)`.
#    - Nếu `database_path` là một thư mục tồn tại, tiến hành bước tiếp theo.
#    - Nếu `database_path` không phải là một thư mục tồn tại, danh sách nhãn sẽ vẫn rỗng.

# 4. Sử dụng `os.listdir(database_path)` để lấy danh sách các tệp tin và thư mục trong `database_path`.
#    - Các tệp tin và thư mục được trả về dưới dạng một danh sách.

# 5. Gán danh sách nhãn bằng danh sách các tệp tin và thư mục đã lấy được.

# 6. Trả về danh sách nhãn.

# Tóm lại, hàm `load_labels` được sử dụng để tải danh sách nhãn từ thư mục chứa dữ liệu và trả về danh sách này.
def load_labels(database_path):
    labels= []
    if os.path.isdir(database_path):
        labels = os.listdir(database_path)
    return labels

# Hàm `get_dataset(database_path)` trong đoạn code trên có chức năng xây dựng dữ liệu huấn luyện và dữ liệu kiểm tra từ thư mục chứa dữ liệu. Dưới đây là mô tả tóm tắt của hàm này:

# 1. Tham số đầu vào:
#    - `database_path`: Đường dẫn đến thư mục chứa dữ liệu.

# 2. Kiểm tra xem `database_path` có tồn tại hay không bằng cách sử dụng `os.path.exists(database_path)`.
#    - Nếu `database_path` không tồn tại, ném ra một ngoại lệ `IOError` với thông báo "not exist path".

# 3. Khởi tạo hai danh sách rỗng để lưu trữ tên tệp tin của hai loại dữ liệu: `no_data` và `fi_data`.

# 4. Sử dụng vòng lặp `for` để lặp qua các tệp tin và thư mục trong `database_path` bằng cách sử dụng `os.listdir(database_path)`.
#    - Nếu tên của tệp tin bắt đầu bằng "fi", thực hiện một vòng lặp lồng để lặp qua các tệp tin trong thư mục "fi".
#    - Nếu tên của tệp tin bắt đầu bằng "no", thực hiện một vòng lặp lồng để lặp qua các tệp tin trong thư mục "no".

# 5. Sử dụng `train_test_split` từ thư viện Scikit-learn để chia các tệp tin thành hai tập: `no_train`, `no_test` và `fi_train`, `fi_test`.
#    - `no_train` và `fi_train` chứa các tệp tin dùng cho huấn luyện.
#    - `no_test` và `fi_test` chứa các tệp tin dùng cho kiểm tra.

# 6. Khởi tạo hai từ điển rỗng: `train_database` và `val_database`.

# 7. Sử dụng vòng lặp `for` để lặp qua từng tệp tin trong `no_train`.
#    - Tách phần tên tệp tin và phần mở rộng bằng `os.path.splitext(file_name)`.
#    - Thêm thông tin về tệp tin vào `train_database` với tên là `name`, thuộc tính `subset` là "training", và thuộc tính `annotations` với nhãn "no".

# 8. Sử dụng vòng lặp `for` để lặp qua từng tệp tin trong `fi_train`.
#    - Tương tự như bước 7, thêm thông tin về tệp tin vào `train_database` với nhãn "fi".

# 9. Sử dụng vòng lặp `for` để lặp qua từng tệp tin trong `no_test`.
#    - Tương tự như bước 7, thêm thông tin về tệp tin vào `val_database

# ` với thuộc tính `subset` là "validation" và thuộc tính `annotations` với nhãn "no".

# 10. Sử dụng vòng lặp `for` để lặp qua từng tệp tin trong `fi_test`.
#    - Tương tự như bước 9, thêm thông tin về tệp tin vào `val_database` với nhãn "fi".

# 11. Trả về `train_database` và `val_database` như là kết quả của hàm.

# Tóm lại, hàm `get_dataset` được sử dụng để xây dựng dữ liệu huấn luyện và dữ liệu kiểm tra từ thư mục chứa dữ liệu, dựa trên quy tắc chia tệp tin vào các tập huấn luyện và kiểm tra, và trả về hai từ điển chứa thông tin về dữ liệu huấn luyện và dữ liệu kiểm tra.
def get_dataset(database_path):
    if not os.path.exists(database_path):
        raise IOError('not exist path')

    no_data = []
    fi_data = []
    for i in os.listdir(database_path):
        if i[:2]  == 'fi':
            for j in os.listdir(database_path + '\\fi'):
              if j[:2]  == 'fi':
                  fi_data.append(j)
        elif i[:2] == 'no':
            for j in os.listdir(database_path + '\\no'):
              if j[:2]  == 'no':
                  no_data.append(j)
    no_train, no_test = train_test_split(no_data, test_size=0.2, shuffle=True)
    fi_train, fi_test = train_test_split(fi_data, test_size=0.2, shuffle=True)

    train_database = {}
    for file_name in no_train:
        name, _ = os.path.splitext(file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'no'}
    for file_name in fi_train:
        name, _ = os.path.splitext(file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'fi'}

    val_database = {}
    for file_name in no_test:
        name, _ = os.path.splitext(file_name)
        val_database[name] = {}
        val_database[name]['subset'] = 'validation'
        val_database[name]['annotations'] = {'label': 'no'}
    for file_name in fi_test:
        name, _ = os.path.splitext(file_name)
        val_database[name] = {}
        val_database[name]['subset'] = 'validation'
        val_database[name]['annotations'] = {'label': 'fi'}

    return train_database, val_database

# Hàm `generate_annotation(database_path, dst_json_path)` trong đoạn code trên có chức năng tạo và lưu trữ các thông tin về dữ liệu và nhãn vào một tệp JSON. Dưới đây là mô tả tóm tắt của hàm này:

# 1. Tham số đầu vào:
#    - `database_path`: Đường dẫn đến thư mục chứa dữ liệu.
#    - `dst_json_path`: Đường dẫn đến tệp JSON đích để lưu trữ thông tin.

# 2. Gọi hàm `load_labels(database_path)` để tải danh sách nhãn từ `database_path`.

# 3. Gọi hàm `get_dataset(database_path)` để tạo dữ liệu huấn luyện và dữ liệu kiểm tra từ `database_path`.

# 4. Khởi tạo một từ điển rỗng `dst_data` để lưu trữ thông tin dữ liệu và nhãn.

# 5. Thêm danh sách nhãn vào `dst_data` với khóa 'labels'.

# 6. Khởi tạo một từ điển rỗng `database` trong `dst_data` để lưu trữ thông tin về cơ sở dữ liệu.

# 7. Sử dụng phương thức `update()` để thêm thông tin về dữ liệu huấn luyện từ `train_database` vào `database`.

# 8. Sử dụng phương thức `update()` để thêm thông tin về dữ liệu kiểm tra từ `val_database` vào `database`.

# 9. Mở tệp JSON đích `dst_json_path` để ghi dữ liệu vào.

# 10. Sử dụng `json.dump()` để ghi dữ liệu từ `dst_data` vào tệp JSON.

# Tóm lại, hàm `generate_annotation` tải danh sách nhãn, tạo dữ liệu huấn luyện và kiểm tra, sau đó lưu trữ thông tin dữ liệu và nhãn vào một tệp JSON theo đường dẫn chỉ định.
def generate_annotation(database_path, dst_json_path):
    labels = load_labels(database_path)
    train_database, val_database = get_dataset(database_path)

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    database_path = "F:\AVSS2019\src\VioDB\hockey_jpg"
    dst_json_path = database_path + '1.json'

    generate_annotation(database_path, dst_json_path)
