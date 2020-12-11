import numpy as np

import struct

import matplotlib.pyplot as plt

# 訓練集檔案

train_images_idx3_ubyte_file = 'C:\\Users\\wala481652\\Documents\\VSCode\\Python\\CNNs\\bin\\train-images-idx3-ubyte.gz'

# 訓練集標籤檔案

train_labels_idx1_ubyte_file = 'C:\\Users\\wala481652\\Documents\\VSCode\\Python\\CNNs\\bin\\train-labels-idx1-ubyte.gy'

# 測試集檔案

test_images_idx3_ubyte_file = 'C:\\Users\\wala481652\\Documents\\VSCode\\Python\\CNNs\\bin\\t10k-images-idx3-ubyte.gy'

# 測試集標籤檔案

test_labels_idx1_ubyte_file = 'C:\\Users\\wala481652\\Documents\\VSCode\\Python\\CNNs\\bin\\t10k-labels-idx1-ubyte.gy'


def decode_idx3_ubyte(idx3_ubyte_file):
    """

    解析idx3檔案的通用函式

    :param idx3_ubyte_file: idx3檔案路徑

    :return: 資料集

    """

    # 讀取二進位制資料

    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析檔案頭資訊，依次為魔數、圖片數量、每張圖片高、每張圖片寬

    offset = 0

    fmt_header = '>iiii'

    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)

    print('魔數:%d, 圖片數量: %d張, 圖片大小: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))

    # 解析資料集

    image_size = num_rows * num_cols

    offset += struct.calcsize(fmt_header)

    fmt_image = '>' + str(image_size) + 'B'

    images = np.empty((num_images, num_rows, num_cols))

    for i in range(num_images):

        if (i + 1) % 10000 == 0:

            print('已解析 %d' % (i + 1) + '張')

        images[i] = np.array(struct.unpack_from(
            fmt_image, bin_data, offset)).reshape((num_rows, num_cols))

        offset += struct.calcsize(fmt_image)

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """

    解析idx1檔案的通用函式

    :param idx1_ubyte_file: idx1檔案路徑

    :return: 資料集

    """

    # 讀取二進位制資料

    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析檔案頭資訊，依次為魔數和標籤數

    offset = 0

    fmt_header = '>ii'

    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    print('魔數:%d, 圖片數量: %d張' % (magic_number, num_images))

    # 解析資料集

    offset += struct.calcsize(fmt_header)

    fmt_image = '>B'

    labels = np.empty(num_images)

    for i in range(num_images):

        if (i + 1) % 10000 == 0:

            print('已解析 %d' % (i + 1) + '張')

        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]

        offset += struct.calcsize(fmt_image)

    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """

    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

    :param idx_ubyte_file: idx檔案路徑

    :return: n*row*col維np.array物件，n為圖片數量

    """

    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """

    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

    :param idx_ubyte_file: idx檔案路徑

    :return: n*1維np.array物件，n為圖片數量

    """

    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """

    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):

    :param idx_ubyte_file: idx檔案路徑

    :return: n*row*col維np.array物件，n為圖片數量

    """

    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """

    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):

    :param idx_ubyte_file: idx檔案路徑

    :return: n*1維np.array物件，n為圖片數量

    """
