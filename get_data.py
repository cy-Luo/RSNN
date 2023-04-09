import os
import gzip
import shutil
import urllib.request
import struct
import numpy as np

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}

def download_and_extract_mnist():
    if not os.path.exists('data'):
        os.mkdir('data')

    for file_type, file_name in FILES.items():
        gz_file_path = os.path.join('data', file_name)
        file_path = os.path.join('data', file_name[:-3])

        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(DATA_URL + file_name, gz_file_path)
            print(f"Extracting {file_name}...")
            with gzip.open(gz_file_path, 'rb') as gz_f:
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(gz_f, f)

def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images

def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_mnist():
    """加载MNIST数据集"""
    data_dir = './data'
    X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    download_and_extract_mnist()
    print("MNIST dataset downloaded and extracted successfully!")