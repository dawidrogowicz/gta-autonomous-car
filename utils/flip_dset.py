import numpy as np
import cv2
import h5py
from keras.utils.io_utils import HDF5Matrix

batch_size = 256
data_path = 'data/training_data.hdf5'
data_dest_path = 'data/training_data.hdf5'
x_name = 'state'
y_name = 'action'
state_dim = (60, 80, 1)
action_dim = (1,)


def save_batch(x, y):
    assert len(x) == len(y) == batch_size

    with h5py.File(data_dest_path) as f:
        if x_name not in f:
            f.create_dataset(x_name, data=x, chunks=(batch_size, *state_dim),
                             maxshape=(None, *state_dim)),
        else:
            f[x_name].resize(f[x_name].shape[0] + batch_size, axis=0)
            f[x_name][-batch_size:] = x

        if y_name not in f:
            f.create_dataset(y_name, data=y, chunks=(batch_size, *action_dim),
                             maxshape=(None, *action_dim)),
        else:
            f[y_name].resize(f[y_name].shape[0] + batch_size, axis=0)
            f[y_name][-batch_size:] = y

        assert f[x_name].shape[0] == f[y_name].shape[0]


def read_batch():
    x = HDF5Matrix(data_path, x_name)
    y = HDF5Matrix(data_path, y_name)
    assert x.end == y.end
    for i in range(0, x.end, batch_size):
        batch_end = len if i + batch_size > x.end else i + batch_size
        yield x[i:batch_end], y[i:batch_end]


def process_batch(x, y):
    # save_batch(x, y)
    x = [cv2.flip(img, 1) for img in x]
    y = [-angle for angle in y]
    x = np.expand_dims(x, -1)
    # y = np.expand_dims(y, -1)
    save_batch(x, y)


if __name__ == '__main__':
    for x, y in read_batch():
        process_batch(x, y)
    with h5py.File(data_path) as f:
        print(np.shape(f[x_name]))
        print(np.shape(f[y_name]))
    with h5py.File(data_dest_path) as f:
        print(np.shape(f[x_name]))
        print(np.shape(f[y_name]))
