import os
import torch
import scipy.io as sio
import numpy as np


def load_data(data_path):

    if not os.path.exists(data_path):

        raise FileNotFoundError(
            f"Data file {data_path} not found."
        )


    data = sio.loadmat(data_path)


    X_raw = data["X"].astype(np.float32)

    Y_label = data["Y"]


    # 保持和原始代码一致
    # 不进行标准化
    X_scaled = X_raw


    # feature × sample
    A = torch.FloatTensor(
        X_raw.T
    )


    n_classes = len(
        np.unique(Y_label)
    )


    return (
        A,
        n_classes,
        Y_label,
        X_scaled
    )