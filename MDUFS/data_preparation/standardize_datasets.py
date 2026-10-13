import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler


# =====================================================
# Path
# =====================================================

# 原始数据目录
RAW_DIR = "../raw"

# 处理后数据保存目录
PROCESSED_DIR = "../processed"



# =====================================================
# 数据集文件名
# =====================================================

DATASETS = [
    "COIL20.mat",
    # "ISOLET.mat",
    # "USPS.mat",
    # "PIE.mat",
    # "LUNG.mat",
    # "MSTAR.mat",
    # "MNIST.mat",
    # "HAR.mat",
]



# =====================================================
# 统一读取 feature 和 label
# =====================================================

def find_feature_label(data, mat_file):
    """
    统一把不同变量名的数据转换成 X 和 Y。

    支持：
    1) X / Y
    2) fea / gnd
    """


    if "X" in data and "Y" in data:

        X = data["X"]
        Y = data["Y"]


    elif "fea" in data and "gnd" in data:

        X = data["fea"]
        Y = data["gnd"]


    else:

        keys = [
            k for k in data.keys()
            if not k.startswith("__")
        ]

        raise ValueError(
            f"{mat_file}: cannot find feature/label variables. "
            f"Available keys: {keys}. "
            f"Expected X/Y or fea/gnd."
        )


    return X.astype(np.float32), Y



# =====================================================
# 标准化处理
# =====================================================

def standardize_mat(dataset_name):


    # -----------------------------
    # 输入路径
    # -----------------------------

    mat_file = os.path.join(
        RAW_DIR,
        dataset_name
    )


    if not os.path.exists(mat_file):

        print(
            f"skip: {mat_file} not found"
        )

        return



    # 读取mat

    data = sio.loadmat(
        mat_file
    )


    X_raw, Y = find_feature_label(
        data,
        mat_file
    )



    # -----------------------------
    # Z-score标准化
    # -----------------------------

    scaler = StandardScaler()

    X_std = scaler.fit_transform(
        X_raw
    ).astype(np.float32)



    # -----------------------------
    # 创建输出目录
    # -----------------------------

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )



    # -----------------------------
    # 输出文件
    # -----------------------------

    out_file = os.path.join(

        PROCESSED_DIR,

        os.path.splitext(dataset_name)[0]
        + "_std.mat"

    )



    # 保存统一格式 X/Y

    save_dict = {

        "X": X_std,

        "Y": Y

    }



    sio.savemat(
        out_file,
        save_dict
    )



    print(
        f"{dataset_name:25s} "
        f"X={X_raw.shape} "
        f"Y={Y.shape} "
        f"saved -> {out_file}"
    )



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":


    for f in DATASETS:

        standardize_mat(f)