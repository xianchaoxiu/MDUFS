import os
import torch

from train import (
    set_seed,
    train_model,
    evaluate_and_save
)


def demo():

    # ==========================
    # 1. Set seed
    # ==========================

    set_seed(2023)


    # ==========================
    # 2. Device
    # ==========================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    # ==========================
    # 3. Dataset
    # ==========================

    data_file = os.path.join(
        "datasets",
        "COIL20_std1.mat"
    )


    # ==========================
    # 4. Train model
    # ==========================

    model, G = train_model(

        model_type="MDUFS",

        file_path=data_file,

        num_stages=5,

        num_epochs=20,

        lr=1e-4,

        device=device
    )


    # ==========================
    # 5. Save learned representation
    # ==========================

    evaluate_and_save(

        model=model,

        model_type="MDUFS",

        G=G,

        file_name="demo_result"

    )


    print(
        "Demo finished!"
    )



if __name__=="__main__":

    demo()