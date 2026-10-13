import os
import torch


from train import (
    set_seed,
    train_model,
    evaluate_and_save
)



if __name__ == "__main__":


    # ==========================
    # Random seed
    # ==========================

    set_seed(2023)



    # ==========================
    # Device
    # ==========================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Using device: {device}"
    )



    # ==========================
    # Model
    # ==========================

    model_type = "MDUFS"



    # ==========================
    # Dataset
    # ==========================

    data_file = os.path.join(
        "datasets",
        "COIL20_std1.mat"
    )



    # ==========================
    # Train
    # ==========================

    model, G = train_model(

        model_type=model_type,

        file_path=data_file,

        num_stages=15,

        num_epochs=100,

        lr=1e-4,

        device=device

    )



    # ==========================
    # Save result
    # ==========================

    evaluate_and_save(

        model=model,

        model_type=model_type,

        G=G,

        file_name=os.path.splitext(
            os.path.basename(data_file)
        )[0]

    )