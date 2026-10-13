import os
import random

import torch
import numpy as np
import scipy.io as sio


from models import (
    MDUFS_Net,
    MDUFS_Net_plus
)


from utils.data import load_data


from utils.loss import (
    calculate_reconstruction_loss,
    calculate_l20_surrogate,
    calculate_l0_surrogate,
    calculate_orth_loss
)



# ==========================================================
# Random seed
# ==========================================================

def set_seed(seed=2023, deterministic=True):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


    if deterministic:

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False




# ==========================================================
# Train model
# ==========================================================

def train_model(
        model_type,
        file_path,
        num_stages=15,
        num_epochs=100,
        lr=1e-4,
        device="cuda"):



    file_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]


    print(
        f"Processing: {file_name}"
    )



    # ==========================
    # Load data
    # ==========================

    A, n_classes, Y_label, X_scaled = load_data(
        file_path
    )


    d, n = A.shape


    A = A.to(device)



    print(
        "Computing Gram Matrix..."
    )


    G = (
        A @ A.t()
    ).to(device)



    # ==========================
    # Model
    # ==========================


    if model_type == "MDUFS":


        model = MDUFS_Net(
            d,
            n_classes,
            num_stages=num_stages
        )


    elif model_type == "MDUFS_plus":


        model = MDUFS_Net_plus(
            d,
            n_classes,
            num_stages=num_stages
        )


    else:

        raise ValueError(
            f"Unknown model type {model_type}"
        )



    model = model.to(device)



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )



    print(
        f"Training {model_type}"
    )



    # ======================================================
    # Training loop
    # ======================================================


    for epoch in range(num_epochs):


        model.train()


        optimizer.zero_grad()



        # forward

        if model_type == "MDUFS":


            X_pred, Y_pred = model(G)


        else:


            X_pred, Y_pred, Z_pred = model(G)



        # reconstruction

        loss_recon = calculate_reconstruction_loss(
            A,
            X_pred
        )


        layer = model.layers[-1]


        sigma = max(
            0.10,
            0.50 *
            (
                1.0 -
                epoch /
                max(1, num_epochs-1)
            )
        )



        loss_sparse_y = (
            layer.lam *
            calculate_l20_surrogate(
                Y_pred,
                sigma=sigma
            )
        )



        w_sparse = (
            0.3 *
            min(
                1.0,
                epoch/10
            )
        )



        if model_type == "MDUFS_plus":


            loss_sparse_z = (
                layer.mu *
                calculate_l0_surrogate(
                    Z_pred,
                    sigma=sigma
                )
            )


            loss = (
                loss_recon
                +
                w_sparse *
                (
                    loss_sparse_y
                    +
                    loss_sparse_z
                )
            )


        else:


            loss = (
                loss_recon
                +
                w_sparse *
                loss_sparse_y
            )



        loss.backward()



        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()



        if (epoch+1)%10==0:


            orth = calculate_orth_loss(
                X_pred
            )


            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss:{loss.item():.4f} "
                f"| Orth:{orth.item():.2e}"
            )



    print(
        "Training finished."
    )


    return model, G





# ==========================================================
# Save result
# ==========================================================


def evaluate_and_save(
        model,
        model_type,
        G,
        file_name):


    model.eval()


    with torch.no_grad():


        if model_type=="MDUFS":


            X_final,Y_final=model(G)


            save_dict={

                "X":
                X_final.cpu().numpy(),


                "Y":
                Y_final.cpu().numpy()

            }



        else:


            X_final,Y_final,Z_final=model(G)


            save_dict={

                "X":
                X_final.cpu().numpy(),


                "Y":
                Y_final.cpu().numpy(),


                "Z":
                Z_final.cpu().numpy()

            }



    save_name = (
        f"{file_name}_{model_type}.mat"
    )


    sio.savemat(
        save_name,
        save_dict
    )


    print(
        f"Saved: {save_name}"
    )