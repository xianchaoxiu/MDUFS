import torch



# ==========================================================
# Reconstruction loss
# ==========================================================

def calculate_reconstruction_loss(A, X):

    """
    ||A - XX^T A||_F^2
    """

    XTA = X.t() @ A

    recon = X @ XTA

    diff = A - recon

    loss = torch.norm(
        diff,
        p='fro'
    ) ** 2

    return loss



# ==========================================================
# Row-wise L2,0 surrogate
# For MDUFS-Net
# ==========================================================

def calculate_l20_surrogate(
        Y,
        sigma=0.5,
        eps=1e-12):

    """
    Smooth approximation of ||Y||_{2,0}

    Row sparsity:
    each row corresponds to a feature
    """

    row_sq = torch.sum(
        Y * Y,
        dim=1
    )

    denom = (
        2.0 * sigma ** 2
        + eps
    )

    return torch.sum(
        1.0 - torch.exp(
            -row_sq / denom
        )
    )



# ==========================================================
# Element-wise L0 surrogate
# For MDUFS-Net+
# ==========================================================

def calculate_l0_surrogate(
        Z,
        sigma=0.5,
        eps=1e-12):

    """
    Smooth approximation of element-wise L0
    """

    sq = Z * Z


    denom = (
        2.0 * sigma ** 2
        + eps
    )


    return torch.sum(
        1.0 - torch.exp(
            -sq / denom
        )
    )



# ==========================================================
# Orthogonality monitoring
# ==========================================================

def calculate_orth_loss(X):

    """
    ||X^T X-I||_F^2

    Only for monitoring,
    not included in total loss.
    """

    m = X.shape[1]


    I = torch.eye(
        m,
        device=X.device,
        dtype=X.dtype
    )


    XtX = X.t() @ X


    return torch.norm(
        XtX - I,
        p='fro'
    ) ** 2