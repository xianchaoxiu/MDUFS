import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(x: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(x))))


class MDUFSLayer(nn.Module):
    def __init__(self, init_eta=1e-2, init_alpha=1.0, init_beta=1.0, init_lambda=1e-3, init_mu=1e-3, eps=1e-8):
        super().__init__()
        self.eps = eps

        # 可学习参数
        self.raw_eta = nn.Parameter(inv_softplus(init_eta))
        self.raw_alpha = nn.Parameter(inv_softplus(init_alpha))
        self.raw_beta  = nn.Parameter(inv_softplus(init_beta))
        self.raw_lambda = nn.Parameter(inv_softplus(init_lambda))
        self.raw_mu = nn.Parameter(inv_softplus(init_mu))

    @property
    def eta(self):
        return F.softplus(self.raw_eta) + self.eps

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha) + self.eps

    @property
    def beta(self):
        # Penalty parameter for constraint X = Z (MDUFS_Net+ uses a separate beta)
        return F.softplus(self.raw_beta) + self.eps

    @property
    def lam(self):
        lam_min = 0.001
        return lam_min + F.softplus(self.raw_lambda)

    @property
    def mu(self):
        mu_min = 0.0095
        return mu_min + F.softplus(self.raw_mu)
    def update_X(self, X, Y, Lambda, Z, Pi, G):
        alpha = self.alpha
        beta = self.beta
        eta = self.eta

        # 计算辅助变量 T_k 和 U_k (MDUFS_Net++)
        #   T_k = Y - Lambda/alpha  (for constraint X = Y)
        #   U_k = Z - Pi/beta       (for constraint X = Z)
        T_k = Y - Lambda / alpha
        U_k = Z - Pi / beta

        grad_euclidean = - (G @ X) + alpha * (X - T_k) + beta * (X - U_k)

        Xt_grad = X.t() @ grad_euclidean
        grad_riemann = grad_euclidean - X @ Xt_grad

        X_temp = X - eta * grad_riemann

        try:
            Q, _ = torch.linalg.qr(X_temp)
        except RuntimeError:
            X_cpu = X_temp.detach().cpu()
            X_cpu = X_cpu + 1e-5 * torch.randn_like(X_cpu)
            Q_cpu, _ = torch.linalg.qr(X_cpu)
            Q = Q_cpu.to(X.device)

        return Q

    def update_Y(self, X, Lambda):
        alpha = self.alpha
        lam = self.lam
        V = X + Lambda / alpha
        row_norms = torch.norm(V, p=2, dim=1, keepdim=True)
        tau = torch.sqrt(2.0 * lam / alpha)

        mask_hard = (row_norms > tau).float()
        scale = 10.0
        mask_soft = torch.sigmoid(scale * (row_norms - tau))
        mask = mask_hard + mask_soft - mask_soft.detach()

        Y_new = V * mask
        return Y_new

    def update_Z(self, X, Pi):
        """Element-wise L0 hard-thresholding update for Z (with straight-through estimator)."""
        mu = self.mu
        beta = self.beta
        W = X + Pi / beta

        # Element-wise threshold (L0 on entries)
        absW = torch.abs(W)
        tau = torch.sqrt(2.0 * mu / beta)

        mask_hard = (absW > tau).float()
        scale = 10.0
        mask_soft = torch.sigmoid(scale * (absW - tau))
        mask = mask_hard + mask_soft - mask_soft.detach()

        Z_new = W * mask
        return Z_new

    def update_Lambda(self, X, Y, Lambda):
        """Dual ascent for constraint X = Y."""
        alpha = self.alpha
        return Lambda + alpha * (X - Y)

    def update_Pi(self, X, Z, Pi):
        """Dual ascent for constraint X = Z."""
        beta = self.beta
        return Pi + beta * (X - Z)


    def forward(self, X, Y, Lambda, Z, Pi, G):
        X_new = self.update_X(X, Y, Lambda, Z, Pi, G)
        Y_new = self.update_Y(X_new, Lambda)
        Z_new = self.update_Z(X_new, Pi)
        Lambda_new = self.update_Lambda(X_new, Y_new, Lambda)
        Pi_new = self.update_Pi(X_new, Z_new, Pi)
        return X_new, Y_new, Z_new, Lambda_new, Pi_new


# ==========================================
# 3. 深度展开网络：MDUFS_Net+
# ==========================================

class MDUFS_Net_plus(nn.Module):
    def __init__(self, d, m, num_stages=10):
        super().__init__()
        self.d = d
        self.m = m
        self.layers = nn.ModuleList([
            MDUFSLayer(
                init_eta=0.01,
                init_alpha=1.0,
                init_beta=1.0,
                init_lambda=0.08,
                init_mu=0.1
            ) for _ in range(num_stages)
        ])

        self.X0 = nn.Parameter(torch.randn(d, m) * 0.01)

        self.final_X = None
        self.final_Y = None
        self.final_Z = None

    def _init_orthogonal(self, X):
        if not torch.isfinite(X).all():
            X = torch.randn_like(X) * 0.01
        Q, _ = torch.linalg.qr(X)
        return Q

    def forward(self, G):
        X = self.X0
        X = self._init_orthogonal(X)
        Y = X.clone()
        Z = X.clone()
        Lambda = torch.zeros_like(X)
        Pi = torch.zeros_like(X)

        for layer in self.layers:
            X, Y, Z, Lambda, Pi = layer(X, Y, Lambda, Z, Pi, G)

        self.final_X = X
        self.final_Y = Y
        self.final_Z = Z
        return X, Y, Z


