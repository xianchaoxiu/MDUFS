import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(x: float) -> torch.Tensor:
    """Softplus的逆函数，用于参数初始化"""
    return torch.log(torch.expm1(torch.tensor(float(x))))


class MDUFSLayer(nn.Module):
    def __init__(self, init_eta=1e-2, init_alpha=1.0, init_lambda=1e-3, eps=1e-8):
        super().__init__()
        self.eps = eps

        # 可学习参数
        self.raw_eta = nn.Parameter(inv_softplus(init_eta))
        self.raw_alpha = nn.Parameter(inv_softplus(init_alpha))
        self.raw_lambda = nn.Parameter(inv_softplus(init_lambda))

    @property
    def eta(self):
        return F.softplus(self.raw_eta) + self.eps

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha) + self.eps

    @property
    def lam(self):
        lam_min = 0.001
        return lam_min + F.softplus(self.raw_lambda)

    def update_X(self, X, Y, Lambda, G):

        alpha = self.alpha
        eta = self.eta

        # 1. 计算辅助变量 V_k
        V_k = Y - Lambda / alpha

        # 2. 计算欧氏梯度 (Euclidean Gradient)
        # ∇F(X) = -GX + alpha * (X - V_k)
        grad_euclidean = - (G @ X) + alpha * (X - V_k)

        # 3.
        Xt_grad = X.t() @ grad_euclidean
        grad_riemann = grad_euclidean - X @ Xt_grad

        # 4. 沿负梯度方向更新 (Gradient Descent Step)
        X_temp = X - eta * grad_riemann

        # 5. 回缩操作 (Retraction)
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

    def update_Lambda(self, X, Y, Lambda):
        alpha = self.alpha
        Lambda_new = Lambda + alpha * (X - Y)
        return Lambda_new

    def forward(self, X, Y, Lambda, G):
        X_new = self.update_X(X, Y, Lambda, G)
        Y_new = self.update_Y(X_new, Lambda)
        Lambda_new = self.update_Lambda(X_new, Y_new, Lambda)
        return X_new, Y_new, Lambda_new


# ==========================================
# 3. 深度展开网络：MDUFS_Net
# ==========================================

class MDUFS_Net(nn.Module):
    def __init__(self, d, m, num_stages=10):
        super().__init__()
        self.d = d
        self.m = m
        self.layers = nn.ModuleList([
            MDUFSLayer(
                init_eta=0.01,
                init_alpha=1.0,
                init_lambda=0.08
            ) for _ in range(num_stages)
        ])

        # 初始化为正交矩阵 (使用 QR 分解随机矩阵)
        self.X0 = nn.Parameter(torch.randn(d, m) * 0.01)

        self.final_X = None
        self.final_Y = None

    def _init_orthogonal(self, X):
        """确保初始 X0 是正交的"""
        if not torch.isfinite(X).all():
            X = torch.randn_like(X) * 0.01
        Q, _ = torch.linalg.qr(X)
        return Q

    def forward(self, G):
        X = self.X0
        X = self._init_orthogonal(X)  # 强制初始正交
        Y = X.clone()
        Lambda = torch.zeros_like(X)

        for layer in self.layers:
            X, Y, Lambda = layer(X, Y, Lambda, G)

        self.final_X = X
        self.final_Y = Y
        return X, Y


